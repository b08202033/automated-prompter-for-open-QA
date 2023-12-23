import os
import sys
import argparse
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, load_from_disk
import transformers, datasets
from peft import PeftModel
from colorama import *

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import GenerationConfig
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training
)

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# dataset = load_dataset("xiyuez/im-feeling-curious")
dataset = load_dataset("xiyuez/im-feeling-curious", split="train[95%:]").rename_column("question","query")

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)

# 生成輸入提示，包含指令、輸入和回覆信息（若有輸入）
def generate_prompt(data_point):
    if data_point["input"]:
        return f"""下方是一個關於任務的指令，以及一個提供與任務相關之資訊的輸入。請撰寫一個能適當地完成該任務指令需求的回覆。
### 指令:
{data_point["instruction"]}
### 輸入:
{data_point["input"]}
### 回覆:
{data_point["output"]}"""
    else:
        return f"""下方是一個關於任務的指令。請撰寫一個能適當地完成該任務指令需求的回覆。
### 輸入:
{data_point["instruction"]}
### 回覆:
{data_point["output"]}"""

# def generate_prompt(data_point):
#     if data_point["input"]:
#         return ("以下是一個描述任務的指令，以及一個與任務資訊相關的輸入。請撰寫一個能適當完成此任務指令的回覆\n\n"
#         f'### 指令：\n{data_point["instruction"]}\n\n### 輸入：\n{data_point["input"]}\n\n'
#         f'### 回覆：\n{data_point["output"]}')
#     else:
#         return ("以下是一個描述任務的指令。請撰寫一個能適當完成此任務指令的回覆\n\n"
#         f'### 指令：\n{data_point["instruction"]}\n\n### 回覆：\n{data_point["output"]}')

# 將提示文本轉換成模型所需的數字表示形式
def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }

# 生成推理用的提示文本，包含指令和輸入（若有）
def generate_prompt_inference(instruction, input=None):
    if input:
        return f"""下方是一個關於任務的指令，以及一個提供與任務相關之資訊的輸入。請撰寫一個能適當地完成該任務指令需求的回覆。
### 指令:
{instruction}

### 輸入:
{input}

### 回覆:"""
    else:
        return f"""{instruction}"""

# def generate_prompt_inference(instruction, input=None):
#     if input:
#         return f"""以下是一個描述任務的指令，以及一個與任務資訊相關的輸入。請撰寫一個能適當完成此任務指令的回覆\n\n
# ### 指令:
# {instruction}

# ### 輸入:
# {input}

# ### 回覆:"""
#     else:
#         return f"""以下是一個描述任務的指令。請撰寫一個能適當完成此任務指令的回覆\n\n
# ### 輸入:
# {instruction}

# ### 回覆:"""

# 進行生成回覆的評估
def evaluate(instruction, generation_config, max_len, input=None):
    # 根據指令和輸入生成提示文本
    prompt = generate_prompt_inference(instruction, input)
    # 將提示文本轉換為模型所需的數字表示形式
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    # 使用模型進行生成回覆
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_len,
    )
    # 將生成的回覆解碼並印出
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        # print(output)
        # print(f"{Fore.GREEN}回覆:{Style.RESET_ALL}")
        # print(output.split("### 回覆:")[1].strip() + '\n')
    return output.strip()

model_name = "meta-llama/Llama-2-7b-chat-hf"  # 設定模型名稱或路徑
# ckpt_name = "./lora_RL"  # 從特定 checkpoint 載入模型權重的檔案名稱 (你也可以選擇不同step的checkpoint)
cache_dir = "./cache"  # 設定快取目錄路徑

seed = 42  # 設定隨機種子，用於重現結果
max_len = 256
temperature = 0.1  # 設定生成回覆的隨機度，值越小生成的回覆越穩定
top_p = 0.3  # Top-p (nucleus) 抽樣的機率閾值，用於控制生成回覆的多樣性
num_beams = 1  # 設定束搜索 (beam search) 的束寬
no_repeat_ngram_size = 3  # 設定禁止重複 Ngram 的大小，用於避免生成重複片段

# 使用 tokenizer 將模型名稱轉換成模型可讀的數字表示形式
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=False,
    device_map={'': 0}, 
    cache_dir=cache_dir
)

# 從指定的 checkpoint 載入模型權重
# model = PeftModel.from_pretrained(model, ckpt_name, device_map={'': 0})

model.bfloat16()

# for name, param in model.named_parameters():
#     print(name, param.data)

# 設定生成配置，包括隨機度、束搜索等相關參數
generation_config = GenerationConfig(
    do_sample=False,
    temperature=temperature,
    num_beams=num_beams,
    top_p=top_p,
    no_repeat_ngram_size=no_repeat_ngram_size,
)

# Initialize variables for F1 score calculation
total_f1_score = 0.0
total_samples = 0

for data_point in tqdm(dataset, desc="Processing testing dataset"):
    instruction = data_point["query"]
    instruction = instruction

    # Generate response
    generated_response = evaluate(instruction, generation_config, max_len)

    # print(generated_response)

    # Get the ground truth response from the dataset
    ground_truth_response = data_point["answer"]

    # Calculate F1 score for the generated response
    f1_score = compute_f1(generated_response, ground_truth_response)

    # Print and accumulate F1 score
    print(f"{Fore.BLUE}Ground Truth: {ground_truth_response}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Generated Response: {generated_response}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}F1 Score: {f1_score}{Style.RESET_ALL}\n")

    total_f1_score += f1_score
    total_samples += 1

# Calculate and print the average F1 score
avg_f1_score = total_f1_score / total_samples if total_samples > 0 else 0
print(f"{Fore.CYAN}Average F1 Score on Testing Dataset: {avg_f1_score}{Style.RESET_ALL}")


# 進入無限迴圈，等待使用者輸入指令並進行生成回覆
# while(True):
#     evaluate(input(f"\n{'-'*10}\n{Fore.BLUE}指令: {Style.RESET_ALL}"), generation_config, max_len)

# output_results = []
# for data_point in tqdm(dataset["train"], desc="Processing dataset"):
#     instruction = data_point["question"]
#     instruction = "input: " + instruction + " instruction: Question Answering"
#     output = evaluate(instruction, generation_config, max_len)
#     # print(output)
#     output_results.append({"input": instruction, "generated": output})

# with open("instruct_QA_new_ins.json", "w", encoding="utf-8") as json_file:
#     json.dump(output_results, json_file, ensure_ascii=False, indent=4)