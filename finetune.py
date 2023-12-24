import os
import sys
import argparse

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

os.environ['TRANSFORMERS_CACHE'] = '/work/guangming413/.cache'
seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

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

# 生成並轉換提示文本用於訓練數據
def generate_and_tokenize_prompt(data_point):
    user_prompt = (
        (
            f"""下方是一個關於任務的指令，以及一個提供與任務相關之資訊的輸入。請撰寫一個能適當地完成該任務需求的回覆。
### 指令:
{data_point["instruction"]}
### 輸入:
{data_point["input"]}
### 回覆:
"""
        )
        if data_point["input"]
        else (
            f"""下方是一個關於任務的指令。請撰寫一個能適當地完成該任務需求的回覆。
### 輸入:
{data_point["instruction"]}
### 回覆:
"""
        )
    )
    len_user_prompt_tokens = (
        len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
                padding="max_length",
            )["input_ids"]
        )
        - 1
    )
    full_tokens = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]


    # print(full_tokens)
    # print([-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:])
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
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
        return f"""下方是一個關於任務的指令。請撰寫一個能適當地完成該任務指令需求的回覆。
### 輸入:
{instruction}

### 回覆:"""

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
        print(f"{Fore.GREEN}回覆:{Style.RESET_ALL}")
        print(output.split("### 回覆:")[1].strip() + '\n')


model_name = "meta-llama/Llama-2-7b-chat-hf"  # 設定模型名稱或路徑facebook/opt-1.3b
cache_dir = "./cache"  # 設定快取目錄路徑
from_ckpt = False  # 是否從checkpoint載入模型的權重，預設為否
ckpt_name = None  # 從特定checkpoint載入權重時使用的檔案名稱，預設為無
dataset_dir = "./alpaca_data_pro_ins_25.json"  # 設定資料集的目錄或檔案路徑
output_dir = "./test_opt_lora_hd"  # 設定實驗結果輸出目錄
num_epoch = 3  # 設定訓練的總Epoch數
logging_steps = 20  # 定義訓練過程中每隔多少步驟輸出一次訓練日誌
save_steps = 20  # 定義訓練過程中每隔多少步驟保存一次模型
save_total_limit = 3  # 控制最多保留幾個模型checkpoint
report_to = None  # 設定上報實驗指標的目標，預設為無
MICRO_BATCH_SIZE = 4  # 定義微批次的大小
BATCH_SIZE = 16  # 定義一個批次的大小
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE  # 計算每個微批次累積的梯度步數
LEARNING_RATE = 3e-4  # 設定學習率
CUTOFF_LEN = 256  # 設定文本截斷的最大長度
LORA_R = 8  # 設定LORA（Layer-wise Random Attention）的R值
LORA_ALPHA = 16  # 設定LORA的Alpha值
LORA_DROPOUT = 0.05  # 設定LORA的Dropout率
VAL_SET_SIZE = 0  # 設定驗證集的大小，預設為無
# TARGET_MODULES = [
#     "q_proj",
#     "v_proj",
# ]  # 設定目標模組，這些模組的權重將被保存為checkpoint
TARGET_MODULES = [
    "q_proj",
    "up_proj",
    "o_proj",
    "k_proj",
    "down_proj",
    "gate_proj",
    "v_proj"
]  # 設定目標模組，這些模組的權重將被保存為checkpoint
device_map = "auto"  # 設定設備映射，預設為"auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))  # 獲取環境變數"WORLD_SIZE"的值，若未設定則預設為1
ddp = world_size != 1  # 根據world_size判斷是否使用分散式數據處理(DDP)，若world_size為1則不使用DDP
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float16
)

# 從指定的模型名稱或路徑載入預訓練的語言模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir
)

# 創建 tokenizer 並設定結束符號 (eos_token)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_eos_token=True,
    cache_dir=cache_dir
)

# # 從指定的模型名稱或路徑載入預訓練的語言模型
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     cache_dir=cache_dir,
#     quantization_config=nf4_config
# )

# # 創建 tokenizer 並設定結束符號 (eos_token)
# tokenizer = AutoTokenizer.from_pretrained(
#     model_name,
#     add_eos_token=True,
#     cache_dir=cache_dir,
#     quantization_config=nf4_config
# )

# 根據 from_ckpt 標誌，從 checkpoint 載入模型權重
if from_ckpt:
    model = PeftModel.from_pretrained(model, ckpt_name)

# 將模型準備好以使用 INT8 訓練
model = prepare_model_for_int8_training(model)

# 使用 LoraConfig 配置 LORA 模型
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

# 將 tokenizer 的 padding token 設定為 0
tokenizer.pad_token_id = 0

# 載入並處理訓練數據
data = load_dataset('json', data_files=dataset_dir, download_mode="force_redownload")

# 將訓練數據分為訓練集和驗證集（若 VAL_SET_SIZE 大於 0）
if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
else:
    train_data = data['train'].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

# 使用 Transformers Trainer 進行模型訓練
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=50,
        num_train_epochs=num_epoch,
        learning_rate=LEARNING_RATE,
        fp16=False,  # 使用混合精度訓練
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        ddp_find_unused_parameters=False if ddp else None,  # 是否使用 DDP，控制梯度更新策略
        report_to=report_to,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 禁用模型的 cache 功能
model.config.use_cache = False

# 若使用 PyTorch 2.0 版本以上且非 Windows 系統，進行模型編譯
if torch.__version__ >= "2" and sys.platform != 'win32':
    model = torch.compile(model)

# 開始模型訓練
trainer.train()

# 將訓練完的模型保存到指定的目錄中
model.save_pretrained(output_dir)

# 印出訓練過程中可能的缺失權重的警告信息
print("\n If there's a warning about missing keys above, please disregard :)")