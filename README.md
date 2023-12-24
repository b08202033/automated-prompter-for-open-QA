# automated-prompter-for-open-QA
This is the main repository for RL course final project.

## Environment
We provide our environment in ``requirements.txt`` and ``environment.yml``. You can install through any way you want. For example:

``` pip install -r requirements.txt```

## Supervised fine-tuning
- finetune.py
  This file is for doing supervised finetuning on prompter model, the result of this model will generate the instruction format when we give it certain prompt.
- alpaca_data_pro_ins_25.json
  This file is the data for SFT, which is preprocessed from Alpaca dataset, and we choose the first 10% to do SFT.

## Reinforcement learning for (d) and (e)
- training_eval_for_(d).py
  This file is for training (d) and it will evaluate the result of (d).
- training_for_(e).py
  This file is for training (e)
## Result
After SFT and RL,

- result_of_(a).py is to evaluate prompter (a).
- (b) (c) (e) is the same.

## Execution
All the code can be executed by direct running. For example:

```python finetune.py```
