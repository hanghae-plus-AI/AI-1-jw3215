import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from typing import cast
from peft.mapping import get_peft_model
from peft.utils.peft_types import TaskType
from peft.tuners.lora.config import LoraConfig
import wandb

MODEL_NAME = "facebook/opt-350m"
DATASET_NAME = "lucasmccabe-lmi/CodeAlpaca-20k"

# LoRA 파라미터
lora_r: int = 8
# lora_r: int = 128
# lora_r: int = 256
lora_dropout: float = 0.1
lora_alpha: int = 32

MAX_LENGTH = 300

FP16_FLAG = False
FP_LABEL = "fp16" if FP16_FLAG else "fp32"
project_name = "lightweight-model"
wandb.init(project=project_name, name=f"faster-TA-{FP_LABEL}")

dataset = load_dataset("json", data_files="./corpus.json", split="train")
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["note"])):
        text = f"### Question: {example['note'][i]}\n ### Answer: {example['expectation'][i]}"
        output_texts.append(text)
    return output_texts


response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

target_modules = set()

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        names = name.split(".")
        target_modules.add(names[0] if len(names) == 1 else names[-1])

if "lm_head" in target_modules:
    target_modules.remove("lm_head")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, peft_config)

# Trainer 설정 및 실행
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir=f"/tmp/{project_name}/{FP_LABEL}",
        max_seq_length=MAX_LENGTH,
        fp16=FP16_FLAG,
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)
trainer.train()


trainer.evaluate()
wandb.finish()
