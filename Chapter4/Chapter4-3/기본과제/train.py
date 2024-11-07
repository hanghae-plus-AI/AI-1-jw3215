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

project_name = "LoRA-instruction-tuning"
wandb.init(project=project_name, name=f"LoRA-rank-{lora_r}")

dataset = cast(Dataset, load_dataset(DATASET_NAME, split="train"))
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# 데이터셋의 상위 95%에 해당하는 최대 토큰 길이: 276
MAX_LENGTH = 300
tokenizer.model_max_length = MAX_LENGTH


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts


response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

target_modules = set()

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        names = name.split(".")
        target_modules.add(names[0] if len(names) == 1 else names[-1])

# print(target_modules) # {'project_out', 'v_proj', 'project_in', 'q_proj', 'fc2', 'fc1', 'k_proj', 'out_proj'}
# target_modules
# - q_proj, k_proj, v_proj: Attention의 Q, K, V 벡터 생성.
# - fc1, fc2: Feed-Forward Network로, Transformer 레이어 내 비선형성을 추가.
# - project_in, project_out: Transformer의 입력과 출력을 위한 차원 변환.
# - out_proj: Attention의 출력을 다음 레이어로 전달하기 위해 변환.
# Q와 V가 모델의 표현을 대표하므로, LoRA 목적에 부합함.

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
    train_dataset=dataset,
    args=SFTConfig(
        output_dir=f"/tmp/{project_name}/rank-{lora_r}", max_seq_length=MAX_LENGTH
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)
trainer.train()

# 메모리 점유율
print("Max Alloc:", round(torch.cuda.max_memory_allocated(0) / 1024**3, 1), "GB")

wandb.finish()
