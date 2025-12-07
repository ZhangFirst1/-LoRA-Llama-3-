# src/train.py
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def train():
    # 1. 配置参数
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # 基座模型路径
    output_dir = "./model/lora_adapter"
    data_path = "./data/adgen_train.json"

    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. 加载基座模型 (4-bit 量化加载 - 显存优化关键)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_4bit=True,   # QLoRA 核心
        torch_dtype=torch.float16
    )

    # 4. 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                 # Rank
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() # 打印可训练参数量 (0.1%)

    # 5. 加载数据
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    def process_func(example):
        # 简单的数据处理逻辑 (Alpaca格式)
        instruction = example["instruction"]
        input_text = example["input"]
        output_text = example["output"]
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = tokenizer(prompt + output_text + "<|eot_id|>", return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    tokenized_ds = dataset.map(process_func)

    # 6. 配置训练参数
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4, # 梯度累积
        learning_rate=2e-4,            # 学习率
        num_train_epochs=5,
        logging_steps=10,
        fp16=True,                     # 混合精度
        save_strategy="epoch"
    )

    # 7. 开始训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    print("Training Completed! Model saved.")

if __name__ == "__main__":
    train()