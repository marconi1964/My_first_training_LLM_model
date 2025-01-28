import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from datasets import load_dataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 加載資料集（WikiText-103）
dataset = load_dataset("wikitext", "wikitext-103-v1")

def filter_empty(example):
    return example["text"] != "" and len(example["text"]) > 50

dataset = dataset.filter(filter_empty)
train_dataset = dataset["train"].select(range(len(dataset["train"]) // 10))
valid_dataset = dataset["validation"]

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 確保使用 GPT2LMHeadModel（帶有語言模型頭）
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt"
    )
    # 明確添加 labels 欄位（與 input_ids 相同）
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

# 分詞處理
train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=8).select(range(1000))
valid_dataset = valid_dataset.map(tokenize_function, batched=True, batch_size=8).select(range(200))

# 修正後的 TrainingArguments
training_args = TrainingArguments(
    output_dir="./gpt2_output",
    evaluation_strategy="epoch",    # 舊版參數名（若 transformers<4.31）
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_total_limit=1,
    logging_steps=10,
    load_best_model_at_end=True,
    use_mps_device=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# 開始訓練
trainer.train()

# 保存模型
model.save_pretrained("./gpt2_finetuned")
tokenizer.save_pretrained("./gpt2_finetuned")
