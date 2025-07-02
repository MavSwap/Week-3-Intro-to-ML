# sentiment_pipeline.py

import torch
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
import evaluate
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the IMDb dataset
dataset = load_dataset("imdb")

# 2. Preprocess the dataset with the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# 3. Define the model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

# 5. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=100,
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"].shuffle(seed=42).select(range(20000)),
    eval_dataset=encoded_dataset["test"].select(range(5000)),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 7. Train the model
trainer.train()

# 8. Save the model
model_path = "./sentiment-bert-imdb"
os.makedirs(model_path, exist_ok=True)
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# 9. Load model for inference and test on sample input
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

