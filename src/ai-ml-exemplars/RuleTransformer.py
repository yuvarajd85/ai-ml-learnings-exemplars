'''
Created on 2/28/2025 at 1:28 AM
By yuvaraj
Module Name: RuleTransformer
'''
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Load dataset
df = pd.read_csv(f"../resources/datasets/rule_train_dataset.csv")
print(df.head())

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


# Prepare dataset
class DQDLDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = self.data.iloc[idx, 0]
        target_text = self.data.iloc[idx, 1]

        source = self.tokenizer(
            source_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        target = self.tokenizer(
            target_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": target["input_ids"].squeeze(),
        }


# Create dataset and dataloader
dataset = DQDLDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print(f"Starting to train the model")
# Training
epochs = 3
for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}")
    model.train()
    print(f"training executed")
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Save the model
model.save_pretrained("dqdl_parser_model")
tokenizer.save_pretrained("dqdl_parser_model")

print("Model training complete!")
