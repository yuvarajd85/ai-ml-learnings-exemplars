'''
Created on 4/26/2025 at 2:39 AM
By yuvaraj
Module Name: DQDLRuleWordGenerator
'''
from dotenv import load_dotenv

load_dotenv()


import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Step 1: Data Preparation
data = {
    "Function": ["Length", "Value", "Exists"],
    "Field": ["port_id", "port_id", "effective_date"],
    "Operation": ["==", "not", "IN"],
    "Value": ["4", "null", "table_name"],
    "Output": [
        "ColumnLength `port_id` == 4",
        "ColumnValue `port_id` not null",
        "ColumnExists in table_name"
    ]
}

df = pd.DataFrame(data)

# Combine columns into input sequences
df["input_seq"] = df["Function"] + " " + df["Field"] + " " + df["Operation"] + " " + df["Value"]
df["output_seq"] = df["Output"]

# Create word vocab
all_words = set()
for text in df["input_seq"].tolist() + df["output_seq"].tolist():
    all_words.update(text.strip().split())

word2idx = {word: idx+2 for idx, word in enumerate(sorted(all_words))}
word2idx["<PAD>"] = 0
word2idx["<SOS>"] = 1
idx2word = {idx: word for word, idx in word2idx.items()}

vocab_size = len(word2idx)

def encode_words(text, max_len):
    tokens = text.strip().split()
    encoded = [word2idx.get(tok, 0) for tok in tokens]
    encoded = [1] + encoded  # prepend SOS token
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    return encoded[:max_len]

class RuleDataset(Dataset):
    def __init__(self, df, max_len=15):
        self.inputs = df["input_seq"].tolist()
        self.outputs = df["output_seq"].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = encode_words(self.inputs[idx], self.max_len)
        y = encode_words(self.outputs[idx], self.max_len)
        return torch.tensor(x), torch.tensor(y)

dataset = RuleDataset(df)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 2: Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src).transpose(0, 1)  # (seq_len, batch, embed)
        tgt_emb = self.embedding(tgt).transpose(0, 1)
        memory = self.transformer(src_emb, tgt_emb)
        output = self.fc(memory).transpose(0, 1)  # (batch, seq_len, vocab)
        return output

model = TransformerModel(vocab_size=vocab_size)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 3: Training Loop
for epoch in range(100):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        optimizer.zero_grad()
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output = model(src, tgt_input)
        loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {total_loss:.4f}")

# Step 4: Prediction function
def predict(model, input_text, max_len=15):
    model.eval()
    input_encoded = torch.tensor([encode_words(input_text, max_len)])
    decoder_input = torch.tensor([[1] + [0]*(max_len-1)])  # start with SOS

    with torch.no_grad():
        output = model(input_encoded, decoder_input)
        pred_tokens = output.argmax(dim=-1).squeeze().tolist()
        decoded = [idx2word[idx] for idx in pred_tokens if idx > 1]
    return " ".join(decoded)

# Test the model
test_input = "value port_id > 0000"
print("Prediction:", predict(model, test_input))
