'''
Created on 4/26/2025 at 2:27 AM
By yuvaraj
Module Name: DQDLRuleGenerator
'''
from dotenv import load_dotenv

load_dotenv()


import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# Sample data from your table
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

# Combine inputs into a single string sequence
df["input_seq"] = df["Function"] + " " + df["Field"] + " " + df["Operation"] + " " + df["Value"]
df["output_seq"] = df["Output"]

# Label encode characters (or use tokenizer if you want words)
all_text = " ".join(df["input_seq"]) + " " + " ".join(df["output_seq"])
chars = sorted(set(all_text))
char2idx = {ch: i+1 for i, ch in enumerate(chars)}  # +1 for padding index 0
idx2char = {i: ch for ch, i in char2idx.items()}

def encode_text(text, max_len):
    return [char2idx[c] for c in text.ljust(max_len)]

class RuleDataset(Dataset):
    def __init__(self, df, max_len=50):
        self.x = df["input_seq"].tolist()
        self.y = df["output_seq"].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = encode_text(self.x[idx], self.max_len)
        y = encode_text(self.y[idx], self.max_len)
        return torch.tensor(x), torch.tensor(y)

dataset = RuleDataset(df)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model: Simple Encoder-Decoder with LSTM
class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_size, padding_idx=0)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size + 1)

    def forward(self, src, tgt):
        embed_src = self.embedding(src)
        _, (h, c) = self.encoder(embed_src)

        embed_tgt = self.embedding(tgt)
        out, _ = self.decoder(embed_tgt, (h, c))
        out = self.fc(out)
        return out

vocab_size = len(char2idx)
model = Seq2SeqModel(vocab_size)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    for x, y in loader:
        optimizer.zero_grad()
        y_input = y[:, :-1]
        y_target = y[:, 1:]

        out = model(x, y_input)
        loss = criterion(out.view(-1, vocab_size + 1), y_target.reshape(-1))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# Function to decode predictions
def predict(model, input_str, max_len=50):
    model.eval()
    input_encoded = torch.tensor([encode_text(input_str, max_len)])
    y_input = torch.tensor([[char2idx[' ']] * max_len])  # decoder input

    with torch.no_grad():
        out = model(input_encoded, y_input)
        pred = out.argmax(dim=-1).squeeze().tolist()
        decoded = "".join([idx2char[i] for i in pred if i in idx2char])
    return decoded.strip()

# Test the model
test_input = "Length port_id == 4"
print("Prediction:", predict(model, test_input))

