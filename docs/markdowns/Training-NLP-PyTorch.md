### **🚀 Hands-On Project: Training an NLP Model with PyTorch**
In this project, we’ll build a simple **text classification model** using **PyTorch and an LSTM (Long Short-Term Memory)** network. The model will classify movie reviews as **positive** or **negative** using the **IMDb dataset**.

---

## **1️⃣ Install Required Libraries**
If you haven’t installed them yet, run:
```bash
pip install torch torchtext numpy nltk
```

---

## **2️⃣ Import Dependencies**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.legacy import data, datasets
import random
import nltk
nltk.download('punkt')
```

---

## **3️⃣ Load and Preprocess IMDb Dataset**
We will use `torchtext` to load and process the **IMDb movie review dataset**.
```python
SEED = 42
torch.manual_seed(SEED)

# Define Fields for text (X) and labels (Y)
TEXT = data.Field(tokenize="spacy", lower=True, include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)

# Load IMDb dataset
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# Split train data into training & validation (80-20)
train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))

print(f"Train size: {len(train_data)}, Validation size: {len(valid_data)}, Test size: {len(test_data)}")
```

---

## **4️⃣ Build Vocabulary and Load Word Embeddings**
We use **GloVe (Global Vectors for Word Representation)** for embedding words.
```python
# Build vocabulary using pre-trained word embeddings (GloVe)
TEXT.build_vocab(train_data, max_size=25_000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

# Define batch size and create iterators
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=device
)
```

---

## **5️⃣ Define the LSTM Model**
We create an **LSTM-based classifier** for text sentiment analysis.
```python
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(SentimentLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)
```

---

## **6️⃣ Initialize Model, Loss, and Optimizer**
```python
# Define model parameters
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
DROPOUT = 0.5

model = SentimentLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)

# Load pre-trained embeddings
model.embedding.weight.data.copy_(TEXT.vocab.vectors)

# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Move to GPU if available
model = model.to(device)
criterion = criterion.to(device)
```

---

## **7️⃣ Train the Model**
```python
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss, epoch_acc = 0, 0

    for batch in iterator:
        optimizer.zero_grad()
        
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# Train for 5 epochs
N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
```

---

## **8️⃣ Evaluate the Model**
```python
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

test_loss = evaluate(model, test_iterator, criterion)
print(f"Test Loss = {test_loss:.4f}")
```

---

## **9️⃣ Save and Load the Model**
```python
# Save model
torch.save(model.state_dict(), "sentiment_lstm.pth")

# Load model
model.load_state_dict(torch.load("sentiment_lstm.pth"))
```

---

### **🎯 Final Summary**
| Step | Code |
|------|------|
| Load IMDb Data | `datasets.IMDB.splits(TEXT, LABEL)` |
| Preprocess Text | `TEXT.build_vocab(train_data, vectors="glove.6B.100d")` |
| Define LSTM Model | `class SentimentLSTM(nn.Module)` |
| Train Model | `train(model, train_iterator, optimizer, criterion)` |
| Evaluate Model | `evaluate(model, test_iterator, criterion)` |
| Save Model | `torch.save(model.state_dict(), "sentiment_lstm.pth")` |

### **🚀 Next Steps**
✅ Fine-tune **BERT for NLP classification**  
✅ Use **Transformer models (like GPT-3, T5) with Hugging Face**  
✅ Deploy the model using **TorchServe**  

Would you like a tutorial on **fine-tuning transformers (BERT, GPT, etc.)**? 😊