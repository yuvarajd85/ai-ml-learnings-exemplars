### **🚀 Fine-Tuning BERT and GPT for NLP Tasks**
This tutorial will walk you through **fine-tuning BERT for text classification** and **GPT for text generation** using **Hugging Face's `transformers` library**. We’ll use **PyTorch** as our backend.

---

## **1️⃣ Install Required Libraries**
```bash
pip install torch transformers datasets nltk
```

---

## **2️⃣ Fine-Tuning BERT for Text Classification**
BERT (Bidirectional Encoder Representations from Transformers) is ideal for **text classification, sentiment analysis, and question answering**.

### **Step 1: Load Pre-trained BERT Model and Tokenizer**
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model and tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # Binary classification

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

---

### **Step 2: Load and Tokenize Dataset**
We use the **IMDb dataset** for sentiment classification.

```python
from datasets import load_dataset

# Load IMDb dataset
dataset = load_dataset("imdb")
train_texts, train_labels = dataset["train"]["text"], dataset["train"]["label"]
test_texts, test_labels = dataset["test"]["text"], dataset["test"]["label"]

# Tokenize the dataset
def tokenize_data(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    return torch.utils.data.TensorDataset(torch.tensor(encodings["input_ids"]),
                                          torch.tensor(encodings["attention_mask"]),
                                          torch.tensor(labels))

train_data = tokenize_data(train_texts[:2000], train_labels[:2000])  # Subset for quick training
test_data = tokenize_data(test_texts[:500], test_labels[:500])  # Subset for quick testing

train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=8)
```

---

### **Step 3: Define Training Function**
```python
import torch.optim as optim
import torch.nn.functional as F

optimizer = optim.AdamW(model.parameters(), lr=5e-5)

def train_model(model, dataloader, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(dataloader):.4f}")

# Train the model
train_model(model, train_loader)
```

---

### **Step 4: Evaluate Model**
```python
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy: {correct / total:.4f}")

evaluate_model(model, test_loader)
```

---

### **Step 5: Save and Load Fine-Tuned Model**
```python
# Save the model
model.save_pretrained("bert_imdb_finetuned")

# Load the fine-tuned model
fine_tuned_model = BertForSequenceClassification.from_pretrained("bert_imdb_finetuned")
fine_tuned_model.to(device)
```

---
## **3️⃣ Fine-Tuning GPT for Text Generation**
GPT (Generative Pre-trained Transformer) is ideal for **text generation, chatbots, and creative AI writing**.

### **Step 1: Load GPT Model and Tokenizer**
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_NAME = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.to(device)
```

---

### **Step 2: Generate Text Using Pretrained GPT**
```python
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example: Generate text
print(generate_text("Once upon a time"))
```

---

### **Step 3: Fine-Tune GPT on Custom Text Data**
To fine-tune GPT, we need custom text data.

```python
from datasets import load_dataset

# Load a small text dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Tokenize the dataset
def tokenize_gpt(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")

tokenized_dataset = dataset.map(tokenize_gpt, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids"])
```

---

### **Step 4: Train GPT on New Data**
```python
from torch.utils.data import DataLoader
from transformers import AdamW

train_loader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

def train_gpt(model, dataloader, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(dataloader):.4f}")

# Train GPT
train_gpt(model, train_loader)
```

---

### **Step 5: Generate Text with Fine-Tuned GPT**
```python
print(generate_text("The future of AI is"))
```

---

### **🚀 Summary of BERT vs. GPT Fine-Tuning**
| Model | Task | Code Example |
|-------|------|-------------|
| **BERT** | Sentiment Classification | `BertForSequenceClassification` |
| **BERT** | Text Embeddings | `BertModel` |
| **GPT** | Text Generation | `GPT2LMHeadModel` |
| **GPT** | Fine-Tuning on New Text Data | `train_gpt(model, dataloader)` |

### **Next Steps**
✅ Use **T5 for text summarization**  
✅ Deploy fine-tuned models using **TorchServe**  
✅ Optimize GPT **with LoRA or Quantization for faster inference**  

Would you like a tutorial on **deploying fine-tuned BERT/GPT models**? 😊