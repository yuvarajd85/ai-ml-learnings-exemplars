### **🚀 Deploying Fine-Tuned BERT & GPT Models Using TorchServe**
In this tutorial, we'll deploy **fine-tuned BERT (for text classification) and GPT (for text generation)** using **TorchServe** for scalable inference.

---

## **1️⃣ Install TorchServe**
First, install **TorchServe** and related dependencies:
```bash
pip install torchserve torch-model-archiver torchtext transformers
```

---

## **2️⃣ Save the Fine-Tuned Model**
Before deployment, we need to save the fine-tuned model in a format that TorchServe can use.

### **For BERT (Text Classification)**
```python
from transformers import BertForSequenceClassification

# Load fine-tuned BERT model
MODEL_PATH = "bert_imdb_finetuned"
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Save in TorchScript format
import torch
example_input = torch.randint(0, 1000, (1, 512))  # Example input for tracing
traced_model = torch.jit.trace(model, example_input)

# Save model
traced_model.save("bert_traced.pt")
```

### **For GPT (Text Generation)**
```python
from transformers import GPT2LMHeadModel

# Load fine-tuned GPT model
MODEL_PATH = "gpt_finetuned"
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# Save in TorchScript format
example_input = torch.randint(0, 1000, (1, 512))  # Example input
traced_model = torch.jit.trace(model, example_input)

# Save model
traced_model.save("gpt_traced.pt")
```

---

## **3️⃣ Create Custom Inference Handler**
TorchServe requires a **handler script** to define how the model processes requests.

### **Create `bert_handler.py` for BERT**
```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class TextClassifierHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def initialize(self, model_dir):
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model.eval()

    def preprocess(self, request):
        text = request["text"]
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        return tokens

    def inference(self, input_data):
        input_ids = input_data["input_ids"]
        attention_mask = input_data["attention_mask"]
        with torch.no_grad():
            output = self.model(input_ids, attention_mask=attention_mask)
        return torch.argmax(output.logits, dim=1).item()

    def postprocess(self, inference_output):
        return {"prediction": "Positive" if inference_output == 1 else "Negative"}
```

### **Create `gpt_handler.py` for GPT**
```python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TextGeneratorHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def initialize(self, model_dir):
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model.eval()

    def preprocess(self, request):
        text = request["prompt"]
        return self.tokenizer(text, return_tensors="pt")

    def inference(self, input_data):
        with torch.no_grad():
            output = self.model.generate(input_data["input_ids"], max_length=50, temperature=0.7)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def postprocess(self, inference_output):
        return {"generated_text": inference_output}
```

---

## **4️⃣ Package the Models**
TorchServe uses **.mar files** (Model Archive) for deployment.

### **Package BERT Model**
```bash
torch-model-archiver --model-name bert_sentiment --version 1.0 --serialized-file bert_traced.pt \
--handler bert_handler.py --extra-files "bert_imdb_finetuned/config.json"
```

### **Package GPT Model**
```bash
torch-model-archiver --model-name gpt_textgen --version 1.0 --serialized-file gpt_traced.pt \
--handler gpt_handler.py --extra-files "gpt_finetuned/config.json"
```

---

## **5️⃣ Start TorchServe**
```bash
mkdir model_store
mv bert_sentiment.mar gpt_textgen.mar model_store/

torchserve --start --ncs --model-store model_store --models bert_sentiment=gpt_textgen
```
- `--ncs` runs TorchServe **without gRPC support** (optional).
- `--models` loads both BERT and GPT.

---

## **6️⃣ Send Requests to the Deployed Models**
Use **cURL or Python** to send HTTP requests.

### **Test BERT Sentiment Model**
```bash
curl -X POST "http://127.0.0.1:8080/predictions/bert_sentiment" -H "Content-Type: application/json" \
-d '{"text": "The movie was absolutely fantastic!"}'
```

### **Test GPT Text Generation Model**
```bash
curl -X POST "http://127.0.0.1:8080/predictions/gpt_textgen" -H "Content-Type: application/json" \
-d '{"prompt": "Once upon a time in a futuristic city,"}'
```

---

## **7️⃣ Stop TorchServe**
```bash
torchserve --stop
```

---

### **🚀 Final Summary**
| Step | Action | Command |
|------|--------|---------|
| **1** | Install TorchServe | `pip install torchserve` |
| **2** | Save Fine-Tuned Models | `traced_model.save("model.pt")` |
| **3** | Create Inference Handlers | `bert_handler.py`, `gpt_handler.py` |
| **4** | Package Models | `torch-model-archiver --model-name ...` |
| **5** | Start TorchServe | `torchserve --start ...` |
| **6** | Send Requests | `curl -X POST ...` |
| **7** | Stop Server | `torchserve --stop` |

---

### **✅ Next Steps**
- Deploy on **AWS, GCP, or Azure**  
- Optimize **inference speed with quantization (TorchScript, ONNX)**  
- Use **FastAPI for a custom REST API**  

Would you like a tutorial on **deploying to AWS Lambda or optimizing inference speed**? 🚀