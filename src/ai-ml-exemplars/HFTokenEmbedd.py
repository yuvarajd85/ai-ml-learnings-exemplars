'''
Created on 3/28/26 at 8:14 PM
By yuvarajdurairaj
Module Name HFTokenEmbedd
'''
import torch
from transformers import BertTokenizer, BertModel
from dotenv import load_dotenv

load_dotenv()

tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
model = BertModel.from_pretrained('google-bert/bert-base-uncased')
model.eval()

text = "This is an example sentence."

tokens = tokenizer(
    text,
    add_special_tokens=True,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

with torch.no_grad():
    output = model(
        input_ids=tokens['input_ids'],
        attention_mask=tokens['attention_mask']
    )

# Token-level embeddings — one vector per token
word_embeddings = output.last_hidden_state
print("Token embeddings shape:", word_embeddings.shape)  # [1, 512, 768]

# Sentence-level — mean pooling
mask = tokens['attention_mask'].unsqueeze(-1).float()
mean_embedding = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
print("Mean pooled embedding shape:", mean_embedding.shape)  # [1, 768]

vector = mean_embedding.squeeze().tolist()
print("Dimension:", len(vector))
print("vector:", vector)


