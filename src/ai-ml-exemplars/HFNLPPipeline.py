'''
Created on 3/29/26 at 12:39 AM
By yuvarajdurairaj
Module Name HFNLPPipeline
'''

from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

# Sentiment Analysis — 'sentiment-analysis' is mapped to 'text-classification' in v5
sentiment = pipeline("text-classification", model="siebert/sentiment-roberta-large-english")
print(sentiment("I absolutely loved this product!"))
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Named Entity Recognition — 'ner' is still supported in v5
ner = pipeline("ner", aggregation_strategy="simple", model="dbmdz/bert-large-cased-finetuned-conll03-english")
print(ner("Apple was founded by Steve Jobs in Cupertino."))
# [{'entity_group': 'ORG', 'word': 'Apple'},
#  {'entity_group': 'PER', 'word': 'Steve Jobs'},
#  {'entity_group': 'LOC', 'word': 'Cupertino'}]

# Text Summarization — use 'text2text-generation' for encoder-decoder models like BART
summarizer = pipeline("text-generation", model="facebook/bart-large-cnn")
long_text = """Hugging Face is a company that develops tools for 
               building machine learning applications..."""
print(summarizer(long_text, max_new_tokens=60))
# Note: max_length/min_length replaced by max_new_tokens in v5

# Question Answering — not a pipeline task in v5, use model + tokenizer directly
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    score = (
        torch.softmax(outputs.start_logits, dim=1)[0][start].item() +
        torch.softmax(outputs.end_logits, dim=1)[0][end - 1].item()
    ) / 2
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
    )
    return {"answer": answer, "score": round(score, 4)}

print(answer_question(
    question="Where is Hugging Face based?",
    context="Hugging Face is a company based in New York City."
))
# {'answer': 'New York City', 'score': 0.98}

# Translation — use 'text2text-generation' instead of 'translation_en_to_fr'
translator = pipeline("text-generation", model="Helsinki-NLP/opus-mt-en-fr")
print(translator("Hello, how are you?"))
# [{'generated_text': 'Bonjour, comment allez-vous?'}]

# Zero-shot Classification — still supported in v5
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print(classifier(
    "The stock market crashed today",
    candidate_labels=["finance", "sports", "technology", "politics","investment"]
))