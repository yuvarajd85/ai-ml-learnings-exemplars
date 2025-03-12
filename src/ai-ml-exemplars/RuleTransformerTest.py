'''
Created on 2/28/2025 at 2:40 AM
By yuvaraj
Module Name: RuleTransformerTest
'''
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(text, model, tokenizer):
    model.eval()
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).input_ids
    print(input_ids)
    input_ids = input_ids.to(device)

    output_ids = model.generate(input_ids, max_length=128)
    print(output_ids)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)

    return output_text

# Load model
model = T5ForConditionalGeneration.from_pretrained("dqdl_parser_model").to(device)
tokenizer = T5Tokenizer.from_pretrained("dqdl_parser_model")

# Test examples
test_cases = ["age > 50", "length of ticker_symbol = 5", "type of salary = float", "holdings <= 467722"]
for case in test_cases:
    print(f"Input: {case}")
    print(f"Output: {predict(case, model, tokenizer)}\n")