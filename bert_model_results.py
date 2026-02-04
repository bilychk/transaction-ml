from transformers import pipeline

# Load the fine-tuned model and tokenizer
model_path = "bert_finetuned_model"  
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)


examples = [
    "Netflix subscription payment",
    "Paid rent for January",
    "Bought groceries at the store",
    "Flight tickets to New York"
]


predictions = classifier(examples)


for text, pred in zip(examples, predictions):
    print(f"Text: {text}")
    print(f"Prediction: {pred['label']}, Confidence: {pred['score']:.4f}")
    print()