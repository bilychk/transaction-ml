from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from datasets import Dataset
import evaluate


df = pd.read_csv("financial_data_cleaned.csv")

#Map transaction_type to numeric labels
label_mapping = {label: idx for idx, label in enumerate(df['transaction_type'].unique())}
df['label'] = df['transaction_type'].map(label_mapping)

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['purpose_text'],  
    df['label'], 
    test_size=0.2,
    random_state=42
)
X_train = X_train.astype(str)
X_test = X_test.astype(str)


y_train = y_train.astype(int)
y_test = y_test.astype(int)

#Prepare the dataset for Hugging Face
train_data = Dataset.from_dict({"text": X_train, "label": y_train})
test_data = Dataset.from_dict({"text": X_test, "label": y_test})

#Prepare the dataset for Hugging Face
train_data = Dataset.from_dict({"text": X_train, "label": y_train})
test_data = Dataset.from_dict({"text": X_test, "label": y_test})

#Load pre-trained tokenizer and model

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping),ignore_mismatched_sizes=True)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

#training arguments
training_args = TrainingArguments(
    output_dir="./results",
    do_eval=True,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500
)

#evaluation
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=labels, average="weighted")["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average="weighted")["recall"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics
)

trainer.train()


results = trainer.evaluate()
print("Evaluation Results:", results)


model.save_pretrained("bert_finetuned_model")
tokenizer.save_pretrained("bert_finetuned_model")