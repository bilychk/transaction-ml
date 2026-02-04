import pandas as pd
import re
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


df = pd.read_csv("financial_data.csv")
df['purpose_text'] = df['purpose_text'].fillna("unknown")

#Remove rows with empty or whitespace-only 'purpose_text'
df = df[df['purpose_text'].str.strip() != ""]


def clean_text(text):
    text = text.lower()  
    text = re.sub(r"[^a-z\s]", "", text)  # Remove special characters, numbers, and punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

df['purpose_text'] = df['purpose_text'].apply(clean_text)

#Tokenize the cleaned 'purpose_text' field using the Hugging Face tokenizer
def tokenize_with_transformers(text):
    return tokenizer.tokenize(text)

df['tokens'] = df['purpose_text'].apply(tokenize_with_transformers)


output_file = "financial_data_cleaned.csv"
df.to_csv(output_file, index=False)

print("Preprocessing completed. Cleaned data saved to financial_data_cleaned.csv")
print("Dataset Statistics:")
print(f"Total Rows: {len(df)}")
print(f"Unique Purpose Texts: {df['purpose_text'].nunique()}")
print(f"Empty Purpose Texts: {(df['purpose_text'].fillna('').str.strip() == '').sum()}")
print("Transaction Type Distribution:")
print(df['transaction_type'].value_counts())
