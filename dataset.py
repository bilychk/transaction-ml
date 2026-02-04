import pandas as pd
import random

random.seed(42)

# Define possible values for transaction types and dynamic components
transaction_types = ["rent", "groceries", "utilities", "subscription", "entertainment", "travel", "other"]
stores = ["lidl", "rimi", "selver", "maxima", "prisma", "coop", "spar", "aldi", "local market", "online store", "wolt market", "bolt market"]
services = ["Netflix", "Spotify", "Amazon Prime", "Hulu", "Disney+", "Apple", "YouTube", "Google One", "Microsoft 365", "Adobe"]
cities = ["New York", "Boston", "Paris", "London", "Tokyo", "Tallinn", "Tartu", "Helsinki", "Riga", "Berlin"]
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
utilities = ["Electricity", "Water", "Gas", "Heating", "Internet", "Mobile"]
providers = ["Enefit", "Telia", "Elisa", "CityWater", "UtilityCo"]
payment_words = ["payment", "transfer", "fee", "invoice", "bill"]

# Generate a single purpose text
def generate_one(label: str) -> str:
    if label == "rent":
        return f"rent {random.choice(months)} {random.choice(payment_words)}"
    if label == "groceries":
        return f"{random.choice(stores)} groceries {random.choice(payment_words)}"
    if label == "utilities":
        return f"{random.choice(utilities)} {random.choice(providers)} bill {random.choice(months)}"
    if label == "subscription":
        return f"{random.choice(services)} subscription {random.choice(payment_words)}"
    if label == "entertainment":
        return f"{random.choice(['restaurant', 'cinema', 'concert', 'museum', 'tickets'])} {random.choice(payment_words)}"
    if label == "travel":
        return f"{random.choice(['flight', 'train', 'hotel', 'taxi'])} {random.choice(cities)} {random.choice(payment_words)}"
    return f"{random.choice(['misc', 'general', 'unexpected'])} expense {random.choice(payment_words)}"

# Generate dataset
n_rows = 900  # Increase the number of rows
edge_frac = 0.1  # 10% empty texts or noise
data = []
seen_texts = set()

while len(data) < n_rows:
    label = random.choice(transaction_types)
    text = generate_one(label)

    #random amounts in 25% of cases
    if random.random() < 0.25:
        amount = round(random.uniform(3, 300), 2)
        text = f"{text} {amount:.2f} EUR"

    #occasional uppercase in 10% of cases
    if random.random() < 0.1:
        text = text.upper()

    text = text.strip()

    #high uniqueness
    if text in seen_texts:
        continue
    seen_texts.add(text)
    data.append((text, label))

#edge cases: empty texts or noise
for _ in range(int(n_rows * edge_frac)):
    idx = random.randint(0, len(data) - 1)
    if random.random() < 0.5:
        data[idx] = ("", data[idx][1])  # Empty purpose_text
    else:
        dirty_text = random.choice(["!!!", "12345", "@@@", "###", "UNKNOWN", "N/A"])
        data[idx] = (dirty_text, data[idx][1])  # random noise


random.shuffle(data)
df = pd.DataFrame(data, columns=["purpose_text", "transaction_type"])

output_file = "financial_data.csv"
df.to_csv(output_file, index=False)

print(f"Dataset saved to {output_file}")
print("Rows:", len(df))
print("Unique texts:", df["purpose_text"].nunique())
print("Empty texts:", (df["purpose_text"].fillna("").str.strip() == "").sum())
print(df["transaction_type"].value_counts())
