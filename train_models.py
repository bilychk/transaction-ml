import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("financial_data_cleaned.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df['tokens'].apply(lambda tokens: " ".join(eval(tokens))),  # Convert tokens back to strings
    df['transaction_type'],
    test_size=0.2,
    random_state=42
)

#Converting text to numerical features using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

#cross-validation for Logistic Regression and training
lr_model = LogisticRegression(class_weight="balanced")
lr_cv_scores = cross_val_score(lr_model, X_train_vectorized, y_train, cv=5, scoring="accuracy")
print(f"Logistic Regression Cross-Validation Accuracy: {lr_cv_scores.mean():.2f} (+/- {lr_cv_scores.std():.2f})")

lr_model.fit(X_train_vectorized, y_train)
lr_predictions = lr_model.predict(X_test_vectorized)

#cross-validation for Naive Bayes and training
nb_model = MultinomialNB()
nb_cv_scores = cross_val_score(nb_model, X_train_vectorized, y_train, cv=5, scoring="accuracy")
print(f"Naive Bayes Cross-Validation Accuracy: {nb_cv_scores.mean():.2f} (+/- {nb_cv_scores.std():.2f})")

nb_model.fit(X_train_vectorized, y_train)
nb_predictions = nb_model.predict(X_test_vectorized)

#Evaluating
print("Logistic Regression Performance:")
print(classification_report(y_test, lr_predictions))
lr_accuracy = accuracy_score(y_test, lr_predictions)

print("Naive Bayes Performance:")
print(classification_report(y_test, nb_predictions))
nb_accuracy = accuracy_score(y_test, nb_predictions)

#Comparing
results = {
    "Model": ["Logistic Regression", "Naive Bayes"],
    "Accuracy": [lr_accuracy, nb_accuracy]
}
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

#Save the best model and vectorizer
joblib.dump(lr_model, "logistic_regression_model.pkl")
joblib.dump(nb_model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

