import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import mlflow
import mlflow.sklearn
mlflow.set_experiment("transaction-classifier")

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

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

#cross-validation for logistic regression and training
with mlflow.start_run(run_name="logistic_regression"):
 
    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000)
 
    lr_cv_scores = cross_val_score(lr_model, X_train_vectorized, y_train, cv=5, scoring="accuracy")
    print(f"LR CV Accuracy: {lr_cv_scores.mean():.2f} (+/- {lr_cv_scores.std():.2f})")
 
    lr_model.fit(X_train_vectorized, y_train)
    lr_predictions = lr_model.predict(X_test_vectorized)
 
    lr_accuracy  = accuracy_score(y_test, lr_predictions)
    lr_f1        = f1_score(y_test, lr_predictions, average="weighted")
    lr_precision = precision_score(y_test, lr_predictions, average="weighted")
    lr_recall    = recall_score(y_test, lr_predictions, average="weighted")
 
    # log params
    mlflow.log_params({
        "model":        "LogisticRegression",
        "class_weight": "balanced",
        "cv_folds":     5,
        "test_size":    0.2,
        "vectorizer":   "TfidfVectorizer",
    })
 
    # log metrics
    mlflow.log_metrics({
        "cv_accuracy_mean": round(lr_cv_scores.mean(), 4),
        "cv_accuracy_std":  round(lr_cv_scores.std(),  4),
        "test_accuracy":    round(lr_accuracy,          4),
        "test_f1":          round(lr_f1,                4),
        "test_precision":   round(lr_precision,         4),
        "test_recall":      round(lr_recall,            4),
    })
 
    mlflow.sklearn.log_model(lr_model, "model")
    joblib.dump(lr_model, "logistic_regression_model.pkl")
 
    print("Logistic Regression Performance:")
    print(classification_report(y_test, lr_predictions))

#cross-validation for Naive Bayes and training
with mlflow.start_run(run_name="naive_bayes"):
 
    nb_model = MultinomialNB()
 
    nb_cv_scores = cross_val_score(nb_model, X_train_vectorized, y_train, cv=5, scoring="accuracy")
    print(f"NB CV Accuracy: {nb_cv_scores.mean():.2f} (+/- {nb_cv_scores.std():.2f})")
 
    nb_model.fit(X_train_vectorized, y_train)
    nb_predictions = nb_model.predict(X_test_vectorized)
 
    nb_accuracy  = accuracy_score(y_test, nb_predictions)
    nb_f1        = f1_score(y_test, nb_predictions, average="weighted")
    nb_precision = precision_score(y_test, nb_predictions, average="weighted")
    nb_recall    = recall_score(y_test, nb_predictions, average="weighted")
 
    mlflow.log_params({
        "model":      "MultinomialNB",
        "cv_folds":   5,
        "test_size":  0.2,
        "vectorizer": "TfidfVectorizer",
    })
 
    mlflow.log_metrics({
        "cv_accuracy_mean": round(nb_cv_scores.mean(), 4),
        "cv_accuracy_std":  round(nb_cv_scores.std(),  4),
        "test_accuracy":    round(nb_accuracy,          4),
        "test_f1":          round(nb_f1,                4),
        "test_precision":   round(nb_precision,         4),
        "test_recall":      round(nb_recall,            4),
    })
 
    mlflow.sklearn.log_model(nb_model, "model")
    joblib.dump(nb_model, "naive_bayes_model.pkl")
 
    print("Naive Bayes Performance:")
    print(classification_report(y_test, nb_predictions))

#Comparing
results_df = pd.DataFrame({
    "Model":    ["Logistic Regression", "Naive Bayes"],
    "Accuracy": [lr_accuracy, nb_accuracy],
    "F1":       [lr_f1, nb_f1],
})
print("\nModel Comparison:")
print(results_df)
print("\nRun  `mlflow ui`  to view experiment results in the browser.")


#Save the best model and vectorizer
joblib.dump(lr_model, "logistic_regression_model.pkl")
joblib.dump(nb_model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

