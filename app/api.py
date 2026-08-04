from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


lr_model = LogisticRegression()
vectorizer = TfidfVectorizer()


model_path = "logistic_regression_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


app = FastAPI()


class InputText(BaseModel):
    purpose_text: str


@app.post("/classify")
def classify(input_text: InputText):
    """
    Classify the purpose of a financial transaction.

    Args:
        input_text (InputText): JSON object with a single key "purpose_text" containing the transaction description.

    Returns:
        dict: JSON object with the predicted class ("predicted_type").
    """
    try:
        
        transformed_text = vectorizer.transform([input_text.purpose_text])

      
        prediction = model.predict(transformed_text)[0]

        
        return {
            "predicted_type": prediction
        }
    except Exception as e:
        return {
            "error": str(e),
            "message": "An error occurred while processing your request. Please check your input."
        }
