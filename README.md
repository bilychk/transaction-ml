# **Financial Transaction Classifier**

This project builds a text classification pipeline to predict the "transaction type" of financial transactions based on their "purpose text." The solution includes data preprocessing, model training, evaluation, and deployment as a REST API.

---

## **Features**
1. **Synthetic Dataset Generation**: 
   - Generates a dataset of financial transactions with realistic "purpose text" and "transaction type" labels.
   - Includes noise and edge cases (e.g., empty texts, random characters) for robustness.

2. **Data Preprocessing**:
   - Handles missing or irrelevant data.
   - Cleans and tokenizes text for machine learning models.
   - Prepares data for both classical ML models and transformer-based models.

3. **Model Training and Evaluation**:
   - Trains and evaluates multiple classical ML models (Logistic Regression, Naive Bayes).
   - Demonstrates a conceptual and partial implementation of a transformer-based model (DistilBERT).

4. **REST API**:
   - Exposes the trained Logistic Regression model via a FastAPI-based REST API.
   - Accepts transaction descriptions and returns predicted transaction types.

---

## **Usage**

### **1. Generate Dataset**
Run the `dataset.py` script to generate a synthetic dataset:
```bash
python dataset.py
```
This will create a file named `financial_data.csv`.

---

### **2. Preprocess Data**
Run the `preprocessing.py` script to clean and tokenize the dataset:
```bash
python preprocessing.py
```
This will create a file named `financial_data_cleaned.csv`.

---

### **3. Train and Evaluate Models**
Run the `train_models.py` script to train and evaluate Logistic Regression and Naive Bayes models:
```bash
python train_models.py
```
The script will output evaluation metrics (accuracy, precision, recall, F1-score) and save the trained models and vectorizer.

---

### **4. Fine-Tune Transformer Model**
Run the `transformer_model.py` script to fine-tune a pre-trained transformer model (e.g., DistilBERT):
```bash
python transformer_model.py
```
This will save the fine-tuned model and tokenizer to the `bert_finetuned_model` directory.

---

### **5. Deploy REST API**
Run the `apikeys.py` script to start the REST API:
```bash
uvicorn apikeys:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

#### **API Endpoint**
- **POST /classify**
  - **Input**: JSON object with a single key `"purpose_text"` containing the transaction description.
  - **Output**: JSON object with the predicted transaction type.

**Example Request**:
```bash
curl -X POST "http://127.0.0.1:8000/classify" \
-H "Content-Type: application/json" \
-d '{"purpose_text": "Netflix subscription payment"}'
```

**Example Response**:
```json
{
  "predicted_type": "subscription"
}
```

---

## **Project Structure**
- `dataset.py`: Generates a synthetic dataset of financial transactions.
- `preprocessing.py`: Cleans and tokenizes the dataset.
- `train_models.py`: Trains and evaluates classical ML models.
- `transformer_model.py`: Fine-tunes a transformer model for classification.
- `apikeys.py`: REST API for serving predictions using the trained Logistic Regression model.
- `bert_model_results.py`: Demonstrates inference using the fine-tuned transformer model.
- `financial_data.csv`: Generated synthetic dataset.
- `financial_data_cleaned.csv`: Preprocessed dataset.

---

## **Evaluation Results**

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 91%      | 0.92      | 0.91   | 0.91     |
| Naive Bayes           | 88%      | 0.89      | 0.87   | 0.88     |

---

## **Future Work**
- Experiment with additional classical ML models (e.g., SVM, Decision Trees).
- Perform hyperparameter tuning for the transformer model.
- Deploy the transformer model as a REST API for real-time predictions.
- Explore additional features, such as transaction amount or date, to improve classification accuracy.

