# 📊 Model Selection for Logistic Regression

This project demonstrates how to build, tune, evaluate, and persist a **Logistic Regression** model using **scikit-learn**. It uses a preprocessed Airbnb dataset (`airbnbData_train.csv`) for binary classification — predicting whether a listing is of "great quality".

---

## ✅ Features

- Load and prepare data using `pandas`
- Split data into training and test sets
- Build baseline logistic regression model
- Tune the `C` hyperparameter using `GridSearchCV`
- Evaluate model performance with:
  - **Confusion Matrix**
  - **Precision-Recall Curve**
  - **ROC Curve** and **AUC**
- Select top 5 features using `SelectKBest`
- Save and load model with `pickle`

---

## 🧪 Technologies Used

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- pickle

---

## 📝 How to Run

1. Place the file `airbnbData_train.csv` inside a folder named `data_LR/`
2. Open and run the notebook `ModelSelectionForLogisticRegression.ipynb` step-by-step
3. Final trained model will be saved as `model_best.pkl`

### Load and Use the Saved Model

```python
import pickle

with open("model_best.pkl", "rb") as file:
    model = pickle.load(file)

predictions = model.predict(X_test)
