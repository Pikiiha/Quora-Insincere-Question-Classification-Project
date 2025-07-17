# 🧠 Quora Insincere Question Classification Project

## 📌 Project Title
**Improving Online Question Quality on Quora Using AI**

---

## 🔍 Problem Statement

Quora is one of the largest platforms where users ask and answer questions on a wide range of topics. However, due to the scale of activity, **identifying insincere (toxic or misleading) questions** is difficult for human moderators. 

As a data scientist, my goal in this project was to apply **AI models to automatically classify questions as sincere or insincere**, improving the quality and safety of the platform.

---

## 🎯 Objective

- Build a binary classifier to detect **insincere (1)** and **sincere (0)** questions.
- Use machine learning models to assist Quora moderators in filtering harmful or low-quality content.

---

## 📁 Dataset Overview

- **Target variable**: `0` for sincere, `1` for insincere
- **Highly imbalanced** (majority class = sincere)
- **Features engineered**:
  - Text length
  - Word count
  - TF-IDF and Count vectorization

### 📊 Exploratory Data Analysis

- Class 0 dominates the dataset (~92%)
- Insincere questions are slightly longer and more wordy
- High correlation between text length and word count

📌 **Target and Text Feature Distribution**  
<img width="1489" height="957" alt="download (5)" src="https://github.com/user-attachments/assets/5c8ad92e-73ef-4000-b00c-336335719510" />




---

## ⚙️ Model Building & Evaluation

I trained and compared multiple models with both TF-IDF and Count Vectorizers:

| Model                | Vectorizer     | CV Accuracy |
|---------------------|----------------|-------------|
| Logistic Regression | TF-IDF         | 0.9495      |
| Logistic Regression | Count          | **0.9496** ✅ |
| Naive Bayes         | TF-IDF         | 0.9454      |
| Naive Bayes         | Count          | 0.8984      |
| Linear SVM          | TF-IDF         | 0.9492      |
| Linear SVM          | Count          | 0.9494      |

📌 **Model Comparison Chart**  
<img width="1774" height="489" alt="download (6)" src="https://github.com/user-attachments/assets/51410181-ed6c-4fe4-8596-1276b5105dd9" />


---

## 🔍 Hyperparameter Tuning (Grid Search)

I used `GridSearchCV` with 3-fold cross-validation to find optimal model + vectorizer combinations.

- **Logistic Regression**: `C=10`, `ngram_range=(1,2)`, `max_features=5000`
- **Linear SVM**: `C=1`, `ngram_range=(1,2)`, `max_features=5000`

---

## ✅ Final Model Performance

**Best Model:** Logistic Regression with CountVectorizer  
**Test Accuracy:** `0.9494`  
**AUC Score:** `0.9269`

📌 **Confusion Matrix**  
<img width="583" height="490" alt="download (7)" src="https://github.com/user-attachments/assets/b5019f87-c770-4b23-874f-a0a1c1eaaad5" />


```text
               Predicted
              |   0   |   1
         ---------------------
Actual  0   | 242174 | 2888
Actual  1   | 10318  | 5843
📌 ROC Curve

📈 Classification Report
              precision    recall  f1-score
Sincere (0):     0.96       0.99     0.97
Insincere (1):  0.67       0.36     0.47

Precision (class 1): 67% of predicted insincere questions are actually insincere.

Recall (class 1): Only 36% of real insincere questions were caught.

AUC = 0.9269: There is a 92.7% probability that the model correctly ranks an insincere question above a sincere one.

💡 Conclusion
The Logistic Regression with CountVectorizer model achieved the highest accuracy and AUC.

The model is highly effective at detecting sincere questions but still misses some insincere ones.

This model could serve as a pre-filter to assist moderators in managing question quality on Quora.

🔧 Future Improvements
Use class balancing (e.g., SMOTE, class weights)

Fine-tune transformer models like BERT

Explore ensemble methods like XGBoost or Stacking

| Image               | Suggested Name         | Replace With Your Link                                                                                |
| ------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------- |
| Target Distribution | `distribution.png`     | <img width="1489" height="957" alt="download (5)" src="https://github.com/user-attachments/assets/76543ccc-cd59-4bee-8553-39772e98bac1" />
          |
| Correlation Matrix  | `correlation.png`      |            |
| Model Comparison    | `model_comparison.png` |<img width="1774" height="489" alt="download (6)" src="https://github.com/user-attachments/assets/18f48323-77f9-49c3-ab00-3b53daa497bb" />
 |
| Confusion Matrix    | `confusion_matrix.png` <img width="583" height="490" alt="download (7)" src="https://github.com/user-attachments/assets/fef0b4a5-31ab-4c50-9c63-4b515720a6d2" />

| ROC Curve           | `roc_curve.png`        <img width="590" height="490" alt="download (8)" src="https://github.com/user-attachments/assets/6a179c53-b289-4c66-8204-47445a88fcf1" />
             |


