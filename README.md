# 💳 Fraudulent Transactions Classification

## 📌 Project Overview
This project focuses on detecting **fraudulent financial transactions** using machine learning models.  
The dataset simulates 30 days of transactions, where a small portion are labeled as fraud.  
Our main goal is to build models that can identify fraud effectively while exploring different preprocessing and modeling strategies.

---

## 🛠️ What We Did
1. **Data Exploration**
   - Explored transaction types, amounts, and patterns of fraud.
   - Visualized distributions with histograms, pie charts, and correlation heatmaps.

2. **Preprocessing**
   - Encoded categorical features (`type`, `nameOrig`, `nameDest`).
   - Removed highly collinear features (`newbalanceDest`, `newbalanceOrig`).
   - Balanced the dataset by sampling to reduce bias.

3. **Model Training**
   - Tested **Naive Bayes**, **Logistic Regression**, and **Random Forest** classifiers.
   - Compared models with and without certain features (e.g., `nameOrig`, `nameDest`).
   - Used an 80/20 train-test split with feature scaling.

---

## 📊 Results
- **Naive Bayes**: Weak performance; struggled with data complexity.  
- **Logistic Regression**: Improved performance, but not optimal.  
- **Random Forest**: Achieved the **best accuracy**, especially when dropping high-cardinality features (`nameOrig`, `nameDest`).  

✅ **Chosen Model:** Random Forest (with selected features removed).  

---

## 🔮 Future Improvements
While this is a simple baseline project, here are some areas for enhancement:

1. **Handle Imbalanced Data Better**
   - Use SMOTE (oversampling) or class weighting instead of undersampling.

2. **Advanced Evaluation Metrics**
   - Include precision, recall, F1-score, ROC-AUC, and confusion matrices, not just accuracy.

3. **Model Optimization**
   - Tune Random Forest hyperparameters.
   - Try advanced models like XGBoost, LightGBM, or CatBoost.

4. **Feature Engineering**
   - Create time-based features (day/hour of transaction).
   - Explore transformations for high-cardinality features.

5. **Deployment**
   - Save the trained model (`joblib`/`pickle`).
   - Build a pipeline or API for real-time fraud detection.

6. **Explainability**
   - Add feature importance plots to highlight key factors behind fraud prediction.

---

## 👥 Contributors
- Anfal Alkuraydis  
- Modhi Alhamdan  
- Haya Alhodaib  
- Nada Almutairi  
- Norah Almezied  

---

✨ This project serves as a **starting point for fraud detection** with clear opportunities for improvement.  
It’s simple, yet demonstrates key steps in a real-world ML workflow.
