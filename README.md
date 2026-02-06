# Day 39 – AdaBoost: Classification & Regression (End-to-End)


## 📌 Project Overview

This project is part of my **Daily Machine Learning Practice (Day 39)** and focuses on implementing **AdaBoost (Adaptive Boosting)** for both **classification** and **regression** problems. The goal is to understand how boosting improves model performance by combining multiple weak learners into a strong ensemble model.

---

##  🚀 What is AdaBoost?

AdaBoost (Adaptive Boosting) is an **ensemble learning technique** that:
- Combines multiple **weak learners** (usually decision stumps)
- Assigns **higher weights to misclassified samples**
- Iteratively improves model accuracy
- Reduces bias and improves generalization

AdaBoost can be applied to:

- **Classification problems** → `AdaBoostClassifier`
- **Regression problems** → `AdaBoostRegressor`

---

##  🧠 Objectives

- Understand the working principle of AdaBoost
- Implement AdaBoost for **classification**
- Implement AdaBoost for **regression**
- Compare model performance with evaluation metrics
- Build an end-to-end ML workflow

---

## 🗂️ Project Structure

`Day_39_ML_Ada-Boost_Classification-Regression/
│
├── AdaBoost_Classification.ipynb
├── AdaBoost_Regression.ipynb
├── dataset/
│ ├── classification_data.csv
│ └── regression_data.csv
├── README.md
└── requirements.txt`


---

##  🛠️ Tech Stack & Libraries
- **Programming Language:** Python  
- **Libraries Used:**
  - NumPy
  - Pandas
  - Matplotlib / Seaborn
  - Scikit-learn

---

##  📊 AdaBoost Classification
### Workflow
1. Load and explore dataset
2. Perform data preprocessing
3. Split data into training & testing sets
4. Train `AdaBoostClassifier`
5. Evaluate model performance

###  Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

---

##  📈 AdaBoost Regression
### Workflow
1. Load and analyze regression dataset
2. Data preprocessing
3. Train `AdaBoostRegressor`
4. Make predictions
5. Evaluate results

###  Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

---

##  ⚙️ Model Parameters Used
- `n_estimators`
- `learning_rate`
- `base_estimator` (Decision Tree Regressor / Classifier)

These parameters were tuned to observe their impact on model performance.

---

##  📌 Key Learnings
- AdaBoost focuses more on **hard-to-classify samples**
- Boosting reduces bias compared to single models
- Works well with weak learners
- Sensitive to noisy data and outliers

---

##  📉 Results & Observations
- AdaBoost significantly improved accuracy over baseline models
- Classification results showed improved recall and F1-score
- Regression model achieved better error reduction compared to single regressors

---

##  🔮 Future Improvements
- Hyperparameter tuning using GridSearchCV
- Comparison with other ensemble methods (Random Forest, Gradient Boosting, XGBoost)
- Handling noisy data more effectively
- Feature importance visualization

---

##  📚 References
- Scikit-learn Documentation  
- Machine Learning by Andrew Ng  
- Hands-On Machine Learning with Scikit-Learn  

---

##  👨‍💻 Author

**Laxman Bhimrao Khedkar**  
- LinkedIn: https://www.linkedin.com/in/laxman-khedkar  
- GitHub: https://github.com/laxmankhedkar  
- Portfolio: https://beacons.ai/laxmankhedkar  

---
