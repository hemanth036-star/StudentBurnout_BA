# 🎓 Student Burnout & Dropout Risk Prediction System

An AI-driven hybrid machine learning system designed to identify at-risk students early by analyzing academic and behavioral engagement patterns. The system predicts burnout risk and dropout probability to enable timely institutional intervention.

---

## 📌 Project Overview

Educational institutions often rely on traditional exam-based evaluation methods, which delay the identification of struggling students. This project introduces a data-driven predictive analytics system that analyzes LMS engagement, attendance, assignment behavior, and sentiment data to detect early warning signs of academic disengagement and burnout.

The goal is to transform reactive academic management into a proactive AI-powered student support system.

---

## 📊 Key Features

- Student engagement analysis using LMS activity  
- Burnout Risk Score (0–100 scale)  
- Dropout probability prediction (0–1 scale)  
- Behavioral clustering (Low, Medium, High Risk groups)  
- Explainable AI using SHAP for transparent decision-making  

---

## 🧠 Machine Learning Models Used

### 🔹 K-Means Clustering
- Segments students into behavioral risk groups  
- Identifies natural engagement patterns  

### 🔹 Logistic Regression
- Predicts dropout probability  
- Provides interpretable risk estimation  

### 🔹 Random Forest Regression
- Generates continuous burnout risk score  
- Captures nonlinear feature relationships  

### 🔹 SHAP (Explainable AI)
- Explains individual prediction factors  
- Enhances trust and transparency  

---

## 📈 Model Performance

- Logistic Regression Accuracy: 85%  
- Random Forest Accuracy: 92%  
- Precision: 90%  
- Recall: 88%  

Random Forest performed best due to its ability to model complex behavioral interactions.

---

## 📂 Dataset Features

- LMS logins per week  
- Login trend changes  
- Assignment submission delays  
- Missed assignments count  
- Attendance percentage  
- Attendance trend changes  
- Feedback sentiment score  
- Activity variance  
- Late-night activity ratio  

---

## 🔍 Key Insights

- Declining LMS activity is an early disengagement signal  
- Attendance below 70–75% strongly correlates with dropout risk  
- High submission delays increase burnout probability  
- Multiple risk indicators significantly increase vulnerability  

---

## 🌍 Practical Impact

- Early identification of at-risk students  
- Data-driven academic decision-making  
- Improved retention rates  
- Reduced dropout rates  
- Proactive counseling and support system  

---

## 🚀 Future Scope

- Deep learning models (LSTM / Transformer)  
- Real-time LMS & ERP integration  
- Automated intervention recommendations  
- Institution-wide analytics dashboard  
- Cross-university benchmarking  

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- SHAP  
- Matplotlib / Seaborn  

---

## 📌 Long-Term Vision

To develop a fully automated intelligent academic monitoring ecosystem that continuously analyzes student engagement and ensures academic success through AI-driven insights.
