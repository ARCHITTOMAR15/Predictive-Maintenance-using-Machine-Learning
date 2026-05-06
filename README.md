<p align="center"> <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python"> <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn"> <img src="https://img.shields.io/badge/XGBoost-Classifier-green?style=for-the-badge"> <img src="https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit"> <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-black?style=for-the-badge&logo=pandas"> <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?style=for-the-badge&logo=numpy"> <img src="https://img.shields.io/badge/Matplotlib-Visualization-yellow?style=for-the-badge"> <img src="https://img.shields.io/badge/Machine%20Learning-End%20to%20End-purple?style=for-the-badge"> </p>
# Predictive-Maintenance-using-Machine-Learning
A Machine Learning-based Predictive Maintenance System developed to predict machine failures and reduce downtime using data-driven insights. The project includes data preprocessing, feature engineering, imbalance handling with SMOTE, and model training using XGBoost. Deployed with Streamlit for real-time machine failure prediction.
📌 Project Overview

This project focuses on building an end-to-end Predictive Maintenance System using Machine Learning to predict whether an industrial machine is likely to fail based on sensor and operational data.

The complete workflow includes:

Data Cleaning & Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering
Handling Imbalanced Dataset
Model Building & Evaluation
Hyperparameter Tuning
Streamlit Deployment

The final deployed model uses XGBoost Classifier, achieving high prediction performance for machine failure detection.

🎯 Business Problem

Unexpected machine failures can lead to:

Increased operational costs
Production downtime
Reduced efficiency
Higher maintenance expenses

The goal of this project is to proactively predict machine failures so industries can perform preventive maintenance before breakdowns occur.

📂 Dataset Information

The dataset contains machine operational parameters and sensor readings used to predict machine failure.

Features Used
Feature	Description
Type	Machine Type
Air Temperature [K]	Surrounding air temperature
Process Temperature [K]	Manufacturing process temperature
Rotational Speed [rpm]	Machine rotational speed
Torque [Nm]	Torque generated
Tool Wear [min]	Tool wear duration
Target	Failure Indicator (0 = No Failure, 1 = Failure)

Project reference and dataset context:

⚙️ Tech Stack
🖥️ Programming & Libraries
Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Seaborn
Streamlit
Pickle
🔍 Machine Learning Workflow
✅ Data Preprocessing
Missing value handling
Duplicate value removal
Outlier treatment
Label Encoding
Feature Scaling using StandardScaler
Box-Cox Transformation for skewness reduction
Feature Selection
📊 Exploratory Data Analysis (EDA)

Performed detailed EDA to understand:

Feature distributions
Correlation between variables
Failure patterns
Class imbalance analysis
Outlier detection
⚖️ Handling Imbalanced Dataset

Since machine failures were highly imbalanced, imbalance handling techniques were applied:

SMOTE
SMOTETomek
Class balancing strategies
🤖 Model Building

Multiple machine learning models were trained and evaluated:

Logistic Regression
Random Forest
Decision Tree
KNN
XGBoost Classifier
🏆 Final Model Selected
✅ XGBoost Classifier

Reasons for selection:

High accuracy
Better generalization
Strong performance on imbalanced data
Robust handling of non-linear relationships
📈 Model Evaluation Metrics

The model was evaluated using:

Accuracy Score
Precision
Recall
F1-Score
Confusion Matrix
Cross Validation
🌐 Streamlit Deployment

The trained machine learning model was deployed using Streamlit to provide an interactive web application where users can:

Enter machine parameters
Predict machine failure instantly
View prediction results in real-time
📁 Project Structure
Predictive-Maintenance/
│
├── data/
├── notebooks/
├── models/
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── predictive_maintenance.ipynb
▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/predictive-maintenance.git
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Streamlit App
streamlit run app.py
📸 Application Preview

Add screenshots of your Streamlit application here.

💡 Key Highlights

✅ End-to-End ML Project
✅ Industry Use Case
✅ Imbalanced Data Handling
✅ Feature Engineering
✅ XGBoost Optimization
✅ Streamlit Deployment
✅ Production-Oriented Workflow

📚 Learning Outcomes

Through this project, I gained practical experience in:

Real-world Machine Learning workflow
Data preprocessing techniques
Model optimization
Imbalanced classification problems
Deployment of ML applications
Building production-ready ML projects
👨‍💻 Author
Archit Tomar

Machine Learning Enthusiast passionate about building real-world AI/ML solutions and deploying scalable applications.
