# 📊 Customer Churn Prediction System
### 🧩 Problem Statement
Understanding and predicting customer churn is essential for the long-term success of customer-facing businesses. Churn refers to when customers stop doing business with a company. Our goal is to create a predictive system that can identify which customers are at risk of leaving (label 0) and which are likely to stay (label 1) based on the following factors:

Gender

Age

Amount Spent

By identifying customers at risk of churning, the business can implement retention strategies to reduce loss and improve customer satisfaction.

### 🎯 Project Objective
Build a binary classification model to predict churn.

Use customer demographic and behavioral data.

Compare Logistic Regression, Random Forest, XGBoost, and a Deep Learning model (ANN).

Deploy the best model using Dash and Render.

### 📂 Dataset Overview
Column	Type	Description
CustomerID	int64	Unique identifier for each customer (ignored)
Gender	category	Customer gender (e.g., Male = 0, Female = 1)
Age	int64	Customer age
AmountSpent	float64	Total amount spent by the customer
Churn	int64	Target variable: 0 = Leaving, 1 = Staying

### ⚙️ System Setup
### 🔧 Tools Used
Visual Studio Code

Python 3.10+

Git + GitHub

Render.com for deployment

Dash (Plotly)

### 🐍 Python Libraries
pandas, numpy – Data manipulation

matplotlib, seaborn – Visualization

scikit-learn – ML models and preprocessing

xgboost – Gradient boosting

tensorflow – Deep learning model

joblib – Save/load models

skopt – Bayesian hyperparameter tuning

### 📊 Exploratory Data Analysis (EDA)
✅ Univariate Analysis
Checked class balance for the Churn column.

Analyzed distributions of Age, AmountSpent, and Gender.

✅ Bivariate Analysis
Compared churn vs. features using:

Boxplots for numeric features.

Barplots for categorical (Gender).

Correlation heatmaps.

✅ Data Cleaning
Removed CustomerID column.

Converted Gender to binary (e.g., Male = 0, Female = 1).

Handled outliers using the IQR method.

Checked and treated missing values.

### 🤖 Model Development
🔸 Logistic Regression
Applied StandardScaler on Age and AmountSpent.

Encoded Gender as binary.

Trained using 80/20 train-test split with stratification.

Evaluated with classification report and confusion matrix.

🔸 Random Forest
Used default parameters with 80/20 split.

Feature importance assessed.

Evaluated model using accuracy, F1-score, and confusion matrix.

🔸 XGBoost
Objective: binary:logistic

Balanced class weights using scale_pos_weight.

Tuned hyperparameters with Bayesian optimization (skopt).

Achieved high accuracy with explainable feature influence.

🔸 ANN (Deep Learning)
Input: Gender, Age, AmountSpent

Architecture:

1 input layer

1–2 hidden layers (ReLU)

1 output layer (Sigmoid)

Used BinaryCrossentropy as loss function.

Evaluated using metrics like accuracy, precision, and recall.

### 📈 Evaluation Metrics
Metric	Description
Accuracy	Overall correctness of the model
Precision	Correctly predicted stayers / all predicted stayers
Recall	Correctly predicted stayers / all actual stayers
F1 Score	Harmonic mean of precision and recall
Confusion Matrix	Breakdown of correct/incorrect predictions per class

### 💾 Feature Engineering
Dropped: CustomerID

Encoded: Gender to 0/1

Scaled: Age, AmountSpent

Balanced: Used SMOTE for class imbalance during training

### 🌐 Model Deployment
🔧 Stack
Web App: Built with Dash (Python)

Hosting: Render (auto-deploy from GitHub)

### 📄 Pages
Home Page: Input form for predicting churn

Analysis Page: Visuals and performance metrics

### 🔧 Virtual Environment Setup
Used rye and poetry for:

Python version locking

Dependency management

Environment reproducibility

### 🧠 Model Export
Best model saved to .joblib or .h5 and loaded at runtime in the Dash app for real-time prediction.

### 📁 Project Structure
graphql
Copy
Edit
📦 customer-churn-predictor
├── app/
│   ├── app.py               # Dash app entry point
│   ├── home.py              # User input and prediction
│   ├── analysis.py          # EDA and evaluation visualizations
├── models/
│   ├── churn_model.h5       # Trained best model
├── notebooks/
│   ├── EDA.ipynb            # Exploratory analysis
│   ├── model_training.ipynb # Model training and evaluation
├── data/
│   ├── raw_data.csv
│   ├── cleaned_data.csv
├── requirements.txt         # Dependencies
├── README.md                # Project overview
└── .render.yaml             # Render deployment config
### ✅ Final Notes
Business Value: The model empowers marketing and support teams to proactively engage with at-risk customers.

Next Steps: Incorporate behavioral data (e.g., frequency of purchases, feedback ratings) to improve accuracy.
