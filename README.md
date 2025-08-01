#  Telco Customer Churn Prediction

This project is a **complete machine learning solution** to predict customer churn in a telecom company using customer demographic and service-related features. We used a structured approach starting from data loading, cleaning, EDA, feature engineering, model training, model comparison, and finally deploying the solution via a **Streamlit web application**.

---

##  Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Techniques Used](#techniques-used)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
- [Modeling & Evaluation](#modeling--evaluation)
- [Model Comparison](#model-comparison)
- [Deployment](#deployment)
- [Libraries Used](#libraries-used)
- [Folder Structure](#folder-structure)
- [Conclusion](#conclusion)

---

##  Project Overview

The goal of this project is to predict customer churn using historical Telco customer data. Churn prediction enables businesses to proactively retain customers and minimize revenue loss.

---

##  Problem Statement

Churn is one of the biggest challenges in the telecom industry. Companies spend significant money acquiring new users. If we can **predict which customers are likely to churn**, retention teams can take early action to reduce churn and increase profitability.

---

##  Dataset Information

- **Source**: [Kaggle – Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Rows**: ~7,000+
- **Columns**: 21
- **Target**: `Churn` (Yes/No)


###  Key Features:

| Feature              | Description |
|----------------------|-------------|
| `customerID`         | Unique ID for each customer |
| `gender`             | Gender (Male/Female) |
| `SeniorCitizen`      | 1 if senior citizen, 0 otherwise |
| `Partner`            | Has a partner (Yes/No) |
| `Dependents`         | Has dependents (Yes/No) |
| `tenure`             | Number of months with the company |
| `PhoneService`       | Has phone service (Yes/No) |
| `MultipleLines`      | Has multiple lines (Yes/No/No phone service) |
| `InternetService`    | Type of internet (DSL/Fiber optic/No) |
| `OnlineSecurity`     | Has online security service |
| `OnlineBackup`       | Has online backup service |
| `DeviceProtection`   | Has device protection |
| `TechSupport`        | Has tech support |
| `StreamingTV`        | Has streaming TV |
| `StreamingMovies`    | Has streaming movies |
| `Contract`           | Contract type (Month-to-month, One year, Two year) |
| `PaperlessBilling`   | Opted for paperless billing (Yes/No) |
| `PaymentMethod`      | Mode of payment |
| `MonthlyCharges`     | Monthly billing amount |
| `TotalCharges`       | Lifetime billing amount |
| `Churn`              | Target label (Yes/No) |

---

###  Data Types Handled


| Feature Type       | Examples                                       | Notes                                   |
| ------------------ | ---------------------------------------------- | --------------------------------------- |
| **Categorical**    | `gender`, `Partner`, `Dependents`, `Contract`  | Used for one-hot encoding               |
| **Numerical**      | `tenure`, `MonthlyCharges`, `TotalCharges`     | Scaled using `StandardScaler`           |
| **Binary**         | `SeniorCitizen`, `PhoneService`, `Churn`       | Label encoded (0 or 1)                  |
| **Multi-class**    | `InternetService`, `PaymentMethod`, `Contract` | One-hot encoded with `drop_first=True`  |
| **Text (Ignored)** | `customerID`                                   | Not used in modeling due to irrelevance |


##  Techniques Used

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Label Encoding & One-Hot Encoding
- Feature Scaling (StandardScaler)
- Model Training & Comparison
- Evaluation Metrics (Precision, Recall, F1, Accuracy)
- Model Deployment (Streamlit)

### 🔷 Data Preprocessing
- `Drop` unnecessary columns 
- `Convert` TotalCharges to numeric
- `Label Encoding` for binary categories
- `One-Hot Encoding` for multiclass features
- `Standard Scaling` of numerical columns



- Removed customerID

- Converted TotalCharges to numeric (with null handling)

- Applied Label Encoding on binary categorical columns

- Applied One-Hot Encoding with drop_first=True on multi-class categorical features

- Applied StandardScaler on numerical columns: tenure, MonthlyCharges, TotalCharges

- After encoding, total columns became 30.

---

##  Exploratory Data Analysis (EDA)

- Churn distribution: highly imbalanced
- Contract type strongly correlated with churn
- Visualized:
  - Univariate: histograms, countplots
  - Bivariate: churn vs contract/tenure/gender
  - Boxplots for outlier detection
  - Correlation heatmap for multicollinearity

---

##  Preprocessing & Feature Engineering

- Converted all categorical variables via encoding
- Dropped unneeded columns like `customerID`
- Used `StandardScaler` to scale numerical features
- Post-encoding → **30 features total**

---

##  Modeling & Evaluation

###  **Model 1: Logistic Regression (Final)**
- Best performance on imbalanced data
- Great precision and good balance with recall

###  **Model 2: Random Forest (For Comparison)**
- Slightly better recall but lower precision

---

##  Model Comparison

| Metric      | Logistic Regression | Random Forest |
|-------------|---------------------|----------------|
| Accuracy    | 80%                 | 78%            |
| Precision   | 65%                 | 62%            |
| Recall      | 57%                 | 50%            |
| F1 Score    | 61%                 | 56%            |



** Final Model Chosen**: **Logistic Regression**  
**Why?** For business-focused applications like churn, precision matters more (i.e., fewer false positives).

- Logistic Regression performs better overall, especially for detecting churn (class 1), with higher precision, recall, and F1-score.

- Even though Random Forest is more complex, it underperformed on recall and F1 in this imbalanced setting.

- Hence, Logistic Regression is the better choice for this use case

---

##  Deployment

- **Frontend**: Streamlit Web App
- **Backend**:
  - Saved model using `pickle`
  - Saved column transformer and scaler for input consistency

##  Streamlit Features:

- User-friendly interface to input customer details

- Real-time churn prediction with confidence score

- Works locally in your browser

## How It Works

Loads a pre-trained Logistic Regression model using pickle

Accepts user inputs for all required features

Applies one-hot encoding and scaling

Displays prediction and confidence level

---

##  Libraries Used


pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
pickle


## Conclusion:

This project successfully built a Telco Customer Churn Prediction System using machine learning. I performed data preprocessing, handled imbalanced classes, and applied both Logistic Regression and Random Forest models. After comparing the models, Logistic Regression was selected for its better balance of precision, recall, and F1-score for the minority class (churned customers).

Best Model: Logistic Regression

Accuracy: 80%

Churn Class (1) - Precision: 65%, Recall: 57%, F1-Score: 61%

Deployed using a Streamlit web app with an interactive UI for real-time churn predictions.

This solution can help telecom companies proactively identify at-risk customers and take retention actions.

