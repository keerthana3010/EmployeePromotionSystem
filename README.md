# 🚀 Employee Promotion Prediction System

### An Intelligent Machine Learning System for Employee Promotion Prediction, Analytics & Personalized Recommendations

The **Employee Promotion Prediction System** is an intelligent machine-learning-based web application designed to analyze employee performance and predict promotion eligibility.

The system uses employee-related factors such as **performance ratings, training scores, KPIs, awards, experience, department, education, and other employee attributes** to generate a promotion prediction.

In addition to prediction, the application provides **promotion probability, analytics, employee reports, and personalized recommendations** through an interactive Streamlit dashboard.

---

## 📌 Project Overview

Employee promotion decisions often require analyzing multiple performance and organizational factors.

Traditional evaluation methods may require significant manual effort and may not provide personalized guidance to employees.

This project addresses this problem by developing a **data-driven employee promotion prediction system** that combines:

* Machine Learning
* Data Analytics
* Interactive Visualization
* Prediction
* Personalized Recommendations

The system helps transform employee data into meaningful insights that can support promotion-readiness assessment.

---

## 🎯 Objectives

The main objectives of this project are:

* To develop a machine-learning model for employee promotion prediction.
* To analyze factors influencing employee promotion.
* To estimate an employee's promotion probability.
* To provide personalized recommendations for improvement.
* To visualize promotion-related trends and patterns.
* To build an interactive and user-friendly web application.
* To provide data-driven insights through an analytics dashboard.

---

# ✨ Key Features

## 🔮 1. Employee Promotion Prediction

Users can enter employee-related information and obtain a prediction regarding their promotion eligibility.

The system analyzes the provided information using the trained machine-learning model.

---

## 📊 2. Promotion Probability

Along with the prediction, the system provides the estimated probability associated with the prediction.

This provides additional information beyond a simple promoted/not-promoted classification.

---

## 💡 3. Personalized Recommendations

The system provides recommendations based on employee-related factors.

These recommendations can help identify areas where an employee may improve their promotion readiness.

---

## 📈 4. Promotion Analytics Dashboard

The analytics module provides interactive visualizations for understanding employee promotion patterns.

The dashboard can be used to explore relationships between promotion and factors such as:

* Department
* Training
* Performance rating
* KPI achievement
* Awards
* Experience
* Age
* Education

---

## 📋 5. Employee Promotion Reports

The application provides employee-level promotion information that can be used to understand prediction results and supporting factors.

---

## 🖥️ 6. Interactive Streamlit Interface

The complete system is implemented as an interactive web application using **Streamlit**.

The interface provides easy navigation between prediction, analytics, and recommendation modules.

---

# 🧠 Machine Learning Workflow

The project follows a structured machine-learning pipeline:

```text
Dataset Collection
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Missing Value Handling
        ↓
Duplicate Removal
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Preparation
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Serialization
        ↓
Streamlit Deployment
```

---

# 🤖 Machine Learning Model

The prediction system uses an **XGBoost classification model**.

XGBoost is a gradient-boosting algorithm that is effective for structured/tabular datasets and classification problems.

The trained model is stored using Joblib:

```text
promotion_model.pkl
```

The application loads the saved model during execution and uses it to generate predictions.

---

# 📊 Dataset

The project uses an employee promotion dataset containing employee performance, demographic, training, and organizational information.

### Major Features

| Feature                | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `department`           | Employee department                              |
| `region`               | Employee region                                  |
| `education`            | Employee education level                         |
| `gender`               | Employee gender                                  |
| `recruitment_channel`  | Recruitment source                               |
| `no_of_trainings`      | Number of trainings completed                    |
| `age`                  | Employee age                                     |
| `previous_year_rating` | Previous year performance rating                 |
| `length_of_service`    | Years of service                                 |
| `KPIs_met >80%`        | Whether key performance indicators were achieved |
| `awards_won?`          | Whether the employee received an award           |
| `avg_training_score`   | Average training score                           |
| `is_promoted`          | Promotion outcome                                |

The dataset used by the application is:

```text
final_employee_promotion_dataset.csv
```

---

# 🔍 Important Promotion Factors

The project analyzes several factors that can contribute to employee promotion outcomes.

Important factors include:

* Previous year performance rating
* Average training score
* KPI achievement
* Awards won
* Length of service
* Age
* Department
* Region
* Number of trainings
* Education
* Recruitment channel
* Gender

These factors are used by the machine-learning pipeline to identify patterns in employee promotion outcomes.

---

# 🛠️ Technology Stack

## Programming Language

* **Python**

## Machine Learning

* **XGBoost**
* **Scikit-learn**

## Data Processing

* **Pandas**
* **NumPy**

## Data Visualization

* **Plotly**

## Web Application

* **Streamlit**

## Model Serialization

* **Joblib**

## Development Environment

* **VS Code**
* **Git**
* **GitHub**

---

# 📁 Project Structure

```text
EmployeePromotionSystem/
│
├── app.py                          # Home page / hero + summary
│
├── pages/
│   ├── prediction.py                # Single-employee prediction (sectioned form)
│   ├── recommendations.py           # Model-connected, what-if recommendations
│   ├── analytics.py                 # Filterable analytics dashboard
│   ├── report.py                    # Full employee report (download .txt)
│   ├── model_info.py                # Model performance / AI insights (NEW)
│   └── batch_prediction.py          # CSV batch prediction (NEW)
│
├── utils/                           # Shared logic (NEW - removes duplication)
│   ├── data.py                      # Cached dataset loading
│   ├── preprocessing.py             # Loads the SAVED pipeline; no runtime refitting
│   ├── model_utils.py               # Explainability, confidence, what-if analysis
│   └── recommendations.py           # Model-connected recommendation engine
│
├── training/                        # Training & evaluation (NEW, run offline)
│   ├── train_model.py               # Trains/evaluates LR, RF, XGBoost; saves artifacts
│   └── model_comparison.json        # Full metrics for every candidate model
│
├── models/
│   └── promotion_pipeline.joblib    # Bundled encoder + imputers + model + threshold
│
├── backup_original/                 # Untouched copies of every original file
│
├── final_employee_promotion_dataset.csv
├── promotion_model.pkl              # Original model file (kept, superseded by models/)
├── requirements.txt                 # Pinned versions
└── README.md
```

### File Description

| File                                    | Description                                          |
| ---------------------------------------- | ----------------------------------------------------- |
| `app.py`                                 | Main Streamlit application / home page                |
| `pages/prediction.py`                    | Employee promotion prediction with explainability      |
| `pages/recommendations.py`               | Model-connected, what-if personalized recommendations  |
| `pages/analytics.py`                     | Filterable promotion analytics dashboard               |
| `pages/report.py`                        | Full downloadable employee report                       |
| `pages/model_info.py`                    | Transparent model performance / AI insights page       |
| `pages/batch_prediction.py`              | Upload a CSV, get predictions for many employees        |
| `utils/data.py`                          | Cached dataset loading                                  |
| `utils/preprocessing.py`                 | Loads the saved inference pipeline (no runtime refit)   |
| `utils/model_utils.py`                   | Explainability + what-if + input validation helpers     |
| `utils/recommendations.py`               | Recommendation engine driven by real model output       |
| `training/train_model.py`                | Full training/evaluation pipeline (run offline)          |
| `training/model_comparison.json`         | Saved metrics for every candidate model                 |
| `models/promotion_pipeline.joblib`       | Bundled encoder + imputers + model + threshold          |
| `final_employee_promotion_dataset.csv`   | Employee promotion dataset                               |
| `requirements.txt`                       | Pinned required Python packages                          |
| `README.md`                              | Project documentation                                    |

---

# ⚙️ Installation

## 1. Navigate to the Project Directory

```bash
cd EmployeePromotionSystem
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. (Optional) Retrain the Model

The trained pipeline (`models/promotion_pipeline.joblib`) is already included.
To retrain from scratch — e.g. after changing the dataset — run:

```bash
python training/train_model.py
```

This regenerates `models/promotion_pipeline.joblib` and
`training/model_comparison.json`, which the app reads directly.

---

# ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

After executing the command, Streamlit will provide a local URL.

Open the URL in your browser to access the application.

---

# 🧪 A Note on Model Selection (Transparency)

During the upgrade, three candidate models (Logistic Regression, Random
Forest, XGBoost) were trained and validated using a strict, leak-free
80/20 stratified split. The **original** pre-upgrade model was also
evaluated on the exact same held-out test data for a fair comparison.

That comparison revealed the original model's reported test metrics were
nearly identical to its metrics on the *entire* dataset — a strong
indication it was trained on 100% of the data with no genuine held-out
set, so its high scores aren't a trustworthy estimate of real-world
performance. The new XGBoost model, trained and validated with no such
leakage, is what's deployed in `models/promotion_pipeline.joblib`. Full
numbers for every candidate are visible on the in-app **Model Performance**
page and in `training/model_comparison.json`.

---

# 🖥️ Application Modules

The application is organized into multiple modules.

### 🏠 Main Dashboard

Provides the primary interface and navigation for the system.

### 🔮 Prediction

Allows users to enter employee information and obtain:

* Promotion prediction
* Promotion probability
* Employee-level prediction details

### 📊 Analytics

Provides interactive charts and statistical insights related to employee promotion patterns.

### 💡 Recommendations

Provides personalized suggestions based on employee-related factors.

---

# 🔄 System Architecture

```text
                 Employee Data
                       │
                       ▼
              Data Preprocessing
                       │
                       ▼
             Feature Transformation
                       │
                       ▼
              XGBoost ML Model
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      Promotion Prediction   Probability
             │                   │
             └─────────┬─────────┘
                       ▼
             Personalized Insights
                       │
                       ▼
             Streamlit Dashboard
```

---

# 📈 Expected System Output

For a given employee, the system can provide:

```text
Employee Information
        ↓
Promotion Prediction
        ↓
Promotion Probability
        ↓
Performance Insights
        ↓
Personalized Recommendations
```

This allows users to understand not only the prediction but also potential areas for improvement.

---

# 💼 Potential Applications

The system can be used as a decision-support tool for:

* HR departments
* Employee performance analysis
* Promotion-readiness assessment
* Workforce analytics
* Training planning
* Employee development programs

The system is intended to **support human decision-making**, rather than replace organizational HR policies or professional judgment.

---

# 🔮 Future Enhancements

Future versions of the system could include:

* 👤 Employee authentication and role-based access
* ☁️ Cloud deployment
* 📧 Automated email reports
* 📄 PDF employee reports
* 📊 Advanced HR analytics
* 🔎 Explainable AI for individual predictions
* 📈 Employee performance tracking over time
* 🏢 Integration with HR management systems
* 🤖 Comparison of multiple machine-learning algorithms
* 🔐 Secure database integration
* 📱 Responsive interface for mobile devices

---

# 📚 Learning Outcomes

This project demonstrates practical implementation of:

* Machine Learning
* Classification
* Data Preprocessing
* Exploratory Data Analysis
* Feature Engineering
* Model Training
* Model Evaluation
* Model Serialization
* Data Visualization
* Streamlit Application Development
* Git & GitHub Version Control

---

# 👩‍💻 Author

### Keerthana R

**B.Tech – Artificial Intelligence and Data Science**

**Chennai Institute of Technology**

---

# 📌 Project Status

**Status:** Completed / Academic Project

**Repository:** EmployeePromotionSystem

---

# 📄 License

This project is developed for **academic and educational purposes**.

---

## ⭐ Acknowledgement

This project was developed as part of an academic machine-learning project to explore the application of artificial intelligence and data analytics in employee promotion assessment.
