# Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting customer churn using machine learning classification algorithms. Customer churn occurs when customers stop using a company's products or services.

The project uses the **Churn Modelling dataset** to train and compare three machine learning models:

- Random Forest Classifier
- Logistic Regression
- Gradient Boosting Classifier

The goal is to identify customers who are likely to leave and evaluate model performance using classification metrics, confusion matrices, and ROC-AUC.

## 🎯 Objectives

- Analyze customer data for churn prediction.
- Clean and preprocess the dataset.
- Encode categorical features.
- Apply feature scaling where appropriate.
- Train multiple classification models.
- Handle class imbalance using class weights and prediction threshold adjustment.
- Evaluate model performance on unseen test data.
- Visualize model performance using an ROC curve.

## 🛠️ Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- NumPy

## 📂 Dataset

**Dataset:** `Churn_Modelling.csv`

The dataset contains customer information, including demographic details, account balance, credit score, product usage, and customer activity.

The target variable is:

| Column | Description |
|---|---|
| `Exited` | Indicates whether the customer left the bank (1) or stayed (0) |

The following identifier columns are removed because they are not used as predictive features:

- `RowNumber`
- `CustomerId`
- `Surname`

The `Geography` column is converted into dummy variables using one-hot encoding, while `Gender` is converted into numeric labels using `LabelEncoder`.

**Note:** The dataset file is not included in this repository. Add your own copy locally if you have the right to use and distribute it.

## ⚙️ Project Workflow

### 1. Data Loading and Preprocessing

- Load the dataset using Pandas.
- Remove identifier columns.
- Apply one-hot encoding to `Geography`.
- Encode `Gender` numerically.
- Separate the input features and target variable.

### 2. Train-Test Split

The dataset is split into training and testing sets using an 80:20 ratio.

```python
train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Stratification helps preserve the proportion of churn and non-churn cases in both sets.

### 3. Feature Scaling

`StandardScaler` is used for Logistic Regression and Gradient Boosting in this implementation.

The scaler is fitted only on the training data and then applied to the test data to avoid data leakage during scaling.

### 4. Random Forest Classifier

Random Forest combines multiple decision trees to make classification predictions.

Configuration:

- `n_estimators=100`
- `max_depth=5`
- `class_weight='balanced'`
- `random_state=42`

The model is evaluated using test accuracy, ROC-AUC, and a classification report.

### 5. Logistic Regression

Logistic Regression is trained with standardized features and balanced class weights.

Configuration:

- `class_weight='balanced'`
- `max_iter=2000`
- Prediction threshold: `0.3`

Instead of using the default threshold of 0.5, the model classifies a customer as likely to churn when the predicted probability exceeds 0.3.

This lower threshold can increase recall, but it may also increase false-positive predictions.

### 6. Gradient Boosting Classifier

Gradient Boosting builds an ensemble of decision trees sequentially, with each stage attempting to improve on the previous ensemble.

Configuration:

- `n_estimators=100`
- `learning_rate=0.1`
- `max_depth=3`
- `random_state=42`

The model is evaluated using a confusion matrix, classification report, and ROC curve.

## 📊 Evaluation Metrics

The project uses the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correct predictions |
| Precision | Measures how many predicted churn cases are actually churn cases |
| Recall | Measures how many actual churn cases are correctly identified |
| F1 Score | Balances precision and recall |
| ROC-AUC | Measures the model's ability to rank churn cases above non-churn cases |
| Confusion Matrix | Shows true positives, true negatives, false positives, and false negatives |

Recall is particularly important in churn prediction because failing to identify a customer who will leave may mean missing an opportunity to retain them.

However, precision must also be considered because incorrectly flagging customers can waste retention resources.

## 📈 Visualization

The project plots the ROC curve for the Gradient Boosting model.

The ROC curve compares the true positive rate against the false positive rate at different classification thresholds. The ROC-AUC score summarizes the model's ranking performance.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/customer-churn-prediction.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Navigate to the project folder

```bash
cd customer-churn-prediction
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 5. Configure the dataset path

Place `Churn_Modelling.csv` in your project folder and update the code:

```python
df = pd.read_csv("Churn_Modelling.csv")
```

### 6. Run the project

Save your script as `churn_prediction.py` and execute:

```bash
python churn_prediction.py
```

The program prints evaluation results and displays the Gradient Boosting ROC curve.

## 🔬 Future Improvements

- Compare all models using the same metrics and a single evaluation table.
- Use cross-validation for more reliable performance estimates.
- Tune hyperparameters using `GridSearchCV` or `RandomizedSearchCV`.
- Compare precision-recall curves and evaluate different thresholds.
- Analyze feature importance and explain individual predictions.
- Build an interactive dashboard using Streamlit.
- Deploy the selected model as a web application or API.
- Track model performance on new customer data.

## ⚠️ Limitations

- The current implementation uses a single train-test split.
- The models have not yet been compared using a consistent set of evaluation metrics.
- The Logistic Regression threshold of 0.3 is manually selected and should be validated against business objectives.
- Scaling is generally not necessary for tree-based models such as Random Forest and Gradient Boosting.
- Results depend on dataset quality, class imbalance, and model configuration.

## 👩‍💻 Author

**Your Name**

GitHub: https://github.com/YOUR-USERNAME

---

⭐ If you find this project useful, consider giving the repository a star.
