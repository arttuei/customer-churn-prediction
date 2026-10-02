# Customer Churn Prediction

Machine learning project that predicts if a customer is likely to leave a telecom service.

## Dataset and License

The dataset is based on the IBM Telco Customer Churn dataset,
obtained from https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset.

The dataset is available under the https://creativecommons.org/licenses/by/4.0/ license.

Original dataset source: https://data.mendeley.com/datasets/phsxg9ssrf/1

The dataset contains information about telecom customers, including:

- Customer demographics
- Contract type
- Length of the customer relationship
- Internet service
- Additional services
- Payment method
- Monthly charges
- Total charges
- Churn status

The `customerID` column is removed because it does not provide useful information for the prediction.

## Technologies

- Python
- Pandas
- Scikit-learn
- Logistic Regression
- One-Hot Encoding
- StandardScaler

## Project structure

```text
customer-churn/
│
├── data/
│   ├── customer_churn.csv
│   └── customer_churn_processed.csv
│
├── clean.py
├── train.py
└── README.md
```

## Data preprocessing

The `clean.py` script performs the initial data cleaning:

1. Reads the original CSV file.
2. Removes the `customerID` column.
3. Converts `TotalCharges` to a numeric value.
4. Removes rows with missing `TotalCharges` values.
5. Converts the target variable `Churn` from `Yes/No` to `1/0`.
6. Saves the processed dataset.

Further preprocessing is performed inside the machine learning pipeline in `train.py`.

Categorical features are converted using One-Hot Encoding, while numerical features are imputed and standardized.

## Machine learning model

The project uses Logistic Regression.

The preprocessing and model are combined into a Scikit-learn `Pipeline`:

```text
Raw data
   │
   ▼
Initial data cleaning
   │
   ▼
Processed dataset
   │
   ▼
Train / test split
   │
   ▼
Preprocessing
   ├── Numerical features
   │   ├── Missing value imputation
   │   └── StandardScaler
   │
   └── Categorical features
       ├── Missing value imputation
       └── One-Hot Encoding
   │
   ▼
Logistic Regression
   │
   ▼
Churn prediction
```

The dataset is divided into training and testing sets using an 80/20 split.

## Model results

The model achieved the following results on the test set:

| Metric          |    Result |
| --------------- | --------: |
| Accuracy        | **0.805** |
| ROC-AUC         | **0.836** |
| Churn precision |  **0.65** |
| Churn recall    |  **0.57** |
| Churn F1-score  |  **0.61** |

### Classification report

```text
              precision    recall  f1-score   support

   No Churn       0.85      0.89      0.87      1033
      Churn       0.65      0.57      0.61       374

    accuracy                          0.80      1407
   macro avg      0.75      0.73      0.74      1407
weighted avg      0.80      0.80      0.80      1407
```

### Confusion matrix

```text
[[917 116]
 [159 215]]
```

The model correctly identified 215 of the 374 customers who churned.

## Feature analysis

The Logistic Regression coefficients were also examined to understand which features were most strongly associated with the model's predictions.

### Features associated with higher churn

Some of the strongest positive coefficients were:

| Feature                          | Coefficient |
| -------------------------------- | ----------: |
| Contract: Month-to-month         |      +0.719 |
| Internet service: Fiber optic    |      +0.703 |
| Total charges                    |      +0.641 |
| Streaming TV: Yes                |      +0.301 |
| Streaming Movies: Yes            |      +0.287 |
| Online Security: No              |      +0.270 |
| Payment method: Electronic check |      +0.260 |
| Tech Support: No                 |      +0.249 |

### Features associated with lower churn

Some of the strongest negative coefficients were:

| Feature               | Coefficient |
| --------------------- | ----------: |
| Tenure                |      -1.350 |
| Contract: Two year    |      -0.670 |
| Monthly charges       |      -0.559 |
| Internet service: DSL |      -0.515 |

## How to run

Clone the repository and install the required packages:

```bash
pip install pandas scikit-learn
```

Run the data cleaning:

```bash
python clean.py
```

Then train and evaluate the model:

```bash
python train.py
```

The program prints the evaluation metrics and feature coefficients.
