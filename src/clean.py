import pandas as pd

df = pd.read_csv('data/customer_churn.csv')

# Drop the 'customerID' as it is not useful
df = df.drop(columns=['customerID'])

# Convert 'TotalCharges' to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing values in 'TotalCharges'
df = df.dropna(subset=['TotalCharges'])

# Convert target variable 'Churn' to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df.to_csv("data/customer_churn_processed.csv", index=False)