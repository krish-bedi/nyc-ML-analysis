import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Load the dataset
data = pd.read_csv('./New_York_City_Leading_Causes_of_Death_20240328.csv', na_values='.')

# Drop rows with any missing values
data = data.dropna()

# drop duplicate rows
data = data.drop_duplicates()

# Convert the appropriate columns to numeric
data['Deaths'] = pd.to_numeric(data['Deaths'], errors='coerce')
data['Death Rate'] = pd.to_numeric(data['Death Rate'], errors='coerce')
data['Age Adjusted Death Rate'] = pd.to_numeric(data['Age Adjusted Death Rate'], errors='coerce')

# Create consistency between capitalization
data['Race Ethnicity'] = data['Race Ethnicity'].str.title()  # Capitalizes each word
data['Leading Cause'] = data['Leading Cause'].str.upper()  # Converts to uppercase

# Standardize numerical columns
scaler = StandardScaler(with_mean=False)
data[['Deaths', 'Death Rate', 'Age Adjusted Death Rate']] = scaler.fit_transform(data[['Deaths', 'Death Rate', 'Age Adjusted Death Rate']])

# Data Visualization

# Histogram for Age-Adjusted Death Rate
plt.figure(figsize=(8, 6))
sns.histplot(data['Age Adjusted Death Rate'], bins=10, kde=True)
plt.title('Histogram: Age-Adjusted Death Rate')
plt.xlabel('Age-Adjusted Death Rate')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Bar Chart for Count of Deaths by Leading Cause:
plt.figure(figsize=(12, 8))
sns.countplot(x='Leading Cause', data=data, hue='Leading Cause', palette='Set3', order=data['Leading Cause'].value_counts().index, legend=False)
plt.title('Count of Deaths by Leading Cause')
plt.xlabel('Leading Cause')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Boxplot for Age-Adjusted Death Rate by Race/Ethnicity
plt.figure(figsize=(10, 6))
sns.boxplot(x='Race Ethnicity', y='Age Adjusted Death Rate', data=data, hue='Race Ethnicity', palette='Set3', legend=False)
plt.title('Boxplot: Age-Adjusted Death Rate by Race/Ethnicity')
plt.xlabel('Race/Ethnicity')
plt.ylabel('Age Adjusted Death Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Line Chart for Deaths Over Time (Year)
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Deaths', data=data)
plt.title('Deaths Over Time')
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.tight_layout()
plt.show()
