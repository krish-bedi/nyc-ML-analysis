import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('./New_York_City_Leading_Causes_of_Death_20240328.csv', na_values='.')

# Year:                     The year of death.
# Leading Cause:            The cause of death.
# Sex:                      The decedent's sex.
# Race Ethnicity:           The decedent's ethnicity.
# Deaths:                   The number of people who died due to cause of death within the specified gender and race.
# Death Rate:               Refers to the number of deaths per 100,000 population in the specified gender and race in the specified year.
# Age Adjusted Death Rate:  Same as death rate but adjusted for age.

# Drop rows with missing values in relevant columns
data.dropna(subset=['Deaths', 'Age Adjusted Death Rate', 'Race Ethnicity'], inplace=True)

# One-hot encode the 'Race Ethnicity' column
data_encoded = pd.get_dummies(data, columns=['Race Ethnicity'], drop_first=True)

# Select features and target variable
selected_features = ['Age Adjusted Death Rate', 'Race Ethnicity_White Non-Hispanic']
X = data_encoded[selected_features]
y = data_encoded['Deaths']

# Split the data into training 80% and testing sets 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
# 100,200,300 picked to provide good balance between computation time and model performance
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10, 20]}
model = GridSearchCV(RandomForestRegressor(random_state=69), param_grid, cv=5)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model performance
r2 = r2_score(y_test, y_pred)

print(f"R-squared Score: {r2}")
print(f"Best Parameters: {model.best_params_}")

# Plotting feature importance
feature_importance = model.best_estimator_.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
