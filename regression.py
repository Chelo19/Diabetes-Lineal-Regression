import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

data_train = pd.read_csv('diabetes_test.csv')   # 1 = Female, 2 = Male, 3 = Other, 0 = No info, 1 = never, 2 = not current, 3 = former, 4 = ever, 5 = current
data_train = data_train.dropna()

data_test = pd.read_csv('test_data.csv')
data_test = data_test.dropna()

X_train = data_train[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y_train = data_train['diabetes']

clf = LinearRegression()
clf.fit(X_train, y_train)

X_test = data_test[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y_test = data_test['diabetes']

predicted = clf.predict(X_test)

mse = mean_squared_error(y_test, predicted)
r2 = r2_score(y_test, predicted)

threshold = 0.4
predicted_binary = [1 if p >= threshold else 0 for p in predicted]
count_ones = predicted_binary.count(1)

accuracy = accuracy_score(y_test, predicted_binary)

print('MSE:', mse)
print('R2:', r2)
print(f'{round(accuracy * 100, 2)}%')