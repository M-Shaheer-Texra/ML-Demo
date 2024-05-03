import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder

print(end="\n")
print("Golf Weather Data")
for i in range(50):
    print("--", end="")
print()

golf_data = pd.read_csv("data/golf.csv")

print(golf_data)

X_golf = golf_data.drop(columns=["Play", "Temperature", "Humidity"], axis=1)


oe = OneHotEncoder()
X_golf = oe.fit_transform(X_golf)

print(X_golf)

y_golf = golf_data["Play"]

le = LabelEncoder()
y_golf = le.fit_transform(y_golf)

X_golf_train, X_golf_test, y_golf_train, y_golf_test = train_test_split(X_golf, y_golf, test_size=0.3, random_state=22)

linear_regression_model.fit(X_golf_train, y_golf_train)

y_golf_pred = linear_regression_model.predict(X_golf_test)

print("Mean squared error For golf using linear regression:",mean_squared_error(y_golf_test, y_golf_pred))

logistic_regression_model.fit(X_golf_train, y_golf_train)

y_golf_pred = logistic_regression_model.predict(X_golf_test)
