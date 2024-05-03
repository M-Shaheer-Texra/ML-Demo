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
print("Diabetes Dataset")
for i in range(50):
    print("--", end="")
print()


diabetes_data = datasets.load_diabetes()

X_diabetes = diabetes_data.data
y_diabetes = diabetes_data.target

X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test = train_test_split(X_diabetes, y_diabetes, test_size=0.3, random_state=32)

decision_tree_model.fit(X_diabetes_train, y_diabetes_train)

y_diabetes_pred = decision_tree_model.predict(X_diabetes_test)

print("Accuracy score For diabetes using Decision Tree:",accuracy_score(y_diabetes_test, y_diabetes_pred))

k_neighbors_model.fit(X_diabetes_train, y_diabetes_train)

y_diabetes_pred = k_neighbors_model.predict(X_diabetes_test)

print("Accuracy score For diabetes using K-Neighbors:",accuracy_score(y_diabetes_test, y_diabetes_pred))

naive_bayes_model = GaussianNB()

naive_bayes_model.fit(X_diabetes_train, y_diabetes_train)

y_diabetes_pred = naive_bayes_model.predict(X_diabetes_test)

print("Accuracy score For diabetes using Naive Bayes:",accuracy_score(y_diabetes_test, y_diabetes_pred))
