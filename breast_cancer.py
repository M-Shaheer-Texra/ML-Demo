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
print("Breast Cancer DataSet")
for i in range(50):
    print("--", end="")
print()

breast_cancer_data = datasets.load_breast_cancer()

X_breast_cancer = breast_cancer_data.data

y_breast_cancer = breast_cancer_data.target

X_breast_cancer_train, X_breast_cancer_test, y_breast_cancer_train, y_breast_cancer_test = train_test_split(X_breast_cancer, y_breast_cancer, test_size=0.3, random_state=42)

linear_regression_model = LinearRegression()

linear_regression_model.fit(X_breast_cancer_train, y_breast_cancer_train)

y_breast_cancer_pred = linear_regression_model.predict(X_breast_cancer_test)

print("Mean squared error For breast cancer using linear regression:",mean_squared_error(y_breast_cancer_test, y_breast_cancer_pred))

decision_tree_model = DecisionTreeClassifier()

decision_tree_model.fit(X_breast_cancer_train, y_breast_cancer_train)

y_breast_cancer_pred = decision_tree_model.predict(X_breast_cancer_test)

print("Accuracy score For breast cancer using Decision Tree:",accuracy_score(y_breast_cancer_test, y_breast_cancer_pred))

k_neighbors_model = KNeighborsClassifier(n_neighbors=4)

k_neighbors_model.fit(X_breast_cancer_train, y_breast_cancer_train)

y_breast_cancer_pred = k_neighbors_model.predict(X_breast_cancer_test)

print("Accuracy score For breast cancer using K-Neighbors:",accuracy_score(y_breast_cancer_test, y_breast_cancer_pred))

naive_bayes_model = GaussianNB()

naive_bayes_model.fit(X_breast_cancer_train, y_breast_cancer_train)

y_breast_cancer_pred = naive_bayes_model.predict(X_breast_cancer_test)

print("Accuracy score For breast cancer using Naive Bayes:",accuracy_score(y_breast_cancer_test, y_breast_cancer_pred))
