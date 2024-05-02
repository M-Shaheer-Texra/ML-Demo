#Performing different types of models on sklearn toy datasets


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

print("Wine Dataset")
for i in range(50):
    print("--", end="")
print()

wine_data = datasets.load_wine()
X_wine = wine_data.data
y_wine = wine_data.target

X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_wine_train, y_wine_train)
y_wine_pred = linear_regression_model.predict(X_wine_test)

print("Mean squared error For wine using linear regression:", mean_squared_error(y_wine_test, y_wine_pred))

#Since this is a classification problen we will use logistic regression

logistic_regression_model = LogisticRegression(max_iter=5000)

logistic_regression_model.fit(X_wine_train, y_wine_train)

y_wine_pred = logistic_regression_model.predict(X_wine_test)

print("Mean squared error For wine using logistic regression:", mean_squared_error(y_wine_test, y_wine_pred))

#Since this is a classification problen we will use K-Neighbors

k_neighbors_model = KNeighborsClassifier(n_neighbors=4)

k_neighbors_model.fit(X_wine_train, y_wine_train)

y_wine_pred = k_neighbors_model.predict(X_wine_test)

print("Accuracy score For wine using K-Neighbors:",accuracy_score(y_wine_test, y_wine_pred))

#Since this is a classification problen we will use Decision Tree

decision_tree_model = DecisionTreeClassifier()

decision_tree_model.fit(X_wine_train, y_wine_train)

y_wine_pred = decision_tree_model.predict(X_wine_test)

print("Accuracy score For wine using Decision Tree:",accuracy_score(y_wine_test, y_wine_pred))

print("Iris Dataset")
for i in range(50):
    print("--", end="")
print()

iris_data = datasets.load_iris()

X_iris = iris_data.data
y_iris = iris_data.target

#perform train test split on iris dataset

X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=2)



linear_regression_model.fit(X_iris_train, y_iris_train)

y_iris_pred = linear_regression_model.predict(X_iris_test)


print("Mean squared error For iris using linear regression:",mean_squared_error(y_iris_test, y_iris_pred))

logistic_regression_model.fit(X_iris_train, y_iris_train)

y_iris_pred = logistic_regression_model.predict(X_iris_test)



print("Accuracy score For iris using Logistic regression:",accuracy_score(y_iris_test, y_iris_pred))


k_neighbors_model.fit(X_iris_train, y_iris_train)

y_iris_pred = k_neighbors_model.predict(X_iris_test)

print("Accuracy score For iris using K-Neighbors:",accuracy_score(y_iris_test, y_iris_pred))

decision_tree_model.fit(X_iris_train, y_iris_train)

y_iris_pred = decision_tree_model.predict(X_iris_test)

print("Accuracy score For iris using Decision Tree:",accuracy_score(y_iris_test, y_iris_pred))

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

print(end="\n")
print("Breast Cancer DataSet")
for i in range(50):
    print("--", end="")
print()

breast_cancer_data = datasets.load_breast_cancer()

X_breast_cancer = breast_cancer_data.data

y_breast_cancer = breast_cancer_data.target

X_breast_cancer_train, X_breast_cancer_test, y_breast_cancer_train, y_breast_cancer_test = train_test_split(X_breast_cancer, y_breast_cancer, test_size=0.3, random_state=42)


linear_regression_model.fit(X_breast_cancer_train, y_breast_cancer_train)

y_breast_cancer_pred = linear_regression_model.predict(X_breast_cancer_test)

print("Mean squared error For breast cancer using linear regression:",mean_squared_error(y_breast_cancer_test, y_breast_cancer_pred))

decision_tree_model.fit(X_breast_cancer_train, y_breast_cancer_train)

y_breast_cancer_pred = decision_tree_model.predict(X_breast_cancer_test)

print("Accuracy score For breast cancer using Decision Tree:",accuracy_score(y_breast_cancer_test, y_breast_cancer_pred))

k_neighbors_model.fit(X_breast_cancer_train, y_breast_cancer_train)

y_breast_cancer_pred = k_neighbors_model.predict(X_breast_cancer_test)

print("Accuracy score For breast cancer using K-Neighbors:",accuracy_score(y_breast_cancer_test, y_breast_cancer_pred))

naive_bayes_model.fit(X_breast_cancer_train, y_breast_cancer_train)

y_breast_cancer_pred = naive_bayes_model.predict(X_breast_cancer_test)

print("Accuracy score For breast cancer using Naive Bayes:",accuracy_score(y_breast_cancer_test, y_breast_cancer_pred))


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

print("Accuracy score For golf using Logistic regression:",accuracy_score(y_golf_test, y_golf_pred))



