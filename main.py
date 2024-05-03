#Performing different types of models on sklearn toy datasets

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

