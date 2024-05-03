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
