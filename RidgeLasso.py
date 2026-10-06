'''
Implement Ridge and Lasso regression on the Diabetes dataset. Compare the performance
of these regularized models with standard linear regression.
Tasks:
● Load and preprocess the dataset.
● Implement Ridge and Lasso regression.
● Tune hyperparameters using cross-validation.
● Compare performance metrics (MSE, R-squared) with standard linear regression
'''

#Ridge and Lasso

from sklearn.datasets import load_diabetes
d = load_diabetes()
X = d.data
y = d.target
print(X.shape[0])
print(X.shape[1])
print(d.feature_names)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

results = []
alpha = []

from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)

ridge = Ridge(alpha = 1.0)
ridge.fit(X_train, y_train)
pred_ridge = ridge.predict(X_test)

lasso = Lasso(alpha = 0.1, max_iter = 10000)
lasso.fit(X_train, y_train)
pred_lasso = lasso.predict(X_test)

alphas = [0.001, 0.01, 0.1, 1, 10, 100]

ridgecv = RidgeCV(alphas = alphas, cv = 5)
ridgecv.fit(X_train, y_train)
ridgecv_pred = ridgecv.predict(X_test)

lassocv = LassoCV(alphas = alphas, cv = 5, max_iter = 10000, random_state = 42)
lassocv.fit(X_train, y_train)
lassocv_pred = lassocv.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
print("Linear Regression")
print("MSE: ", mean_squared_error(y_test, y_pred))
print("R2: ", r2_score(y_test, y_pred))
print("\n")
print("Normal Ridge Regression")
print("Best alpha: ", ridge.alpha)
print("MSE: ", mean_squared_error(y_test, pred_ridge))
print("R2: ", r2_score(y_test, pred_ridge))
print("\n")
print("Normal Lasso Regression")
print("Best alpha: ", lasso.alpha)
print("MSE: ", mean_squared_error(y_test, pred_lasso))
print("R2: ", r2_score(y_test, pred_lasso))
print("\n")
print("Tuned ridge Regression")
print("Best alpha: ", ridgecv.alpha_)
print("MSE: ", mean_squared_error(y_test, ridgecv_pred))
print("R2: ", r2_score(y_test, ridgecv_pred))
print("\n")
print("Tuned Lasso Regression")
print("Best alpha: ", lassocv.alpha_)
print("MSE: ", mean_squared_error(y_test, lassocv_pred))
print("R2: ", r2_score(y_test, lassocv_pred))
