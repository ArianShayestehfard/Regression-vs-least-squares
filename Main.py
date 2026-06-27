import numpy
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pandas .read_csv(r'C:\Users\user\Downloads\dataset.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

def least_squares(X, y):
    x_b = numpy.c_[numpy .ones(X.shape[0]), X]
    return numpy .linalg.pinv(x_b) @ y

w_ls = least_squares(X, y)
X_b = numpy .c_[numpy .ones(X.shape[0]), X]
y_pred_ls = X_b @ w_ls

lr = LinearRegression()
lr.fit(X, y)
y_pred_lr = lr.predict(X)

print("Least Squares:")
print(f"MSE: {mean_squared_error(y, y_pred_ls):.6f}")
print(f"R2: {r2_score(y, y_pred_ls):.6f}\n")

print("Regression:")
print(f"MSE: {mean_squared_error(y, y_pred_lr):.6f}")
print(f"R2: {r2_score(y, y_pred_lr):.6f}")