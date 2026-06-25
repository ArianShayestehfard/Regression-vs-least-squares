import numpy
import pandas
import matplotlib.pyplot as mplt
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

print("Linear Regression (sklearn):")
print(f"MSE: {mean_squared_error(y, y_pred_lr):.6f}")
print(f"R2: {r2_score(y, y_pred_lr):.6f}")

mplt.figure(figsize=(12, 5))
mplt.subplot(1, 2, 1)
mplt.scatter(y, y_pred_ls, alpha=0.5)
mplt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
mplt.title('Least Squares')
mplt.xlabel('True')
mplt.ylabel('Predicted')

mplt.subplot(1, 2, 2)
mplt.scatter(y, y_pred_lr, alpha=0.5)
mplt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
mplt.title('Regression')
mplt.xlabel('True')
mplt.ylabel('Predicted')
mplt.tight_layout()
mplt.show()