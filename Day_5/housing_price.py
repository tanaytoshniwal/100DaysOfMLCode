from sklearn.model_selection import train_test_split
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import sklearn.metrics

import numpy as np

data = datasets.load_boston()

features_x = data.data
labels_y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(features_x, labels_y, test_size = 0.21, random_state = 5)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, Y_train)

Y_pred = linear_regression.predict(X_test)

print(sklearn.metrics.mean_squared_error(Y_test, Y_pred))

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()
