from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np

data = np.genfromtxt('data.csv', delimiter=',')

X = data[1:, 1:]
Y = data[1:, 0].reshape((data.shape[0] - 1, 1))

# X feature matrix is of the shape (569, 30)
# Y lable vector is of the shape (569, 1)

# Let's take first 500 examples for training set

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.21, random_state = 42)

clf = GaussianNB()

clf = clf.fit(X_train, Y_train)

X_predict = clf.predict(X_test)

c=0

for i in range(X_predict.shape[0]):
    if X_predict[i] == Y_test[i, 0]:
        c = c+1

print('accuracy:' + str(c/120*100) + '%')