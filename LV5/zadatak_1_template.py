import numpy as np
import matplotlib
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn . metrics import accuracy_score, confusion_matrix , ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#A)
plt.scatter(X_train[:,0], X_train[:,1], c="red", cmap="hot", marker="x")
plt.scatter(X_test[:,0], X_test[:,1], c="blue", cmap="cool")
plt.legend()
plt.show()

#B)
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_test_p = logistic_regression.predict(X_test)

#C)
b = logistic_regression.intercept_[0]
w1,w2 = logistic_regression.coef_.T

c = -b/w2
m = -w1/w2

xmin, xmax = -4, 4
ymin, ymax = -4, 4
xd = np.array([xmin, xmax])
yd = m*xd+c

plt.plot(xd,yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin,color='orange', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='blue', alpha=0.2)
plt.show()

#D)
print (" Tocnost : " , accuracy_score ( y_test , y_test_p ) )
confusion_matrix=ConfusionMatrixDisplay ( confusion_matrix ( y_test , y_test_p ) )
confusion_matrix.plot()
plt.show()

#E)
x_false=[]
x_true=[]
for i in range(len(y_test)):
    if y_test[i] != y_test_p[i]:
        x_false.append([X_test[i,0],X_test[i,1]])
    else:
        x_true.append([X_test[i,0],X_test[i,1]])
x_true=np.array(x_true)
x_false=np.array(x_false)
plt.scatter(x_true[:,0],x_true[:,1],c="green")
plt.scatter(x_false[:,0],x_false[:,1], c="red")
plt.show()