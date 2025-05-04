import numpy as np
import matplotlib
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn . metrics import accuracy_score, confusion_matrix , ConfusionMatrixDisplay

data = np.loadtxt("pima-indians-diabetes.csv",delimiter=",", dtype=float)
data = [list(i) for i in data]
data = np.unique(data, axis=0)
bmi=data[:,5]
data = data[bmi!=0]

# Extract Age (column 7) and BMI (column 5)
age_bmi = data[:, [7, 5]]  

# Find unique (age, bmi) row combinations
_, unique_indices = np.unique(age_bmi, axis=0, return_index=True)

# Keep only rows with unique (age, bmi)
filtered_data = data[sorted(unique_indices)]
print(filtered_data)
y=filtered_data[:,8]
X=np.delete(filtered_data,8,1)


print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train, y_train)
y_test_p = logistic_regression.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score

print (" Tocnost : " , accuracy_score ( y_test , y_test_p ) )
print (" Preciznost : " , precision_score ( y_test , y_test_p ) )
print (" Odziv : " , recall_score ( y_test , y_test_p ) )



confusion_matrix=ConfusionMatrixDisplay ( confusion_matrix ( y_test , y_test_p ) )
confusion_matrix.plot()
plt.show()

