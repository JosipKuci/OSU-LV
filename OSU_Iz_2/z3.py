import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
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

#a)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = keras.Sequential()
model.add(layers.Dense(12, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

#b)
model.compile(loss='crossentropy',
              optimizer = 'adam',
              metrics=['accuracy',])


# TODO: provedi ucenje mreze
model.fit(X_train,y_train, batch_size=10,epochs=150, validation_split=0.2)

predictions = model.predict(X_test)

model.save("NNmodel.keras")

score = model.evaluate(X_test, y_test, verbose= 0)

print(f'Evaluation score: {score}')

y_pred_int = (predictions > 0.5).astype(int)

confusion_matrix=ConfusionMatrixDisplay ( confusion_matrix ( y_test , y_pred_int ) )
confusion_matrix.plot()
plt.show()