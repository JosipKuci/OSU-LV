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


data = pd.read_csv("titanic.csv")

data.columns = data.columns.str.strip()

#Priprema dataseta        
data.dropna()
data=data[['Pclass','Embarked','Sex', 'Fare','Survived']]
X=data.drop(columns=['Survived'])
y=data['Survived']

X_numerical=X[['Pclass','Fare']]
X_categorical=X[['Sex', 'Embarked']]

sc = StandardScaler()
X_numerical_n = sc.fit_transform(X_numerical)
X_numerical_n=pd.DataFrame({'Pclass':X_numerical_n[:,0], 'Fare':X_numerical_n[:,1]})


encoder = OneHotEncoder(sparse_output=False)
X_categorical_n = encoder.fit_transform(X_categorical)
X_categorical_n=pd.DataFrame({'Sex':X_categorical_n[:,0], 'Embarked':X_categorical_n[:,1]})
print(X_categorical_n)

X_n=pd.concat([X_numerical_n,X_categorical_n],axis=1)


from sklearn.neighbors import KNeighborsClassifier
# podijeli podatke u omjeru 80-20%
X_train_n, X_test_n, y_train, y_test = train_test_split(X_n, y, test_size = 0.2, stratify=y, random_state = 10)

model = keras.Sequential()
model.add(layers.Dense(12, activation='relu', input_shape=(X_train_n.shape[1],)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.summary()

model.compile(loss='binary_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy',])


print("TensorFlow version:", tf.__version__)
# TODO: provedi ucenje mreze
model.fit(X_train_n,y_train, batch_size=5,epochs=100, validation_split=0.2)

predictions = model.predict(X_test_n)

print(predictions)

score = model.evaluate(X_test_n, y_test, verbose= 0)

y_pred_int = (predictions > 0.5).astype(int)

y_test_int = y_test

print(f"Evaluation score: {score}")

cm = confusion_matrix(y_test_int, y_pred_int)

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(2), yticklabels=np.arange(2))
plt.title('Confusion Matrix for MNIST Test Set')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

print(X_n)