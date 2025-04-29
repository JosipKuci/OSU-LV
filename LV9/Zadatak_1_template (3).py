import numpy as np
from tensorflow import keras as keras
from keras import layers
from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib import pyplot as plt


# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]),plt.yticks([])
    plt.imshow(X_train[i])

plt.show()


# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# CNN mreza
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn_dropout',
                                update_freq = 100),
    keras . callbacks . EarlyStopping ( monitor ="val_loss" ,
        patience = 5)
]

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X_train_n,
            y_train,
            epochs = 40,
            batch_size = 300,
            callbacks = my_callbacks,
            validation_split = 0.1)


score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')

#Zadatak 9.4.1
#1)
print("Vidimo da model ima 11 slojeva")

#2 i 3)
print("Krivulja točnosti se izravnala nakon 10,000-tog koraka na otprilike 0.98 (98%)\n")
print("Gubitak se normalizirao nakon 10000-tog koraka na otprilike 0.05 (5%)")
print(f'Tocnost na testnom skupu podataka: 73.28%')

#Zadatak 9.4.2
print("Vidimo marginalno veću točnost na testom skupu od otprilike 2%")
print("Također preko grafova vidimo manju točnost pri skupu za trening")

#Zadatak 9.4.3
print("Točnost na testnom skuku se povećala na 76%")

#Zadatak 9.4.4
#1) 
##Jako veliki batch size: 300
##Vidimo manju točnost na testnom skupu podataka
##Jako mali batch size: 10
##Vidimo sporije izvršavanje 
