import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

model = keras.models.load_model("NNmodel.keras")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_test_s = x_test.astype("float32") / 255

x_test_s = np.expand_dims(x_test_s, -1)

predictions = model.predict(x_test_s)

y_pred = np.argmax(predictions, axis=1)

wrongly_classified_indices = np.where(y_pred != y_test)[0]


num_wrong = 10
plt.figure(figsize=(20, 5))

for i in range(num_wrong):
    index = wrongly_classified_indices[i]
    
    plt.subplot(1, num_wrong, i + 1)
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.title(f'True: {y_test[index]}\n Predicted: {y_pred[index]}')
    plt.axis('off')

plt.show()
