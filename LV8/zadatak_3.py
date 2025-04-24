import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image

model = keras.models.load_model("NNmodel.keras")

image = Image.open("test.png")

plt.imshow(image)
plt.title("Image")
plt.axis('off')
plt.show()


img = image.convert('L')  

img_array = np.array(img).astype("float32") / 255

img_array = np.expand_dims(img_array, axis=-1)

img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)

print("Predikcije po klasama:", predictions)

predicted_class = np.argmax(predictions, axis=1)



print(f"Predikcija same znamenke: {predicted_class}")
