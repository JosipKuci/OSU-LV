import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

def recolor_image(img,n_clusters,isRGBA=False):

    if isRGBA:
        img = img.astype(np.float64)
    else:
        img = img.astype(np.float64)/255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()

    kmeans = KMeans(n_clusters,random_state=0).fit(img_array_aprox)

    labels = kmeans.predict(img_array_aprox)

    identified_palette = np.array(kmeans.cluster_centers_)

    recolored_img = np.copy(img_array_aprox)
    for index in range(len(recolored_img)):
        recolored_img[index] = identified_palette[labels[index]]

    return recolored_img.reshape(w,h,d)
    
def showImage(original,recolored_image):
    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(original)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title("Kvantizirana slika")
    plt.imshow(recolored_image)
    plt.show()

# ucitaj sliku
img1 = Image.imread("imgs/test_1.jpg")

recolored_image1 = recolor_image(img1,5)

showImage(img1,recolored_image1)

img2 = Image.imread("imgs/test_2.jpg")
img3 = Image.imread("imgs/test_3.jpg")
img4 = Image.imread("imgs/test_4.jpg")
img5 = Image.imread("imgs/test_5.jpg")
img6 = Image.imread("imgs/test_6.jpg")

recolored_image2 = recolor_image(img2,5)
recolored_image3 = recolor_image(img3,5)
recolored_image4 = recolor_image(img4,5, True)
recolored_image5 = recolor_image(img5,5)
recolored_image6 = recolor_image(img6,5)

showImage(img2,recolored_image2)
showImage(img3,recolored_image3)
showImage(img4,recolored_image4)
showImage(img5,recolored_image5)
showImage(img6,recolored_image6)