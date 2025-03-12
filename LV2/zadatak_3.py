import numpy as np
import matplotlib . pyplot as plt
def a():
    img = plt.imread("road.jpg")
    img = img[ :,:,0].copy ()
    plt . figure ()
    plt .imshow ( img , cmap ="gray", vmax=100)
    plt . show ()

def b():
    img = plt.imread("road.jpg")
    img = img[:int(img.shape[0]/2):,int(img.shape[1]/2)::,0].copy ()
    plt . figure ()
    plt .imshow ( img , cmap ="gray")
    plt . show ()

def c():
    img = plt.imread("road.jpg")
    img = img[ :,:,0].transpose().copy()
    img = np.array([list(reversed(row)) for row in img])
    plt . figure ()
    plt .imshow ( img , cmap ="gray")
    plt . show ()

def d():
    img = plt.imread("road.jpg")
    img = img[ :,:,0].copy()
    img = np.array([list(reversed(row)) for row in img])
    plt . figure ()
    plt .imshow ( img , cmap ="gray")
    plt . show ()

a() #a) Povecan kontrast
b() #b) Gornji desni kut
c() #c) 90* udesno
d() #d) Zrcaljenje
