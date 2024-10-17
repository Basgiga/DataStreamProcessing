import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.stats

obr = cv2.imread("dozaladowania.jpg")

def z1():
    cv2.imshow("Image", obr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def z2():
    #histogram szary
    plt.figure()
    plt.hist(obr.ravel(), 256, [0, 256])
    plt.title('histogram szary')
    plt.xlabel("koszyki")
    plt.ylabel("ilość pixeli")
    plt.show()

    #histogram RGB
    chans = cv2.split(obr)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("RGB histogram")
    plt.xlabel("koszyki")
    plt.ylabel("ilość pixeli")
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


def z3(image=None,jak_bardzo_zaszumic=1):

    #pusty obraz
    if image is None or image.size == 0:
        # rozmiar obrazu szumu sol pieprz
        w = 1000
        s = 1000
        image = np.zeros((w, s), dtype=np.uint8)
    else:
        if len(image.shape) > 2:
            w,s,_ = image.shape # obrazek kolorowy
        else:
            w,s = image.shape

    ile_pixeli = random.randint(100, w*s/jak_bardzo_zaszumic)

    #bialy
    for i in range(ile_pixeli):
        y = random.randint(0, w-1)
        x = random.randint(0, s-1)
        image[y,x] = 255
    #czarny
    for i in range(ile_pixeli):
        y = random.randint(0, w-1)
        x = random.randint(0,s-1)
        image[y,x] = 0

    plt.imshow(image, cmap='gray')
    plt.title('Szum "sol i pieprz (z lub bez obrazka)')
    plt.axis('off')
    plt.show()


def z4(image):

    cv2.imshow("oryginal", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(image.shape) > 2:
        w, s, _ = image.shape  # obrazek kolorowy
    else:
        w, s = image.shape

    m = cv2.getRotationMatrix2D((s//2, w//2), 90, 1.0)

    #z racji takiej ze obracamy obrazek a nie ramke trzeba wyciagnac z macierzy rotacji apre wartosci i przesunac obrazek w ramce
    cosinus_z_macierzy = np.abs(m[0, 0])
    sinus_z_macierzy = np.abs(m[0, 1])
    sskala = int((w * sinus_z_macierzy) + (s * cosinus_z_macierzy))
    wskala= int((w * cosinus_z_macierzy) + (s * sinus_z_macierzy))

    m[0, 2] += (sskala / 2) - s//2
    m[1, 2] += (wskala / 2) - w//2

    obrot = cv2.warpAffine(image, m, (sskala, wskala))

    cv2.imshow("obrocony obrazek o 90 stopnii", obrot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def z5():
    # Wymiary obrazu
    w = 100
    s = 100

    #szum riciana
    rician_noise = scipy.stats.rice.rvs(1.0, size=(w, s))

    # szum poissona
    poisson_noise = np.random.poisson(lam=1.0, size=(w, s))

    #szum rayleigha
    rayleigh_noise = scipy.stats.rayleigh.rvs(scale=1.0, size=(w, s))

    plt.figure(figsize=(10, 4))

    plt.subplot(2, 3, 1)
    plt.imshow(rician_noise, cmap='gray')
    plt.title('Szum Riciana')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(poisson_noise, cmap='gray')
    plt.title('Szum Poissona')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(rayleigh_noise, cmap='gray')
    plt.title('Szum Rayleigha')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.hist(rician_noise.ravel(), bins=50, color='blue', alpha=0.7)
    plt.title('Histogram szumu Riciana')

    plt.subplot(2, 3, 5)
    plt.hist(poisson_noise.ravel(), bins=50, color='green', alpha=0.7)
    plt.title('Histogram szumu Poissona')

    plt.subplot(2, 3, 6)
    plt.hist(rayleigh_noise.ravel(), bins=50, color='red', alpha=0.7)
    plt.title('Histogram szumu Rayleigha')

    plt.tight_layout()
    plt.show()



#z1()

#z2()

#z3(obr,1)

#obr2 = cv2.imread('dozaladowania.jpg', cv2.IMREAD_GRAYSCALE)
#z3(obr2,2)

#obr3 = cv2.imread('dozaladowania2.jpg')
#z4(obr3)

#z5()