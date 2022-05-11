from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

def implot():

    directory = "ToyModelImagesBestOrder2/"
    title = "finalreswave"
    extension = ".fits" 

    image_file = directory + title + extension

    directorybis = "ToyModelImagespdf/"
    titlebis = "finalreswave"
    extensionbis = "order2bestcoupledbad.pdf" 

    image_filebis = directorybis + titlebis + extensionbis


    hdu_list = fits.open(image_file)

    image_data_whole = hdu_list[0].data

    hdu_list.close()

    image_data = image_data_whole

    plt.imshow(image_data, cmap='gray')
    #plt.title(title)

    plt.xlabel("$\lambda$")
    plt.ylabel('x')

    plt.xlim([1, 450])

    cbar = plt.colorbar(orientation="horizontal", shrink = 0.5,aspect = 50, label="ADU")


    plt.savefig(image_filebis, bbox_inches='tight')


def contlot():

    directory = "ToyModelImages/"
    title = "ContrastWave"
    extension = ".fits" 

    image_file = directory + title + extension

    directorybis = "ToyModelImagespdf/"
    titlebis = "ContrastWave"
    extensionbis = ".pdf" 

    image_filebis = directorybis + titlebis + extensionbis


    hdu_list = fits.open(image_file)

    image_data_whole = hdu_list[0].data

    hdu_list.close()

    image_data = image_data_whole

    image_data = np.transpose(image_data)


    plt.plot(image_data)

    plt.xlabel("$\lambda$")
    #plt.xlabel('x')
    plt.ylabel('ADU')
    plt.xlim([1600, 2048])





    plt.savefig(image_filebis, bbox_inches='tight')

implot()