from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ImageNormalize
from astropy.visualization import ZScaleInterval
from astropy.visualization.stretch import LinearStretch




def implot():

    directory = ""#"ToyModelImagesBestOrder2/"
    title = "folded"
    extension = ".fits" 

    image_file = directory + title + extension

    directorybis = "ToyModelImagespdf/"
    titlebis = "folded"
    extensionbis = ".pdf" 

    image_filebis = directorybis + titlebis + extensionbis


    hdu_list = fits.open(image_file)

    image_data_whole = hdu_list[0].data

    hdu_list.close()

    image_data = image_data_whole

    plt.imshow(image_data, cmap='gray')
    #plt.title(title)

    plt.xlabel("$\lambda$")
    plt.ylabel('x')

    plt.xlim([1000, 1600])

    cbar = plt.colorbar(orientation="horizontal", shrink = 0.5,aspect = 50, label="ADU")


    plt.savefig(image_filebis, bbox_inches='tight')

def implot2(image_file):


    title = image_file.split("/")[-1]
    directory = "/home/unahzaal/Documents/StageM2/autoscreenshot2/"
    extension = ".pdf" 

    image_filebis = directory + title + extension


    hdu_list = fits.open(image_file)

    image_data_whole = hdu_list[0].data

    hdu_list.close()



    image_data = image_data_whole

    m = image_data.max()
    interval = ZScaleInterval()
    stretch = LinearStretch(slope=1.5, intercept=0.)
    norm = ImageNormalize(image_data, interval=interval,stretch = stretch)

    #plt.imshow(image_data, cmap='grey',norm = norm)

    #image_data = np.power(image_data,0.3)
    im = plt.imshow(image_data, cmap='magma')

    cbar = plt.colorbar(im,label="Original signal - deformed signal (ADU)",shrink = 0.5)


    #plt.title(title)

    plt.xlabel("y")
    plt.ylabel('x')

    plt.xlim([1000, 1600])

    plt.clim(vmin = -0.05, vmax = 0.05)



    plt.savefig(image_filebis)
    plt.clf()
"""
    title = image_file.split("/")[-1]
    directory = "/home/unahzaal/Documents/StageM2/autoscreenshot/"
    extension = ".pdf" 

    image_filebis = directory + title + "slitlet15" + extension 


    hdu_list = fits.open(image_file)

    image_data_whole = hdu_list[0].data

    hdu_list.close()



    image_data = image_data_whole

    m = image_data.max()
    interval = ZScaleInterval()
    stretch = LinearStretch(slope=1.5, intercept=0.)
    norm = ImageNormalize(image_data, interval=interval,stretch = stretch)

    #plt.imshow(image_data, cmap='grey',norm = norm)

    #image_data = np.power(image_data,0.3)
    im = plt.imshow(image_data, cmap='magma')

    #plt.title(title)

    plt.xlabel("x")
    plt.ylabel('y')

    plt.xlim([779, 838])
    plt.ylim([0, 100])


    plt.clim(vmin = 0000., vmax = 20000.)

    cbar = plt.colorbar(im,label="ADU")


    plt.savefig(image_filebis)
    plt.clf()
"""

def contlot():

    directory = ""#"ToyModelImages/"
    title = "contgoodw"
    extension = ".fits" 

    image_file = directory + title + extension

    directorybis = "ToyModelImagespdf/"
    titlebis = "contgoodw"
    extensionbis = ".pdf" 

    image_filebis = directorybis + titlebis + extensionbis


    hdu_list = fits.open(image_file)

    image_data_whole = hdu_list[0].data

    hdu_list.close()

    image_data = image_data_whole

    #image_data = np.transpose(image_data)


    plt.plot(image_data)

    plt.xlabel("$\lambda$")
    #plt.xlabel('x')
    plt.ylabel('ADU')
    plt.xlim([1600, 2048])






    plt.savefig(image_filebis, bbox_inches='tight')




def genplot():
    import os,glob
    folder_path = '/home/unahzaal/Documents/StageM2/sample2'
    for filename in glob.glob(os.path.join(folder_path, '*.fits')):
        print(filename)
        implot2(filename)



genplot()
