from skimage import io
from skimage import util
from skimage.color import rgb2gray
from skimage import exposure
import matplotlib.pyplot as plt

def load_image(image_path):
    r""" Load image from a specified path
    Parameters
    ----------
    image_path : string
                String containing the location of the image.

    Returns
    -------
    image : numpy array
            Numpy array with the values of the pixels in the image
    """
    image = io.imread(image_path)
    return image

def save_image(image, image_path):
    r""" Show individual image
    Parameters
    ----------
    image : numpy array
            Numpy array with the values of the pixels in the image
    image_path : string
            Path location where th eimage will be saved
    """
    image = exposure.rescale_intensity(image)
    io.imsave(image_path, image)

def show_image(image, title=""):
    r""" Show individual image
    Parameters
    ----------
    image : numpy array
            Numpy array with the values of the pixels in the image
    title : string
            Title string to be placed on the header of the image
    """
    io.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_collection(image_list, image_names):
    r""" Show collection of images
    Parameters
    ----------
    image_list : list of numpy array
                List containig the numpy array of the images to be plotted
    image_names : list of string
                List containing the names of the images to be placed as individual titles.
    """
    sz = len(image_list)
    fig, axes = plt.subplots(1,sz,sharex=True,sharey=True)
    ax = axes.ravel()
    i=0
    for image in image_list:
        ax[i].imshow(image,cmap='gray')
        ax[i].set_title(image_names[i])
        i+=1

    for a in ax.ravel():
        a.axis('off')
    plt.show()

def convert_to_grayscale(image):
    gray = rgb2gray(image)
    return gray

def add_gaussian_noise(image, mean=0, var=0.01):
    noisy = util.random_noise(image,mode='gaussian',mean=mean,var=var,seed=None,clip=True)
    return noisy

def add_impulse_noise(image, amount=0.05):
    noisy = util.random_noise(image,'s&p',amount=amount,seed=None,clip=True)
    return noisy
