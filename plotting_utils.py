import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import zoom
import tensorflow as tf

from interlacer import utils

def ssim(img1, img2):
    C1 = (0.01 * 5)**2
    C2 = (0.03 * 5)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map


def simulate_multicoil_k(image, maps):
    """
    image: (x,y) (not-shifted)
    maps: (x,y,coils)
    """
    image = np.repeat(image[:, :, np.newaxis], maps.shape[2], axis=2)
    sens_image = image*maps
    shift_sens_image = np.fft.ifftshift(sens_image, axes=(0,1))

    k = np.fft.fftshift(np.fft.fft2(shift_sens_image, axes=(0,1)), axes=(0,1))
    return k


def rss_image_from_multicoil_k(k):
    """
    k: (x,y,coils) (shifted)
    """
    img_coils = np.fft.ifft2(np.fft.ifftshift(k, axes=(0,1)), axes=(0,1))
    img = np.sqrt(np.sum(np.square(np.abs(img_coils)), axis=2))
    img = np.fft.fftshift(img)
    
    return img


def rss_image_from_multicoil_img(img):
    """
    k: (x,y,coils) (shifted)
    """
    img = np.sqrt(np.sum(np.square(np.abs(img)), axis=2))
    img = np.fft.fftshift(img)
    
    return img


def plot_img(img, axes=None,rotate=True,psx=None,psy=None,vmin=0,vmax=1):
    img = rss_image_from_multicoil_img(img)
    if(rotate):
        img = np.rot90(img,k=3)
    
    if(psx is not None and psy is not None):
        img = zoom(img, zoom=(1/psx, 1/psy))
    if(axes is not None):
        axes.imshow(img,cmap='gray',vmin=vmin,vmax=vmax)
        axes.axis('off')
    else:
        plt.figure()
        plt.imshow(img,cmap='gray',vmin=vmin,vmax=vmax)
        plt.axis('off')


def plot_img_from_k(k,axes=None,rotate=True,psx=None,psy=None,vmin=0,vmax=1):
    img = rss_image_from_multicoil_k(k)
    img = np.rot90(img,k=3)
    if(psx is not None and psy is not None):
        img = zoom(img, zoom=(1/psx, 1/psy))

    if(axes is not None):
        axes.imshow(img,cmap='gray',vmin=vmin,vmax=vmax)
        axes.axis('off')
    else:
        plt.figure()
        plt.imshow(img,cmap='gray',vmin=vmin,vmax=vmax)
        plt.axis('off')
        
        
def plot_img_crop_from_k(k,x,y,dx,dy,axes=None,rotate=True,psx=None,psy=None,vmin=0,vmax=1):
    img = rss_image_from_multicoil_k(k)
    if(rotate):
        img = np.rot90(img,k=3)
    if(psx is not None and psy is not None):
        img = zoom(img, zoom=(1/psx, 1/psy))
    if(axes is not None):
        axes.imshow(img[x:x+dx,y:y+dy],cmap='gray',vmin=vmin,vmax=vmax)
        axes.axis('off')
    else:
        plt.figure()
        plt.imshow(img[x:x+dx,y:y+dy],cmap='gray',vmin=vmin,vmax=vmax)
        plt.axis('off')
        
        
def plot_img_diff_from_k(k1,k2,axes=None,rotate=True,psx=None,psy=None,vmin=-0.5,vmax=0.5):
    img1 = rss_image_from_multicoil_k(k1)
    img2 = rss_image_from_multicoil_k(k2)    
    if(rotate):
        img1 = np.rot90(img1,k=3)
        img2 = np.rot90(img2,k=3)
    if(psx is not None and psy is not None):
        img1= zoom(img1, zoom=(1/psx, 1/psy))
        img2= zoom(img2, zoom=(1/psx, 1/psy))
    if(axes is not None):
        axes.imshow(img1-img2,cmap='seismic',vmin=vmin,vmax=vmax)
        axes.axis('off')
    else:
        plt.figure()
        plt.imshow(img1-img2,cmap='seismic',vmin=vmin,vmax=vmax)
        plt.axis('off')

        
def plot_img_ssim_from_k(k1,k2,axes=None,rotate=True,psx=None,psy=None):
    img1 = rss_image_from_multicoil_k(k1)
    img2 = rss_image_from_multicoil_k(k2)
    if(rotate):
        img1 = np.rot90(img1,k=3)
        img2 = np.rot90(img2,k=3)
    
    if(psx is not None and psy is not None):
        img1= zoom(img1, zoom=(1/psx, 1/psy))
        img2= zoom(img2, zoom=(1/psx, 1/psy))

    if(axes is not None):
        axes.imshow(ssim(img1,img2),cmap='gist_gray',vmin=0,vmax=1)
        axes.axis('off')
    else:
        plt.figure()
        plt.imshow(ssim(img1,img2),cmap='gist_gray',vmin=0,vmax=1)
        plt.axis('off')
        
        
    
def plot_k(k, axes=None, rotate=True, vmin=-20,vmax=20):
    k = k[:,:,0]
    if(rotate):
        k = np.rot90(k,k=3)
    if(axes is not None):
        axes.imshow(np.log(np.abs(k)+1e-7),cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
        axes.axis('off')
    else:
        plt.figure()
        plt.imshow(np.log(np.abs(k)+1e-7),cmap='gray',vmin=vmin,vmax=vmax)
        plt.axis('off')
        
        
def plot_k_from_img(img, axes=None, rotate=True, vmin=-20,vmax=20):
    k = utils.join_reim_channels(utils.convert_channels_to_frequency_domain(img))
    k = k[0,...,0]
    if(rotate):
        k = np.rot90(k,k=3)
    
    if(axes is not None):
        axes.imshow(np.log(np.abs(k)+1e-7),cmap='gray',vmin=vmin,vmax=vmax)
        axes.axis('off')
    else:
        plt.figure()
        plt.imshow(np.log(np.abs(k)+1e-7),cmap='gray',vmin=vmin,vmax=vmax)
        plt.axis('off')
        
        
def plot_k_diff(k1, k2, axes=None, rotate=True, vmin=-20,vmax=20):
    k1 = k1[:,:,0]
    k2 = k2[:,:,0]
    if(rotate):
        k1 = np.rot90(k1,k=3)
        k2 = np.rot90(k2,k=3)
    if(axes is not None):
        axes.imshow(np.log(np.abs(k1-k2)+1e-7),cmap='gray',vmin=vmin,vmax=vmax)
        axes.axis('off')
    else:
        plt.figure()
        plt.imshow(np.log(np.abs(k1-k2)+1e-7),cmap='gray',vmin=vmin,vmax=vmax)
        plt.axis('off')
        
        
def plot_comparison_results(ex_out,ex_in, recons, labels, ind, rotate=True, x=100, y=200, dx=128, dy=128, psx=None, psy=None, vmin=0, vmax=1):
    if(psx == None):
        psx = 1
    if(psy == None):
        psy = 1
    assert(len(recons)==len(labels))
    n = len(recons)
    
    ex_x = ex_in.shape[1]*psx
    ex_y = ex_in.shape[2]*psy

    to_plot = [ex_out,ex_in]
    to_plot.extend(recons)

    titles = ['Ground Truth', 'Motion-Corrupted']
    titles.extend(labels)

    def process_recon(k):
        k = utils.join_reim_channels(tf.convert_to_tensor(k))
        k = k[ind,:,:,:]
        return k
    
    to_plot = [process_recon(recon) for recon in to_plot]

    matplotlib.rcParams.update({'font.size': 22})

    fig_x = ex_y*(n+2)
    fig_y = ex_x*3+ex_y*ex_in.shape[2]/ex_out.shape[1]

    fig,axes = plt.subplots(4,n+2,figsize=(25,25*4/(n+2)))
    
    if(not rotate):
        hold = psx
        psx = psy
        psy = hold
    for i in range(n+2):
        plot_img_from_k(to_plot[i],axes[0][i],rotate=rotate,psx=psx,psy=psy,vmin=vmin,vmax=vmax)
        axes[0][i].set_title(titles[i])
        plot_img_crop_from_k(to_plot[i],x,y,dx,dy,axes[1][i],rotate=rotate,psx=psx,psy=psy,vmin=vmin,vmax=vmax)
        plot_img_ssim_from_k(to_plot[i],to_plot[0],axes[2][i],rotate=rotate,psx=psx,psy=psy)
        plot_k(to_plot[i],axes[3][i])
        axes[3][i].set_aspect(psy/psx)
        
    axes[2][0].axis('off')
    
    
    plt.subplots_adjust(wspace=0,hspace=0)
