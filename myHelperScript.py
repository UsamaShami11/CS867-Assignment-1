import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

#function to load images
#filetype = '.png' , '.jpg' or '.jpeg'
def loadImage(path,filetype):
  images = [os.path.join(path,file)
  for file in os.listdir(path)
   if file.endswith(filetype)]
  return images

#function to display 1 image
def displayImage(image,title):
  plt.imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB))
  plt.xticks([]), plt.yticks([]) #Axis Markers turned off
  plt.title(title)
  plt.show()

#function to exclude R,G,B channel
def rgbEexclusion(image,channel):
  '''
    # Args:
    #   image: 3d input image (numpy array)
    #   channel: channel to be excluded (order is BGR so, b:0, g:1, r:2.)
  '''
  image[:,:,channel] = 0
  return image

#convolution function
def myConv2D(image, kernel):
    """
    In this function convolution operation is implemented from scratch
    # This function which takes an image and a kernel 
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).
    
    """
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = image
    
    # Loop over every pixel of the image and implement convolution operation (element wise multiplication and summation). 
    # You can use two loops. The result is stored in the variable output.
    
    for x in range(image.shape[0]):     # Loop over every pixel of the image
        for y in range(image.shape[1]):
            # element-wise multiplication and summation 
            output[x,y]=(kernel*image_padded[x:x+3,y:y+3]).sum()
    return output


#function to display 2 images in 1x2 Subplot
def displayTwoImages(image1,image2,title1,title2):

  fig = plt.figure()
  fig.add_subplot(1, 2, 1)
  plt.imshow(cv.cvtColor(image1,cv.COLOR_BGR2RGB), cmap = 'gray')
  plt.title(title1), plt.xticks([]), plt.yticks([]) #Axis Markers turned off

  fig.add_subplot(1, 2, 2)
  plt.imshow(cv.cvtColor(image2,cv.COLOR_BGR2RGB), cmap = 'gray')
  plt.title(title2), plt.xticks([]), plt.yticks([]) #Axis Markers turned off
  plt.show()


#function to display 3 images in 1x3 Subplot
def displayThreeImages(image1,image2,image3,title1,title2,title3):
  
  fig = plt.figure(figsize=(10,25)) #zoomed in figure size
  fig.add_subplot(1, 3, 1)
  plt.imshow(image1, cmap = 'gray')
  plt.title(title1), plt.xticks([]), plt.yticks([])

  fig.add_subplot(1, 3, 2)
  plt.imshow(image2, cmap = 'gray')
  plt.title(title2), plt.xticks([]), plt.yticks([])

  fig.add_subplot(1, 3, 3)
  plt.imshow(image3, cmap = 'gray')
  plt.title(title3), plt.xticks([]), plt.yticks([])
  plt.show()


#function to plot 3D Contours of Filter Kernels
def kernel3dPlot(kernel,title,contFreq,cmapVal):
  '''
    # Args:
    #   kernel:   n by n filter kernel (numpy array)
    #   title:    title of 3D plot (string)
    #   contFreq: frequency of Contours for plotting (int)
    #   cmapVal:  color map value (string)
  '''
  i = kernel.shape[0]
  x = np.linspace(-i, i, i)
  y = np.linspace(-i, i, i)
  X, Y = np.meshgrid(x, y)

  plt.figure()
  ax = plt.axes(projection='3d')
  ax.contour3D(X, Y, kernel, contFreq, cmap=cmapVal)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.set_title(title)
  plt.show()
