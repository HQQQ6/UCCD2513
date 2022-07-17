#!/usr/bin/env python
# coding: utf-8

# # Tutorial week 4 
# 
# ## Learning outcomes
# 1. Brief walkthrough on GitHub.
# 2. Python fundamentals: data types, custom function, Numpy.
# 3. Gentle introduction to what is computer vision
# 4. Read and save images
# 5. Read and save videos.
# 6. Basic operations on images
# 7. Image resizing

# # Python fundamentals
# 
# Let's learn some *basic Python syntax, data types, castings, lists, tuples and Numpy arrays*.
# ## Python syntax

# In[1]:


print("Hello World")
# You can comment in the codeblock using # at the front of each statement.


# In[2]:


# Docstring and functions
def simple_add(x, y):
    """ This function is for addition of two numbers, x and y.
    """
    return(x+y)


# In[3]:


simple_add(2, 3)


# ## Data types
# Includes but not limited to *integer, float point number, complex number*

# In[4]:


x = 5
y = 4.2
z = 1j
z1 = "I am human"

print(type(x))
print(type(y))
print(type(z))
print(type(z1))


# In[5]:


# We can do casting, specify the data type by these function int(), float(), str()
y = 8/10
z = "7.6"

print(int(y))
print(float(z))


# ## List and tuple
# List and tuple are both variables that can store multiple items. There are 2 major differences between the two:
# 1. List is written with square brackets, while tuple is written with round brackets (parentheses).
# 2. List is changeable, while tuple is unchangeable.

# In[6]:


mylist = ["a", "b", "c"]
print(mylist)

mytuple = ("a", "b", "c")
print(mytuple)


# In[7]:


mylist[0] = 1
print(mylist)


# In[8]:


mytuple[0] = 1


# ## Numpy functions
# 
# Numpy is a Python modules used for working with arrays. Numpy provides array object that is faster than Python lists. This is due to locality of reference. 

# In[9]:


# First, import Numpy libraries
import numpy as np


# In[10]:


# Create Numpy array
x = np.array([1, 2, 3, 4])
print(x)
print(type(x))


# In[11]:


# Array indexing
# first element of the array x and last element of x
print(x[0])
print(x[-1])


# In[12]:


x1 = np.array([5, 6, 7, 8])
x_long = np.concatenate((x, x1), axis = None)
print(x_long)


# An ndarray is a generic multidimensional container for homogeneous data; that is, all of the elements must be the same type. Every array has a **shape**, a tuple indicating the size of each dimension, and a dtype, an object describing the **data type** of the array. 

# In[13]:


print(f"The shape of variable x_long: {x_long.shape}")
print(x_long.dtype)    # dtype attribute is especially useful for debugging purpose.


# ### Exercises
# 1. Try to write a simple custom function to determine whether a given integer is odd or even number.
# 2. Write a simple example code to show that Numpy is more efficient in numerical computation of large arrays of data than equivalent Python list.
# 3. Run the following codes:
# ```python
#     my_arr = np.arange(10)
#     print(my_arr)
#     my_arr[4:7] = 25
#     print(my_arr)
#     arr_slice = my_arr[4:7]
# 
#     # Change the first element of arr_slice to -1
#     arr_slice[0]= -1
# 
#     print(arr_slice)
#     print(my_arr)
# ```
# What do you notice? 

# # What is digital image processing / computer vision?
# 
# As humans, we perceive the 3D structure of the world around us with ease. For example, looking at a framed group portrait, you can easily count and name all the people in the picture and even guess at their emotions from their facial expressions.
# 
# Perceptual psychologists have spent decades trying to comprehend how visual system works and optical illusions have been discovered to solve the puzzle, a complete solution is still far beyond our reach. 
# 
# Computer vision / digital image processing is being utilized in diverse of real world applications:
# - Optical character recognition (OCR): reading handwritten postal codes on letters and automatic plate recognition.
#     ![OCR](image_embed/OCR.png "OCR example")
# - Medical imaging: registering pre-operative and intra-operative imagery or performing long term studies of internal organ.
#     ![medical imaging](image_embed/medical.png "medical image")
# - Self-driving vehicles.
# - Surveillance: monitoring for intruders, analyzing highway traffic and monitoring pools for drowning victims.
# - Fingerprint recognition and biometrics: automatic access authentication as well as forensic applications.

# Consumer level applications include:
# - Stitching: turning overlapping photos into a seamlessly stitched panorama.
#     ![image stitching](image_embed/stitching.jfif "stitching")
# - Morphing: turning a picture of your friend into another.
#     ![blending](image_embed/blending.jfif "blending")
# - Face detection: for improved camera focusing as well as more relevant image searching.
#     ![face detection](image_embed/face_detection.jfif "face_detection")

# # OpenCV in Python
# 
# Setup procedure:
# 1. Install Python and its IDE, preferably Jupyter notebook.
# 2. Install OpenCV module by the following steps:
#     - Open cmd terminal and type in `pip install opencv-contrib-python`.
# 3. Install *Numpy* and *matplotlib* modules.
# 
# Before we jump into the codes, lets briefly walk through what is OpenCV. Created in 1999, OpenCV currently supports a lot of algorithms related to **Computer Vision** and **Machine Learning** and its is expanding day-by-day. OpenCV supports a wide variety of programming languages like C++, Python, Java etc and is available on different platforms like Windows, Linux and so on. OpenCV-Python is the Python API of OpenCV. It combines the best qualites of OpenCV C++ and Python language. How OpenCV-Python works? It is a *Python wrapper around original C++ implementation*. Another upside of using OpenCV-Python is that *OpenCV array structures are converted to and from Numpy arrays*. So whatever operations you can do in Numpy, you can combine it with OpenCV. All in all, OpenCV-Python is an appropriate tool for fast prototyping of computer vision problems.

# ## Representation of an image in OpenCV and Numpy
# An image is a multidimensional array; it has columns and rows of pixels, and each pixel has a value. For different kinds of image data, the pixel value may be formatted in different ways. We can create a 3x3 square black image from scratch by simply creating a 2D NumPy array as shown in the following cells.

# ## Setup

# In[1]:


import sys
# Python 3.7 is required
assert sys.version_info >= (3,7)

import cv2 as cv
import numpy as np

# For reproducibility,
np.random.seed(99)

# Make sure that optimization is enabled
if not cv.useOptimized():
    cv.setUseOptimized(True)

cv.useOptimized()


# In[15]:


img = np.zeros((3, 3), dtype = np.uint8)
print(img)


# Here, each pixel is represented by a single 8-bit integer, which means that the values for each pixel are in 0-255 range, where 0 is black, 255 is white and the in-between values are shades of gray. This is a **grayscale** image. You can use `cv.cvtColor()` to convert the images from one color spaces to another. We will discuss about image color spaces later.

# In[16]:


img_rgb = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
print(img_rgb)


# As you can see now, each pixel is now represented by a three-element array, with each integer representing one of the 3 color channels: B, G and R respectively. 

# In[17]:


# You can always check the shape of an image
print(img_rgb.shape)


# ## Import different formats of images
# There are a wide variety of image file types out there. In this tutorial, we will just cover a few common image formats, which include:
# 1. TIFF(.tif): lossless image files, allowing for high quality images but larger file sizes. Best for high quality prints.
# 2. Bitmap (.bmp): Bitmap images are created by arranging a grid of differently colored pixels. Creating crisp and highly detailed images requires great deal of computing space. Due to its proprietary nature, it is rarely supported in web browsers. 
# 3. JPEG (.jpg): image is compressed to make a smaller file. JPEG files are very common on the internet. 
# 4. GIF (.gif): Good choice for simple images and animations. Supports up to 256 colors, allow for transparency and can be animated. 
# 5. PNG (.png): lossless - compression without loss of quality. PNG is preferred over JPEG for more precise reproduction of source images. Can handle up to 16 million colors, unlike 256 colors supported by GIF.
# 
# ## Read an image
# Lets try to read all the images into the workspace.
# 
# Before that, lets examine `imread()` function:
# ```python
# cv.imread(filename, flag)
# ```
# It takes two arguments:
# 1. The first argument is the image file name (Note: specify the whole path if the image is not in the same directory as the work path).
# 2. The second argument is optional flag that allows you to specify how image should be represented:
#     - cv.IMREAD_UNCHANGED or -1 
#     - cv.IMREAD_GRAYSCALE or 0 
#     - cv.IMREAD_COLOR or 1 (default)
#     
# More flags argument for `cv.imread()`, please refer to the [online documentation](https://docs.opencv.org/4.x/d8/d6a/group__imgcodecs__flags.html#ga61d9b0126a3e57d9277ac48327799c80)

# ## Display an image
# You can display image using `imshow()` function:
# ```python
# cv.imshow(window_name, image)
# ```
# This function again takes 2 arguments:
# 1. First argument is the window name that will be displayed.
# 2. Second argument is the image.
# 
# To display multiple images at once, you need to <mark>call this function multiple times.</mark>
# 
# The `cv.imshow()` function is designed to used alongside `cv.waitKey()` and `cv.destroyAllWindows()` or `cv.destroyWindow()` functions. 
# 
# The `waitKey()` function is a keyboard-binding function. It takes a single argument, which is the time (miliseconds). If the user presses any key within this period, the program continues. If 0 is passed, the program waits indefinitely for a keystroke. You can set the function to detect a specific keystroke like Esc key or any alphabet.
# 
# The function `destroyAllWindows()` destroys all windows we created. If a specific window needs to be destroyed, give that exact window name as the argument.
# 
# JPEG/JFIF is the currently most popular format for storing and transmitting photographic images on the internet. Sometimes Windows 10 saves JPG files as JFIF files. The image file would remain identical in terms of its quality, compression and structure.

# In[18]:


img = cv.imread('lena.jfif')    # default bgr 
img_grayscale = cv.imread('lena.jfif', 0)    # grayscale

cv.imshow('original',img)
cv.imshow('gray', img_grayscale)
cv.waitKey(0)
cv.destroyAllWindows()


# Lets work with different image formats

# In[19]:


# TIFF
img = cv.imread('lena.tif')
cv.imshow('TIFF image', img)
cv.waitKey(0)
cv.destroyAllWindows()


# In[20]:


# Bitmap
img = cv.imread('lena.bmp')
cv.imshow('Bitmap image', img)
cv.waitKey(0)
cv.destroyAllWindows()


# In[21]:


# JPEG
img = cv.imread('lena.jpg')
cv.imshow('JPEG image', img)
cv.waitKey(0)
cv.destroyAllWindows()


# **Warning**  
# You will run into execution error if you use `cv.imread` on GIF file. The workaround is we need to load the gif to `numpy.ndarray` and change the channel orders. You can refer to this [stackoverflow post](https://stackoverflow.com/questions/48163539/how-to-read-gif-from-url-using-opencv-python) for more info.

# In[2]:


import imageio


# In[3]:


gif = imageio.mimread('lena.gif')
# Convert from RGB to BGR format
imgs = [cv.cvtColor(img, cv.COLOR_RGB2BGR) for img in gif]
cv.imshow('GIF image', imgs[0])
cv.waitKey(0)
cv.destroyAllWindows()


# In[5]:


len(gif)


# In[24]:


# PNG
img = cv.imread('lena.png')
cv.imshow('PNG image', img)
cv.waitKey(0)
cv.destroyAllWindows()


# ## Writing an image
# Let's discuss how to write/save an image into the file directory. The function is `cv.imwrite()`:
# ```python
# imwrite(filename, image)
# ```
# 1. First argument is the filename (Must include the extension, like .png, .jpg, .bmp).
# 2. Second argument is the image you want to save.

# In[25]:


cv.imwrite('lena_save.jpg', img)


# ### Exercises
# 4. Load and display the image 'dog.jfif'. Save the image in .png format.

# ## Deals with Video using OpenCV
# 
# A video is nothing but a series of images that are often referred to as frames. So, all you need to do is loop over all the frames in sequence, and then process one frame at a time.
# 
# ### Read and display video from file
# Use `cv.VideoCapture()` class to invoke a VideoCapture object, which is useful to read video file.
# ```python
# cv.VideoCapture(path, apiPreference)
# ```
# 1. First argument is the path to the video file. Input zero for webcam capture.
# 2. Second argument is API preference (optional)
# 
# Furthermore, there are some methods related to VideoCapture object that are worth mentioning:
# * `isOpened()` method returns a boolean indicating whether a video file is opened successfully. 
# * `get()` method retrive metadata associated with the video. It takes one argument (enumerator). `get(3)` --> width of frame, `get(4)` --> height of frame, `get(5)` --> frame rate. More info, please refer to this [online documentation](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html)
# 
# How to read image frame from the file? Create a loop and read one frame at a time from the video stream using `read()` method. It returns a tuple, where the first element is a boolean and the second argument is the video frame. 
# 
# ### Save videos
# In order to save a video file, you first need to create a video writer object from `cv.VideoWriter()` class. Syntax for `cv.VideoWriter()`:
# ```python
# cv.VideoWriter(filename, fourcc, fps, framesize, isColor)
# ```
# 1. First argument is pathname for output file
# 2. fourcc: 4-character code of codec. Fourcc is a 32 bit (4 byte) ASCII Character code used to uniquely identifies video formats. Below are the video codecs which corresponds to .avi video.
#     - `cv.VideoWriter_fourcc('M', 'J', 'P', 'G')` 
#     - `cv.VideoWriter_fourcc(*'XVID')` 
#     - For MP4 video, use `cv.VideoWriter_fourcc(*'MP4V')`
# 3. fps: frame rate of the video stream
# 4. frame_size: (width, height) of frame
# 5. isColor: if not zero (default: `True`), the encoder will encode color frames.

# In[26]:


# Create a VideoCapture object
cap = cv.VideoCapture('img_pexels.mp4')

# Check if the object has been created successfully
if not cap.isOpened():
    print("Unable to create video")

# Read until the video is completed.
while cap.isOpened():
    ret, frame = cap.read()
    
    # if frame is read then ret is True
    if not ret:
        print("Can't receive frame.")
        break
    
    cv.imshow('frame', frame)
    # Press Esc key to exit (27 is ASCII code for Esc). cv.waitKey() returns 32 bit integer values. You can find the ASCII table
    # on this URL: https://theasciicode.com.ar/
    if cv.waitKey(1) & 0xFF == 27:
        break

# destroy the constructor
cap.release()
cv.destroyAllWindows()


# In[8]:


key = cv.waitKey(1) & 0xFF
print(key)


# ## Video using webcam

# In[27]:


cap = cv.VideoCapture(0)

width = int(cap.get(3))
height = int(cap.get(4))
# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
fps = 20
out = cv.VideoWriter('out.avi', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (streaming end)")
        break
    # Horizontal flip (left and right)
    frame = cv.flip(frame, 1)
    # write the flipped frame
    out.write(frame)
    
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()


# The arguments to the constructor of the VideoWriter class deserve special attention. A video's filename must be specified. Any preexisting file with this name is overwritten. A video codec must also be specified. The available codecs may vary from system to system. 

# ## Basic operations on images
# 
# A good knowledge of Numpy is required to write better optimized code with OpenCV. One thing to keep in mind is image are stored as 3D Numpy array.

# In[28]:


# access pixels
img = cv.imread('lena.jpg')

get_ipython().run_line_magic('timeit', 'a = img[20, 50, 0]')
get_ipython().run_line_magic('timeit', 'b = img.item(20,50,0)')


# ## Python scalar vs Numpy scalar

# In[18]:


x = 4

get_ipython().run_line_magic('timeit', 'y = x**2')
get_ipython().run_line_magic('timeit', 'y = x*x')


# In[19]:


y = np.uint8([4])

get_ipython().run_line_magic('timeit', 'z = y*y')
get_ipython().run_line_magic('timeit', 'z = np.square(y)')


# The takeaway here are:
# 1. Python scalar operations are faster that Numpy scalar operations. Numpy has the advantage when the size of array is bigger.
# 
# Aside from the `%timeit` command shown above, OpenCV function like `cv.getTickFrequency` is also capable of capturing the time of execution. Example are shown as below:
# ```python
# e1 = cv.getTickCount()
# # your code
# e2 = cv.getTickCount()
# time = (e2-e1)/cv.getTickFrequency()
# ```
# 
# ## Numpy array slicing
# We pass slice like this: `[start:end]` or `[start:end:step]`. We need to get used to slicing 2D and 3D array.

# ### Exercise
# 5. Extract the region of interest (flower) from the 'flower.jfif'.

# ## Image resizing
# To resize an image, scale it along each axis (height and width), considering the specified _scale factors_ or set the _desired height and width_.
# 
# When resizing an image:
# - It is important to be mindful of the original aspect ratio of the image.
# - Reducing the size of image requires resampling of pixels.
# - Enlarging requires reconstruction through interpolation. Common interpolation are available in OpenCV:
#     * INTER_NEAREST: nearest neighbor interpolation.
#     * INTER_LINEAR: bilinear interpolation.
#     * INTER_CUBIC: bicubic interpolation (generally slower).
# More info can be found in OpenCV [online documentation](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121)
# 
# The function for image resizing is `cv.resize()`:
# ```python
# cv.resize(src, dsize, fx, fy, interpolation)
# ```
# 1. First argument is the input image.
# 2. dsize: the desired output image dimension.
# 3. fx: scale factor along horizontal axis (width).
# 4. fy: scale factor along vertical axis (height).
# 5. interpolation: option flags stated above.
# 
# ### Example 1: Specify specific output dimension

# In[34]:


img = cv.imread('soccer.jpg')
new_width = 300
new_height = 450
img_resize = cv.resize(img, (new_width, new_height), interpolation = cv.INTER_LINEAR)

cv.imshow('downsize', img_resize)
cv.waitKey(0)
cv.destroyAllWindows()


# ### Example 2: Resizing with a scaling factor
# The advantage of using scaling factor for resizing is that it keeps the aspect ratio intact and preserve the display quality.

# In[35]:


img = cv.imread('lena.jfif')
img_resize = cv.resize(img, None, fx = 1.5, fy = 1.5, interpolation = cv.INTER_LINEAR)

cv.imshow('upsize', img_resize)
cv.waitKey(0)
cv.destroyAllWindows()


# ### Exercise
# 6. Enlarge the image "dog.jfif" by using different techniques: 1) Linear interpolation, 2) Cubic interpolation and 3) nearest neighbor interpolation. Comment on the upscaled of all the outputs.

# ## Weekly activity
# 1. _Suggest two ways and write codes to display two images simultaneously_. You can use any image snapped from your handphone,  downloaded from internet or images from week 4 materials on MS teams. The two images are a color image and its corresponding grayscale image. 
# 2. Write codes that performs the following:
#     - Load the video “img_pexels.mp4” into the Python environment, *resize* it and display the videos with smaller frames (The frames can be of any size, as long as it is smaller). You can specify an arbitrary frame rate.
#     - Save it as a separate files: “smaller_img_pexels.avi” or "smaller_img_pexels.mp4"
# 3. Create a **random noise color and grayscale** image. You can set a custom width and height. (Hint: use Numpy functions like `np.array` and `np.reshape`.)
# 4. Extract the region of interest (flower) from the 'flower.jfif'.
# 5. Enlarge the image "dog.jfif" by using different techniques: 1) Linear interpolation, 2) Cubic interpolation and 3) nearest neighbor interpolation. Comment on the upscaled of all the outputs.

# In[ ]:


#Question 1 first way
import cv2 as cv
img = cv.imread('Flower.jpg')    # default bgr 
img_grayscale = cv.imread('Flower.jpg', 0)    # grayscale

cv.imshow('default',img)
cv.imshow('grayScale', img_grayscale)
cv.waitKey(0)
cv.destroyAllWindows()


# In[ ]:


#Question 1 second way
import cv2 as cv
img = cv.imread('Flower.jfif', cv.imread_color)    # default bgr 
img_grayscale = cv.imread('Flower.jfif', cv.imread_grayscale)   # grayscale

cv.imshow('original',img)
cv.imshow('gray', img_grayscale)
cv.waitKey(0)
cv.destroyAllWindows()


# In[ ]:


#Question 2
import cv2 as cv

cap = cv.VideoCapture('img_pexels.mp4')

# Function for downscale the video with smaller frame
def rescale_frame(frame, percent = 75)
    width = int(frame.shape[1] * percent/100)
    height = int(frame.shape[0] * percent/100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation = cv.INTER_AREA)

while True:
    rect, frame = cap.read()
    frame75 = rescale_frame(frame, percent = 75)
    cv.imshow('frame75', frame75)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(* ‘MP4V’)
fps = 20
out = cv.VideoWriter('smaller_img_pexels.mp4', fourcc, 50, (width, height))

# Check if the object has been created successfully
if not cap.isOpened():
    print("Unable to create video")
    
# Read until the video is completed.
while cap.isOpened():
    ret, frame = cap.read()
    
    # if frame is read then ret is True
    if not ret:
        print("Can't receive frame.")
        break
    cv.imshow('frame', frame)
    # Press Esc key to exit (27 is ASCII code for Esc). cv.waitKey() returns 32 bit integer
    # on this URL: https://theasciicode.com.ar/
    if cv.waitKey(1) & 0xFF == 27:
        break
        
# destroy the constructor
cap.release()
out.release()
cv.destroyAllWindows()


# In[ ]:


#Question 3
import numpy as np
from PIL import image as image

def arrayImage():
     array = np.arange(0, 675420, 1, np.uint8)
     array = np.reshape(array, (1000, 750))
     img = image.fromarray(array)
     img.save(‘image.jpg)

arrayImage()


# In[ ]:


#Question 4
import cv2 as cv

img = cv.imread('flower.jfif')
flower = img[20:80, 25:105]
cv.imshow('flower', img)
cv.waitKey(0)
cv.destroyAllWindows()


# In[ ]:


#Question 5
import cv2 as cv
img = cv.imread('dog.jfif')
new_width = 400
new_height = 450
img_enlarge1 = cv.resize(img, (new_width, new_height), interpolation = cv.INTER_LINEAR)         # linear interpolation
img_enlarge2 = cv.resize(img, (new_width, new_height), interpolation = cv.INTER_CUBIC)           # cubic interpolation
Img_enlarge3 = cv.resize(img, (new_width, new_height), interpolation = cv.INTER_NEAREST)     # nearest neighbour interpolation

cv.imshow('enlarge', img_enlarge1)
cv.imshow('enlarge', img_enlarge2)
cv.imshow('enlarge', img_enlarge3)
cv.waitKey(0)
cv.destroyAllWindows()

