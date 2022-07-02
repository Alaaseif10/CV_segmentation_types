import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
img = cv.imread("4.jpg", cv.IMREAD_GRAYSCALE)

# localOtsuThresholding Takes an original image ,blockSize
# to be thresholded and threshold it
# by otsu thresholding method into blocks
# returns thresholded image

def otsu_threshold(image, nbins=0.1):
    
    #validate grayscale
    if len(image.shape) == 1 or len(image.shape) > 2:
        print("must be a grayscale image.")
        return
    
    #validate multicolored
    if np.min(image) == np.max(image):
        print("the image must have multiple colors")
        return
    
    all_colors = image.flatten()
    total_weight = len(all_colors)
    least_variance = -1
    least_variance_threshold = -1
    
    # create an array of all possible threshold values which we want to loop through
    color_thresholds = np.arange(np.min(image)+nbins, np.max(image)-nbins, nbins)
    
    # loop through the thresholds to find the one with the least within class variance
    for color_threshold in color_thresholds:
        bg_pixels = all_colors[all_colors < color_threshold]
        weight_bg = len(bg_pixels) / total_weight
        variance_bg = np.var(bg_pixels)

        fg_pixels = all_colors[all_colors >= color_threshold]
        weight_fg = len(fg_pixels) / total_weight
        variance_fg = np.var(fg_pixels)

        within_class_variance = weight_fg*variance_fg + weight_bg*variance_bg
        if least_variance == -1 or least_variance > within_class_variance:
            least_variance = within_class_variance
            least_variance_threshold = color_threshold
            print("trace:", within_class_variance, color_threshold)
    return least_variance_threshold        

def optimal_threshold(image):
    Corners = [image[0,0], image[0,-1], image[-1, 0], image[-1, -1]]
    BMean = np.mean(Corners)
    FMean = np.mean(image) - BMean
    threshold = (BMean + FMean) / float(2)
    flag = True
    while flag:
        old_thresh = threshold
        ForeHalf = np.extract(image > threshold, image)
        BackHalf = np.extract(image < threshold, image)
        if ForeHalf.size:
            FMean = np.mean(ForeHalf)
        else:
            FMean = 0
        if BackHalf.size:
            BMean = np.mean(BackHalf)
        else:
            BMean = 0
        threshold = (BMean + FMean) / float(2)
        if old_thresh == threshold:
            flag = False
    return threshold

def Global_threshold(image , thresh_typ):
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    thresh_img = np.zeros(image.shape)
    if thresh_typ == "Otsu":
        threshold = otsu_threshold(image)
        # thresh_img = np.uint8(np.where(image > threshold, 255, 0))
        thresh_img = image > threshold
    elif thresh_typ == "Optimal":
         threshold= optimal_threshold(image)
         thresh_img = image > threshold
         thresh_img = np.uint8(np.where(image > threshold, 255, 0))
    return thresh_img

def Local_threshold(image, block_size , thresh_typ ):
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    thresh_img = np.copy(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            mask = image[row:min(row+block_size,image.shape[0]),col:min(col+block_size,image.shape[1])]
            thresh_img[row:min(row+block_size,image.shape[0]),col:min(col+block_size,image.shape[1])] = Global_threshold(mask, thresh_typ)
    return thresh_img






imgg= Local_threshold(img,128,"Optimal")
plt.imshow(imgg)
plt.set_cmap('gray') 
plt.show() 



globall = Global_threshold(img,"Optimal")
plt.imshow(globall)
plt.set_cmap('gray') 
plt.show()  




