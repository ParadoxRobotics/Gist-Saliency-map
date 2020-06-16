# Itti .al 1998 Saliency map implementation in Python3
# Author : Munch Quentin

import numpy as np
import cv2
from scipy.ndimage.filters import maximum_filter
import matplotlib as plt

# create gaussian pyramid
def gaussianImagePyramid(img):
    pyramid = []
    pyramid.append(img)
    for i in range(1,9):
        pyramid.append(cv2.pyrDown(dst[i-1]))
    return pyramid

# Center and surround feature map in a gaussian pyramid
def CenterSurroundDiff(featureMap):
    CSDPyr = []
    # we only take the feature map between [2,5] in the gaussian pyramid
    for i in range(2,5):
        currSize = featureMap[i].shape
        currSize = (currSize[1], currSize[0]) # shape = [W, H]
        # first centering = |currentMap - resize(nextMap)|
        ResMp = cv2.resize(featureMap[s+3], currSize, interpolation=cv2.INTER_LINEAR)
        CSDPyr.append(cv2.absdiff(featureMap[i], ResMp))
        # second centering = |currentMap - resize(nextMap)|
        ResMp = cv2.resize(featureMap[s+4], currSize, interpolation=cv2.INTER_LINEAR)
        CSDPyr.append(cv2.absdiff(featureMap[i], ResMp))
    return CSDPyr

# get intensity feature map (grayscaling)
def intensityMap(img):
    # create a gaussian pyramid with the intensity and perform CSD
    intensityPyr = gaussianImagePyramid(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    intensityFeature = CenterSurroundDiff(intensityPyr)
    return intensityFeature

# get color feature map (RG and BY)
def colorMap(img):
    # convert image to RGB and split channel
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(img)
    # calculate RGB max
    maxRGB = cv2.max(B, cv2.max(R, G))
    # normalize to avoid 0 division
    maxRGB[maxRGB <= 0] = 0.0001
    # calculate Red/Green map
    RG = (R-G)/maxRGB
    # calculate Blue/Yellow map
    Y = cv2.min(R, G)
    BY = (B - Y)/maxRGB
    # set nagative values to 0
    RG[RG < 0] = 0
    BY[BY < 0] = 0
    # create a gaussian pyramid with (RG,BY) and perform CSD
    RGPyr = gaussianImagePyramid(RG)
    BYPyr = gaussianImagePyramid(BY)
    colorFeature = [CenterSurroundDiff(RGPyr), CenterSurroundDiff()]
    return colorFeature

# Gabor filter th = [0°,45°,90°,135°]
GaborKernel_0 = [\
    [ 1.85212E-06, 1.28181E-05, -0.000350433, -0.000136537, 0.002010422, -0.000136537, -0.000350433, 1.28181E-05, 1.85212E-06 ],\
    [ 2.80209E-05, 0.000193926, -0.005301717, -0.002065674, 0.030415784, -0.002065674, -0.005301717, 0.000193926, 2.80209E-05 ],\
    [ 0.000195076, 0.001350077, -0.036909595, -0.014380852, 0.211749204, -0.014380852, -0.036909595, 0.001350077, 0.000195076 ],\
    [ 0.000624940, 0.004325061, -0.118242318, -0.046070008, 0.678352526, -0.046070008, -0.118242318, 0.004325061, 0.000624940 ],\
    [ 0.000921261, 0.006375831, -0.174308068, -0.067914552, 1.000000000, -0.067914552, -0.174308068, 0.006375831, 0.000921261 ],\
    [ 0.000624940, 0.004325061, -0.118242318, -0.046070008, 0.678352526, -0.046070008, -0.118242318, 0.004325061, 0.000624940 ],\
    [ 0.000195076, 0.001350077, -0.036909595, -0.014380852, 0.211749204, -0.014380852, -0.036909595, 0.001350077, 0.000195076 ],\
    [ 2.80209E-05, 0.000193926, -0.005301717, -0.002065674, 0.030415784, -0.002065674, -0.005301717, 0.000193926, 2.80209E-05 ],\
    [ 1.85212E-06, 1.28181E-05, -0.000350433, -0.000136537, 0.002010422, -0.000136537, -0.000350433, 1.28181E-05, 1.85212E-06 ]\
]
GaborKernel_45 = [\
    [  4.04180E-06,  2.25320E-05, -0.000279806, -0.001028923,  3.79931E-05,  0.000744712,  0.000132863, -9.04408E-06, -1.01551E-06 ],\
    [  2.25320E-05,  0.000925120,  0.002373205, -0.013561362, -0.022947700,  0.000389916,  0.003516954,  0.000288732, -9.04408E-06 ],\
    [ -0.000279806,  0.002373205,  0.044837725,  0.052928748, -0.139178011, -0.108372072,  0.000847346,  0.003516954,  0.000132863 ],\
    [ -0.001028923, -0.013561362,  0.052928748,  0.460162150,  0.249959607, -0.302454279, -0.108372072,  0.000389916,  0.000744712 ],\
    [  3.79931E-05, -0.022947700, -0.139178011,  0.249959607,  1.000000000,  0.249959607, -0.139178011, -0.022947700,  3.79931E-05 ],\
    [  0.000744712,  0.003899160, -0.108372072, -0.302454279,  0.249959607,  0.460162150,  0.052928748, -0.013561362, -0.001028923 ],\
    [  0.000132863,  0.003516954,  0.000847346, -0.108372072, -0.139178011,  0.052928748,  0.044837725,  0.002373205, -0.000279806 ],\
    [ -9.04408E-06,  0.000288732,  0.003516954,  0.000389916, -0.022947700, -0.013561362,  0.002373205,  0.000925120,  2.25320E-05 ],\
    [ -1.01551E-06, -9.04408E-06,  0.000132863,  0.000744712,  3.79931E-05, -0.001028923, -0.000279806,  2.25320E-05,  4.04180E-06 ]\
]
GaborKernel_90 = [\
    [  1.85212E-06,  2.80209E-05,  0.000195076,  0.000624940,  0.000921261,  0.000624940,  0.000195076,  2.80209E-05,  1.85212E-06 ],\
    [  1.28181E-05,  0.000193926,  0.001350077,  0.004325061,  0.006375831,  0.004325061,  0.001350077,  0.000193926,  1.28181E-05 ],\
    [ -0.000350433, -0.005301717, -0.036909595, -0.118242318, -0.174308068, -0.118242318, -0.036909595, -0.005301717, -0.000350433 ],\
    [ -0.000136537, -0.002065674, -0.014380852, -0.046070008, -0.067914552, -0.046070008, -0.014380852, -0.002065674, -0.000136537 ],\
    [  0.002010422,  0.030415784,  0.211749204,  0.678352526,  1.000000000,  0.678352526,  0.211749204,  0.030415784,  0.002010422 ],\
    [ -0.000136537, -0.002065674, -0.014380852, -0.046070008, -0.067914552, -0.046070008, -0.014380852, -0.002065674, -0.000136537 ],\
    [ -0.000350433, -0.005301717, -0.036909595, -0.118242318, -0.174308068, -0.118242318, -0.036909595, -0.005301717, -0.000350433 ],\
    [  1.28181E-05,  0.000193926,  0.001350077,  0.004325061,  0.006375831,  0.004325061,  0.001350077,  0.000193926,  1.28181E-05 ],\
    [  1.85212E-06,  2.80209E-05,  0.000195076,  0.000624940,  0.000921261,  0.000624940,  0.000195076,  2.80209E-05,  1.85212E-06 ]
]
GaborKernel_135 = [\
    [ -1.01551E-06, -9.04408E-06,  0.000132863,  0.000744712,  3.79931E-05, -0.001028923, -0.000279806, 2.2532E-05, 4.0418E-06 ],\
    [ -9.04408E-06,  0.000288732,  0.003516954,  0.000389916, -0.022947700, -0.013561362, 0.002373205, 0.00092512, 2.2532E-05 ],\
    [  0.000132863,  0.003516954,  0.000847346, -0.108372072, -0.139178011, 0.052928748, 0.044837725, 0.002373205, -0.000279806 ],\
    [  0.000744712,  0.000389916, -0.108372072, -0.302454279,  0.249959607, 0.46016215, 0.052928748, -0.013561362, -0.001028923 ],\
    [  3.79931E-05, -0.022947700, -0.139178011,  0.249959607,  1.000000000, 0.249959607, -0.139178011, -0.0229477, 3.79931E-05 ],\
    [ -0.001028923, -0.013561362,  0.052928748,  0.460162150,  0.249959607, -0.302454279, -0.108372072, 0.000389916, 0.000744712 ],\
    [ -0.000279806,  0.002373205,  0.044837725,  0.052928748, -0.139178011, -0.108372072, 0.000847346, 0.003516954, 0.000132863 ],\
    [  2.25320E-05,  0.000925120,  0.002373205, -0.013561362, -0.022947700, 0.000389916, 0.003516954, 0.000288732, -9.04408E-06 ],\
    [  4.04180E-06,  2.25320E-05, -0.000279806, -0.001028923,  3.79931E-05 , 0.000744712, 0.000132863, -9.04408E-06, -1.01551E-06 ]\
]

# filter bank
filters = [GaborKernel_0, GaborKernel_45, GaborKernel_90, GaborKernel_135]

# get gabor filter map
def gaborMap(img, filters):
    # create a gaussian pyramid with the intensity
    intensityPyr = gaussianImagePyramid(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # create gabor feature map
    gaborPyr_0 = [ np.empty((1,1)), np.empty((1,1)) ]
    gaborPyr_45 = [ np.empty((1,1)), np.empty((1,1)) ]
    gaborPyr_90 = [ np.empty((1,1)), np.empty((1,1)) ]
    gaborPyr_135 = [ np.empty((1,1)), np.empty((1,1)) ]
    # find gabor edge in the intensity feature pyramid
    for i in range(2,9)
        gaborPyr_0.append(cv2.filter2D(intensityPyr[i], cv2.CV_32F, filters[0]))
        gaborPyr_45.append(cv2.filter2D(intensityPyr[i], cv2.CV_32F, filters[1]))
        gaborPyr_90.append(cv2.filter2D(intensityPyr[i], cv2.CV_32F, filters[2]))
        gaborPyr_135.append(cv2.filter2D(intensityPyr[i], cv2.CV_32F, filters[3]))
    # perform CSD on every edge orientation feature pyramid
    gaborFeature_0 = CenterSurroundDiff(gaborPyr_0)
    gaborFeature_45 = CenterSurroundDiff(gaborPyr_45)
    gaborFeature_90 = CenterSurroundDiff(gaborPyr_90)
    gaborFeature_135 = CenterSurroundDiff(gaborPyr_135)
    return [gaborFeature_0, gaborFeature_45, gaborFeature_90, gaborFeature_135]

# get Optical flow feature map
def OpticalFlow(lastImg, curImg):
    # calculate optical flow the last and current image
    
