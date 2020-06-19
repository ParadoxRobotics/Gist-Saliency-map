# Itti .al 1998 Saliency map implementation in Python3
# Author : Munch Quentin

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot

# create gaussian pyramid
def gaussianImagePyramid(img):
    pyramid = []
    pyramid.append(img)
    for i in range(1,9):
        pyramid.append(cv2.pyrDown(pyramid[i-1]))
    return pyramid

# Center and surround feature map in a gaussian pyramid
def CenterSurroundDiff(featureMap):
    CSDPyr = []
    # we only take the feature map between [2,5] in the gaussian pyramid
    for i in range(2,5):
        currSize = featureMap[i].shape
        currSize = (currSize[1], currSize[0]) # shape = [W, H]
        # first centering = |currentMap - resize(nextMap)|
        ResMp = cv2.resize(featureMap[i+3], currSize, interpolation=cv2.INTER_LINEAR)
        CSDPyr.append(cv2.absdiff(featureMap[i], ResMp))
        # second centering = |currentMap - resize(nextMap)|
        ResMp = cv2.resize(featureMap[i+4], currSize, interpolation=cv2.INTER_LINEAR)
        CSDPyr.append(cv2.absdiff(featureMap[i], ResMp))
    return CSDPyr

# normalization for conspicuity map
def valueNorm(featureMap):
    # find global maximal/minimal value in the feature map
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(featureMap)
    if maxVal != minVal:
        normalizedMap = featureMap/(maxVal-minVal) + minVal/(minVal-maxVal)
    else:
        normalizedMap = featureMap-minVal
    return normalizedMap

# compute average of local maxima in the feature map
def LocalAvgMaxima(featureMap, stepSize):
    # feature map shape
    FMHeight = featureMap.shape[1]
    FMWidth = featureMap.shape[0]
    # local max and mean init
    nbLoc = 0
    locMeanMax = 0
    # iterate over the whole feature map
    for i in range(0, FMHeight-stepSize, FMHeight):
        for j in range(0, FMWidth-stepSize, FMWidth):
            # get local image patch
            localPatch = featureMap[i:i+stepSize, j:j+stepSize]
            # calculate local min/max value in the local patch
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(localPatch)
            locMeanMax += maxVal
            nbLoc += 1
    return locMeanMax / nbLoc

# calculate saliency for a feature map in a feature pyramid
def saliencyNorm(featureMap, stepSize):
    normalizedMap = valueNorm(featureMap)
    localAvgMax = LocalAvgMaxima(normalizedMap, stepSize)
    NormCoeff = (1-localAvgMax)*(1-localAvgMax)
    return featureMap * NormCoeff

# calculate saliency for every feature map in a feature pyramid + resize
def ApplySaliencyNorm(featureMap, stepSize, inputSize):
    NormalizedFeatureMap = []
    for i in range(0,6):
        salientMap = saliencyNorm(featureMap[i], stepSize)
        NormalizedFeatureMap.append(cv2.resize(salientMap, inputSize))
    return NormalizedFeatureMap

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
    colorFeature = [CenterSurroundDiff(RGPyr), CenterSurroundDiff(BYPyr)]
    return colorFeature

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
    for i in range(2,9):
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
def OpticalFlowMap(lastImg, curImg):
    # convert to grayscale
    lastImg = cv2.cvtColor(lastImg, cv2.COLOR_BGR2GRAY)
    curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)
    # calculate dense optical flow with the last and current image
    OF = cv2.calcOpticalFlowFarneback(prev=lastImg,
                                      next=curImg,
                                      pyr_scale=0.5,
                                      levels=3,
                                      winsize=15,
                                      iterations=3,
                                      poly_n=5,
                                      poly_sigma=1.2,
                                      flags=0,flow=None)
    OFXPyr = gaussianImagePyramid(OF[...,0])
    OFYPyr = gaussianImagePyramid(OF[...,1])
    OFFeatureX = CenterSurroundDiff(OFXPyr)
    OFFeatureY = CenterSurroundDiff(OFYPyr)
    return [OFFeatureX, OFFeatureY]

# get flicker feature map
def flickerMap(lastImg, curImg):
    # convert to grayscale
    lastIntensityPyr = gaussianImagePyramid(cv2.cvtColor(lastImg, cv2.COLOR_BGR2GRAY))
    curIntensityPyr = gaussianImagePyramid(cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY))
    flickerPyr = []
    for i in range(len(curIntensityPyr)):
        flickerPyr.append(cv2.absdiff(curIntensityPyr[i], lastIntensityPyr[i]))
    flickerFeature = CenterSurroundDiff(flickerPyr)
    return flickerFeature

# get intensity gist
def gistExtractionIntensity(intensityFeature):
    gistIntensity = []
    for fm in range(0, 6):
        gistMap = []
        for i in range(0, 4):
            for j in range(0, 4):
                patch = intensityFeature[fm][int(i*intensityFeature[fm].shape[0]/4):int(i*intensityFeature[fm].shape[0]/4 + intensityFeature[fm].shape[0]/4), \
                int(j*intensityFeature[fm].shape[1]/4):int(j*intensityFeature[fm].shape[1]/4 + intensityFeature[fm].shape[1]/4)]
                localMean = np.mean(patch)
                gistMap.append(localMean)
        gistIntensity.append(gistMap)
    return gistIntensity


# get color gist
def gistExtractionColor(colorFeature):
    gistColor = []
    for cm in range(0,2):
        gistColorMap = []
        for fm in range(0, 6):
            gistMap = []
            for i in range(0, 4):
                for j in range(0, 4):
                    patch = colorFeature[cm][fm][int(i*colorFeature[cm][fm].shape[0]/4):int(i*colorFeature[cm][fm].shape[0]/4 + colorFeature[cm][fm].shape[0]/4), \
                    int(j*colorFeature[cm][fm].shape[1]/4):int(j*colorFeature[cm][fm].shape[1]/4 + colorFeature[cm][fm].shape[1]/4)]
                    localMean = np.mean(patch)
                    gistMap.append(localMean)
            gistColorMap.append(gistMap)
        gistColor.append(gistColorMap)
    return gistColor

# get edge gist
def gistExtractionEdge(edgeFeature):
    gistEdge = []
    for cm in range(0,4):
        gistEdgeMap = []
        for fm in range(0, 6):
            gistMap = []
            for i in range(0, 4):
                for j in range(0, 4):
                    patch = edgeFeature[cm][fm][int(i*edgeFeature[cm][fm].shape[0]/4):int(i*edgeFeature[cm][fm].shape[0]/4 + edgeFeature[cm][fm].shape[0]/4), \
                    int(j*edgeFeature[cm][fm].shape[1]/4):int(j*edgeFeature[cm][fm].shape[1]/4 + edgeFeature[cm][fm].shape[1]/4)]
                    localMean = np.mean(patch)
                    gistMap.append(localMean)
            gistEdgeMap.append(gistMap)
        gistEdge.append(gistEdgeMap)
    return gistEdge

# get optical flow gist
def gistExtractionOF(OFFeature):
    gistOF = []
    for cm in range(0,2):
        gistOFMap = []
        for fm in range(0, 6):
            gistMap = []
            for i in range(0, 4):
                for j in range(0, 4):
                    patch = OFFeature[cm][fm][int(i*OFFeature[cm][fm].shape[0]/4):int(i*OFFeature[cm][fm].shape[0]/4 + OFFeature[cm][fm].shape[0]/4), \
                    int(j*OFFeature[cm][fm].shape[1]/4):int(j*OFFeature[cm][fm].shape[1]/4 + OFFeature[cm][fm].shape[1]/4)]
                    localMean = np.mean(patch)
                    gistMap.append(localMean)
            gistOFMap.append(gistMap)
        gistOF.append(gistOFMap)
    return gistOF

# get flicker gist
def gistExtractionFlicker(flickerFeature):
    gistFlicker = []
    for fm in range(0, 6):
        gistMap = []
        for i in range(0, 4):
            for j in range(0, 4):
                patch = flickerFeature[fm][int(i*flickerFeature[fm].shape[0]/4):int(i*flickerFeature[fm].shape[0]/4 + flickerFeature[fm].shape[0]/4), \
                int(j*flickerFeature[fm].shape[1]/4):int(j*flickerFeature[fm].shape[1]/4 + flickerFeature[fm].shape[1]/4)]
                localMean = np.mean(patch)
                gistMap.append(localMean)
        gistFlicker.append(gistMap)
    return gistFlicker

# get Intensity conspicuity
def conspicuityIntensityMap(intensityFeature, stepSize, inputSize):
    ConsInt = ApplySaliencyNorm(intensityFeature, stepSize, inputSize)
    conspicuityIntensity = sum(ConsInt)
    return conspicuityIntensity

# get color conspicuity
def conspicuityColorMap(colorFeature, stepSize, inputSize):
    # calculate conspicuity for RG and BY feature pyramid
    constRG = ApplySaliencyNorm(colorFeature[0], stepSize, inputSize)
    conspicuityRG = sum(constRG)
    constBy = ApplySaliencyNorm(colorFeature[1], stepSize, inputSize)
    conspicuityBy = sum(constBy)
    # merge the 2 conspicuity map
    return conspicuityRG + conspicuityBy

# get edge conspicuity
def conspicuityEdgeMap(edgeFeature, stepSize, inputSize):
    conspicuityEdge = np.zeros((inputSize[1], inputSize[0]))
    for i in range (0,4):
        # extracting a conspicuity map for every angle
        ConspicuityEdgeTheta = ApplySaliencyNorm(edgeFeature[i], stepSize, inputSize)
        ConspicuityEdgeTheta = sum(ConspicuityEdgeTheta)
        # renormalize
        ConspicuityEdgeTheta = saliencyNorm(ConspicuityEdgeTheta, stepSize)
        # accumulate for every angle
        conspicuityEdge += ConspicuityEdgeTheta
    return conspicuityEdge

# get optical flow conspicuity
def conspicuityOFMap(OFFeature, stepSize, inputSize):
    # calculate conspicuity for DX and DY feature pyramid
    constDx = ApplySaliencyNorm(OFFeature[0], stepSize, inputSize)
    conspicuityDx = sum(constDx)
    constDy = ApplySaliencyNorm(OFFeature[1], stepSize, inputSize)
    conspicuityDy = sum(constDy)
    # merge the 2 conspicuity map
    return conspicuityDx + conspicuityDy

# get flicker conspicuity
def conspicuityFlickerMap(flickerFeature, stepSize, inputSize):
    ConsFlicker = ApplySaliencyNorm(flickerFeature, stepSize, inputSize)
    conspicuityFlicker = sum(ConsFlicker)
    return conspicuityFlicker

def saliencyBinarization(saliency, th):
    # convert channel scale
    saliency8bit = np.uint8(saliency*255)
    thresh, binSaliency = cv2.threshold(saliency8bit, thresh=th, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binSaliency

# compute grabcut given saleincy and the main image
def grabCutSaliency(img, saliency, th, iter):
    # saliency binarization
    binSaliency = saliencyBinarization(saliency, th)
    # perform GrabCut
    img = img.copy()
    mask =  np.where((binSaliency!=0), cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
    background = np.zeros((1,65),np.float64)
    foreground = np.zeros((1,65),np.float64)
    rect = (0,0,1,1)
    cv2.grabCut(img, mask=mask, rect=rect, bgdModel=background, fgdModel=foreground, iterCount=iter, mode=cv2.GC_INIT_WITH_MASK)
    # post-processing
    outMask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
    output = cv2.bitwise_and(img, img, mask=outMask)
    return output

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
    [  4.04180E-06,  2.25320E-05, -0.000279806, -0.001028923,  3.79931E-05, 0.000744712, 0.000132863, -9.04408E-06, -1.01551E-06 ]\
]

# filter bank
filters = [np.array(GaborKernel_0), np.array(GaborKernel_45), np.array(GaborKernel_90), np.array(GaborKernel_135)]

# read images
current_img = cv2.imread('im1.png')
current_img = cv2.resize(current_img, (640,480))
last_img = cv2.imread('im0.png')
last_img = cv2.resize(last_img, (640,480))
# extract feature pyramid + CenterSurroundDiff
intensityFeature = intensityMap(current_img)
colorFeature = colorMap(current_img)
edgeFeature = gaborMap(current_img, filters)
OFFeature = OpticalFlowMap(last_img, current_img)
flickerFeature = flickerMap(last_img, current_img)
# colculate gist of the feature pyramids
gistIntensity = gistExtractionIntensity(intensityFeature)
gistColor = gistExtractionColor(colorFeature)
gistEdge = gistExtractionEdge(edgeFeature)
gistOF = gistExtractionOF(OFFeature)
gistFlicker = gistExtractionFlicker(flickerFeature)
# calculate conspicuity map
conspicuityIntensity = conspicuityIntensityMap(intensityFeature, 16, (640,480))
conspicuityColor = conspicuityColorMap(colorFeature, 16, (640,480))
conspicuityEdge = conspicuityEdgeMap(edgeFeature, 16, (640,480))
conspicuityOF = conspicuityOFMap(OFFeature, 16, (640,480))
conspicuityFlicker = conspicuityFlickerMap(flickerFeature, 16, (640,480))
# compute weighted map given conspicuity map
WI = 0.30
WC = 0.10
WE = 0.40
WO = 0.10
WF = 0.10
weightMap = WI*conspicuityIntensity + WC*conspicuityColor + WE*conspicuityEdge + WO*conspicuityOF + WF*conspicuityFlicker
# normalize the weightMap
normSaliency = valueNorm(weightMap)
saliency = normSaliency.astype(np.float32)

plt.imshow(cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB))
plt.show()

plt.imshow(saliency)
plt.show()

plt.imshow(grabCutSaliency(cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB), saliency, 0, 1))
plt.show()
