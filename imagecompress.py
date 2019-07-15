import cv2
import numpy as np
import os
import sys
import scipy.fftpack
import seaborn as sns
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

from pictureAnalysis import showImageFromFile, showImage, getFileSize

def main():
    filePath = 'img11.jpg'
    img = cv2.imread(filePath,0) #Read image
    resizedImage = resizeImage(img,0.25,0.25)
    # showImage(resizedImage) # Show resized image
    fileSize = getFileSize(filePath) #File size in bytes

    # dft = np.fft.fft2(resizedImage) # 2D DFT
    # dct = scipy.fftpack.dct(resizedImage, norm = 'ortho')
    dft = np.fft.fft2(resizedImage)
    dimensions = dft.shape
    # showImage(r)
    # print(r)
    # print('------------')
    # print(resizedImage)

    dftArr = np.asarray(dft).reshape(-1)
    dftArr = np.absolute(dftArr)
    # print(dftArr)
    # Sort in descending order

    sortedDFT = -np.sort(-dftArr)
    # Find argsort

    sortedArgs = np.argsort(-dftArr)
    # print(sortedArgs)

    numPixels = dimensions[0]*dimensions[1]
    # print(numPixels)
    numCoeffs = round(0.1*numPixels)
    # numCoeffs = numPixels
    coeffs = sortedDFT[0:numCoeffs]
    coeffIndices = sortedArgs[0:numCoeffs]

    # Convert coeffIndices into matrix dimensions

    # coeffLocations = [convertToMatrix(x, dimensions) for x in coeffIndices]
    i = 0
    coeffLocations = []

    coeffLocations = [convertToMatrix(x, dimensions) for x in coeffIndices]

    # Reconstruct image
    imageRec = np.zeros(dimensions, dtype = np.csingle)
    for i in range(0, numCoeffs):
        loc = coeffLocations[i]
        imageRec[loc] = dft[loc]


    reconstructedImage = np.fft.ifft2(imageRec)
    r = np.absolute(reconstructedImage/255)
    showImage(r)
    cv2.imwrite('test3.png',np.absolute(reconstructedImage))




def resizeImage(image,scaleX,scaleY):
    return cv2.resize(image, (0,0), fx=scaleX, fy=scaleY)


def convertToMatrix(index,dimensions):
    return np.unravel_index(index, dimensions)



if __name__ == '__main__':
    main()
