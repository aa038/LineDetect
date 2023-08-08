#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 2 03:28:53 2023

@author: daniel
"""
import numpy as np
from typing import Tuple

def featureFinder(wavelength: np.ndarray, flux: np.ndarray, yC: np.ndarray,  sigFlux: np.ndarray, sig_yC: np.ndarray,
    N_sig_continuum: float = 3,  N_sig_limits: float = 0.5) -> np.ndarray:
    """
    Find absorption or emission features in a spectrum to fit Legendre polynomials across them for a better continuum estimation.
    
    Args:
        Lambda (np.ndarray): Array of wavelengths.
        flux (np.ndarray): Array of flux values corresponding to the wavelengths.
        yC (np.ndarray): Array of continuum values.
        sigFlux (np.ndarray): Array of uncertainties in the flux values.
        sig_yC (np.ndarray): Array of uncertainties in the continuum values.
        N_sig_continuum (int): Number of standard deviations in equivalent width, above which a line is deemed statsitically significant. Defaults to 3.
        N_sig_limits (int): Threshold of flux recovery for determining feature limits. 
            Defaults to 3. Can be set to 5 for a higher significant level.

    Returns:
        2D array of feature limits, where each row contains the left and right limit of a feature.
    """

    #Define an empty array that will hold the limits of the features
    featureRange = []

    #Go through the spectrum and check for absorption/emission features
    i = 1
    while i < len(wavelength):
        #Check if there is absorption/emission at the current pixel
        eqWidth, deltaEqWidth = aperturePixelEW(i, wavelength, flux, yC, sigFlux, sig_yC)

        #If there is a statistically significant feature, get the limits of the feature
        if np.abs(eqWidth / deltaEqWidth) >= N_sig_continuum:
            left, right = apertureFeatureLimits(i, wavelength, flux, yC, sigFlux, sig_yC, N_sig_limits=N_sig_limits)
            #Write it to the list of features
            featureRange = featureRange + [left, right]
            #Once a feature is found, skip over to the right of the feature
            i = right + 1
            continue

        #Increment the iterator by 1 if a feature is not found
        i += 1

    featureRange = np.array(featureRange)

    return featureRange

def apertureFeatureLimits(i: int, wavelength: np.ndarray, flux: np.ndarray, yC: np.ndarray, sigFlux: np.ndarray, sig_yC: np.ndarray,
    N_sig_limits: int = 0.5) -> Tuple[int, int]:
    """
    Returns the indices of the leftmost and rightmost pixels of a feature centered at the given index.

    Parameters:
        i (int): Index of the central pixel of the feature.
        Lambda (np.ndarray): Array of wavelength values.
        flux (np.ndarray): Array of flux values.
        yC (np.ndarray): Array of continuum values.
        sigFlux (np.ndarray): Array of uncertainties in the flux values.
        sig_yC (np.ndarray): Array of uncertainties in the continuum values.
        N_sig (int): Threshold of flux recovery for determining feature limits. Defaults to 0.5.

    Returns:
        Two valeues, index of the leftmost pixel of the feature followed by the index of the rightmost pixel of the feature.
    """

    #Define variables for scanning blueward and redward
    j = k = i

    # Scan blueward to find the left limit of the feature
    while j >= 0:
        eqWidth, deltaEqWidth = aperturePixelEW(j, wavelength, flux, yC, sigFlux, sig_yC)
        #Check if the flux recovers sufficiently at this, if so, break
        if eqWidth / deltaEqWidth <= N_sig_limits:
            break
        j -= 1 #If it doesn't recover, move to the preceding pixel
    
    #Scan redward to find the right limit of the feature
    #Ensure k does not run out of bounds of the spectrum
    while k < len(wavelength):
        eqWidth, deltaEqWidth = aperturePixelEW(k, wavelength, flux, yC, sigFlux, sig_yC)
        #Check if the flux recovers sufficiently at this pixel
        if eqWidth / deltaEqWidth <= N_sig_limits:
            break
        k += 1 # If it doesn't recover, move to the next pixel, if so, break

    return j, k

def optimizedFeatureLimits(i: int, Lambda: np.array, flux: np.array, yC: np.array, sigFlux: np.array, sig_yC: np.array, R: np.ndarray, 
    N_sig_limits: float = 0.5, resolution_element: int = 3) -> Tuple[int, int]:
    """
    Finds the left and right limits of a feature based on the recovery of the flux.
    
    Parameters:
        i (int): Index of the pixel.
        Lambda (np.array): Array of wavelength values.
        flux (np.array): Array of flux values.
        yC (np.array): Array of continuum values.
        sigFlux (np.array): Array of flux uncertainties.
        sig_yC (np.array): Array of continuum uncertainties.
        R (np.ndarray): Array of resolving powers.
        N_sig (float): Threshold of flux recovery for determining feature limits. Defaults to 0.5.
        resolution_element (int): The size of the resolution element in pixels. Defaults to 3.
    
    Returns:
        left_index (int): Index of the left limit of the feature.
        right_index (int): Index of the right limit of the feature.
    """
    

    # Define variables for left and right indices
    left_index, right_index = i, i 
    
    #Scan blueward to find the left limit of the feature
    while left_index >= 0:
        eqWidth, deltaEqWidth = optimizedResEleEW(left_index, Lambda, flux, yC, sigFlux, sig_yC, R, resolution_element)

        # Does the flux recover sufficiently at this pixel?
        if eqWidth / deltaEqWidth <= N_sig_limits:
            break #If so, exit the loop
        left_index -= 1 #If not, start again for the preceding pixel i.e. decrement the pixel by 1
        
    #Scan redward to find the right limit of the feature
    while right_index < len(Lambda) - 1:  #Max right_index allowed is less than the ISF width
        eqWidth, deltaEqWidth = optimizedResEleEW(right_index, Lambda, flux, yC, sigFlux, sig_yC, R, resolution_element)
        
        #Does the flux recover sufficiently at this pixel?
        if eqWidth / deltaEqWidth <= N_sig_limits:
            break #If so, exit the loop
        right_index += 1 # If not, start again for the next pixel i.e. increment the pixel by 1
    
    #Handle cases where the function cannot find valid feature limits
    if left_index < 0 or right_index >= len(Lambda) - 1:
        raise ValueError("Could not find valid feature limits.")
    
    #Return the INDICES (NOT WAVELENGTH!!) that form the limits of the feature
    return left_index, right_index

def optimizedResEleEW(i: int, Lambda: np.ndarray, flux: np.ndarray, yC: np.ndarray, sigFlux: np.ndarray, sig_yC: np.ndarray,
    R: np.ndarray, resolution_element: int = 3) -> Tuple[int, int]:
    """
    Calculates the equivalent width per resolution element using the optimized method.

    Args:
        i (int): Index of the pixel.
        Lambda (np.ndarray): Array of wavelength values.
        flux (np.ndarray): Array of flux values.
        yC (np.ndarray): Array of continuum values.
        sigFlux (np.ndarray): Array of flux uncertainties.
        sig_yC (np.ndarray): Array of continuum uncertainties.
        R (np.ndarray): Array of resolving powers.
        resolution_element (int): The size of the resolution element in pixels. Defaults to 3.

    Returns:
        Two values, the equivalent width per resolution element and its uncertainty.
    """

    J0 = 2 * resolution_element

    if i + J0 + 1 >= len(Lambda) - 1:
        return 0, 1

    #Determine the pixel width
    deltaLambda = Lambda[i+1] - Lambda[i]

    #Get the array containing P from i - J0 to i + J0
    P = getP(i, Lambda, R, resolution_element=resolution_element)

    #Compute P^2
    sqP = np.sum(P**2)

    #Calculating EW per res element
    eqWidth = 0
    for j in range(2*J0 + 1):
        eqWidth += P[j] * fluxDec(i + j - J0, flux, yC, sigFlux, sig_yC)[0]

    #EW per res element. Multiplying by the constant coefficient
    coeff = (deltaLambda / sqP)
    eqWidth = coeff * eqWidth

    #Uncertainty - EW per res element
    deltaEqWidth = 0
    for j in range(2*J0 + 1):
        deltaEqWidth += P[j]**2 * fluxDec(i + j - J0, flux, yC, sigFlux, sig_yC)[1]**2
    
    #Uncertainty - EW per res element
    deltaEqWidth = coeff * np.sqrt(deltaEqWidth)

    return eqWidth, deltaEqWidth

def getP(i: int, Lambda: np.ndarray, R: np.ndarray, resolution_element: int = 3) -> np.ndarray:
    """
    Calculates the instrumental spread function around a pixel i using a discrete or continuous Gaussian model.
    
    Note:
        The continuous method is more accurate than the discrete method, since it models the ISF as 
        a continuous function rather than a discrete sum. However, it is also more computationally expensive, 
        since it requires the evaluation of a Gaussian function at every pixel within a certain range. The discrete 
        method is less accurate, but much faster to compute.
    
    Args:
        i (int): Index of the pixel.
        Lambda (np.ndarray): Array of wavelength values.
        R (np.ndarray): Array of resolving powers.
        resolution_element (int): The size of the resolution element in pixels. Defaults to 3.
    
    Returns:
        Array of normalized instrumental spread function values.
    """

    #Define J = 2*resolution_element
    J0 = 2 * resolution_element 

    #Expression for the uncertainty in the ISF
    sigISF = Lambda[i] / (2.35 * R[i])

    #Range of pixels over which the ISF is computed (i - J0 to i + J0)
    x = np.zeros(2*J0 + 1)
    #Gaussian model of the ISF
    P = np.zeros(2*J0 + 1)

    #Compute the x values and corresponding P_n (n is j here) around the pixel i
    for j in range(2*J0 + 1):
        #If the resolution element goes out of bounds of spectrum, end the loop
        if i + j - J0 >= len(Lambda) - 1:
            break
            
        #If not, find the value for the ISF
        x[j] = (Lambda[i] - Lambda[i + j - J0]) / sigISF ### Not j - 1 since j starts at 0, not 1
        P[j] = np.exp(-x[j] ** 2)

    #Return the normalized instrumental spread function
    return P / np.sum(P)

def fluxDec(i: int, flux: np.ndarray, yC: np.ndarray, sigFlux: np.ndarray, sig_yC: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the flux decrement at a pixel i.

    If the flux decrement satisfies some detection threshold at a pixel, 
    go left and right from the pixel to find the points where the continuum 
    recovers sufficiently. 

    Parameters:
        i (int): the index of the pixel to calculate the flux decrement for.
        flux (np.ndarray): Array of flux values.
        yC (np.ndarray): Array of continuum values.
        sigFlux (np.ndarray): Array of uncertainties in the flux values.
        sig_yC (np.ndarray): Array of uncertainties in the continuum values.
        
    Returns:
        Two values, the flux decrement at the given pixel and the corresponding uncertainty.
    """
    
    # Check that the input arrays have the same length
    if not (len(flux) == len(yC) == len(sigFlux) == len(sig_yC)):
        raise ValueError("Input arrays must have the same length")
    
    # Flux decrement per pixel
    # -ve for emission since flux > continuum. +ve for absorption.
    with np.errstate(divide='ignore', invalid='ignore'): #So Numpy ignores division by zero errors, will return NaN
        D = 1 - (flux[i] / yC[i])
        D = np.nan_to_num(D)
    
    # Uncertainty in the flux decrement
    deltaD = (flux[i] / yC[i]) * np.sqrt((sigFlux[i] / flux[i])**2 + (sig_yC[i] / yC[i])**2)

    return D, deltaD

def aperturePixelEW(i: int, Lambda: np.ndarray, flux: np.ndarray, yC: np.ndarray, sigFlux: np.ndarray, sig_yC: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the equivalent width per resolution element for a given pixel i.
    
    Args:
        i (int): the index of the pixel to calculate the equivalent width for
        Lambda (np.ndarray): Array of wavelengths.
        flux (np.ndarray): Array of flux values.
        yC (np.ndarray): Array of continuum values.
        sigFlux (np.ndarray): Array of uncertainties in the flux values.
        sig_yC (np.ndarray): Array of uncertainties in the continuum values.
        
    Returns:
        Two values, the equivalent width per resolution element for the given pixel and its uncertainty.
    """

    #Compute the flux decrement at the pixel using the fluxDec function.
    D, deltaD = fluxDec(i, flux, yC, sigFlux, sig_yC)
        
    # Width of pixel i
    pixelWidth = Lambda[i] - Lambda[i-1] #Addings abs to ensure it's always positive
            
    # Equivalent width per pixel
    eqWidth = pixelWidth * D
        
    # Uncertainty in the equivalent width per pixel
    deltaEqWidth = deltaD * pixelWidth

    return eqWidth, deltaEqWidth

def apertureEW(j: int, k: int, Lambda: np.ndarray, flux: np.ndarray, yC: np.ndarray, sigFlux: np.ndarray, sig_yC: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the equivalent width (EW) of a feature between two limits (j and k) and associated uncertainty.
    
    Unlike the apertureResEleEW function, this routine calculates the equivalent width of a feature between 
    two limits (j and k) by summing up the equivalent width per pixel over the range defined by those limits.

    Args:
        j (int): Index of the starting point of the feature in the Lambda array.
        k (int): Index of the ending point of the feature in the Lambda array.
        Lambda (np.ndarray): Array of wavelengths.
        flux (np.ndarray): Array of flux values.
        yC (np.ndarray): Array of continuum values.
        sigFlux (np.ndarray): Array of uncertainties in the flux values.
        sig_yC (np.ndarray): Array of uncertainties in the continuum values.

    Returns:
        Two values, the equivalent width per resolution element for the given pixel and its uncertainty.
    """

    eqWidth, deltaEqWidth = 0, 0

    #Calculate the equivalent width per resolution element in a loop
    #For a pixel i, go from i - p to i + p
    for i in range(j, k + 1):

        #Compute the flux decrement at the pixel using the fluxDec function. 
        D, deltaD = fluxDec(i, flux, yC, sigFlux, sig_yC)
        
        #Width of pixel i
        pixelWidth = Lambda[i] - Lambda[i-1]
        
        #Equivalent width per pixel
        eqWidth += pixelWidth * D
        
        #Uncertainty in the equivalent width per pixel
        deltaEqWidth += (deltaD * pixelWidth) ** 2
    
    return eqWidth, np.sqrt(deltaEqWidth)
