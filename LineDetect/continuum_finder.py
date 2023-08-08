#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 1 10:14:10 2023

@author: daniel
"""
import numpy as np
from lmfit import Model
from scipy.signal import savgol_filter
from scipy.special import eval_legendre, betainc 
from LineDetect.feature_finder import featureFinder

class Continuum:
    """
    Generates the continuum and continuum error.
   
    Args:
        wavelength (np.ndarray): Array of wavelengths.
        flux (np.ndarray): Array of flux values corresponding to the wavelengths.
        flux_err (np.ndarray): Array of uncertainties in the flux values.          
        halfWindow (int): The half-window is used in estimating the continuum using the median filter.
                          The median filter estimates the continuum at a pixel as the median value of the flux across a window centred at the pixel.
                          halfWindow is the size of half this window i.e. halfWindow * 2 = full window over which the median filter is applied.
                          This could be an array to try multiple medians and take the average of them all.
        N_sig_continuum (int): Number of standard deviations in equivalent width, above which a line is deemed statsitically significant. Defaults to 3.
        region_size (int): The size of the region to apply the polynomial fitting. Defaults to 150 pixels.
        resolution_element (int): The number of pixels per resolution element in the input spectrum. Defaults to 3.
        savgol_window_size (int): Window size for Savitzky-Golay smoothing. Defaults to 100. Can be set to 0 to skip the final savgol filtering.
        savgol_poly_order (int): Polynomial order for Savitzky-Golay smoothing. Defaults to 5. Only applicable if savgol_window_size is greater than 0.

    Methods:
        estimate:
        median_filter:
        legendreContinuum:

    """

    def __init__(self, wavelength, flux, flux_err, halfWindow=25, region_size=150, resolution_element=3, 
        savgol_window_size=100, savgol_poly_order=5, N_sig_limits=0.5, N_sig_continuum=3):

        self.wavelength = wavelength
        self.flux = flux
        self.flux_err = flux_err
        self.halfWindow = halfWindow
        self.region_size = region_size
        self.resolution_element = resolution_element
        self.savgol_window_size = savgol_window_size
        self.savgol_poly_order = savgol_poly_order
        self.N_sig_limits = N_sig_limits
        self.N_sig_continuum = N_sig_continuum

        self.size = len(self.wavelength)

        mask = np.where(np.isfinite(self.flux))[0]
        if len(mask) == 0:
            raise ValueError('WARNING: No non-finite values detected in the flux array!')
        if len(mask) != len(self.flux):
            print(f"WARNING: {len(mask)} non-finite values detected in the flux array, masking away...")
            self.wavelength, self.flux, self.flux_err = self.wavelength[mask], self.flux[mask], self.flux_err[mask]
        
    def estimate(self, fit_legendre=True):
        """
        Estimates the continuum for the input spectrum by first fitting an estimate of the continuum using the median filter, followed by a Legendre polynomial fit (from Sembach and Savage 92)
        which includes a robust error analysis.

        Args:
            fit_legendre (bool): Whether to fit with Legendre polynomials for a more robust estimate. Defaults to True.
        
        Returns:
            Generates the continuum and continuum_err class attributes.
        """
        
        #Apply the robust median filter first, legendre continuum fitter second
        self.median_filter(); self.legendreContinuum(max_order=20, p_threshold=0.05) if fit_legendre else None
        
    def median_filter(self):
        """
        Smooths out the flux array with a robust median-based filter

        Returns:
            Creates the continuum and continuum_err attributes.
        """

        if isinstance(self.halfWindow, int) or isinstance(self.halfWindow, float):
            self.continuum, self.continuum_err = np.empty(self.size), np.empty(self.size)
            # Loop to scan through each pixel in the spectrum
            for i in range(self.size):
                # Condition for the blue region of the spectrum where there is insufficient data to the left of the pixel
                if i < self.halfWindow:
                    self.continuum[i] = np.median(self.flux[0: i + self.halfWindow + 1])
                    self.continuum_err[i] = 0.05 #This is the error of the medium continuum, can also be the median(flux_err)
                    continue

                # Condition for the red region of the spectrum where there is insufficient data to the right of the pixel
                if i >= self.size - self.halfWindow:
                    self.continuum[i] = np.median(self.flux[i - self.halfWindow : ])
                    self.continuum_err[i] = 0.05
                    continue

                # Not the end cases
                self.continuum[i] = np.median(self.flux[i - self.halfWindow : i + self.halfWindow + 1])
                self.continuum_err[i] = 0.05

        else:
            yC_list, sig_yC_list = [], []
            for window_size in self.halfWindow:
                yC, sig_yC = np.empty(self.size), np.empty(self.size)
                # Loop to scan through each pixel in the spectrum
                for i in range(0, self.size):
                    # Condition for the blue region of the spectrum where there is insufficient data to the left of the pixel
                    if i < window_size:
                        yC[i] = np.median(self.flux[0: i + self.halfWindow + 1])
                        sig_yC[i] = 0.05
                        continue

                    # Condition for the red region of the spectrum where there is insufficient data to the right of the pixel
                    if i >= self.size - window_size:
                        yC[i] = np.median(self.flux[i - self.halfWindow : ])
                        sig_yC[i] = 0.05
                        continue

                    # Not the end cases
                    yC[i] = np.median(self.flux[i - window_size : i + window_size + 1])
                    sig_yC[i] = 0.05
                
                yC_list.append(yC); sig_yC_list.append(sig_yC)
                
            #Average across individual pixels
            self.continuum, self.continuum_err = np.median(yC_list, axis=0), np.median(sig_yC_list, axis=0)

        return
    
    def legendreContinuum(self, max_order, p_threshold):
        """
        Fits a Legendre polynomial to a spectrum to locate absorption and/or emission features.

        The function identifies absorption or emission features in the input spectrum using the `featureFinder` function,
        then extends thefl spectrum on both sides to avoid running out of bounds. For each feature, it selects a region
        of size region_size pixels around it and fits a Legendre polynomial to this region. If there are two features less than region_size pixels
        apart, the function computes the gap between them and includes the pixels between the features in the fitting array.
        The function returns an array of the fitted continuum values.

        Args:
            max_order (int): The maximum order of the Legendre polynomial to fit.If the order of the polynomial exceeds max_order, the program assumes the region cannot be fit. Defaults to 20. 
            p_threshold (float): The p-value threshold for the F-test. If the p-value is greater than this threshold,
                the fit is considered acceptable and the continuum level is returned. Defaults to 0.05.

        Returns:
            Updates the existing continuum and continuum_err class attributes.
        """

        # Find the regions of absorption
        featureRange = featureFinder(self.wavelength, self.flux, self.continuum, self.flux_err, self.continuum_err, N_sig_continuum=self.N_sig_continuum, N_sig_limits=self.N_sig_limits)
        featureRange = featureRange.astype(int)

        # If there are no absorption lines, return the unchanged continuum array
        if featureRange.size == 0:
            return 

        clean_pixels = [] #Define an array to hold the regions within 
        #for i in range(len(featureRange) // 2):

        #Iterate through the list of absorption features and start fitting 
        i = 0
        while i < len(featureRange) // 2:

            clean_pixels.clear()

            # Start and end of the current absorption feature
            start = featureRange[2 * i]
            end = featureRange[2 * i + 1]

            # We now have an absorption feature selected. 
            # The next step is to choose 150 clean pixels (i.e. pixels devoid of any absorption or emission) on either side of the abs line.
            # Sometimes an absorption feature might have another absorption feature within 150 pixels of each other.
            # If so we skip the second feature and keep going until we hit 150 pixels.
            # We need to get 150 pixels on the left and right side of each absorption feature.
            # Thus this whole shebang needs to be done twice.

            ##########   Left   ############
            # Let us assume the left boundary is 150 pixels to the left of the absorption line
            left_boundary = start - self.region_size
            # countLeft keeps count of how many pixels we have covered from the left end of the absorption line.
            # This helps tremendously when we skip over a second absorption line and have to keep track of how many more pixels we need.
            countLeft = self.region_size
            
            # Start going to the left until we hit 150
            j = i - 1
            while True: #Checks the left of the absorption line

                # If we have enough pixels to the left of the current absorption, break, and go to the next absorption line 
                if start - featureRange[2 * j + 1] > countLeft:# and start - featureRange[2 * j + 1] > 2*self.resolution_element and left_boundary - featureRange[2 * j + 1] > 2*self.resolution_element:
                    #clean_pixels.extend(range(left_boundary - 2*self.resolution_element, start))
                    clean_pixels.extend(range(left_boundary, start + 1))
                    break
                
                # If we have reached the first absorption line, 
                # check if the we have enough pixels to the left of it.
                # If we don't, stop at the very first pixel.
                elif j < 0:
                    left_boundary = max(left_boundary, 0)
                    clean_pixels.extend(range(left_boundary, start + 1))
                    break
                
                #elif start - featureRange[2 * j + 1] > countLeft:
                #    clean_pixels.extend(range(featureRange[2 * j + 1], start + 1))
                #    countLeft -= start - featureRange[2 * j + 1] + 2*self.resolution_element

                # If we don't have enough pixels and we are not at the first absorption line,
                # cut down the number of pixels we have left.  
                else:
                    clean_pixels.extend(range(featureRange[2 * j + 1], start + 1))
                    countLeft -= start - featureRange[2 * j + 1]

                # For the next iteration, the start pixel becomes the left end of the previous absorption line
                start = featureRange[2 * j]
                # Update j and the left boundary. 
                # This is where countLeft becomes critical.
                left_boundary = start - countLeft
                j -= 1

            ##########   Right   ############
            # Same stuff as the left
            right_boundary = end + self.region_size
            countRight = self.region_size

            j = i + 1
            #leftEdge = 0

            while True: #Checks the right of the absorption line
                if j > len(featureRange) // 2 - 1:
                    right_boundary = min(right_boundary, len(self.Lambda) - 1)
                    clean_pixels.extend(range(end, right_boundary + 1))
                    break

                elif featureRange[2 * j] - end > countRight:# and featureRange[2 * j] - end > 2*self.resolution_element:
                    #clean_pixels.extend(range(end, right_boundary + 1 + 2*self.resolution_element))
                    clean_pixels.extend(range(end, right_boundary + 1))
                    break

                #elif featureRange[2 * j] - end > countRight:
                #    clean_pixels.extend(range(end, featureRange[2 * j] + 1))
                #    countRight -= featureRange[2 * j] - end + 2*self.resolution_element

                else:
                    clean_pixels.extend(range(end, featureRange[2 * j] + 1))
                    countRight -= featureRange[2 * j] - end
                    i += 1

                end = featureRange[2 * j + 1]
                right_boundary = end + countRight
                j += 1

            clean_pixels.sort()

            #Find the functional fit of the continuum in this range
            result = legendreFit(clean_pixels, self.wavelength, self.flux, self.flux_err, region_size=self.region_size, max_order=max_order, p_threshold=p_threshold)

            if result is not None:
                self.continuum[clean_pixels[0] : clean_pixels[-1] + 1] = result[0]
                self.continuum_err[clean_pixels[0] : clean_pixels[-1] + 1] = result[1]

            i += 1

                #if clean_pixels[0] >= leftEdge:
                #    self.continuum[clean_pixels[0] : clean_pixels[-1] + 1] = result[0]
                #    self.continuum_err[clean_pixels[0] : clean_pixels[-1] + 1] = result[1]
                #    leftEdge = clean_pixels[-1]
                #else:
                #    self.continuum[leftEdge : clean_pixels[-1] + 1] = result[0]
                #    self.continuum_err[clean_pixels[0] : clean_pixels[-1] + 1] = result[1]

        self.continuum = savgol_filter(self.continuum, self.savgol_window_size, self.savgol_poly_order) if self.savgol_window_size > 0 else self.continuum

        return 

def legendreFit(indices, Lambda, flux, sigFlux, region_size, max_order, p_threshold):
    """
    Fits a Legendre polynomial to a given spectrum so as to estimate the continuum.

    Args:
        indices (np.ndarray): The x-values of the spectrum to be fit.
        Lambda (np.ndarray): The x-values of the extended spectrum.
        flux (np.ndarray): The y-values of the spectrum to be fit.
        sigFlux (np.ndarray): The uncertainties in the y-values.
        region_size (int): The number of points on either side of a given point to use in the fit.
        max_order (int): The maximum order of the Legendre polynomial to fit. Defaults to 20.
        p_threshold (float): The p-value threshold for the F-test. If the p-value is greater than this threshold,
            the fit is considered acceptable and the continuum level is returned. Defaults to 0.05.

    Returns:
        The continuum level of the spectrum.
    """

    fitLambda, fitFlux, fitsigFlux = np.array(Lambda[indices]), np.array(flux[indices]), np.array(sigFlux[indices])

    #Convert the x-axis to the domain [-1, 1]
    Lambda_L, Lambda_U = fitLambda[0], fitLambda[-1]
    fitLambda = (2*fitLambda - Lambda_L - Lambda_U) / (Lambda_U - Lambda_L)

    ##Start at the first order Legendre Polynomial and compare it with the 0th order, and so on.
    n = 1 
    while True:

        if n >= max_order:
            return None

        #Fit the window with Legendre polynomial of degree n - 1 = m 
        fit_m = np.polynomial.legendre.legfit(fitLambda, fitFlux, n - 1)

        #Fit the window with Legendre polynomial of degree n
        fit_n = np.polynomial.legendre.legfit(fitLambda, fitFlux, n)
            
        # Construct the Vandermonde matrix
        vander = np.polynomial.legendre.legvander(fitLambda, n)

        # Compute the covariance matrix
        covariance = np.linalg.inv(vander.T @ vander)

        #Find chi square for both of the fits
        chiSq_m = legendreChiSq(fit_m, fitLambda, fitFlux, fitsigFlux)
        chiSq_n = legendreChiSq(fit_n, fitLambda, fitFlux, fitsigFlux)

        # Compute the degrees of freedom
        df1 = 2 * region_size - n
        df2 = 2 * region_size - n - 1

        #Get the F-value
        F = FTest(chiSq_m, chiSq_n, df2)

        #Get the p-value
        p = 2 * betainc(0.5*df2, 0.5*df1, df2/(df2 + df1 * F))

        if p > p_threshold:
            left, right = indices[0], indices[-1]

            #Define the region over which the functional fit has to be applied
            lambdaAbsorption = np.array(Lambda[left : right + 1])

            #Convert from lambda to x
            lambdaAbsorption = (2*lambdaAbsorption - Lambda_L - Lambda_U) / (Lambda_U - Lambda_L)
            
            #Find the continuum in this wavelength range
            absFit, absFitErr = np.zeros(len(lambdaAbsorption)), np.zeros(len(lambdaAbsorption))

            for i in range(len(lambdaAbsorption)):
                absFit[i] = legendreLinCom(fit_n, lambdaAbsorption[i])
                for j in range(len(fit_n)):
                    for k in range(len(fit_n)):
                        absFitErr[i] += covariance[j][k] ** 2 * eval_legendre(j, lambdaAbsorption[j]) * eval_legendre(k, lambdaAbsorption[k])
                absFitErr[i] = np.sqrt(absFitErr[i])
            
            return [absFit, absFitErr]

        n += 1

def FTest(chiSq1: float, chiSq2: float, df: int) -> float:
    """
    Calculates the F-test result for two chi-square values and degrees of freedom.
    
    Args:
        chiSq1 (float): The first chi-square value.
        chiSq2 (float): The second chi-square value.
        df (int): The degrees of freedom.

    Returns:
        float: The F-test result.
    """

    return (chiSq1 - chiSq2) / (chiSq2 / df)

def legendreLinCom(coeff: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Calculates a linear combination of Legendre Polynomials up to a given degree.
    
    Parameters:
        coeff (np.ndarray): Coefficients of the Legendre Polynomials.
        x (np.ndarray): Array of x values at which to evaluate the polynomial.

    Returns:
        Values of the polynomial at the given x values.
    """

    degree = np.arange(len(coeff))
    legendreValues = np.zeros(len(coeff))

    for j in range(len(degree)):
        legendreValues[j] = eval_legendre(degree[j], x)

    return np.dot(coeff, legendreValues)

def legendreChiSq(coeff: np.ndarray, x: np.ndarray, y: np.ndarray, sigy: np.ndarray) -> float:
    """
    Calculates the chi-squared error between two fits.
        
    This function works by looping through each element of the x-values (derived from the wavelength) in the 
    window and performing the following steps for each element:    
        - Finding the continuum, which is a linear combination of Legendre polynomials up to a certain degree M. This 
            is done in an inner loop.
        - Using the continuum and the y-values and uncertainties (I, sig I) to calculate the chi-squared error.
        - Repeating these steps for the next x-value.

    Args:
        coeff (np.ndarray): The coefficients of the Legendre polynomial fit.
        x (np.ndarray): The x-values of the data (the wavelength in this context).
        y (np.ndarray): The y-values of the data (the flux in this context).
        sigy (np.ndarray): The uncertainties in the y-values.

    Returns:
        The chi-squared error.
    """

    if len(y) != len(x) or len(sigy) != len(x):
        raise ValueError('y and sigy must have the same length as x.')

    chiSq = 0
    for i in range(len(x)):
        legSum = legendreLinCom(coeff, x[i])
        chiSq += (y[i] - legSum)**2 / (sigy[i])**2
    
    return chiSq


