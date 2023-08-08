import numpy as np
from lmfit import Model
from typing import Tuple
from LineDetect import feature_finder 

def doublet_finder(wavelength: np.array, flux: np.array, yC: np.array, sigFlux: np.array, sig_yC: np.array, R: np.ndarray, 
    N_sig_line1: float = 5, N_sig_line2: float = 3, N_sig_limits: float = 0.5, resolution_element: int = 3, 
    rest_wavelength_1: float = 1548.204, rest_wavelength_2: float = 1550.781, char_vel: float = 498) -> Tuple[int, int]:
    """ 
    Calculates the equivalent width and associated uncertainties of Mg-II doublet absorption features in a given spectrum.

    Parameters:
        Lambda (np.ndarray): Wavelength array of the spectrum
        flux (np.ndarray): Flux array of the spectrum
        yC (np.ndarray): Continuum flux array of the spectrum
        sigFlux (np.ndarray): Array of flux uncertainties
        sig_yC (np.ndarray): Array of continuum flux uncertainties
        R (np.ndarray): Resolution array of the spectrum
        N_sig_line1 (float): Threshold of flux recovery for determining feature limits.
        N_sig_2 (float): Defaults to 3.
        resolution_element (int): The size of the resolution element in pixels. Defaults to 3.
        rest_wavelength_1 (float):
        rest_wavelength_2 (float):
     
    Returns:
        blue_line (np.ndarray): Array of lower limit wavelength values for the blue line in the doublet
        red_line (np.ndarray): Array of upper limit wavelength values for the red line in the doublet
        blue_line_EW (np.ndarray): Array of equivalent widths for each of the blue lines in the doublet
        red_line_EW (np.ndarray): Array of equivalent widths for each of the red lines in the doublet
        blue_sig_EW (np.ndarray): Array of uncertainties in equivalent widths for each of the blue lines in the doublet
        red_sig_EW (np.ndarray): Array of uncertainties in equivalent widths for each of the red lines in the doublet 
    """

    doublet_separation = abs(rest_wavelength_2 - rest_wavelength_1) #7.18 for MgII
    #Define an empty array to hold the line limits, EW and associated uncertainty
    blue_line, red_line, blue_line_EW, red_line_EW, blue_sig_EW, red_sig_EW = [],[],[],[],[],[]
    
    #Instrumental spread function half-width
    J0 = 2 * resolution_element

    #Run through every pixel in the spectrum
    i = 0
    while i < len(wavelength):
        flag = 0
        #Find the equivalent width at the pixel using the Optimized Method
        eqWidth1, deltaEqWidth1 = feature_finder.aperturePixelEW(i, wavelength, flux, yC, sigFlux, sig_yC)

        #Check if the pixel satisfies the selection criterion
        #But first change them to fall fellow the threshold if they are not finite, so as to avoid warnings.
        #eqWidth1 = 1e-3 if (eqWidth1 > 0)==False else eqWidth1
        #deltaEqWidth1 = 1e3 if (deltaEqWidth1 > 0)==False else deltaEqWidth1

        if eqWidth1 / deltaEqWidth1 > N_sig_line1:
            #Congrats! We have located an absorption feature. We need to ensure the absorption feature is indeed Mg-II. 
            #If we assume this feature to be the 2796 line, there must be a second absorption feature at the equilibrium separation.
            #To look for such a pixel, we first find the redshift and then the equilibrium separation.
            z = wavelength[i] / rest_wavelength_1 - 1

            #The separation is then z * 7.18 (which is the separation of the troughs at zero redshift) 
            sep = (z + 1) * doublet_separation

            #Find the index of the first element from the wavelength list that is greater than the required separation. 
            try:
                index = next(j for j, val in enumerate(wavelength) if val > wavelength[i] + sep)
                if index + 10 >= len(wavelength):   #Ensure we do not run out of bounds
                    i += 1
                    continue

            except:
                i += 1
                continue

            #Find the equivalent width and the corresponding uncertainty at a range around the second pixel
            for k in range(index-1, index+2):
                
                #Find the pixel eq width around the second pixel and check if there is a second absorption system
                eqWidth2, deltaEqWidth2 = feature_finder.aperturePixelEW(k, wavelength, flux, yC, sigFlux, sig_yC)

                if eqWidth2 / deltaEqWidth2 > N_sig_line2:

                    #Get the wavelength range of each absorption range now that both the systems are stat. sig.
                    line1B, line1R = feature_finder.apertureFeatureLimits(i, wavelength, flux, yC, sigFlux, sig_yC, R=R, N_sig_limits=N_sig_limits)
                    line2B, line2R = feature_finder.apertureFeatureLimits(k, wavelength, flux, yC, sigFlux, sig_yC, R=R, N_sig_limits=N_sig_limits)

                    if line1B == line2B:
                        EW1, sigEW1, EW2, sigEW2, waveB, waveR = lineFit(line1B, line2R, wavelength, flux, rest_wavelength_1, rest_wavelength_2) # This function already converts the EW to restframe!
                        velocity_sep = ((waveR - waveB) / rest_wavelength_1) * 3e5 
                        velocity_sep = velocity_sep / (1 + z)

                    else:
                        #Calculate the total EW over the two features
                        EW1, sigEW1 = feature_finder.apertureEW(line1B, line1R, wavelength, flux, yC, sigFlux, sig_yC)
                        EW2, sigEW2 = feature_finder.apertureEW(line2B, line2R, wavelength, flux, yC, sigFlux, sig_yC)

                        #Convert EW to rest frame
                        EW1, sigEW1 = EW1/(1+z), sigEW1/(1+z)
                        z2 = 0.5 * (wavelength[line2B] + wavelength[line2R]) / rest_wavelength_2
                        EW2, sigEW2 = EW2/z2, sigEW2/z2

                        velocity_sep = ((wavelength[line2B] - wavelength[line1B]) / rest_wavelength_1) * 3e5 
                        velocity_sep = velocity_sep / (1 + z)

                    if EW1/sigEW1 > N_sig_line1 and EW2/sigEW2 > N_sig_line2 and ((EW1 + sigEW1) / (EW2 - sigEW2) > 1 or (EW1 - sigEW1) / (EW2 + sigEW2) < 2) and char_vel - 100 < velocity_sep < char_vel + 100:
                        blue_line.extend([line1B, line1R])
                        red_line.extend([line2B, line2R])
                        blue_line_EW.append(EW1)
                        red_line_EW.append(EW2)
                        blue_sig_EW.append(sigEW1)
                        red_sig_EW.append(sigEW2)

                    i = line2R + 1
                    flag = 1
                    break

        if flag == 1:
            continue

        i += 1

    return blue_line, red_line, blue_line_EW, red_line_EW, blue_sig_EW, red_sig_EW


def doubleGaussian(x, amplitude_1, amplitude_2, sigma_1, sigma_2, x0_1, x0_2):
    """
    Calculates the value of a double Gaussian function at a given x coordinate.
    
    Parameters:
        x (float): The x coordinate.
        amplitude_1 (float): Amplitude of the first Gaussian component.
        amplitude_2 (float): Amplitude of the second Gaussian component.
        sigma_1 (float): Standard deviation of the first Gaussian component.
        sigma_2 (float): Standard deviation of the second Gaussian component.
        x0_1 (float): Mean of the first Gaussian component.
        x0_2 (float): Mean of the second Gaussian component.
        
    Returns:
        float: The value of the double Gaussian function at the given x coordinate.
    """

    return np.exp((-0.5*(x-x0_1)**2)/sigma_1**2)*amplitude_1 + np.exp((-0.5*(x-x0_2)**2)/sigma_2**2)*amplitude_2 + 1

def eqWidth(amplitude, width, amp_error, width_error):
    EW = amplitude * np.sqrt(2 * np.pi) * width
    dEW_dA = np.sqrt(2 * np.pi) * width
    dEW_dwidth = amplitude * np.sqrt(2 * np.pi)
    delta_EW = np.sqrt((dEW_dA * amp_error)**2 + (dEW_dwidth * width_error)**2)

    return -EW, delta_EW

def lineFit(index1, index2, Lambda, flux, rest_wavelength_1, rest_wavelength_2):
    """
    Fits a line using a double Gaussian model and returns the equivalent width and errors.

    Parameters:
        index1 (int): Start index of the line region.
        index2 (int): End index of the line region.
        Lambda (array-like): Array of wavelengths.
        flux (array-like): Array of flux values.
        rest_wavelength_1 (float):
        rest_wavelength_2 (float):

    Returns:
        tuple: A tuple containing the equivalent widths and errors of the two Gaussian components:
            - EW1 (float): Equivalent width of the first Gaussian component.
            - deltaEW1 (float): Error in the equivalent width of the first Gaussian component.
            - EW2 (float): Equivalent width of the second Gaussian component.
            - deltaEW2 (float): Error in the equivalent width of the second Gaussian component.
    """

    step = (index2 - index1) / 4
    x0_1 = Lambda[round(step)+index1]
    x0_2 = Lambda[3*round(step)+index1]
    #z = x0_1/rest_wavelength_1 - 1

    leftIndex = max(0, index1 - 30)
    rightIndex = min(len(Lambda), index2 + 30)
    wavelength = Lambda[leftIndex:rightIndex]
    fitFlux = np.append(np.ones(index1 - leftIndex), flux[index1:index2])   
    fitFlux = np.append(fitFlux, np.ones(rightIndex - index2))

    fmodel = Model(doubleGaussian)
    params = fmodel.make_params(amplitude_1=-0.5, amplitude_2=-0.5, sigma_1=2, sigma_2=2, x0_1=x0_1, x0_2=x0_2)

    result = fmodel.fit(fitFlux, params, x=wavelength)
    coeff = result.best_values
    perr = np.sqrt(np.diag(result.covar))

    '''
    # Condition to check if the line is unresolved
    if coeff['sigma_1'] + perr[2] < 1.46:
        params['sigma_1'].vary = False
        result = fmodel.fit(fitFlux, params, x=wavelength)

    if coeff['sigma_2'] + perr[3] < 1.46:
        params['sigma_2'].vary = False
        result = fmodel.fit(fitFlux, params, x=wavelength)
    '''

    EW1, deltaEW1 = eqWidth(coeff['amplitude_1'], coeff['sigma_1'], perr[0], perr[2])
    z1 = coeff['x0_1']/rest_wavelength_1 - 1

    EW2, deltaEW2 = eqWidth(coeff['amplitude_2'], coeff['sigma_2'], perr[1], perr[3])
    z2 = coeff['x0_2']/rest_wavelength_2 - 1

    return EW1 / (1 + z1), deltaEW1 / (1 + z1), EW2 / (1 + z2), deltaEW2 / (1 + z2), coeff['x0_1'], coeff['x0_2']
