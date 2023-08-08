#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 1 06:43:54 2023

@author: daniel
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from operator import itemgetter
import matplotlib.pyplot as plt  ## To plot the spectrum
import matplotlib.colors as mcolors 

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.importance import get_param_importances, FanovaImportanceEvaluator

from astropy.io import fits  
from astropy.wcs import WCS
from progress.bar import FillingSquaresBar

from LineDetect.continuum_finder import Continuum
from LineDetect.detect_elements import MgII 

class Spectrum:
    """
    A class for processing spectral data stored in FITS files.

    Can process either a set of .fits files or single spectra.

    Note:
        If the line is detected the spectrum features will be added
        to the DataFrame `df` attribute, which will always append new detections. 
        If no line is detected then nothing will be added to the DataFrame,
        but a message with the object name will print.

    Args:
        halfWindow (int, list, np.ndarray): The half-size of the window/kernel (in Angstroms) used to compute the continuum. 
            If this is a list/array of integers, then the continuum will be calculated
            as the median curve across the fits across all half-window sizes in the list/array.
            Defaults to 25.
        region_size (int):
        savgol_window_size (int):
        savgol_poly_order (int): The order of the polynomial used for smoothing the spectrum.
        resolution_range (tuple): A tuple of the minimum and maximum resolution (in km/s) used to detect MgII absorption.
            Can also be an integer or a float.
        directory (str): The path to the directory containing the FITS files. Defaults to None.
        save_all (bool): Parameter to control whether to save the non-detections. If the spectral feature is not detected 
            and save_all=True, the qso_name will be appended alongside 'None' entries. Defaults to False to save only positive detections.

    Methods:
        process_files(): Process the FITS files in the directory.
        process_spectrum(Lambda, y, sig_y, z, file_name): Process a single instance of spectral data.
        _reprocess(): Re-runs the process_spectrum method using the saved spectrum attributes.
        plot(include, errorbar, xlim, ylim, xlog, ylog, savefig): Plots the spectrum and/or continuum.
        find_MgII_absorption(Lambda, y, yC, sig_y, sig_yC, z, qso_name): Find the MgII lines, if present.
        find_CIV_absorption(Lambda, y, yC, sig_y, sig_yC, z, qso_name): Find the CIV lines, if present.
    """

    def __init__(self, halfWindow=25, resolution_range=(1400, 1700), region_size=100, resolution_element=3,
        savgol_window_size=100, savgol_poly_order=3, N_sig_limits=0.2, N_sig_line1=2.5, N_sig_line2=2.5, N_sig_continuum = 2,
        rest_wavelength_1=1548.204, rest_wavelength_2=1550.781, directory=None, save_all=False):
        
        self.halfWindow = halfWindow
        self.resolution_range = resolution_range
        self.region_size = region_size
        self.resolution_element = resolution_element
        self.savgol_window_size = savgol_window_size
        self.savgol_poly_order = savgol_poly_order

        self.N_sig_limits = N_sig_limits
        self.N_sig_line1 = N_sig_line1
        self.N_sig_line2 = N_sig_line2
        self.N_sig_continuum = N_sig_continuum
        self.rest_wavelength_1 = rest_wavelength_1
        self.rest_wavelength_2 = rest_wavelength_2

        self.directory = directory
        self.save_all = save_all

        #Declare a dataframe to hold the info
        self.df = pd.DataFrame(columns=['QSO', 'Wavelength', 'z', 'W', 'deltaW']) 

    def process_files(self):
        """
        Processes each FITS file in the directory, detecting any Mg II absorption that may be present.

        The method iterates through each FITS file in the directory specified during initialization, 
        reads in the spectrum data and associated header information, applies continuum normalization, 
        identifies Mg II absorption features, and calculates the equivalent widths of said absorptions.
        The results are stored in a pandas DataFrame (df attribute). 
        
        Note:
            Unlike when processing single spectra, this method does not save
            the continuum and continuum_err attributes, therefore the plot()
            method cannot be called. Load a single spectrum using process_spectrum
            to save the continuum attributes.

        Returns:
            None
        """

        for i, (root, dirs, files) in enumerate(os.walk(os.path.abspath(self.directory))):
            progress_bar = FillingSquaresBar('Processing files......', max=len(files))
            for file in files:
                #Read each file in the directory
                try:
                    hdu = fits.open(os.path.join(root, file))
                except OSError:
                    print(); print('Invalid file type, skipping file: {}'.format(file))
                    progress_bar.next(); continue

                #Get the flux intensity and corresponding error array as well as redshift
                flux, flux_err, z = hdu[0].data, np.sqrt(hdu[1].data), hdu[0].header['Z'] 
                #Recreate the wavelength spectrum from the info given in the WCS of the header
                w = WCS(hdu[0].header, naxis=1, relax=False, fix=False)
                Lambda = w.wcs_pix2world(np.arange(len(flux)), 0)[0]

                mask = np.isfinite(Lambda) & np.isfinite(flux) & np.isfinite(flux_err)
                if len(mask) != len(Lambda):
                    print('WARNING: Non-finite values detected, these values will be omitted...')
                    Lambda, flux, flux_err = Lambda[mask], flux[mask], flux_err[mask]
                  
                #Cut the spectrum blueward of the LyAlpha line and less than the rest frame
                mask = np.where((Lambda > (z + 1) * 1310) & (Lambda <= (z + 1) * 1554))[0]
                
                Lambda, flux, flux_err = Lambda[mask], flux[mask], flux_err[mask]
                
                try:
                    #Generate the contiuum
                    continuum = Continuum(Lambda, flux, flux_err, halfWindow=self.halfWindow, region_size=self.region_size, resolution_element=self.resolution_element, 
                        savgol_window_size=self.savgol_window_size, savgol_poly_order=self.savgol_poly_order, N_sig_limits=self.N_sig_limits, N_sig_line2=self.N_sig_line2)
                    continuum.estimate()
                except ValueError: #This will catch the failed to fit message!
                    print(); print('Failed to fit the continuum, skipping file: {}'.format(file))
                    progress_bar.next(); continue
                #Find the MgII Absorption
                self.find_MgII_absorption(Lambda, flux, continuum.continuum, flux_err, continuum.continuum_err, z=z, qso_name=file)
                
                progress_bar.next()

        progress_bar.finish()

        return 

    def process_spectrum(self, Lambda, flux, flux_err, z, qso_name=None):
        """
        Processes a single spectrum, detecting any Mg II absorption that may be present.

        Args:
            Lambda (array-like): An array-like object containing the wavelength values of the spectrum.
            flux (array-like): An array-like object containing the flux values of the spectrum.
            flux_err (array-like): An array-like object containing the flux error values of the spectrum.
            z (float): The redshift of the QSO associated with the spectrum.
            qso_name (str, optional): The name of the QSO associated with the spectrum, will be
                saved in the DataFrame. Defaults to None, in which case 'No_Name' is used.

        Returns:
            None
        """

        mask = np.isfinite(Lambda) & np.isfinite(flux) & np.isfinite(flux_err)
        if len(mask) != len(Lambda):
            print('WARNING: Non-finite values detected, these values will be omitted...')
            Lambda, flux, flux_err = Lambda[mask], flux[mask], flux_err[mask]

        qso_name = 'No_Name' if qso_name is None else qso_name

        #Cut the spectrum blueward of the LyAlpha line and less than the rest frame

        mask = np.where((Lambda > (z + 1) * 1310) & (Lambda <= (z + 1) * 1554))[0]
        
        Lambda, flux, flux_err = Lambda[mask], flux[mask], flux_err[mask]
        
        #Generate the contiuum
        continuum = Continuum(Lambda, flux, flux_err, halfWindow=self.halfWindow, region_size=self.region_size, resolution_element=self.resolution_element, 
            savgol_window_size=self.savgol_window_size, savgol_poly_order=self.savgol_poly_order, N_sig_limits=self.N_sig_limits, N_sig_line2=self.N_sig_line2)
        continuum.estimate()

        #Save the continuum attributes
        self.continuum, self.continuum_err = continuum.continuum, continuum.continuum_err

        #Find the MgII Absorption
        self.find_MgII_absorption(Lambda, flux, self.continuum, flux_err, self.continuum_err, z=z, qso_name=qso_name)
                
        self.Lambda, self.flux, self.flux_err, self.z, self.qso_name = Lambda, flux, flux_err, z, qso_name #For plotting

        return

    def _reprocess(self, qso_name=None):
        """
        Reprocesses the data, intended to be used after running process_spectrum().
        Useful for changing the attributes and quickly re-running the same sample without
        having to re-input the spectra.
        
        Note:
            This will update the DataFrame by appending the new object line features (if found).
        
        Args:
            qso_name (str, optional):

        Returns:
            None
        """

        qso_name = self.qso_name if qso_name is None else 'No_Name'

        #Cut the spectrum blueward of the LyAlpha line and less than the rest frame
        mask = np.where((self.Lambda > (self.z + 1) * 1310) & (self.Lambda <= (self.z + 1) * 1554))[0]

        self.Lambda, self.flux, self.flux_err = self.Lambda[mask], self.flux[mask], self.flux_err[mask]
  
        #Generate the contiuum
        continuum = Continuum(self, self.Lambda, self.flux, self.flux_err, halfWindow=self.halfWindow, region_size=self.region_size, resolution_element=self.resolution_element, 
            savgol_window_size=self.savgol_window_size, savgol_poly_order=self.savgol_poly_order, N_sig_limits=self.N_sig_limits, N_sig_continuum=self.N_sig_continuum)
        continuum.estimate()

        #Save the continuum attributes
        self.continuum, self.continuum_err = continuum.continuum, continuum.continuum_err

        #Find the MgII Absorption
        self.find_MgII_absorption(self.Lambda, self.flux, self.continuum, self.flux_err, self.continuum_err, z=self.z, qso_name=self.qso_name)
        
        return 
        
    def find_MgII_absorption(self, Lambda, y, yC, sig_y, sig_yC, z, qso_name=None):
        """
        Finds Mg II absorption features in the QSO spectrum and adds the line information to the DataFrame,
        including the Equivalent Width and the corresponding error. 

        Args:
            Lambda (array-like): Wavelength array.
            y (array-like): Observed flux array.
            yC (array-like): Estimated continuum flux array.
            sig_y (array-like): Observed flux error array.
            sig_yC (array-like): Estimated continuum flux error array.
            z (float): The redshift of the QSO associated with the spectrum.
            qso_name (str, optional): The name of the QSO associated with the spectrum, will be
                saved in the DataFrame. Defaults to None, in which case 'No_Name' is used.
            
        Returns:
            None
        """

        #Declare an array to hold the resolution at each wavelength
        if isinstance(self.resolution_range, int) or isinstance(self.resolution_range, float):
            R = [[self.resolution_range] * len(Lambda)]
        else:
            R = np.linspace(self.resolution_range[0], self.resolution_range[1], len(Lambda))

        #The MgII function finds the lines
        Mg2796, Mg2803, EW2796, EW2803, deltaEW2796, deltaEW2803 = MgII(Lambda, y, yC, sig_y, sig_yC, R, N_sig_line1=self.N_sig_line1, N_sig_line2=self.N_sig_line2, 
            N_sig_limits=self.N_sig_limits, resolution_element=self.resolution_element, rest_wavelength_1=self.rest_wavelength_1, rest_wavelength_2=self.rest_wavelength_2)
        Mg2796, Mg2803 = np.array(Mg2796), np.array(Mg2803)
        self.Mg2796, self.Mg2803 = Mg2796.astype(int), Mg2803.astype(int)

        if len(self.Mg2796) != 0:
            for i in range(0, len(Mg2796) - 1, 2):
                wavelength = (Lambda[self.Mg2796[i]] + Lambda[self.Mg2796[i+1]])/2
                new_row = {'QSO': qso_name, 'Wavelength': wavelength, 'z': wavelength/self.rest_wavelength_1 - 1, 'W': EW2796[i], 'deltaW': deltaEW2796[i]}
                self.df = pd.concat([self.df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        else: 
            if self.save_all:
                new_row = {'QSO': qso_name, 'Wavelength': 'None', 'z': 'None', 'W': 'None', 'deltaW': 'None'}
                self.df = pd.concat([self.df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
            
        return 

    def optimize(self, Lambda, flux, flux_err, z_qso, z_element, halfWindow, region_size, resolution_element, 
        savgol_window_size, savgol_poly_order, N_sig_limits, N_sig_line1, N_sig_line2, n_trials=100, threshold=0.1, show_progress_bar=False):
        """
        This class method will optimize the element detection parameters according to the
        input constraints
        
        Args:
            resolution_element (bool): A tuple containing the (min, max) parameter range 
                to search through. Can be integer to hard code this parameter and exclude from
                the grid search. 
            halfWindow (tuple): A tuple containing the (min, max) parameter range 
                to search through. Can be integer to hard code this parameter and exclude from
                the grid search. 
            N_sig_limits (bool): A tuple containing the (min, max) parameter range 
                to search through. Can be integer to hard code this parameter and exclude from
                the grid search. 
            N_sig_line1 (bool): A tuple containing the (min, max) parameter range 
                to search through. Can be integer to hard code this parameter and exclude from
                the grid search.
            N_sig_line2 (bool): A tuple containing the (min, max) parameter range 
                to search through. Can be integer to hard code this parameter and exclude from
                the grid search. 
            n_trials (int): Number of trials to run the optimizer for. Defaults to 100.
            threshold (float): The desired threshold. For example, if this is set to 0.1, the
                optimization will stop if the target is within this tolerance. Defaults to 0.0001 which
                will essentially top the optimization when it reaches the exact target value.

        """

        def check_threshold(study, trial):
            """Check if the best value in the study is below the threshold.
            This function compares the best value found in the Optuna study with a threshold value.
            If the best value is below the threshold, a `ThresholdExceeded` exception is raised.

            Args:
                study (optuna.study.Study): The Optuna study object containing the optimization results.
                trial (optuna.trial.FrozenTrial): The Optuna trial object representing the current trial.

            Raises:
                ThresholdExceeded: If the best value in the study is below the threshold.

            Returns:
                None
            """

            if study.best_value > 1 - threshold:
                raise ThresholdExceeded()
            return

        sampler = optuna.samplers.TPESampler(seed=1909)
        study = optuna.create_study(direction='maximize', sampler=sampler)#, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10))

        objective = objective_spec(Lambda, flux, flux_err, z_qso=z_qso, z_element=z_element, rest_wavelength_1=self.rest_wavelength_1, rest_wavelength_2=self.rest_wavelength_2, 
            resolution_range=self.resolution_range, halfWindow=halfWindow, region_size=region_size, resolution_element=resolution_element, savgol_window_size=savgol_window_size,
            savgol_poly_order=savgol_poly_order, N_sig_limits=N_sig_limits, N_sig_line1=N_sig_line1, N_sig_line2=N_sig_line2)

        try:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar, callbacks=[check_threshold])#, gc_after_trial=True)
        except ThresholdExceeded:
            print(); print(f"The optimal value is within the {threshold} tolerance threshold, stopping optimization.")

        self.optimization_results, self.best_params = study, study.best_trial.params
        print(); print(f"Optimization complete with a final score of {study.best_value}! Optimal parameters : {self.best_params}")

        print(); print('Re-setting class attributes to optimal values...')
        self.halfWindow = self.best_params['halfWindow'] if isinstance(halfWindow, tuple) else halfWindow
        self.region_size = self.best_params['region_size'] if isinstance(region_size, tuple) else region_size
        self.resolution_element = self.best_params['resolution_element'] if isinstance(resolution_element, tuple) else resolution_element
        self.savgol_window_size = self.best_params['savgol_window_size'] if isinstance(savgol_window_size, tuple) else savgol_window_size
        self.savgol_poly_order = self.best_params['savgol_poly_order'] if isinstance(savgol_poly_order, tuple) else savgol_poly_order
        self.N_sig_limits = self.best_params['N_sig_limits'] if isinstance(N_sig_limits, tuple) else N_sig_limits
        self.N_sig_line1 = self.best_params['N_sig_line1'] if isinstance(N_sig_line1, tuple) else N_sig_line1
        self.N_sig_line2 = self.best_params['N_sig_line2'] if isinstance(N_sig_line2, tuple) else N_sig_line2

        print('Re-fitting spectra with optimal values, setting qso name to "Optimized".')
        self.process_spectrum(Lambda, flux, flux_err, z_qso, qso_name='Optimized')
        
        return 

    def plot(self, include='both', highlight=True, xlim=None, ylim=None, xlog=False, ylog=False, 
        savefig=False, path=None):
        """
        Plots the spectrum and/or continuum.
    
        Args:
            include (float): Designates what to plot, options include 'spectrum', 'continuum', or 'both.
            highlight (bool): If True then the line will be highlighted with accompanying
                vertical lines to visualize the equivalent width. Defaults to True.
            xlim: Limits for the x-axis. Ex) xlim = (4000, 6000)
            ylim: Limits for the y-axis. Ex) ylim = (0.9, 0.94)
            xlog (boolean): If True the x-axis will be log-scaled.
                Defaults to True.
            ylog (boolean): If True the y-axis will be log-scaled.
                Defaults to False.
            savefig (bool): If True the figure will not disply but will be saved instead.
                Defaults to False.
            path (str, optional): Path in which the figure should be saved, defaults to None
                in which case the image is saved in the local home directory. 

        Returns:
            AxesImage
        """

        if self.continuum is None or self.flux is None:
            raise ValueError('This method only works after a single spectrum has been processed via the process_spectrum method.')

        _set_style_() if savefig else plt.style.use('default')

        if include == 'continuum' or include == 'both':
            plt.errorbar(self.Lambda, self.continuum, yerr=None, label='Continuum', fmt='r--', linewidth=0.6)
        if include == 'spectrum' or include == 'both':
            plt.errorbar(self.Lambda, self.flux, yerr=None, fmt='k-.', linewidth=0.2)
        
        plt.title(self.qso_name, size=14)
        plt.xlabel('Wavelength [Å]', size=12); plt.ylabel('Flux', alpha=1, color='k', size=12)
        plt.xticks(fontsize=10); plt.yticks(fontsize=12)
        plt.xscale('log') if xlog else None; plt.yscale('log') if ylog else None 
        plt.xlim(xlim) if xlim is not None else None; plt.ylim(ylim) if ylim is not None else None
        plt.legend(prop={'size': 12}) if include != 'spectrum' else None
        
        if highlight:
            if len(self.Mg2796) == 0:
                print('The highlight parameter is enabled but no line was detected!')
            else:
                for i in range(0, len(self.Mg2796) - 1, 2):
                    plt.axvline(x = self.Lambda[self.Mg2796[i]], color = 'orange')
                    plt.axvline(x = self.Lambda[self.Mg2796[i+1]], color = 'orange')
                    plt.axvline(x = self.Lambda[self.Mg2803[i]], color = 'red')
                    plt.axvline(x = self.Lambda[self.Mg2803[i + 1]], color = 'red')

        if savefig:
            path = str(Path.home()) if path is None else path 
            path += '/' if path[-1] != '/' else ''
            plt.savefig(path+'Spectrum_'+self.qso_name+'.png', dpi=300, bbox_inches='tight')
            print('Figure saved in: {}'.format(path)); plt.clf()
        else:
            plt.show()

        return 

    def plot_param_opt(self, xlim=None, ylim=None, xlog=True, ylog=False, savefig=False):
        """
        Plots the parameter optimization history.

        Note:
            The Optuna API has its own plot function: plot_optimization_history(self.optimization_results)
    
        Args:
            baseline (float): Baseline accuracy achieved when using only
                the default engine hyperparameters. If input a vertical
                line will be plot to indicate this baseline accuracy.
                Defaults to None.
            xlim: Limits for the x-axis. Ex) xlim = (0, 1000)
            ylim: Limits for the y-axis. Ex) ylim = (0.9, 0.94)
            xlog (boolean): If True the x-axis will be log-scaled.
                Defaults to True.
            ylog (boolean): If True the y-axis will be log-scaled.
                Defaults to False.
            savefig (bool): If True the figure will not disply but will be saved instead.
                Defaults to False. 

        Returns:
            AxesImage
        """

        trials = self.optimization_results.get_trials()
        trial_values, best_value = [], []
        for trial in range(len(trials)):
            value = trials[trial].values[0]
            trial_values.append(value)
            if trial == 0:
                best_value.append(value)
            else:
                if any(y > value for y in best_value): #If there are any numbers in best values that are higher than current one
                    best_value.append(np.array(best_value)[trial-1])
                else:
                    best_value.append(value)

        best_value, trial_values = np.array(best_value), np.array(trial_values)
        best_value[1] = trial_values[1] #Make the first trial the best model, since technically it is.
        for i in range(2, len(trial_values)):
            if trial_values[i] < best_value[1]:
                best_value[i] = best_value[1]
            else:
                break

        _set_style_() if savefig else plt.style.use('default')

        plt.plot(range(len(trials)), best_value, color='r', alpha=0.83, linestyle='-', label='Optimal')
        plt.scatter(range(len(trials)), trial_values, c='b', marker='+', s=35, alpha=0.45, label='Trial')
        plt.xlabel('Trial #', alpha=1, color='k'); plt.ylabel('Objective Value', alpha=1, color='k')
        plt.title('Parameter Optimization History')
        plt.legend(loc='upper center', ncol=2, frameon=False)
        plt.rcParams['axes.facecolor']='white'; plt.grid(False)

        plt.xlim(xlim) if xlim is not None else plt.xlim((1, len(trials)))
        plt.ylim(ylim) if ylim is not None else None 
        plt.xscale('log') if xlog else None
        plt.yscale('log') if ylog else None
        
        if savefig:
            plt.savefig('Ensemble_Hyperparameter_Optimization.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

        return

    def plot_param_importance(self, plot_time=False, savefig=False):
        """
        Plots the hyperparameter optimization history.
    
        Note:
            The Optuna API provides its own plotting function: plot_param_importances(self.optimization_results)

        Args:
            plot_tile (bool): If True, the importance on the duration will also be included. Defaults to False.
            savefig (bool): If True the figure will not display but will be saved instead. Defaults to False. 

        Returns:
            AxesImage
        """

        param_importances = get_param_importances(self.optimization_results)

        time_importance = FanovaImportanceEvaluator()
        duration_importances = time_importance.evaluate(self.optimization_results, target=lambda t: t.duration.total_seconds())

        params, importance, duration_importance = [], [], []
        for key in param_importances:       
            params.append(key)

        for name in params:
            importance.append(param_importances[name])
            duration_importance.append(duration_importances[name])

        xtick_labels = format_labels(params)

        _set_style_() if savefig else plt.style.use('default')

        fig, ax = plt.subplots()
        ax.barh(xtick_labels, importance, label='Importance for Line Detection', color=mcolors.TABLEAU_COLORS["tab:blue"], alpha=0.87)
        if plot_time:
            ax.barh(xtick_labels, duration_importance, label='Impact on Detection Speed', color=mcolors.TABLEAU_COLORS["tab:orange"], alpha=0.7, hatch='/')

        ax.set_ylabel("Hyperparameter"); ax.set_xlabel("Importance Evaluation")
        ax.legend(ncol=2, frameon=False, handlelength=2, bbox_to_anchor=(0.5, 1.1), loc='upper center')
        ax.set_xscale('log'); plt.xlim((0, 1.))
        plt.gca().invert_yaxis()

        if savefig:
            if plot_time:
                plt.savefig('Element_Detection_Parameter_Importance.png', bbox_inches='tight', dpi=300)
            else:
                plt.savefig('Element_Detection_Parameter_Duration_Importance.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

        return

class objective_spec(object):
    """
    This class is used for optimizing the spectra processing parameters,
    using the high-level Optuna API

    Args:
        Lambda (array-like): An array-like object containing the wavelength values of the spectrum.
        flux (array-like): An array-like object containing the flux values of the spectrum.
        flux_err (array-like): An array-like object containing the flux error values of the spectrum.
        z_qso (float): The redshift of the QSO associated with the spectrum.
        z_element (float):
        rest_wavelength_1 (float):
        rest_wavelength_2 (float): 
        resolution_range (tuple): 
        resolution_element (bool): A tuple containing the (min, max) parameter range 
            to search through. Can be integer to hard code this parameter and exclude from
            the grid search. 
        halfWindow (tuple): A tuple containing the (min, max) parameter range 
            to search through. Can be integer to hard code this parameter and exclude from
            the grid search. 
        N_sig_limits (bool): A tuple containing the (min, max) parameter range 
            to search through. Can be integer to hard code this parameter and exclude from
            the grid search. 
        N_sig_line1 (bool): A tuple containing the (min, max) parameter range 
            to search through. Can be integer to hard code this parameter and exclude from
            the grid search. 
        N_sig_line2 (bool): A tuple containing the (min, max) parameter range 
            to search through. Can be integer to hard code this parameter and exclude from
            the grid search. 
    """

    def __init__(self, Lambda, flux, flux_err, z_qso, z_element, rest_wavelength_1, rest_wavelength_2, resolution_range,
        halfWindow, region_size, resolution_element, savgol_window_size, savgol_poly_order, N_sig_limits, N_sig_line1, N_sig_line2):

        # Spectra
        self.Lambda = Lambda 
        self.flux = flux 
        self.flux_err = flux_err 
        self.z_qso = z_qso
        self.z_element = z_element

        # Presets
        self.rest_wavelength_1 = rest_wavelength_1
        self.rest_wavelength_2 = rest_wavelength_2
        self.resolution_range = resolution_range
        
        # Tunable
        self.halfWindow = halfWindow
        self.region_size = region_size
        self.resolution_element = resolution_element
        self.savgol_window_size = savgol_window_size
        self.savgol_poly_order = savgol_poly_order
        self.N_sig_limits = N_sig_limits
        self.N_sig_line1 = N_sig_line1
        self.N_sig_line2 = N_sig_line2

    def __call__(self, trial):
        # Suggest the parameters that have been input as tuples
        halfWindow = trial.suggest_int('halfWindow', self.halfWindow[0], self.halfWindow[1], step=1) if isinstance(self.halfWindow, tuple) else self.halfWindow
        region_size = trial.suggest_int('region_size', self.region_size[0], self.region_size[1], step=1) if isinstance(self.region_size, tuple) else self.region_size
        resolution_element = trial.suggest_int('resolution_element', self.resolution_element[0], self.resolution_element[1], step=1) if isinstance(self.resolution_element, tuple) else self.resolution_element
        savgol_window_size = trial.suggest_int('savgol_window_size', self.savgol_window_size[0], self.savgol_window_size[1], step=1) if isinstance(self.savgol_window_size, tuple) else self.savgol_window_size
        savgol_poly_order = trial.suggest_int('savgol_poly_order', self.savgol_poly_order[0], self.savgol_poly_order[1], step=1) if isinstance(self.savgol_poly_order, tuple) else self.savgol_poly_order
        N_sig_limits = trial.suggest_float('N_sig_limits', self.N_sig_limits[0], self.N_sig_limits[1], step=0.05) if isinstance(self.N_sig_limits, tuple) else self.N_sig_limits
        N_sig_line1 = trial.suggest_float('N_sig_line1', self.N_sig_line1[0], self.N_sig_line1[1], step=0.05) if isinstance(self.N_sig_line1, tuple) else self.N_sig_line1
        N_sig_line2 = trial.suggest_float('N_sig_line2', self.N_sig_line2[0], self.N_sig_line2[1], step=0.05) if isinstance(self.N_sig_line2, tuple) else self.N_sig_line2

        if N_sig_limits > N_sig_line1 or N_sig_limits > N_sig_line2:
            return -999

        trial_spectrum = Spectrum(halfWindow=halfWindow, resolution_range=self.resolution_range, region_size=region_size, resolution_element=resolution_element,
            savgol_window_size=savgol_window_size, savgol_poly_order=savgol_poly_order, N_sig_limits=N_sig_limits, N_sig_line1=N_sig_line1, N_sig_line2=N_sig_line2,
            rest_wavelength_1=self.rest_wavelength_1, rest_wavelength_2=self.rest_wavelength_2, save_all=False)

        try:
            trial_spectrum.process_spectrum(self.Lambda, self.flux, self.flux_err, self.z_qso)
        except:
            return -99

        if len(trial_spectrum.df) != 0:
            return 1 - abs(self.z_element - trial_spectrum.df.z)
        else:
            return -9
       
class ThresholdExceeded(optuna.exceptions.OptunaError):
    """Exception raised when a threshold is exceeded during optimization with Optuna.

    This exception is a subclass of `optuna.exceptions.OptunaError` and is used to
    represent an error condition where a threshold has been exceeded during the
    optimization process with Optuna.

    Args:
        None 
    """
    pass


def _set_style_():
    """
    Function to configure the matplotlib.pyplot style. This function is called before any images are saved,
    after which the style is reset to the default.

    Args:
        None
    """

    plt.rcParams["xtick.color"] = "323034"
    plt.rcParams["ytick.color"] = "323034"
    plt.rcParams["text.color"] = "323034"
    plt.rcParams["lines.markeredgecolor"] = "black"
    plt.rcParams["patch.facecolor"] = "#bc80bd"  # Replace with a valid color code
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["patch.linewidth"] = 0.8
    plt.rcParams["scatter.edgecolors"] = "black"
    plt.rcParams["grid.color"] = "#b1afb5"  # Replace with a valid color code
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["legend.title_fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["font.size"] = 15
    #plt.rcParams["axes.prop_cycle"] = (cycler('color', ['#bc80bd', '#fb8072', '#b3de69', '#fdb462', '#fccde5', '#8dd3c7', '#ffed6f', '#bebada', '#80b1d3', '#ccebc5', '#d9d9d9']))  # Replace with valid color codes
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 6
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.8
    plt.rcParams["legend.fontsize"] = 13
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.borderpad"] = 0.2
    plt.rcParams["legend.columnspacing"] = 1.5
    plt.rcParams["legend.labelspacing"] = 0.4
    plt.rcParams["text.usetex"] = False
    plt.rcParams["axes.labelsize"] = 17
    plt.rcParams["axes.titlelocation"] = "center"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["axes.autolimit_mode"] = "round_numbers"
    plt.rcParams["axes.labelpad"] = 3
    plt.rcParams["axes.formatter.limits"] = (-4, 4)
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["figure.titlesize"] = 18
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 300

    return

def format_labels(labels: list) -> list:
    """
    Takes a list of labels and returns the list with all words capitalized and underscores removed.
    
    Args:
        labels (list): A list of strings.
    
    Returns:
        Reformatted list, of same lenght.
    """

    new_labels = []
    for label in labels:
        label = label.replace("_", " ")
        if label == "halfWindow":
            new_labels.append("Half-Window Size"); continue
        if label == "N_sig_line1":
            new_labels.append(r"N $\sigma$ Line 1"); continue
        if label == "N_sig_line2":
            new_labels.append(r"N $\sigma$ Line 2"); continue
        if label == "N_sig_limits":
            new_labels.append(r"N $\sigma$ Line Limits"); continue
        if label == "savgol_window_size":
            new_labels.append("SG Window Length"); continue
        if label == "savgol_poly_order":
            new_labels.append("SG Poly Order"); continue
        new_labels.append(label.title())

    return new_labels




