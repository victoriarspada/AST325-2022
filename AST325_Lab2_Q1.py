"""
AST325 LAB 2 PART 1: Blackbody and Neon Spectra

@author: Victoria Spada
Last Edit: 2022-10-20
"""

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from fitting import *

# Open blackbody and neon data files
blackbody = np.loadtxt("Group_H_BB.dat")
neon = np.loadtxt("Ne_calib.dat")
pixels = np.arange(len(neon))

# First, open the blackbody and neon data and plot it
# Plot the blackbody spectrum
plt.figure()
plt.plot(blackbody, color='black')
plt.title("Blackbody Spectrum Obtained from Spectrograph", fontsize=12)
plt.xlabel("Pixels")
plt.ylabel("Intensity")
plt.grid(visible=True, color="lightgrey")
ax = plt.gca()
ax.yaxis.set_minor_locator(MultipleLocator(50))
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.xaxis.set_major_locator(MultipleLocator(200))
ax.tick_params(direction="in", which="both", top=True, 
               bottom=True, left=True, right=True)
plt.xlim([0,1000])
plt.savefig("AST325_Lab2_Blackbody.png")
plt.show()

# Plot the neon lamp spectrum
plt.figure()
plt.plot(neon, color='red')
plt.title("Spectrum of Neon Lamp Obtained from USB4000 Spectrograph", fontsize=12)
plt.xlabel("Pixels")
plt.ylabel("Intensity")
plt.grid(visible=True, color="lightgrey")
ax = plt.gca()
ax.yaxis.set_minor_locator(MultipleLocator(50))
ax.yaxis.set_major_locator(MultipleLocator(200))
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.xaxis.set_major_locator(MultipleLocator(200))
ax.tick_params(direction="in", which="both", top=True, 
               bottom=True, left=True, right=True)
plt.xlim([0,1000])
plt.ylim([0,1600])
plt.savefig("AST325_Lab2_Neon.png")
plt.show()

# (1) Identify the wavelengths of the Neon lines in Figure 3 as many as possible 
# using the information in Figure 4 and 5. Some nearby lines may overlap.
neon_lines = [ 540.056, 585.249, 640.225, 650.653, 692.947, 703.241, 717.394, 
              724.512, 743.890, 748.887, 753.577 ]
neon_lines_pixel_estimate = [ 53, 241, 475, 522, 698, 742, 804, 830, 916, 928, 954]

# (2) Determine the centroids (i.e., pixels positions) of the lines. 
peaks = [] # pixels value at peak
peak_heights = [] # Signal value at peak
threshold = 50

# Iterate over signal array in search of neon peaks
for j in range(len(neon)- 4):
    i = j+2
    # Conditions that need to be met
    threshold_met = neon[i] > threshold
    higher_than_neighbour = (neon[i] > neon[i-1])&(neon[i] > neon[i+1])
    higher_than_neighbour2 = (neon[i] > neon[i-2])&(neon[i] > neon[i+2])
    if threshold_met & higher_than_neighbour & higher_than_neighbour2:
        peaks.append(pixels[i])
        peak_heights.append(neon[i])
        
# # Plot the peaks
# plt.plot(pixels, neon, '-', label="Neon Spectrum", color="red")
# plt.title("Neon Lamp Spectrum with Identified Peaks", fontsize=12)
# plt.xlabel("Pixels")
# plt.ylabel("Intensity")
# ax = plt.gca()
# ax.yaxis.set_minor_locator(MultipleLocator(50))
# ax.yaxis.set_major_locator(MultipleLocator(200))
# ax.xaxis.set_minor_locator(MultipleLocator(50))
# ax.xaxis.set_major_locator(MultipleLocator(200))
# ax.tick_params(direction="in", which="both", top=True, 
#                bottom=True, left=True, right=True)
# plt.xlim([0,1000])
# plt.ylim([0,1600])
# for i in range(len(peaks)): 
#     # plt.plot(peaks[i]*np.ones(10), np.linspace(0,1600,10), color="grey",
#     #          linestyle="--", linewidth=1, label="Local Maximum" )
#     plt.plot(peaks[i], peak_heights[i], marker='x', color='black', markersize=5)
# plt.savefig("AST325_Lab2_Neon_Maxima.png")
# plt.show()
    
# Locate the centroids
centroids = []
peakwidth = 6
fwhms = [] 

# Find FWHM for each peak
peaks_covered = 0
for i in range(0,len(neon),1):
    curr_peak, curr_peak_height = peaks[peaks_covered], peak_heights[peaks_covered]
    curr_half_maximum = curr_peak_height/2
    curr_peak_width = 0 
    if neon[i] >= (curr_half_maximum):
        while neon[i] >= (curr_half_maximum):
            curr_peak_width += 1
            i += 1
        fwhms += [curr_peak_width]
        peaks_covered += 1
        if peaks_covered == len(peaks):
            break

# Now compute the centroid using the FWHMs
for i, peak in enumerate(peaks):
    peakwidth = fwhms[i]
    indexmin = int(np.where(pixels==peaks[i])[0][0] - peakwidth/2)
    indexmax = int(np.where(pixels==peaks[i])[0][0] + peakwidth/2)
    x_range = pixels[indexmin: indexmax+1]
    I_range = neon[indexmin: indexmax+1]
    x_range = np.array(x_range)
    I_range = np.array(I_range)
    xcm = np.sum(x_range*I_range) / np.sum(I_range)
    centroids += [xcm] 
    
# Now pick out the peaks that we selected in part (1) so we know which exact
# wavelengths we are dealing with
epsilon = 5
chosen_centroids = []
for i in range(0, len(centroids), 1):
    for j in range(0, len(neon_lines_pixel_estimate), 1):
        if (peaks[i] > neon_lines_pixel_estimate[j] - epsilon) and (peaks[i] < neon_lines_pixel_estimate[j] + epsilon):
            chosen_centroids += [peaks[i]]

# Plot the peaks and select FWMHs
plt.plot(pixels, neon, '-', label="Neon Spectrum", color="red")
plt.title("Neon Lamp Spectrum (USB4000) with Identified Peaks", fontsize=12)
plt.xlabel("Pixels")
plt.ylabel("Intensity")
ax = plt.gca()
ax.yaxis.set_minor_locator(MultipleLocator(50))
ax.yaxis.set_major_locator(MultipleLocator(200))
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.xaxis.set_major_locator(MultipleLocator(200))
ax.tick_params(direction="in", which="both", top=True, 
               bottom=True, left=True, right=True)
plt.xlim([0,1000])
plt.ylim([0,1600])
for i in range(len(peaks)): 
#     # plt.plot(peaks[i]*np.ones(10),np.linspace(0,1600,10), color="grey",
#     #          linestyle="--", linewidth=1, label="Local Maximum" )
#     plt.plot(peaks[i], peak_heights[i], marker='x', color='black', markersize=5)
    for j in chosen_centroids:
        if j == peaks[i]:
            plt.plot(peaks[i], peak_heights[i], marker='x', color='black', markersize=5)
for i in range(len(chosen_centroids)):
    plt.plot(chosen_centroids[i]*np.ones(10),np.linspace(0,1600,10), color="green",
             linestyle="--", linewidth=1, label="Centroids" )
    if i == 0:
        plt.legend()
plt.savefig("AST325_Lab2_Neon_Centroids.png")
plt.show()

# (3) Obtain a linear least square fitting between the pixels positions of the 
# Neon lines that you identified and their wavelengths. How good is the fitting?
# The linear fitting is the wavelength solution.

# Use the OLS function I defined in Lab 1 to get the slope, y-intercept, their 
# uncertainties, and the R^2 value.
m, c, sigma_m, sigma_c, R2 = least_squares_fit(chosen_centroids, neon_lines)
ols_x_array = np.linspace(0,1000,10)
ols_y_array = np.array(chosen_centroids)*m+c

delta_abs, abs_err = mean_abs_diff(neon_lines, ols_y_array)
delta_rel, rel_err = mean_rel_diff(neon_lines, ols_y_array)

slope_text = "m = {} ± {}".format(round(m,3), round(sigma_m,3)) 
const_text = "c = {} ± {}".format(round(c,3), round(sigma_c,3)) 
R2_text = "$R^2$ = {}".format(R2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4))
fig.suptitle('Ordinary Least Squares fit for Calibrating the Neon Lamp Spectrum (USB4000)')
ax1.plot(ols_x_array, ols_x_array*m+c,
         linestyle="--", color="green", linewidth=2, label="OLS fit")
ax1.plot(chosen_centroids, neon_lines,
         linestyle="", marker="o", markeredgecolor="navy", color="red",
         linewidth=2, markersize=8, label="Peak Centroids")
ax1.set_title('OLS Fit')
ax1.set_xlabel("Pixel")
ax1.set_ylabel("Wavelength [nm]")
ax1.legend(loc="lower right")
ax = plt.gca()
ax1.yaxis.set_minor_locator(MultipleLocator(10))
ax1.xaxis.set_minor_locator(MultipleLocator(25))
ax1.tick_params(direction="in", which="both", top=True, 
               bottom=True, left=True, right=True)
ax1.grid(color="lightgrey")

ax2.axhline(y=0)
ax2.plot(chosen_centroids, ols_y_array-np.array(neon_lines),
         linestyle="", marker="o", markeredgecolor="navy", color="red",
         linewidth=2, markersize=8, label="OLS Fit - Neon Lines")
ax2.errorbar(chosen_centroids, ols_y_array-np.array(neon_lines), 
             yerr=sigma_m*np.array(chosen_centroids)+sigma_c,
             marker="o", markeredgecolor="navy", color="red",
             linestyle="", capsize=4)
delta_abs = mean_abs_diff(np.array(chosen_centroids)*m+c, np.array(neon_lines))
ax2.set_ylim([-2,2])
ax2.set_title('Residuals')
ax2.set_xlabel("Pixel")
ax2.set_ylabel("Residual [nm]")
ax2.legend(loc="lower left")
ax2.grid(color="lightgrey")
plt.savefig("AST325_Lab2_Neon_OLS.png")
plt.show()

# (4) Apply the wavelength solution that you obtained in step 3 above to the
# blackbody spectrum. Now you know the wavelengths of the blackbody spectrum.
# What is the temperature of the blackbody?
def blackbody_radiation(wavelengths, uncertainties):
    """
    Equation for blackbody radiation

    Returns
    -------
    Temperature of blackbody.

    """
    max_lambda = max(wavelengths)
    lambda_uncertainty = uncertainties[np.argmax(wavelengths)]

    A = (2.898e-3)*1e9 # [ nm K ]
    T = A/max_lambda # Temperature [K]
    T_uncertainty = T*(lambda_uncertainty/max_lambda)
    return T, T_uncertainty


# Plot the intensity vs wavelength for the blackbody
blackbody_wavelengths = m*pixels + c
blackbody_wavelengths_uncertainty = sigma_m*pixels
T, sigma_T = blackbody_radiation(blackbody_wavelengths, blackbody_wavelengths_uncertainty)
print("(4) Blackbody temperature is {} ± {} K.".format(T, sigma_T))

plt.figure(figsize=(6.6,5))
# plt.plot(blackbody_wavelengths+blackbody_wavelengths_uncertainty, blackbody,
#          '-', linewidth=1, label="Uncertainty", color="lightgrey")
# plt.plot(blackbody_wavelengths-blackbody_wavelengths_uncertainty, blackbody,
#          '-', linewidth=1, label="Uncertainty", color="lightgrey")
plt.plot(blackbody_wavelengths, blackbody, '-', label="Blackbody Spectrum", color="red")
plt.errorbar(blackbody_wavelengths[np.argmax(blackbody)], max(blackbody),
             xerr=blackbody_wavelengths_uncertainty[np.argmax(blackbody)],
             marker='o', linestyle="", capsize=7,
             label="Maximum Intensity", color="green", markeredgecolor='black')
print(" {} +/- {} ".format(blackbody_wavelengths[np.argmax(blackbody)], blackbody_wavelengths_uncertainty[np.argmax(blackbody)]))
plt.legend()
plt.title("Blackbody Spectrum Obtained from USB4000 Spectrograph, Calibrated for Wavelength",
          fontsize=15)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Intensity")
ax = plt.gca()
ax.yaxis.set_minor_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.tick_params(direction="in", which="both", top=True, 
               bottom=True, left=True, right=True)
plt.grid(color="lightgrey")
plt.savefig("AST325_Lab2_Blackbody_Wavelengths.png")
plt.show()

