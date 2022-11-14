"""
AST325 LAB2 PART 3: Nighttime Observations
@author: Victoria Spada
Last Edit: 2022-10-20
"""

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from astropy.io import fits
from fitting import *

body_files = ["Mars_1s.fit", "aldebaran_10s.fit", "Jupiter_10s.fit", "elnath_10s.fit", "Neon_8s.fit"]
flat_files = ["flat_1s.fit"]
bias_files = ["bias01.fit", "bias02.fit", "bias03.fit"]
dark_files = ["dark_1s_01.fit", "dark_1s_02.fit", "dark_1s_03.fit",
              "dark_5s_01.fit", "dark_5s_02.fit", "dark_5s_02.fit",
              "dark_10s_01.fit", "dark_10s_02.fit", "dark_10s_03.fit",
              "dark_8s_01.fit", "dark_8s_02.fit", "dark_8s_03.fit"]

# Make master dark files variables 
# 1 second darks
hdulist1, hdulist2, hdulist3 = fits.open(dark_files[0]), fits.open(dark_files[1]), fits.open(dark_files[2])
dark0, dark1, dark2  = hdulist1[0].data, hdulist2[0].data, hdulist3[0].data
dark_1s_data = np.array([dark0, dark1, dark2])
dark_1s_master = np.median(dark_1s_data, axis=0)
# 5 second darks
hdulist1, hdulist2, hdulist3 = fits.open(dark_files[3]), fits.open(dark_files[4]), fits.open(dark_files[5])
dark3, dark4, dark5 = hdulist1[0].data, hdulist2[0].data, hdulist3[0].data
dark_5s_data = np.array([dark3, dark4, dark5])
dark_5s_master = np.median(dark_5s_data, axis=0)
# 10 second darks
hdulist1, hdulist2, hdulist3 = fits.open(dark_files[6]), fits.open(dark_files[7]), fits.open(dark_files[8])
dark6, dark7, dark8 = hdulist1[0].data, hdulist2[0].data, hdulist3[0].data
dark_10s_data = np.array([dark6, dark7, dark8])
dark_10s_master = np.median(dark_10s_data, axis=0)
# 8 second darks
hdulist1, hdulist2, hdulist3 = fits.open(dark_files[9]), fits.open(dark_files[10]), fits.open(dark_files[11])
dark9, dark10, dark11 = hdulist1[0].data, hdulist2[0].data, hdulist3[0].data
dark_8s_data = np.array([dark9, dark10, dark11])
dark_8s_master = np.median(dark_8s_data, axis=0)

# Open the flat .fits file
hdulist = fits.open(flat_files[0])
flat = hdulist[0].data

# Make a master bias variable
hdulist1, hdulist2, hdulist3 = fits.open(bias_files[0]), fits.open(bias_files[1]), fits.open(bias_files[2])
bias0, bias1, bias2  = hdulist1[0].data, hdulist2[0].data, hdulist3[0].data
bias_data = np.array([bias0, bias1, bias2])
bias_master = np.median(bias_data, axis=0)


##############################################
# (1) Perform a Neon Line Spectrum calibration
i = 4
titles = ["Mars Spectrum", "Aldebaran Spectrum", "Jupiter Spectrum", "Elnath Spectrum", "Neon Spectrum (Shelyak Alpy)"]
hdulist = fits.open(body_files[4])
neon_data = hdulist[0].data

# # Plot the emission as a heatmap
# pixel_plot = plt.figure()
# plt.title(titles[i])
# pixel_plot = plt.imshow(neon_data, cmap='plasma')
# plt.colorbar(pixel_plot, label="Intensity", orientation="horizontal")
# plt.show()

neon_data = neon_data - dark_8s_master # Subtract master dark measurement from object spectrum
flat_minus_dark = flat - dark_8s_master # Subtract dark from flat    
neon_data_reduced = np.divide(neon_data, flat_minus_dark)
y_min = 150
y_max = 250

# Plot the emission as a heatmap
pixel_plot = plt.figure()
plt.title("{} with Data Reduction".format(titles[4]))
pixel_plot = plt.imshow(neon_data, cmap='plasma')
plt.ylim([y_min,y_max])
plt.colorbar(pixel_plot, label="Intensity", orientation="horizontal")
plt.savefig("AST325_Lab2 {} with Dark & Bias Subtracted, Flat Divided.jpg")
plt.show()

# Take the median values between pixels y=[150,350]
# neon_data_cut = neon_data_reduced[150:300, :] # select columns 103-123
neon_data_medians = []
for col in range(0,695,1):
    neon_data_medians += [np.median(neon_data_reduced[y_min:y_max, col])]
    
# # Plot the neon spectrum
# plt.figure()
# plt.plot(neon_data_medians, color='red')
# plt.title("Neon Spectrum Obtained from Shelyak Alpy Spectrograph", fontsize=12)
# plt.xlabel("Pixels")
# plt.ylabel("Intensity")
# plt.grid(visible=True, color="lightgrey")
# ax = plt.gca()
# ax.yaxis.set_minor_locator(MultipleLocator(0.1))
# ax.yaxis.set_major_locator(MultipleLocator(0.5))
# ax.xaxis.set_minor_locator(MultipleLocator(10))
# ax.xaxis.set_major_locator(MultipleLocator(100))
# ax.tick_params(direction="in", which="both", top=True, 
#                 bottom=True, left=True, right=True)
# plt.ylim([-0.1,3.0])
# plt.xlim([-50,745])
# plt.savefig("AST325_Lab2_Shelyak_Alpy_Neon.png")
# plt.show()

# Identify the wavelengths of the Neon lines
neon_lines = [ 540.056, 585.249, 640.225, 650.653, 659.895, 692.947, 703.241, 717.394]
neon_lines_pixel_estimate = [ 178, 412, 575, 590, 639, 657, 677, 691 ]

# (2) Determine the centroids (i.e., pixels positions) of the lines. 
pixels = np.arange(0,695,1)
neon_peaks = [] # pixels value at peak
neon_peak_heights = [] # Signal value at peak
threshold = 0.025

# Iterate over signal array in search of neon peaks
for j in range(len(neon_data_medians)- 4):
    i = j+2
    # Conditions that need to be met
    threshold_met = neon_data_medians[i] > threshold
    higher_than_neighbour = (neon_data_medians[i] > neon_data_medians[i-1])&(neon_data_medians[i] > neon_data_medians[i+1])
    higher_than_neighbour2 = (neon_data_medians[i] > neon_data_medians[i-2])&(neon_data_medians[i] > neon_data_medians[i+2])
    if threshold_met & higher_than_neighbour & higher_than_neighbour2:
        if neon_data_medians[i] > 0.2:
            neon_peaks.append(pixels[i])
            neon_peak_heights.append(neon_data_medians[i])
        
# # Plot the peaks
# plt.plot(pixels, neon_data_medians, '-', label="Neon Spectrum", color="red")
# plt.title("Neon Spectrum with Identified Peaks", fontsize=12)
# plt.xlabel("Pixels")
# plt.ylabel("Intensity")
# plt.grid(visible=True, color="lightgrey")
# ax = plt.gca()
# ax.yaxis.set_minor_locator(MultipleLocator(1000))
# ax.yaxis.set_major_locator(MultipleLocator(5000))
# ax.xaxis.set_minor_locator(MultipleLocator(10))
# ax.xaxis.set_major_locator(MultipleLocator(100))
# ax.tick_params(direction="in", which="both", top=True, 
#                bottom=True, left=True, right=True)
# plt.xlim([400,720])
# plt.ylim([-500,32000])
# for i in range(len(neon_peaks)): 
#     plt.plot(neon_peaks[i]*np.ones(10), np.linspace(0,36000,10), color="grey",
#              linestyle="--", linewidth=1, label="Local Maximum" )
#     plt.plot(neon_peaks[i], neon_peak_heights[i], marker='x', color='black', markersize=5)
# plt.savefig("AST325_Lab2_Shelyak_Alpy_Neon_Maxima.png")
# plt.show()
    
# Locate the centroids
neon_centroids = []
peakwidth = 6
neon_fwhms = [] 

# Find FWHM for each peak
peaks_covered = 0
for i in range(0,len(neon_data_medians),1):
    curr_peak, curr_peak_height = neon_peaks[peaks_covered], neon_peak_heights[peaks_covered]
    curr_half_maximum = curr_peak_height/2
    curr_peak_width = 0 
    if neon_data_medians[i] >= (curr_half_maximum):
        while neon_data_medians[i] >= (curr_half_maximum):
            curr_peak_width += 1
            i += 1
        neon_fwhms += [curr_peak_width]
        peaks_covered += 1
        if peaks_covered == len(neon_peaks):
            break

# Now compute the centroid using the FWHMs
for i, peak in enumerate(neon_peaks):
    peakwidth = neon_fwhms[i]
    indexmin = int(np.where(pixels==neon_peaks[i])[0][0] - peakwidth/2)
    indexmax = int(np.where(pixels==neon_peaks[i])[0][0] + peakwidth/2)
    x_range = pixels[indexmin: indexmax+1]
    I_range = neon_data_medians[indexmin: indexmax+1]
    x_range = np.array(x_range)
    I_range = np.array(I_range)
    xcm = np.sum(x_range*I_range) / np.sum(I_range)
    neon_centroids += [xcm] 

# Now pick out the peaks that we selected in part (1) so we know which exact
# wavelengths we are dealing with
epsilon = 5
chosen_neon_centroids = []
for i in range(0, len(neon_centroids), 1):
    for j in range(0, len(neon_lines_pixel_estimate), 1):
        if (neon_peaks[i] > neon_lines_pixel_estimate[j] - epsilon) and (neon_peaks[i] < neon_lines_pixel_estimate[j] + epsilon):
            chosen_neon_centroids += [neon_peaks[i]]

# Plot the peaks and select FWMHs
plt.plot(pixels, neon_data_medians, '-', label="Neon Spectrum", color="red")
plt.title("Neon Spectrum (Shelyak Alpy) with Identified Peaks", fontsize=12)
plt.xlabel("Pixels")
plt.ylabel("Intensity")
plt.grid(visible=True, color="lightgrey")
ax = plt.gca()
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(50))
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.tick_params(direction="in", which="both", top=True, 
               bottom=True, left=True, right=True)
plt.ylim([-0.1,2.7])
plt.xlim([-5,745])
for i in range(len(neon_peaks)): 
    plt.plot(neon_peaks[i]*np.ones(10),np.linspace(-1,3.0,10), color="grey",
             linestyle="--", linewidth=1, label="Local Maximum" )
    plt.plot(neon_peaks[i], neon_peak_heights[i], marker='x', color='black', markersize=5)
for i in range(len(chosen_neon_centroids)):
    plt.plot(chosen_neon_centroids[i]*np.ones(10),np.linspace(-1,3.0,10), color="green",
             linestyle="--", linewidth=1, label="Centroids" )
plt.savefig("AST325_Lab2_Shelyak_Alpy_Neon_Centroids.png")
plt.show()

# (3) Obtain a linear least square fitting between the pixels positions of the 
# Neon lines that you identified and their wavelengths. How good is the fitting?
# The linear fitting is the wavelength solution.
chosen_neon_centroids = np.array(chosen_neon_centroids, dtype=float)
m, c, sigma_m, sigma_c, R2 = least_squares_fit(chosen_neon_centroids, neon_lines)
q, r, s, sigma_q, sigma_r, sigma_s, R2_quad = quadratic_regression_fit(chosen_neon_centroids, neon_lines) 

x_array = np.linspace(0,1000,10)
ols_y_array = np.array(chosen_neon_centroids)*m+c
y_array_quad = np.multiply(x_array,x_array)*q+x_array*r + s
y_array_linear_res = chosen_neon_centroids*m+c
y_array_quad_res = np.multiply(chosen_neon_centroids,chosen_neon_centroids)*q+chosen_neon_centroids*r + s

delta_abs_ols, abs_err_ols = mean_abs_diff(neon_lines, y_array_linear_res)
delta_abs_quad, abs_err_quad = mean_abs_diff(neon_lines, y_array_quad_res)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4))
fig.suptitle('Ordinary Least Squares fit for Calibrating the Neon Spectrum (Shelyak Alpy)')
ax1.plot(x_array, x_array*m+c,
         linestyle="--", color="green", linewidth=2, label="OLS fit")
ax1.plot(x_array, y_array_quad,
         linestyle="--", color="purple", linewidth=2, label="Quadratic fit")
ax1.plot(chosen_neon_centroids, neon_lines,
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
ax2.plot(chosen_neon_centroids, ols_y_array-np.array(neon_lines),
         linestyle="", marker="o", markeredgecolor="green", color="green",
         linewidth=2, markersize=8, label="OLS Fit - Neon Lines")
ax2.errorbar(chosen_neon_centroids, ols_y_array-np.array(neon_lines),
         yerr=sigma_m*np.array(chosen_neon_centroids)+sigma_c,
         linestyle="", marker="o", markeredgecolor="green", color="green",
         linewidth=1, markersize=8, capsize=4)

ax2.plot(chosen_neon_centroids, y_array_quad_res-np.array(neon_lines),
         linestyle="", marker="o", markeredgecolor="purple", color="purple",
         linewidth=2, markersize=8, label="Quadratic Fit") 
yerr = np.sqrt((sigma_q/q)**2 + (sigma_r/r)**2 + sigma_c**2)
ax2.errorbar(chosen_neon_centroids, y_array_quad_res-np.array(neon_lines),
         yerr=yerr,
         linestyle="", marker="o", markeredgecolor="purple", color="purple",
         linewidth=1, markersize=8, capsize=4)   
ax2.set_title('Residuals')
ax2.set_xlabel("Pixel")
ax2.set_ylabel("Residual [nm]")
ax2.legend(loc="lower right")
ax2.grid(color="lightgrey")
ax2.grid(color="lightgrey")
plt.savefig("AST325_Lab2_Neon_OLS_Shelyak Alpy.png")
plt.show()

# (2) Perform analysis on other sepctra
# Index of spectrum file
i = 2
hdulist = fits.open(body_files[i])
data = hdulist[0].data

# Plot the emission as a heatmap
pixel_plot = plt.figure()
plt.title(titles[i])
pixel_plot = plt.imshow(data, cmap='plasma')
plt.colorbar(pixel_plot, label="Intensity", orientation="horizontal")
plt.savefig("AST325_Lab2 {}.jpg".format(titles[i]))
plt.show()

# We will extract spectra of the observed objects after applying the standard data reduction 
# process, including subtraction of dark (bias), flat-fielding, and wavelength calibration 
# * Since Dark image already has Bias in it, Dark subtraction includes Bias subtraction.

pixels = np.arange(0,695,1)
wavelengths = pixels*m + c
if i==0: # 1s Mars
    data = data - dark_1s_master # Subtract master dark measurement from object spectrum
    flat_minus_dark = flat - dark_1s_master # Subtract dark from flat
    y_min = 150
    y_max = 160
    spectra = [wavelengths[561], wavelengths[678]]
    indices = [561, 672]
if i==1 or i==2 or i==3: # 10s Aldebaran Elnath Jupiter
    data = data - dark_10s_master # Subtract master dark measurement from object spectrum
    flat_minus_dark = flat - dark_10s_master # Subtract dark from flat   
    
    if i==1: # aldebaran
        y_min = 225
        y_max = 235
        spectra = [wavelengths[315], wavelengths[418], wavelengths[468], wavelengths[561], wavelengths[609], wavelengths[677]]    
        indices = [ 315, 418, 468, 561, 609, 673]
    if i==2: # jupiter
        y_min = 175
        y_max = 200
        indices = [164, 193, 272, 621, 673]
    if i==3: # Elnath
        y_min = 270
        y_max = 280
        spectra = [wavelengths[270], wavelengths[197], wavelengths[163], wavelengths[144]]
        indices = [270, 197, 163, 144]
data_reduced = np.divide(data, flat_minus_dark)

# Plot the emission as a heatmap
pixel_plot = plt.figure()
pixel_plot = plt.imshow(neon_data, cmap='plasma')
plt.colorbar(pixel_plot, label="Intensity", orientation="horizontal")
plt.title("{} with Dark and Bias Subtracted, and Flat Divided".format(titles[i]))
pixel_plot = plt.imshow(data, cmap='plasma')
plt.ylim([y_min,y_max])
plt.savefig("AST325_Lab2 {} with Dark,Bias Subtracted, Flat Divided.jpg")
plt.show()

# Take the median values between pixels
data_cut = data_reduced[y_min:y_max, :] # select rows
data_medians = []
for col in range(0,695,1):
    data_medians += [np.median(data_reduced[y_min:y_max, col])]
    
wavelengths = pixels*m + c
wavelengths_uncertainties = np.sqrt( (sigma_m/pixels)**2 + sigma_c**2 )
#wavelengths = np.multiply(pixels,pixels)*q + pixels*r + s
# Locate the centroids
# (2) Determine the centroids (i.e., pixels positions) of the lines. 
pixels = np.arange(0,695,1)
peaks = [] # pixels value at peak
peak_heights = [] # Signal value at peak
threshold = 0.005

# Iterate over signal array in search of neon peaks
for j in range(len(data_medians)- 4):
    k = j+2
    # Conditions that need to be met
    threshold_met = data_medians[k] > threshold
    lower_than_neighbour = (data_medians[k] < data_medians[k-1])&(data_medians[i] < data_medians[k+1])
    lower_than_neighbour2 = (data_medians[k] < data_medians[k-2])&(data_medians[i] < data_medians[k+2])
    if threshold_met & lower_than_neighbour & lower_than_neighbour2:
            peaks.append(wavelengths[k])
            peak_heights.append(data_medians[k])
    
    
# Plot the spectrum
plt.figure()
plt.plot(wavelengths, data_medians, color='red')
plt.title("{} Obtained from Shelyak Alpy Spectrograph, Calibrated".format(titles[i]), fontsize=12)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Intensity")
plt.grid(visible=True, color="lightgrey")
ax = plt.gca()
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
# ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.tick_params(direction="in", which="both", top=True, 
                bottom=True, left=True, right=True)
for j in indices:
    plt.plot(wavelengths[j], data_medians[j], marker='x', color='black', markersize=5)
# for j in range(len(peaks)): 
#     if peaks[j] in spectra:
#         plt.plot(peaks[j], peak_heights[j], marker='x', color='black', markersize=5)
if True: #i ==3:
    print(wavelengths[np.array(data_medians).argmax()], max(data_medians), np.array(data_medians).argmax())
    plt.plot(wavelengths[np.array(data_medians).argmax()], max(data_medians),
             marker='^',color='green')
plt.savefig("AST325_Lab2_Shelyak_Alpy_{}.png".format(titles[i]))
plt.show()
