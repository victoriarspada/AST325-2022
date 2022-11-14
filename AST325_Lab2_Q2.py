"""
AST325 LAB2 PART 2: Iron gas velocity
@author: Victoria Spada
Last Edit: 2022-10-29
"""

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from astropy.io import fits
from fitting import *

hdulist = fits.open('Near-infrared.fits')
hdulist.info()

# Calculate the velocity of the iron gas in Figure 6 using the observed 
# wavelength of the [Fe II] 1.644 micron line. 

# 1. In addition to the five already identified OH lines, identify as many more OH lines as 
# possible from a region near the [Fe II] emission (e.g., the rectangular box) in Figure 6.
OH_lines = [ 16128.608, 16194.615, 16235.376, 16317.161, 16350.650, 16442.155, 16502.365 ] # [Angstroms]
OH_lines_pixel_estimate = [ 71, 84, 98, 115, 126, 157, 182 ]
len_spectrum = 256 # [pixels]

data = hdulist[0].data[0]
# In Figure 7, they took the mean values between pixels x=[135,150]
data_cut = data[:, 134:149] # select columns 135-150
data_medians = []
for col in range(0,len_spectrum,1):
    data_medians += [np.median(data[col, 134:149])]
pixels = np.linspace(1,256,256)

# # Plot the data as in the lab handout 
# plt.figure(figsize=(8,5))
# plt.plot(pixels, data_medians, color='navy')
# plt.title("OH Telluric Sky Line Spectrum")
# plt.xlabel("Y-Axis Pixels")
# plt.ylabel("OH Line Intensity")
# plt.grid(visible=True, color="lightgrey")
# plt.xlim([0,256])
# ax = plt.gca()
# ax.yaxis.set_minor_locator(MultipleLocator(500))
# ax.yaxis.set_major_locator(MultipleLocator(2000))
# ax.xaxis.set_minor_locator(MultipleLocator(10))
# ax.xaxis.set_major_locator(MultipleLocator(50))
# ax.tick_params(direction="in", which="both", top=True, 
#                bottom=True, left=True, right=True)
# plt.savefig("AST325_Lab3_OH_Spectrum.png")
# plt.show()

# 2. Determine the central positions (in terms of y-axis pixel numbers) of the identified lines, 
# then conduct polynomial fitting of the central positions to the wavelengths. This 
# gives a wavelength solution, which is mapping between the pixel positions and 
# wavelengths. Choose the degree of the polynomial fit between 1 (= linear least fit) and 3. 

peaks = [] # Pixels value at peak
peak_heights = [] # Signal value at peak
threshold = 50
# Iterate over spectrum in search of peaks
for j in range(len_spectrum - 4):
    i = j+2
    # Conditions that need to be met
    threshold_met = data_medians[i] > threshold
    higher_than_neighbour = (data_medians[i] > data_medians[i-1])&(data_medians[i] > data_medians[i+1])
    higher_than_neighbour2 = (data_medians[i] > data_medians[i-2])&(data_medians[i] > data_medians[i+2])
    if threshold_met & higher_than_neighbour & higher_than_neighbour2:
        peaks.append(pixels[i])
        peak_heights.append(data_medians[i])
        
# # Plot the peaks
# plt.plot(pixels, data_medians, '-', label="OH Spectrum", color="red")
# plt.title("OH telluric Sky Line Spectrum with Identified Peaks", fontsize=12)
# plt.xlabel("Y-Axis Pixels")
# plt.ylabel("Intensity")
# ax = plt.gca()
# ax.yaxis.set_minor_locator(MultipleLocator(500))
# ax.yaxis.set_major_locator(MultipleLocator(2000))
# ax.xaxis.set_minor_locator(MultipleLocator(10))
# ax.xaxis.set_major_locator(MultipleLocator(50))
# ax.tick_params(direction="in", which="both", top=True, 
#                bottom=True, left=True, right=True)
# plt.xlim([0,256])
# plt.ylim([0,11000])
# for i in range(len(peaks)): 
#     plt.plot(peaks[i]*np.ones(10), np.linspace(0,11000,10), color="grey",
#              linestyle="--", linewidth=1, label="Local Maximum" )
#     plt.plot(peaks[i], peak_heights[i], marker='x', color='black', markersize=5)
# plt.show()

# Locate the centroids
centroids = []
peakwidth = 6
fwhms = [] 

peaks_cp, peak_heights_cp = peaks, peak_heights
peaks = peaks[:-4]
peak_heights = peak_heights[:-4]
# Find FWHM for each peak
peaks_covered = 0
for i in range(0, len_spectrum-1, 1):
    curr_peak, curr_peak_height = peaks[peaks_covered], peak_heights[peaks_covered]
    curr_half_maximum = curr_peak_height/2
    curr_peak_width = 0 
    if data_medians[i] >= curr_half_maximum:
        while data_medians[i] >= curr_half_maximum:
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
    I_range = data_medians[indexmin: indexmax+1]
    x_range = np.array(x_range)
    I_range = np.array(I_range)
    xcm = np.sum(x_range*I_range) / np.sum(I_range)
    centroids += [xcm] 
    
# Now pick out the peaks that we selected in part (1) so we know which exact
# wavelengths we are dealing with
epsilon = 1
chosen_centroids = []
for i in range(0, len(centroids), 1):
    for j in range(0, len(OH_lines_pixel_estimate), 1):
        if (peaks[i] > OH_lines_pixel_estimate[j] - epsilon) and (peaks[i] < OH_lines_pixel_estimate[j] + epsilon):
            chosen_centroids += [peaks[i]]

# Plot the peaks and select FWMHs
plt.plot(pixels, data_medians, '-', label="OH Line Spectrum", color="red")
plt.title("OH Telluric Sky Lines with Identified Peaks and Centroids", fontsize=12)
plt.xlabel("Y-Axis Pixels")
plt.ylabel("Intensity")
ax = plt.gca()
#ax.yaxis.set_minor_locator(MultipleLocator(500))
ax.yaxis.set_major_locator(MultipleLocator(2000))
#ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.tick_params(direction="in", which="both", top=True, 
               bottom=True, left=True, right=True)
plt.xlim([0,256])
plt.ylim([0,11000])
for i in range(len(peaks_cp)): 
    # plt.plot(peaks_cp[i]*np.ones(10),np.linspace(0,11000,10), color="grey",
    #          linestyle="--", linewidth=1, label="Local Maximum" )
    epsilon = 5
    for j in chosen_centroids:
        if j == peaks_cp[i]:
            plt.plot(peaks_cp[i], peak_heights_cp[i], marker='x', color='black', markersize=5)
for i in range(len(chosen_centroids)):
    plt.plot(chosen_centroids[i]*np.ones(10),np.linspace(0,11000,10), color="green",
             linestyle="--", linewidth=2, label="Centroids" )
    if i == 0:
        plt.legend()
plt.savefig("AST325_Lab2_OH_Centroids.png")
plt.show()

# Use the OLS function I defined in Lab 1 to get the slope, y-intercept, their 
# uncertainties, and the R^2 value.
m, c, sigma_m, sigma_c, R2_OLS = least_squares_fit(chosen_centroids, OH_lines)
q, r, s, sigma_q, sigma_r, sigma_s, R2_quad = quadratic_regression_fit(chosen_centroids, OH_lines) 
chosen_centroids = np.array(chosen_centroids)

x_array = np.linspace(0,256,10)
y_array_linear = x_array*m+c
y_array_quad = np.multiply(x_array,x_array)*q+x_array*r + s
y_array_linear_res = (chosen_centroids)*m+c
y_array_quad_res = np.multiply(chosen_centroids,chosen_centroids)*q+chosen_centroids*r + s

delta_abs_ols, abs_err_ols = mean_abs_diff(OH_lines, y_array_linear_res)
delta_rel_ols, rel_err_ols = mean_rel_diff(OH_lines, y_array_linear_res)
print("The mean abs. difference between the OLS fit and the centroids is {} +/- {}".format(delta_abs_ols, abs_err_ols))
delta_abs_quad, abs_err_quad = mean_abs_diff(OH_lines, y_array_quad_res)
delta_rel_ols, rel_err_ols = mean_rel_diff(OH_lines, y_array_quad_res)
print("The mean abs. difference between the quadratic fit and the centroids is {} +/- {}".format(delta_abs_quad, abs_err_quad))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4))
ax1.grid(color="lightgrey")
ax1.plot(x_array, y_array_linear,
         linestyle="--", color="green", linewidth=2, label="OLS fit")
ax1.plot(x_array, y_array_quad,
         linestyle="--", color="purple", linewidth=2, label="Quadratic fit")
ax1.plot(chosen_centroids, OH_lines,
         linestyle="", marker="o", markeredgecolor="crimson", color="red",
         linewidth=2, markersize=8, label="Peak Centroids")
fig.suptitle('Fits for Calibrating the OH Telluric Line Spectrum')
ax1.set_xlabel("Y-Axis Pixel")
ax1.set_ylabel("Wavelength [Å]")
ax1.legend(loc="lower right")
ax1 = plt.gca()
# ax1.yaxis.set_minor_locator(MultipleLocator(50))
# ax1.xaxis.set_minor_locator(MultipleLocator(10))
ax1.tick_params(direction="in", which="both", top=True, 
               bottom=True, left=True, right=True)

ax2.grid(color="lightgrey")
ax2.axhline(y=0)
ax2.plot(chosen_centroids, y_array_linear_res-np.array(OH_lines),
         linestyle="", marker="o", markeredgecolor="green", color="green",
         linewidth=2, markersize=8, label="OLS Fit")   
yerr = np.sqrt((sigma_m/m)**2 + sigma_c**2)
ax2.errorbar(chosen_centroids, y_array_linear_res-np.array(OH_lines),
         yerr=yerr,
         linestyle="", marker="o", markeredgecolor="green", color="green",
         linewidth=1, markersize=8, capsize=4)   

ax2.plot(chosen_centroids, y_array_quad_res-np.array(OH_lines),
         linestyle="", marker="o", markeredgecolor="purple", color="purple",
         linewidth=2, markersize=8, label="Quadratic Fit")   
yerr = np.sqrt((sigma_q/q)**2 + (sigma_r/r)**2 + sigma_c**2)
ax2.errorbar(chosen_centroids, y_array_quad_res-np.array(OH_lines),
         yerr=yerr,
         linestyle="", marker="o", markeredgecolor="purple", color="purple",
         linewidth=1, markersize=8, capsize=4)   
ax2.set_title('Residuals')
ax2.set_xlabel("Pixel")
ax2.set_ylabel("Residual [Å]]")
ax2.legend(loc="lower right")
plt.savefig("AST325_Lab2_OH_Lines_OLS.png")
plt.show()

# 4. Determine the central position of the [Fe II] emission in Figure 6 in y-axis, and then 
# apply the wavelength solution that you already obtained above using OH sky lines to 
# estimate the wavelength of [Fe II] emission in Figure 6. The intrinsic wavelength of the 
# [Fe II] 1.644 µm line emission is 1.6439981 µm. What’s the velocity of the gas emitting 
# the [Fe II] emission in Figure 6

# x_centroid = 113
# y_centroid = 180
y_centroid_err, x_centroid_err = 2,2
Fe = data[177:185, 100:123]

# Take the median values between pixels y=[177,185]
data_cut = data[:, 178:183] # select rows 177-185
data_medians = []
for col in range(0,121-103,1):
    data_medians += [np.median(data[103+col, 178:183])]
x_range = np.arange(103,121,1)
I_range = data_medians
x_centroid = np.sum(x_range*I_range) / np.sum(I_range) 

# Take the median values between pixels y=[177,185]
data_cut = data[103:121, :] # select columns 103-123
data_medians = []
for col in range(0,183-178,1):
    data_medians += [np.median(data[103:121, 178+col])]
y_range = np.arange(178,183,1)
I_range = data_medians
y_centroid = np.sum(y_range*I_range) / np.sum(I_range)

# Plot the emission as a heatmap
pixel_plot = plt.figure()
plt.title("Fe [II] Emission")
pixel_plot = plt.imshow(data, cmap='plasma', vmin=2000, vmax=15000)
plt.xlim([103,121])
plt.ylim([178,183])
plt.colorbar(pixel_plot, label="Intensity", orientation="horizontal")
# plt.plot(x_centroid*np.ones(3),np.linspace(160,190,3),
#          color="white",linestyle="--",linewidth=3)
# plt.plot(np.linspace(90,130,3),y_centroid*np.ones(3),
#          color="white",linestyle="--",linewidth=3)
plt.savefig("AST325_Lab2_Fe_Centroid Marked")

# Estimate the wavelength of the Fe [II] Emission
lambda_est = q*y_centroid**2 + r*y_centroid + s
lambda_est_err =np.sqrt( np.sqrt( (y_centroid_err/y_centroid)**2 + (sigma_q/q)**2 ) + \
    np.sqrt( (y_centroid_err/y_centroid)**2 + (sigma_r/r)**2 ) +\
    sigma_s )
    
# lambda_est = m*y_centroid + c
# lambda_est_err = np.sqrt( (y_centroid_err/y_centroid)**2 + (sigma_m/m)**2 + \
#     sigma_s )
# Convert wavelength from Angstroms to um
lambda_est = 0.0001*lambda_est
lambda_est_err = 0.0001*lambda_est_err
    
# Use the Doppler shift equation to find the velocity
lambda_0 = 1.6439981 # intrinsic wavelength [um]
delta_lambda = lambda_est - lambda_0
c = 3e8 # speed of light [m/s]
v = delta_lambda*c/lambda_0
v_err = lambda_est_err*c/lambda_0
print("The estimated velocity is {} +/- {} km/s".format(v*1e-3, v_err*1e-3))




