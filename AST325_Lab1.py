# -*- coding: utf-8 -*-
"""
This python code was written for Lab 1 of AST325H1, Fall 2022.

Author: Victoria Spada
Last Edit: 2022-09-18
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import scipy.special
from scipy.stats import poisson, norm


# Part 1 of the Lab requires recreating the histogram figure shown in the lab handout.
def get_histogram(points):
    """
    This function returns an  array of each of the unique values in the input 
    points array as well the number of occurrences each value had in the input array 

    Parameters
    ----------
    points : array
        Array of ints.

    Returns
    -------
    values : array
        Array of unique values found in the input points array
    counts : array
        Array of number of occurrences of each of the values in the input points array

    """

    values = []
    counts = []
    for i in points:
        # Check if already in possible values list
        if i not in values:
            values += [i]
            counts += [1]
        else:
            index = values.index(i)
            counts[index] += 1
    
    return values, counts

def poisson(x_points, mean):
    """
    This function gives the Poisson distribution for an input array of x values.

    Parameters
    ----------
    x_points : array
        x-axis points for a Poisson distribution.
    mean : float
        mean value for the Poisson distribution.

    Returns
    -------
    y_points : array
        Poisson distribution for input x_points array

    """
    
    a = mean**x_points
    b = scipy.special.factorial(x_points)
    c = np.divide(a, b)
    d = np.exp(-mean)
    
    return np.multiply(c, d)

def gauss(x_points, mean, stddev):
    """
    This function gives the Gaussian distribution for an input array of x values.

    Parameters
    ----------
    x_points : array
        x-axis points for the Gaussian distribution.
    mean : float
        Mean of the Gaussian distribution.
    stddev : float
        Standard deviation for the Gaussian distribution.

    Returns
    -------
    y_points : array
        Gaussian distribution for input x_points array

    """
    
    a = 1/(stddev*np.sqrt(2*np.pi))
    b = ((x_points - mean)/stddev)
    c = np.multiply(b, b)
    d = np.exp(-0.5*c)
    
    return a*d

def least_squares_fit(x,y):
    """
    This function gives the slope and y intercept of a 2-array dataset for an
    ordinary least squares fitting.

    Parameters
    ----------
    x : 1D array of floats or ints
        x-axis points.
    y : 1D array of floats or ints
        Mean of the Gaussian distribution.

    Returns
    -------
    m : float
        Resulting slope for ordinary least squares fitting.
    c : float
        Resulting y-intercept for ordinary leats squares fitting.
    sigma_m : float
        The standard deviation of the slope m.
    sigma_c : float
        The standard deviation of the slope m

    """
    # Define variables for slope and y intercept
    m, c, sigma_m, sigma_c = 0, 0, 0, 0
    
    length = np.size(x)
    if length == np.size(y): # Check that input arrays are the same length
        # Construct the matrices, [m c] = A^-1 * B
        # Start with matrix A
        a, b, d, e = 0, 0, 0, length
        for i in range(0, length, 1):
            a += x[i]**2
            b += x[i]
            d += x[i]
        A = np.array([[a, b], [d, e]]) 
        # Get the inverse of matrix A
        A_inv = np.linalg.inv(A)
        
        # Construct matrix B
        f, g = 0, 0
        for i in range(0, length, 1):
            f += x[i]*y[i]
            g += y[i]
        B = np.array([[f], [g]])
        
        # Find m and c resulting from [m c] = A_inv * B
        h, j, k, l = A_inv[0,0], A_inv[0,1], A_inv[1,0], A_inv[1,1]
        m = h*f + j*g
        c = k*f + l*g
        # Alternatively, can use numpy.matmul()
        # m, c = np.matmul(A_inv, B)
        
        # Now find the errors for the slope and y intercept
        # First estimate the standard deviation sigma 
        C = 0
        for i in range(0, length, 1):
            C += (y[i] - (m*x[i] + c))**2
        sigma_2 = (1/(length-2))*C
        sigma = np.sqrt(sigma_2)
        
        # Now find sigma_m
        sigma_m_2 = length*sigma_2 / ( length*a - (sum(x))**2 )
        sigma_m = np.sqrt(sigma_m_2)
        
        # Now find sigma_c
        sigma_c_2 = sigma_2*a / ( length*a - (sum(x))**2 )
        sigma_c = np.sqrt(sigma_c_2)
        
        print("Slope m [km s^-1 / Mpc]:", m)
        print("Slope m [km s^-1 / Mpc] variance, σ^2_m:", sigma_m_2 )
        print("Slope m [km s^-1 / Mpc] standard deviation, σ_m:", sigma_m)
        print("Y-intercept m [km s^-1]:", c)
        print("Y-intercept [km s^-1] variance, σ^2_c:", sigma_c_2 )
        print("Y-intercept [km s^-1] standard deviation, σ_c:", sigma_c)
        
    return np.array([m, c, sigma_m, sigma_c])     

if __name__ == "__main__":
    data_points = np.array([13, 17, 18, 14, 11, 8, 21, 18, 9, 12, 9, 17, 14, 
                            6, 10, 16, 16, 11, 10, 12, 8, 20, 14, 10, 14, 17, 
                            13, 16, 12, 10])
    n_points = len(data_points)
    bins = np.linspace(4, 25, 22)
    values, counts = get_histogram(data_points)
    
    # Calculate Poisson probablilities with an average of 12
    x = np.linspace(0, 29, 30)
    # Normalize the probabilities for the results we have produced
    y = poisson(x, 12)
    y = y/np.sum(y)
    # Then, multiply by no. of data points)
    y = y*30
    
    # y = scipy.stats.poisson.pmf(x, mu=12)
    # y = y/np.sum(y)
    # # Then, multiply by no. of data points)
    # y = z*30    

    plt.hist(values, bins=bins, weights=counts, facecolor="white", edgecolor="black")
    plt.plot(x, y, linestyle="--", color="black", linewidth=1)
    plt.xlabel("Number of Measurements")
    plt.ylabel("Photon Rates")
    plt.xticks(np.linspace(5,25,5))
    plt.yticks(np.linspace(0,4,5))
    plt.xlim(4,24)
    plt.ylim(0,4.5)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(1)) 
    ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
    ax.tick_params(direction="in", which="both", top=True, 
                            bottom=True, left=True, right=True)  

    plt.savefig("AST325_Lab1_Fig1.png")
    plt.show()    
    
    # Part 3 of the lab requires calculating the mean and standard deviation of the 
    # given dataset. There was no file with my name, so I used the file “Noname-noname*.”
    small_file = "Noname-noname-Small.txt"
    large_file = "Noname-noname-Large.txt"

    small_data = np.loadtxt(small_file)
    large_data = np.loadtxt(large_file)

    small_mean = np.mean(small_data)
    small_stddev = np.std(small_data)
    large_mean = np.mean(large_data)
    large_stddev = np.std(large_data)
    
    print("Small Dataset Mean:", small_mean)
    print("Small Dataset Standard Deviation:", small_stddev)
    print("Large Dataset Mean:", large_mean)
    print("Large Dataset Standard Deviation:", large_stddev)

    # Part 4 of the lab requires plotting the datasets, first in sequence and 
    # then as a histogram.
    
    # Make plots of measurement sequences
    # Small dataset
    plt.figure(figsize=(12, 3))
    plt.plot(small_data, linestyle='None', color='red', marker='o')
    plt.title("Distribution of the measured distances for Small dataset")
    plt.xlabel("Measurements")
    plt.ylabel("Distance (pc)")
    plt.grid(visible=True)
    plt.savefig("AST325_Lab1_Small_Sequence.png")
    plt.show()
    
    # Large Dataset
    plt.figure(figsize=(12, 3))
    plt.plot(large_data, linestyle='None', color='blue', marker='o')
    plt.title("Distribution of Measured Distances for Large Dataset")
    plt.xlabel("Measurements")
    plt.ylabel("Distance (pc)")  
    plt.grid(visible=True)
    plt.savefig("AST325_Lab1_Large_Sequence.png")
    plt.show()
    
    # Make plots for histograms of each measurement set
    s_min, s_max = int(min(small_data)), int(max(small_data))
    l_min, l_max = int(min(large_data)), int(max(large_data))
    bins_small = np.linspace(s_min,s_max,s_max-s_min+1)
    bins_large = np.linspace(l_min,l_max,l_max-l_min+1)
    small_values, small_counts = get_histogram(small_data)
    large_values, large_counts = get_histogram(large_data)
    
    # Small dataset
    plt.hist(small_values, bins=bins_small, weights=small_counts, facecolor="white", edgecolor="red")
    plt.title('Histogram of Measured Distances for Small Dataset')
    plt.xlabel("Value")
    plt.ylabel("Number of Measurements")
    plt.xlim(-1,9)
    plt.ylim(0,280)
    ax = plt.gca()
    ax.yaxis.set_minor_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.tick_params(direction="in", which="both", top=True, 
                            bottom=True, left=True, right=True)
    plt.savefig("AST325_Lab1_Small_Histogram.png")
    plt.show()    
    
    # Large dataset
    plt.hist(large_values, bins=bins_large, weights=large_counts, facecolor="white", edgecolor="blue")
    plt.title('Histogram of Measured Distances for Large Dataset')
    plt.xlabel("Value")
    plt.ylabel("Number of Measurements")
    plt.xlim(950,1200,250)
    plt.ylim(0,19)
    plt.xticks(rotation=45, ha='right')
    ax = plt.gca()
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(direction="out", which="both", top=True, 
                            bottom=True, left=True, right=True)
    plt.savefig("AST325_Lab1_Large_Histogram.png")
    plt.show()
    
    
    # Part 5 of the lab requires us to overplot the Gaussian distribution expected
    # from the mean and standard deviation found in Part 3 on the histograms that 
    # created in Step 4.
    n_small, n_large = 1000, 1000    
    x_small, x_large = np.linspace(-1,999,n_small), np.linspace(950,1950,n_large)

    y_large = gauss(x_large, large_mean, large_stddev)
    y_small = gauss(x_small, small_mean, small_stddev)

    # With SciPy functions
    y_small = scipy.stats.norm(loc=float(small_mean), scale=float(small_stddev)).pdf(x_small)
    y_large = scipy.stats.norm(loc=float(large_mean), scale=float(large_stddev)).pdf(x_large)

    # Normalize the probabilities for the results we have produced
    y_small = y_small/np.sum(y_small)
    y_large = y_large/np.sum(y_large)
    # Then, multiply by no. of data points)
    y_small = y_small*n_small
    y_large = y_large*n_large

    # Small dataset
    plt.hist(small_values, bins=bins_small, weights=small_counts, facecolor="white", edgecolor="red")
    plt.plot(x_small, y_small, linestyle="--", color="green", linewidth=2)
    plt.title('Histogram of Measured Distances for Small Dataset')
    plt.xlabel("Value")
    plt.ylabel("Number of Measurements")
    plt.xlim(-1,9)
    plt.ylim(0,280)
    ax = plt.gca()
    ax.yaxis.set_minor_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.tick_params(direction="in", which="both", top=True, 
                            bottom=True, left=True, right=True)
    plt.savefig("AST325_Lab1_Small_Histogram_Gauss.png")
    plt.show()    
    
    # Large dataset
    plt.hist(large_values, bins=bins_large, weights=large_counts, facecolor="white", edgecolor="blue")
    plt.plot(x_large, y_large, linestyle="--", color="crimson", linewidth=3)
    plt.title('Histogram of Measured Distances for Large Dataset')
    plt.xlabel("Value")
    plt.ylabel("Number of Measurements")
    plt.xlim(950,1200,250)
    plt.ylim(0,19)
    plt.xticks(rotation=45, ha='right')
    ax = plt.gca()
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(direction="out", which="both", top=True, 
                            bottom=True, left=True, right=True)
    plt.savefig("AST325_Lab1_Large_Histogram_Gauss.png")
    plt.show()

    # Part 6 of the lab requires calculating the Hubble constant using the straight-line
    # linear least squares fitting.
    hubble_file = "Noname-noname-Hubble.txt"
    hubble_data = np.loadtxt(hubble_file) 
    
    distances = hubble_data[:,0]
    velocities = hubble_data[:,1]
    mc = least_squares_fit(distances, velocities)
    m, c, sigma_m, sigma_c = mc[0], mc[1], mc[2], mc[3]
    
    plt.plot(distances, velocities,
             linestyle="-", marker="o", markeredgecolor="black", color="crimson",
             linewidth=2, markersize=4, label="Data")
    slope_text = "m = " + "{:.3f}".format(m) 
    const_text = "c = " + "{:.3f}".format(c) 
    plt.text(100, 1.10e5, slope_text, fontsize="large")
    plt.text(100, 0.95e5, const_text, fontsize="large")
    plt.plot(distances, distances*m+c, linestyle="--", color="green", linewidth=2, label="OLS fit")
    plt.title('Ordinary Least Squares fit for the Hubble Constant')
    plt.xlabel("Distance [Mpc]")
    plt.ylabel("Velocity [$km s^{-1}$]")
    ax = plt.gca()
    ax.yaxis.set_minor_locator(MultipleLocator(1e4))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.tick_params(direction="in", which="both", top=True, 
                            bottom=True, left=True, right=True)
    plt.grid(color="lightgrey")
    plt.savefig("AST325_Lab1_Hubble.png")
    plt.show()

