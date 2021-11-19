#_______________________________________________________To construct mass maps or convergence maps from weak lensing shear values___________________________________________________


#Theory

#There would be E and B mode convergence map. The real part of the convergence values would give and E-mode map and imaganary part would give the B-mode map. Ideally the B-mode map
#should give zero result. Refer: 1705.06792, 1504.03002, Pires et. al. 2020

#Constructing mass maps using the shear data is done using following steps;
#step 1: Put the shear values on a grid g1_2d and g2_2d
#step 2: do the fast fourier transform of the shear grids g1_k and g2_k
#step 3: calculate multipole lx and ly 
#step 4: calculate the rotation angles cos(phi) and sin(phi)
#step 5: calculate and E and B mode convergence in fourier space: Kk_E, Kk_B
#step 6: do the inverse fourier transform to get real space K_E and K_B
#step 7: Plot the real part of the K_E and K_B

#Importing required packages
import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.io import fits 

#Initial data is in the h5py formate. To read this type of dataset put the h5 file in the same directory as the script;

hf = h5py.File('nzEUCLID_00_123456.h5', 'r') #Importing the file

xi = np.array(hf["xi_coordinate"][:]) # x cordinates
yi = np.array(hf["yi_coordinate"][:]) #y cordinates
nz = np.array(hf["redshift"][:]) #Redshift distribution between (0,3)
kappa = np.array(hf["convergence"][:]) #convergence values
g_1 = np.array(hf["gamma_1"][:]) #gamma tangential
g_2 = np.array(hf["gamma_2"][:]) #gamma cross component

#Input values 

npix = int(3072) #Number of pixels each side of the map
map_size = float(10.) #map size in units of degrees

#setting the map limits
xmin = xi.min()
xmax = xi.max()
ymin = yi.min()
ymax = yi.max()

h = map_size/npix #resolution of the map

#Initializing the grid

g1_2d = np.zeros((npix+1, npix+1), dtype=np.float64) #Empty gamma1 grid of size (npix, npix), must increase it to 3072 for full data set. 
g2_2d = np.zeros((npix+1, npix+1), dtype=np.float64) #Empty gamma2 grid of size (npix, npix), must increase it to 3072 for full data set.

#putting shear values on a grid

p = 0 #Initialising the counters

while p < len(xi): #considering only 16000 points of the data because of memory limits. Should have been len(xi) to conver full data set.
	i = round( (xi[p] - xi.min() ) / h) #calculating the index number of co-ordinates on the grid along x-axis
	j = round( ( yi[p] - yi.min() ) / h) #calculating the index number of co-ordinates on the grid along y-axis
	print(p)
	g1_2d[i, j] = g_1[p] #putting the shear values on grid position (i, j)
	g2_2d[i,j] =  g_2[p]
	p = p + 1 #updating the counter


#Traansforming the shear grid into fourier space
g1_k = np.fft.fft2(g1_2d)
g2_k = np.fft.fft2(g2_2d)
#print(gk)

#calculating the multipoles for the map

lx = np.fft.fftfreq(g1_2d.data.shape[0])[None] #multipole corresponding to the x-direction as the row vector
ly = np.fft.fftfreq(g1_2d.data.shape[0])[:, None] #multipole corresponding to the y-direction as the column vector
lsq = lx**2 + ly**2 #calculating the multipole squared values
lsq[0,0] = 1. #assigning the [0,0]  component to 1.0 to avoid the mass-sheet degeneracy (Bartelmann 1995)  

#Estimating the convergence using the Kaiser-Squires inversion method

sin_2phi = (2.0 * lx * ly ) / lsq
cos_2phi = (lx**2 - ly**2 ) / lsq #rotation angles 

#calculating the E and B mode convergence in fourier space

Kk_E = (cos_2phi * g1_k) + (sin_2phi * g2_k) #Fourier space E-mode
Kk_B = (-1.0 * sin_2phi * g1_k) + (cos_2phi * g2_k) #Fourier space B-mode

#estimating the real space E and B mode values

K_E = np.fft.ifft2(Kk_E) #Real space E-mode
K_B = np.fft.ifft2(Kk_B) #Real space B-mode

#print(K_E.shape, K_B.shape)
#saving E an B mode convergence as a fits table
hdu  = fits.PrimaryHDU(K_E.real)
hdu.writeto('Kappa_E.fits')
hdu  = fits.PrimaryHDU(K_B.real)
hdu.writeto('Kappa_B.fits')


#Plotting the E and B mode convergence map

plt.imshow(K_E.real, origin="lower", interpolation="gaussian", extent=[xmin, xmax, ymin, ymax], cmap="jet")
plt.colorbar().set_label=(r"$\kappa_E$")
plt.title("E-mode Convergence")
plt.xlabel("x (deg)")
plt.ylabel("y (deg)")
plt.figure()
plt.imshow(K_B.real, origin="lower", interpolation="gaussian", extent=[xmin, xmax, ymin, ymax], cmap="jet")
plt.colorbar().set_label=(r"$\kappa_B$")
plt.title("B-mode Convergence")
plt.xlabel("x (deg)")
plt.ylabel("y (deg)")
plt.show()



















