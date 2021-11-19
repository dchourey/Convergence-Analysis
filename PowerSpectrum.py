#__________________________________To calculate the power spectrum from the convergencece map, C_EE(ell), C_EB(ell), C_BB(ell)______________________________________________


#Importing the required packages
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.stats as stats


#________________________________________________________________User Inputs____________________________________________________________

#Add the power spectrum to calculate by putting true/false
EE = True
BB = False
EB = False

#Importing the E and B mode convergence map
hdul_1 = fits.open('Kappa_E_4194000.fits')
hdul_2 = fits.open('Kappa_B_4194000.fits')
map_e = hdul_1[0].data
map_b = hdul_2[0].data
mapsize_deg = float(10.) #put map size in degree units

#Add the extent of multipoles for which power spectrum will be calculated

ell_min = float(200) 
ell_max = float(20000)
ell_steps = int(200) 

#_________________________________________________________________________________________________________________________________________

#converting map as numpy array
map_e = np.asarray(map_e)
map_b = np.asarray(map_b)

#calculating the multipole arrays 
ell_ext = np.arange(ell_min, ell_max, ell_steps) #multipole array given by user, for best resolution choose ell_step = ell_pix
ell = 0.5*(ell_ext[:-1] + ell_ext[1:])  #output averaged multipole values

#Calculating the convergence maps into fourier space

map_e_k = np.fft.fft2(map_e) #E-mode map in fourier space
map_b_k = np.fft.fft2(map_b) #B-mode map in fourier space

if map_e_k.data.shape != map_b_k.data.shape:
	raise "E and B mode map dimensions are not same"  #checking the input array dimension equality

#Calculating the map prpoperties


size_x = map_e_k.data.shape[0] #pixel numbers along x-axis
size_y = map_e_k.data.shape[1] ##pixel numbers along y-axis
ell_values = int(len(ell_ext)) #number of multipoles requested
ell_bins = ell_values - 1 #number of multipole bins
ell_pix = 360/mapsize_deg #factor converting fft frequency to multipole 

#calculating the unitless fast fourier transform frequencies

freq_x = np.fft.fftfreq(map_e.data.shape[0])
freq_y = np.fft.fftfreq(map_e.data.shape[1])


#------------------------------------------------------------------
#calculating the EE mode power spectrum
#------------------------------------------------------------------

if EE == True:
	C_ee = np.empty(ell_bins) #Defining the empty array for power spectrum	
	i = j = k = 0 #running counters
	fall = np.empty(ell_bins)  #averaging array, it hold the number of l values falling between ell_ext[k] and ell_ext[k+1]
	#calculating the multipole for corrosponding pixels of fourier image
	
	while i < size_x:		
		lx =  freq_x[i] * size_x    #pixel frequency  corrosponding to the i'th pixel
		j = 0 # reseting the y counter
		while j < size_y:
			ly = freq_y[j] * size_y   #pixel frequency corrosponding to the j'th pixel			
			l = np.sqrt(lx**2 + ly**2) * ell_pix  #converting pixel frequency to spherical harmonics
			k = 0 # reseting the k counter			
			while k < ell_bins:
				if( l > ell_ext[k] and l <= ell_ext[k+1]):
					C_ee[k] += (map_e_k[i,j].real * map_e_k[i,j].real ) + (map_e_k[i,j].imag * map_e_k[i,j].imag) #averaging all power values in each ell bin
					fall[k] = fall[k] + 1
				k = k + 1
			j = j + 1
		i = i + 1
		#print(i)
	#fourier normalization factor
	
	norm = ((mapsize_deg * np.pi / 180) / ( size_x * size_y ))**2
	
	#calculating the power spectrum EE mode
	
	C_ee = norm * (C_ee / fall) 


#-------------------------------------------------------------------
#calculating the BB mode power spectrum
#-------------------------------------------------------------------

if BB == True:
	C_bb = np.empty(ell_bins)  #Defining the empty array for power spectrum	
	i = j = k = 0 #running counters
	fall = np.empty(ell_bins)  #averaging array, it hold the number of l values falling between ell_ext[k] and ell_ext[k+1]
	#calculating the multipole for corrosponding pixels of fourier image
	
	while i < size_x:
		lx = freq_x[i] * size_x  
		j = 0 # reseting the y counter
		while j < size_y:
			ly = freq_y[j] * size_y
			l = np.sqrt(lx**2 + ly**2) * ell_pix
			k = 0 
			while k < ell_bins:
				if( l > ell_ext[k] and l <=ell_ext[k+1]):
					C_bb[k] = C_bb[k] + (map_b_k[i,j].real * map_b_k[i,j].real ) + (map_b_k[i,j].imag * map_b_k[i,j].imag) #averaging all power values in each ell bin
					fall[k] = fall[k] + 1
				k = k + 1
			j = j + 1
		i = i + 1
		
	#fourier normalization factor
	
	norm = ( ( mapsize_deg * np.pi / 180) / ( size_x * size_y ) ) **2
	
	#calculating the power spectrum BB mode
	
	C_bb = norm * C_bb / fall

#--------------------------------------------------------------------
#calculating the EB mode power spectrum     
#---------------------------------------------------------------------

if EB == True:
	C_eb = np.empty(ell_bins)  #Defining the empty array for power spectrum	
	i = j = k = 0 #running counters
	fall = np.empty(ell_bins)  #averaging array, it hold the number of l values falling between ell_ext[k] and ell_ext[k+1]
	#calculating the multipole for corrosponding pixels of fourier image
	
	while i < size_x:
		lx =freq_x[i] * size_x  
		j = 0 # reseting the y counter
		while j < size_y:
			ly = freq_y[j] * size_y  
			l = np.sqrt(lx**2 + ly**2)* ell_pix
			k = 0 
			while k < ell_bins:
				if( l > ell_ext[k] and l <=ell_ext[k+1]):
					C_eb[k] = C_eb[k] + (map_e_k[i,j].real * map_b_k[i,j].real ) + (map_e_k[i,j].imag * map_b_k[i,j].imag) #averaging all power values in each ell bin
					fall[k] = fall[k] + 1
				k = k + 1
			j = j + 1
		i = i + 1
		
	#fourier normalization factor
	
	norm = ( ( mapsize_deg * np.pi / 180) / ( size_x * size_y ) ) **2
	
	#calculating the power spectrum EB mode
	
	C_eb = norm * C_eb / fall


#------------------------------------------------
#Plotting and saving the power spectrums
#------------------------------------------------

if EE == True:
	np.savetxt('C_ell_EE.dat', np.c_[ell, C_ee])
	plt.plot(ell, ell*(ell +1)* C_ee/(2.0*np.pi), marker='o', label=r"$C_{\ell}^{EE}$")
	plt.xlabel(r"$\ell$")
	plt.ylabel(r"$l(l+1)C_\ell/2\pi$")
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.savefig('C_ell_EE')
	
if BB == True:
	np.savetxt('C_ell_EE.dat', np.c_[ell, C_bb])
	plt.figure()
	plt.plot(ell, ell*(ell +1)* C_bb/(2.0*np.pi), marker='d', label=r"$C_{\ell}^{BB}$")
	plt.xlabel(r"$\ell$")
	plt.ylabel(r"$l(l+1)C_\ell/2\pi$")
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.savefig('C_ell_BB')
	
if EB == True:
	np.savetxt('C_ell_EE.dat', np.c_[ell, C_eb])
	plt.figure()
	plt.plot(ell, ell*(ell +1)* C_eb/(2.0*np.pi), marker='X', label=r"$C_{\ell}^{EB}$")
	plt.xlabel(r"$\ell$")
	plt.ylabel(r"$l(l+1)C_\ell/2\pi$")
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.savefig('C_ell_EB')
	
plt.show()
