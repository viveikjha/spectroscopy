#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  11 00:15:25 2020

@author: ganesh
"""
import astropy.io.fits as fits
import numpy as np
from scipy import stats
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
import quad
import matplotlib.pyplot as plt

# 3) Set variables
datadir = '/home/ganesh/Desktop/spec_test/'

HD32991_im = 'hd32991_100_gr7_58.0.fits'
He_Ne_im= 'he_ne_sc7_4.0.fits'
#NeAr_im  = 'ao-018.fit'
flat_ims = ['flat_sl100_ifosc7_3.0.fits','flat_sl100_ifosc7_4.0.fits',
	    'flat_sl100_ifosc7_4.1.fits','flat_sl100_ifosc7_5.1.fits',
	    'flat_sl100_ifosc7_5.0.fits','flat_sl100_sc7_11.0.fits']
'''flat_sl100_sc7_7.1.fits','flat_sl100_sc7_11.1.fits',
	    'flat_sl100_sc7_8.0.fits','flat_sl100_sc7_12.0.fits',
	    'flat_sl100_sc7_8.1.fits','flat_sl100_sc7_12.1.fits',
	    'flat_sl100_sc7_9.0.fits','flat_sl100_sc7_6.0.fits',
	    'flat_sl100_sc7_9.1.fits','flat_sl100_sc7_10.0.fits',
	    'flat_sl100_sc7_6.1.fits','flat_sl100_sc7_10.1.fits',
	    'flat_sl100_sc7_7.0.fits'''
  
bias_ims = ['bias_1.0.fits','bias_2.0.fits','bias_2.1.fits']

# 4) Read in the data
hdulist = fits.open(datadir+HD32991_im)
HD32991_d = hdulist[0].data
header  = hdulist[0].header
hdulist.close()

hdulist = fits.open(datadir+He_Ne_im)
He_Ne_d  = hdulist[0].data
hdulist.close()

numims  = len(flat_ims)
hdulist = fits.open(datadir+flat_ims[0])
data    = hdulist[0].data
hdulist.close()
xsz     = data.shape[0]
ysz     = data.shape[1]
flatarr = np.zeros((xsz,ysz,numims)).copy()
for i in range(numims):
    hdulist        = fits.open(datadir+flat_ims[i])
    data           = hdulist[0].data
    flatarr[:,:,i] = data
    hdulist.close()

numims  = len(bias_ims)
biasarr = np.zeros((xsz,ysz,numims)).copy()
for i in range(numims):
    hdulist = fits.open(datadir+bias_ims[i])
    data    = hdulist[0].data
    biasarr[:,:,i] = data
    hdulist.close()

#numims  = len(dark_ims)
#darkarr = np.zeros((xsz,ysz,numims)).copy()
#for i in range(numims):
#    hdulist        = fits.open(datadir+dark_ims[i])
#    data           = hdulist[0].data
#    darkarr[:,:,i] = data
#    hdulist.close()

# 5) Data reduction

# 5.a) Compute the bias level
biasval = np.median(biasarr)
#biasval = np.median(biasarr[0:399,60:2108])
# 5.b) Create a median-value dark image
#md_dark_im = np.median(darkarr,axis=2)

# 5.c) Create a flatfield image
md_flat_im = np.median(flatarr,axis=2)
#md_flat_im = np.median(flatarr[0:399,60:2108],axis=2)

flat_norm  = (md_flat_im-biasval)/stats.mode((md_flat_im-biasval)[0:399,60:2108],axis=None)[0][0]

# 5.d) Reduce the science image: (Raw image - Dark Image)/(Flatfield - bias)
HD32991_final = (HD32991_d) / flat_norm
#HD32991_final = (HD32991_d[0:399,60:2108]) / flat_norm

# 5.e) Modify the header of the image: data reduction complete
header.append('REDUCED')
header['REDUCED'] = '(Image) / normalized flatfield'

# 6) Spectral extraction

# 6.a) Find the spectral trace
trace = np.median(HD32991_final,axis=1)

plt.figure()
plt.plot(trace)
plt.xlabel('Pixel Number (Dispersion Direction)')
plt.ylabel('Intensity (ADU)')
plt.title('Dispersion Profile')
plt.show(block=False)

# Find the peak in the dispersion profile:
idx = np.where(trace == np.max(trace))
mx  = idx[0][0]

sigma = 3.0 #pixels
plt.figure()
plt.plot(trace)
plt.plot([mx,mx],[0,max(trace)])
plt.plot([mx+2*sigma,mx+2*sigma,mx-2*sigma,mx-2*sigma],[0,max(trace),max(trace),0])
plt.show(block=False)

loval = int(mx-2*sigma)
hival = int(mx+2*sigma)

# 6.b) Spectral extraction
sp = np.zeros(ysz).copy()
# Extract the spectrum, element by element. Final spectrum will be in ADU
for i in range(ysz):
    sp[i] = np.nansum(HD32991_final[loval:hival,i])

# 6.c) Modify the header: extraction parameters
header.append('EXTRACT')
header['EXTRACT'] = 'Column extraction, centered at XCENTER, width of 4SIGMA'
header.append('XCENTER')
header['XCENTER'] = mx
header.append('SIGMA')
header['SIGMA'] = sigma

plt.figure()
plt.plot(sp)
plt.xlabel('Pixel Number')
plt.ylabel('Intensity (ADU)')
plt.title('HD 32991, raw spectrum')
#plt.xlim(2000,0)
plt.show(block=False)

# 7) Wavelength calibration
lamp = np.zeros(ysz).copy()
for i in range(ysz):
    lamp[i] = np.nansum(He_Ne_d[loval:hival,i])

plt.figure()
plt.plot(lamp)
plt.show(block=False)

plt.figure()
plt.plot(lamp)
plt.plot(sp)
plt.show(block=False)

# X-values of the Balmer features
pixvals = [456,637,858,1448,1464,1512,1533,1897,1915,1959,2013,2039]
#1464,1826,1897,2013,1699,1826,1571
# Wavelength values for lines in the lamp spectrum
cal_wvs = [4471.48,4713.14,5015.67,5852.49,5875.62,5944.03,6030.00,6506.53,6532.88,6598.95,6678.20,6717.04]
#5944.03,6334.43,6402.25,6598.95,6217.28,6402.25,6030.00
# Fit a function of wavelengths to the pixel values
params = curve_fit(quad.quad,pixvals,cal_wvs,p0=[0.0,0.0,0.0])
# Make an array of wavelength values, corresponding to pixel values
pixarr = np.arange(ysz)
wvarr  = params[0][0]*pixarr**2 + params[0][1]*pixarr + params[0][2]

# Plot the raw spectrum, with wavelengths
plt.figure()
plt.plot(wvarr,sp,linewidth=2)
plt.xlabel(r'$\lambda$ ($\AA$)',fontsize=)
plt.ylabel('Intensity (ADU)',fontsize=11)
plt.title('Spectrum of HD 32991',fontsize=11)
plt.show(block=False)

# Plot the raw spectrum, with normalised intensity
'''sp_norm = sp / np.sqrt(np.sum(sp**2))
plt.figure()
plt.plot(wvarr,sp_norm,linewidth=2)
plt.xlabel(r'$\lambda$ ($\AA$)',fontsize=11)
plt.ylabel('Normalised Intensity (ADU)',fontsize=11)
plt.title('Normalised Spectrum of HD 32991',fontsize=11)
plt.show(block=False)
'''
# 8) Save our spectrum to a FITS file
'''hdu = fits.PrimaryHDU(np.vstack((wvarr,sp)))
hdulist = fits.HDUList([hdu])
hdulist[0].header = header
hdulist.writeto('HD32991_ext.fits')'''
np.savetxt('/home/ganesh/Desktop/spect1.txt',np.column_stack((wvarr,sp)), newline='\n',delimiter=',')
