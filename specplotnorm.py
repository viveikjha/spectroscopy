#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:00:25 2021

@author: ganesh
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models
from astropy import units as u
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from astropy.table import QTable

d=pd.read_csv('/home/ganesh/Desktop/spect1.txt',header=None)
s=d[120:2098]
np.savetxt('/home/ganesh/Desktop/spect2.txt',s, newline='\n',delimiter=',')
data=pd.read_csv('/home/ganesh/Desktop/spect2.txt',header=None)
l=data[0:964]
r=data[965:1977]
#lm,rm=np.median(l[1]) , np.median(r[1])
#f=l[1][962]/r[1][966]
#f=rm/lm
#Calibrate CCD
t=r[1]*1.317
w=l[0].append(r[0])
sf=l[1].append(t)
plt.figure()
plt.plot(w,sf,'m', linewidth=2)

plt.xlabel(r'$\lambda$ ($\AA$)',fontsize=11)
plt.ylabel('ADU',fontsize=11)
plt.title('Spectrum of HD 32991',fontsize=11)
plt.show(block=False)

spectra=np.column_stack((w,sf))
np.savetxt('/home/ganesh/Desktop/spectpf.txt',spectra, newline='\n',delimiter=',')

#fitting spectrum
data2=pd.read_csv('/home/ganesh/Desktop/spectpf.txt',header=None)

y=u.Quantity(data2[1],u.dimensionless_unscaled)
x=u.Quantity(data2[0],u.angstrom)

spectrum = Spectrum1D(spectral_axis=x,flux=y)
g1_fit = fit_generic_continuum(spectrum)
y_continuum_fitted = g1_fit(x)

#continuum fitted spectrum
plt.figure()
plt.plot(x, y,'k')
plt.plot(x, y_continuum_fitted, 'orange')
plt.xlabel(r'$\lambda$ ($\AA$)',fontsize=11)
plt.show(block=False)

#Plot continuum normalised intensity
spec_normalized = spectrum / y_continuum_fitted

t=QTable([spec_normalized.spectral_axis,spec_normalized.flux])
t.write('/home/ganesh/Desktop/spectf.txt',format='ascii', delimiter=',')

plt.figure()
plt.plot(spec_normalized.spectral_axis, spec_normalized.flux, 'k-')
plt.title('Continuum normalized spectrum HD 32991',fontsize=11)
plt.xlabel(r'$\lambda$ ($\AA$)',fontsize=11)
plt.ylabel('Normalized flux',fontsize=11)
plt.show(block=False)

