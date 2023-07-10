"""
Verify that monthly max is equal to average of daily max per month

Author    : Zachary M. Labe
Date      : 4 November 2022
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import cmocean
import cmasher as cmr
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import sys
import scipy.stats as sts
import read_SPEAR_MED as SP

### Parameters
vari = 'TMIN'
variSPEAR = 't_ref_min'
slicenan = 'nan'
sliceperiod = 'none'
sliceshape = 5
numOfEns = 30
timeper = 'all'

### Years for each simulation
years_med = np.arange(1921,2100+1,1)
lat,lon,var = SP.read_SPEAR_MED('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/monthly/',vari,sliceperiod,sliceshape,slicenan,numOfEns,timeper)

### Import daily files
if vari == 'TMAX':
    data1 = Dataset('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/daily/%s/%s_01_1921-1930.nc' % (vari,vari))
    pre = data1.variables[variSPEAR][:] - 273.15
    timepre = data1.variables['time'][:]
    data1.close()
    
    comp = var[0,0,0,:,:]

if vari == 'TMIN':
    data2 = Dataset('/work/Zachary.Labe/Data/SPEAR/SPEAR_MED/daily/%s/%s_01_2091-2100.nc' % (vari,vari))
    post = data2.variables[variSPEAR][:] - 273.15
    data2.close()
    
    comp = var[0,0,0,:,:]
            
