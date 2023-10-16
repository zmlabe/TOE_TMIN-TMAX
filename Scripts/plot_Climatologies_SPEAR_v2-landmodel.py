#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:05:35 2023

@author: Zachary.Labe
"""

"""
Calculate climatologies for different spear runs and LM42p2

Author    : Zachary M. Labe
Date      : 17 February 2023
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

### Read in data files from server
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/Bias/AllModels/' 

### Parameters
monthq = ['JAN','FEB','MAR','ARP','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u"]
variq = 'T2M'
slicenan = 'nan'
reg_name = 'Globe'
lat_bounds,lon_bounds = UT.regions(reg_name)
monthlychoiceq = ['JFM','AMJ','JAS','OND']
slicemonthnamen = np.repeat(['JAN-MAR','APR-JUN','JUL-SEP','OCT-DEC'],4)
modelsnamen = ['SPEAR_MED','SPEAR_MED_LM42p2','SPEAR_MED_FA','SPEAR_HI']
scenario = 'SSP585'
dataset_obs = 'ERA5_MEDS'

### Masking and preprocessing arguments
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
land_only = False
ocean_only = False
CONUS_only = False

###############################################################################
###############################################################################
###############################################################################
### Read in climate models      
def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons  

### Get data
def readModel(variq,monthlychoice,scenario,lat_bounds,lon_bounds):

    lat_bounds,lon_bounds = UT.regions(reg_name)
    data_spear,lats,lons = read_primary_dataset(variq,'SPEAR_MED',monthlychoice,scenario,lat_bounds,lon_bounds)
    data_spearlm,lats,lons = read_primary_dataset(variq,'SPEAR_MED_LM42p2',monthlychoice,scenario,lat_bounds,lon_bounds)
    data_spearFA,lats,lons = read_primary_dataset(variq,'SPEAR_MED_FA',monthlychoice,scenario,lat_bounds,lon_bounds)
    data_obs_all,lats_obs,lons_obs = read_primary_dataset(variq,dataset_obs,monthlychoice,scenario,lat_bounds,lon_bounds)
    
    ### Select years
    yearsobs = np.arange(1979,2021+1,1)
    yearsq = np.where((yearsobs >= 1979) & (yearsobs <= 2020))[0]
    obs = data_obs_all[yearsq,:,:]
        
    ### Model years
    yearsspear = np.arange(1921,2100+1,1)
    yearspearq = np.where((yearsspear >= 1979) & (yearsspear <= 2020))[0]
    yearslmr = np.arange(1921,2070+1,1)
    yearlmq = np.where((yearslmr >= 1979) & (yearslmr <= 2020))[0]
    
    data_spearn = np.nanmean(data_spear[:,yearspearq,:,:] - obs,axis=(0,1))
    data_spearlmn = np.nanmean(data_spearlm[:,yearlmq,:,:] - obs,axis=(0,1))
    data_FAn = np.nanmean(data_spearFA[:,yearspearq,:,:] - obs,axis=(0,1))
    
    ### Read in high
    obshigh,lats_obs,lons_obs = read_primary_dataset(variq,'ERA5_HIGHS',monthlychoice,scenario,lat_bounds,lon_bounds)
    yearsobshigh = np.arange(1979,2021+1,1)
    yearshighq = np.where((yearsobshigh >= 1979) & (yearsobshigh <= 2020))[0]
    obshighn = obshigh[yearshighq,:,:]
    
    data_spearhigh,latshigh,lonshigh = read_primary_dataset(variq,'SPEAR_HIGH',monthlychoice,scenario,lat_bounds,lon_bounds)
    yearsspearhigh = np.arange(1921,2100+1,1)
    yearspearqhigh = np.where((yearsspearhigh >= 1979) & (yearsspearhigh <= 2020))[0]
    
    data_highn = np.asarray(np.nanmean(data_spearhigh[:,yearspearqhigh,:,:] - obshighn,axis=(0,1)))
    
    ### Calculate RMSE
    spear_rmse = UT.calc_RMSE(np.nanmean(obs,axis=0),np.nanmean(data_spear[:,yearspearq,:,:],axis=(0,1)),lats,lons,'yesnan')
    spearlm_rmse = UT.calc_RMSE(np.nanmean(obs,axis=0),np.nanmean(data_spearlm[:,yearlmq,:,:],axis=(0,1)),lats,lons,'yesnan')
    spearFA_rmse = UT.calc_RMSE(np.nanmean(obs,axis=0),np.nanmean(data_spearFA[:,yearspearq,:,:],axis=(0,1)),lats,lons,'yesnan')
    spearhigh_rmse = UT.calc_RMSE(np.nanmean(obshighn,axis=0),np.nanmean(data_spearhigh[:,yearspearqhigh,:,:],axis=(0,1)),latshigh,lonshigh,'yesnan')
    
    ### Calculate Pattern Correlation
    spear_corr = UT.calc_spatialCorr(np.nanmean(obs,axis=0),np.nanmean(data_spear[:,yearspearq,:,:],axis=(0,1)),lats,lons,'yesnan')
    spearlm_corr = UT.calc_spatialCorr(np.nanmean(obs,axis=0),np.nanmean(data_spearlm[:,yearlmq,:,:],axis=(0,1)),lats,lons,'yesnan')
    spearFA_corr = UT.calc_spatialCorr(np.nanmean(obs,axis=0),np.nanmean(data_spearFA[:,yearspearq,:,:],axis=(0,1)),lats,lons,'yesnan')
    spearhigh_corr = UT.calc_spatialCorr(np.nanmean(obshighn,axis=0),np.nanmean(data_spearhigh[:,yearspearqhigh,:,:],axis=(0,1)),latshigh,lonshigh,'yesnan')
    
    return data_spearn,data_spearlmn,data_highn,data_FAn,lats,lons,latshigh,lonshigh,spear_rmse,spearlm_rmse,spearFA_rmse,spearhigh_rmse,spear_corr,spearlm_corr,spearFA_corr,spearhigh_corr

data_spearn_1,data_spearlmn_1,data_highn_1,data_FAn_1,lats,lons,latshigh,lonshigh,spear_rmse_1,spearlm_rmse_1,spearFA_rmse_1,spearhigh_rmse_1,spear_corr_1,spearlm_corr_1,spearFA_corr_1,spearhigh_corr_1 = readModel(variq,monthlychoiceq[0],scenario,lat_bounds,lon_bounds)
data_spearn_2,data_spearlmn_2,data_highn_2,data_FAn_2,lats,lons,latshigh,lonshigh,spear_rmse_2,spearlm_rmse_2,spearFA_rmse_2,spearhigh_rmse_2,spear_corr_2,spearlm_corr_2,spearFA_corr_2,spearhigh_corr_2 = readModel(variq,monthlychoiceq[1],scenario,lat_bounds,lon_bounds)
data_spearn_3,data_spearlmn_3,data_highn_3,data_FAn_3,lats,lons,latshigh,lonshigh,spear_rmse_3,spearlm_rmse_3,spearFA_rmse_3,spearhigh_rmse_3,spear_corr_3,spearlm_corr_3,spearFA_corr_3,spearhigh_corr_3 = readModel(variq,monthlychoiceq[2],scenario,lat_bounds,lon_bounds)
data_spearn_4,data_spearlmn_4,data_highn_4,data_FAn_4,lats,lons,latshigh,lonshigh,spear_rmse_4,spearlm_rmse_4,spearFA_rmse_4,spearhigh_rmse_4,spear_corr_4,spearlm_corr_4,spearFA_corr_4,spearhigh_corr_4 = readModel(variq,monthlychoiceq[3],scenario,lat_bounds,lon_bounds)

dataplot = [data_spearn_1,data_spearlmn_1,data_FAn_1,data_highn_1,
            data_spearn_2,data_spearlmn_2,data_FAn_2,data_highn_2,
            data_spearn_3,data_spearlmn_3,data_FAn_3,data_highn_3,
            data_spearn_4,data_spearlmn_4,data_FAn_4,data_highn_4]
latsall = [lats,lats,lats,latshigh,
            lats,lats,lats,latshigh,
            lats,lats,lats,latshigh,
            lats,lats,lats,latshigh]
lonsall = [lons,lons,lons,lonshigh,
            lons,lons,lons,lonshigh,
            lons,lons,lons,lonshigh,
            lons,lons,lons,lonshigh]
rmses = [spear_rmse_1,spearlm_rmse_1,spearFA_rmse_1,spearhigh_rmse_1,
          spear_rmse_2,spearlm_rmse_2,spearFA_rmse_2,spearhigh_rmse_2,
          spear_rmse_3,spearlm_rmse_3,spearFA_rmse_3,spearhigh_rmse_3,
          spear_rmse_4,spearlm_rmse_4,spearFA_rmse_4,spearhigh_rmse_4]
corrs = [spear_corr_1,spearlm_corr_1,spearFA_corr_1,spearhigh_corr_1,
          spear_corr_2,spearlm_corr_2,spearFA_corr_2,spearhigh_corr_2,
          spear_corr_3,spearlm_corr_3,spearFA_corr_3,spearhigh_corr_3,
          spear_corr_4,spearlm_corr_4,spearFA_corr_4,spearhigh_corr_4]
    

###############################################################################
###############################################################################
###############################################################################
### Plot biases
fig = plt.figure(figsize=(9,6))


if variq == 'T2M':
    label = r'\textbf{Bias -- GCM minus %s [%s; $^{\circ}$C]}' % (dataset_obs,variq)
    limit = np.arange(-5,5.01,0.05)
    barlim = np.round(np.arange(-5,5.1,1),2)
    cmap = cmocean.cm.balance
elif variq == 'PRECT':
    label = r'\textbf{Bias -- GCM minus %s [%s; mm/day]}' % (dataset_obs,variq)
    limit = np.arange(-4,4.01,0.2)
    barlim = np.round(np.arange(-4,4.1,1),2)
    cmap = cmocean.cm.curl_r
    
for i in range(len(dataplot)):
    ax = plt.subplot(4,4,i+1)
    
    var = dataplot[i]
    lat1 = latsall[i]
    lon1 = lonsall[i]
    
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='l',
                area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=1)
    m.drawstates(color='darkgrey',linewidth=0.5)
    m.drawcountries(color='darkgrey',linewidth=0.5)

    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgrey',
                      linewidth=0.7)
    circle.set_clip_on(False)
        
    lon2,lat2 = np.meshgrid(lon1,lat1)
    
    cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
    
    cs1.set_cmap(cmap)
    
    ax.annotate(r'\textbf{[%s]}' % (letters[i]),xy=(0,0),xytext=(0.0,1.04),
              textcoords='axes fraction',color='k',fontsize=7,
              rotation=0,ha='center',va='center')
    
    if any([i==0,i==4,i==8,i==12]):
        ax.annotate(r'\textbf{%s}' % (slicemonthnamen[i]),xy=(0,0),xytext=(-0.1,0.5),
                  textcoords='axes fraction',color='dimgrey',fontsize=11,
                  rotation=90,ha='center',va='center')     
    if any([i==0,i==1,i==2,i==3]):
        ax.annotate(r'\textbf{%s}' % (modelsnamen[i]),xy=(0,0),xytext=(0.5,1.25),
                  textcoords='axes fraction',color='k',fontsize=15,
                  rotation=0,ha='center',va='center')       
    
cbar_ax1 = fig.add_axes([0.305,0.06,0.4,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=8,color='k',labelpad=5)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.13,top=0.93)
        
plt.savefig(directoryfigure + 'Bias_AllGFDLModels-LM42p2_USA_%s-%s_Seasons_v2.png' % (variq,dataset_obs),dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot biases
fig = plt.figure(figsize=(9,6))

for i in range(len(dataplot)):
    ax = plt.subplot(4,4,i+1)
    
    var = dataplot[i]
    lat1 = latsall[i]
    lon1 = lonsall[i]
    
    m = Basemap(projection='robin',lon_0=0,resolution='l',area_thresh=10000)
    m.drawcoastlines(color='k',linewidth=0.3)

    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgrey',
                      linewidth=0.7)
    circle.set_clip_on(False)
        
    lon2,lat2 = np.meshgrid(lon1,lat1)
    
    cs1 = m.contourf(lon2,lat2,var,limit,extend='both',latlon=True)
    
    cs1.set_cmap(cmap)
    
    ax.annotate(r'\textbf{[%s]}' % (letters[i]),xy=(0,0),xytext=(0.0,0.98),
              textcoords='axes fraction',color='k',fontsize=7,
              rotation=0,ha='center',va='center')
    ax.annotate(r'\textbf{RMSE=%s}' % (np.round(rmses[i],3)),xy=(0,0),xytext=(1,1.04),
              textcoords='axes fraction',color='darkred',fontsize=6,
              rotation=0,ha='center',va='center')
    ax.annotate(r'\textbf{R=%s}' % (np.round(corrs[i],3)),xy=(0,0),xytext=(1,0.9),
              textcoords='axes fraction',color='darkred',fontsize=6,
              rotation=0,ha='center',va='center')
    
    if any([i==0,i==4,i==8,i==12]):
        ax.annotate(r'\textbf{%s}' % (slicemonthnamen[i]),xy=(0,0),xytext=(-0.1,0.5),
                  textcoords='axes fraction',color='dimgrey',fontsize=11,
                  rotation=90,ha='center',va='center')     
    if any([i==0,i==1,i==2,i==3]):
        ax.annotate(r'\textbf{%s}' % (modelsnamen[i]),xy=(0,0),xytext=(0.5,1.3),
                  textcoords='axes fraction',color='k',fontsize=15,
                  rotation=0,ha='center',va='center')       
    
cbar_ax1 = fig.add_axes([0.305,0.07,0.4,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=8,color='k',labelpad=5)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=7)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.13,top=0.93)
        
plt.savefig(directoryfigure + 'Bias_AllGFDLModels-LM42p2_Global_%s-%s_Seasons_v2.png' % (variq,dataset_obs),dpi=300)
    
