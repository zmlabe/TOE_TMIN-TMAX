"""
Functions are useful statistical untilities for data processing in the ANN
 
Notes
-----
    Author : Zachary Labe
    Date   : 7 September 2022
    
Usage
-----
    [1] rmse(a,b)
    [3] remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs)
    [4] remove_merid_mean(data,data_obs)
    [5] remove_observations_mean(data,data_obs,lats,lons)
    [6] calculate_anomalies(data,data_obs,lats,lons,baseline,yearsall)
    [7] remove_ensemble_mean(data,ravel_modelens,ravelmodeltime,rm_standard_dev,numOfEns)
    [8] remove_ocean(data,data_obs,resolution)
    [9] mask_CONUS(data,data_obs,resolution)
    [10] remove_land(data,data_obs,resolution)
    [11] standardize_data(Xtrain,Xtest)
    [12] standardize_dataVal(Xtrain,Xtest,Xval,baselineSTD)
    [13] rm_standard_dev(var,window,ravelmodeltime,numOfEns)
    [14] rm_variance_dev(var,window)
    [15] read_InferenceLargeEnsemble(variq,dataset,dataset_obs,monthlychoice,scenario,resolution,lat_bounds,lon_bounds,rm_annual_mean,rm_merid_mean,land_only,ocean_only,CONUS_only,baselineSTD,yrminb,yrmaxb)
"""

def rmse(a,b):
    """
    Calculates the root mean squared error
    takes two variables, a and b, and returns value
    """
    
    ### Import modules
    import numpy as np
    
    ### Calculate RMSE
    rmse_stat = np.sqrt(np.mean((a - b)**2))
    
    return rmse_stat

###############################################################################

def remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs):
    """
    Removes annual mean from data set
    """
    
    ### Import modulates
    import numpy as np
    import calc_Utilities as UT
    
    ### Create 2d grid
    lons2,lats2 = np.meshgrid(lons,lats)
    lons2_obs,lats2_obs = np.meshgrid(lons_obs,lats_obs)
    
    ### Calculate weighted average and remove mean
    data = data - UT.calc_weightedAve(data,lats2)[:,:,np.newaxis,np.newaxis]
    data_obs = data_obs - UT.calc_weightedAve(data_obs,lats2_obs)[:,np.newaxis,np.newaxis]
    
    return data,data_obs

###############################################################################

def remove_merid_mean(data,data_obs,lats,lons,lats_obs,lons_obs):
    """
    Removes meridional mean from data set
    """
    
    ### Import modules
    import numpy as np
    
    ### Remove mean of latitude
    data = data - np.nanmean(data,axis=3)[:,:,:,np.newaxis,:]
    data_obs = data_obs - np.nanmean(data_obs,axis=1)[:,np.newaxis,:]

    return data,data_obs

###############################################################################

def remove_observations_mean(data,data_obs,lats,lons):
    """
    Removes observations to calculate model biases
    """
    
    ### Import modules
    import numpy as np
    
    ### Remove observational data
    databias = data - data_obs[np.newaxis,np.newaxis,:,:,:]

    return databias

###############################################################################

def calculate_anomalies(data,data_obs,lats,lons,baseline,yearsall):
    """
    Calculates anomalies for each model and observational data set. Note that
    it assumes the years at the moment
    """
    
    ### Import modules
    import numpy as np
    
    ### Select years to slice
    minyr = baseline.min()
    maxyr = baseline.max()
    yearq = np.where((yearsall >= minyr) & (yearsall <= maxyr))[0]
    
    if data.ndim == 5:
        
        ### Slice years
        modelnew = data[:,:,yearq,:,:]
        obsnew = data_obs[yearq,:,:]
        
        ### Average climatology
        meanmodel = np.nanmean(modelnew[:,:,:,:,:],axis=2)
        meanobs = np.nanmean(obsnew,axis=0)
        
        ### Calculate anomalies
        modelanom = data[:,:,:,:,:] - meanmodel[:,:,np.newaxis,:,:]
        obsanom = data_obs[:,:,:] - meanobs[:,:]
    else:
        obsnew = data_obs[yearq,:,:]
        
        ### Average climatology
        meanobs = np.nanmean(obsnew,axis=0)
        
        ### Calculate anomalies
        obsanom = data_obs[:,:,:] - meanobs[:,:]
        modelanom = np.nan
        print('NO MODEL ANOMALIES DUE TO SHAPE SIZE!!!')

    return modelanom,obsanom

###############################################################################

def remove_ensemble_mean(data,ravel_modelens,ravelmodeltime,rm_standard_dev):
    """
    Removes ensemble mean
    """
    
    ### Import modulates
    import numpy as np
    
    ### Remove ensemble mean
    if data.ndim == 4:
        datameangoneq = data - np.nanmean(data,axis=0)
    elif data.ndim == 5:
        ensmeanmodel = np.nanmean(data,axis=1)
        datameangoneq = np.empty((data.shape))
        for i in range(data.shape[0]):
            datameangoneq[i,:,:,:,:] = data[i,:,:,:,:] - ensmeanmodel[i,:,:,:]
            print('Completed: Ensemble mean removed for model %s!' % (i+1))
    
    if ravel_modelens == True:
        datameangone = np.reshape(datameangoneq,(datameangoneq.shape[0]*datameangoneq.shape[1],
                                                 datameangoneq.shape[2],
                                                 datameangoneq.shape[3],
                                                 datameangoneq.shape[4]))
    else: 
        datameangone = datameangoneq
    if rm_standard_dev == False:
        if ravelmodeltime == True:
            datameangone = np.reshape(datameangoneq,(datameangoneq.shape[0]*datameangoneq.shape[1]*datameangoneq.shape[2],
                                                      datameangoneq.shape[3],
                                                      datameangoneq.shape[4]))
        else: 
            datameangone = datameangoneq
    
    return datameangone

###############################################################################

def remove_ocean(data,data_obs,lat_bounds,lon_bounds,resolution):
    """
    Masks out the ocean for land_only == True
    """
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_dataFunctions as df
    import sys
    
    ### Read in land mask
    if resolution == 'LOWS':
        directorydata = '/work/Zachary.Labe/Data/masks/'
        filename = 'land_maskcoarse_SPEAR_LOW.nc'
        datafile = Dataset(directorydata + filename)
        maskq = datafile.variables['land_mask'][:]
        lats = datafile.variables['lat'][:]
        lons = datafile.variables['lon'][:]
        datafile.close()
    elif resolution == 'MEDS':
        directorydata = '/work/Zachary.Labe/Data/masks/'
        filename = 'land_maskcoarse_SPEAR_MED.nc'
        datafile = Dataset(directorydata + filename)
        maskq = datafile.variables['land_mask'][:]
        lats = datafile.variables['lat'][:]
        lons = datafile.variables['lon'][:]
        datafile.close()
    else:
        print(ValueError('WRONG RESOLUTION SELECTED FOR MASK!'))
        sys.exit()
    
    mask,lats,lons = df.getRegion(maskq,lats,lons,lat_bounds,lon_bounds)
    
    ### Mask out model and observations
    datamask = data * mask
    data_obsmask = data_obs * mask
    
    ### Check for floats
    datamask[np.where(datamask==0.)] = 0
    data_obsmask[np.where(data_obsmask==0.)] = 0
    
    return datamask, data_obsmask

###############################################################################

def mask_CONUS(data,data_obs,resolution,lat_bounds,lon_bounds):
    """
    Only plot values over CONUS for LOWS or MEDS resolution
    """
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import sys
    import calc_dataFunctions as df
    
    ### Read in land mask
    if resolution == 'LOWS':
        directorydata = '/work/Zachary.Labe/Data/NClimGrid_LOWS/'
        filename = 'T2M_1895-2021.nc'
        datafile = Dataset(directorydata + filename)
        maskq = np.asarray(datafile.variables['T2M'][:])
        lats = datafile.variables['lat'][:]
        lons = datafile.variables['lon'][:]
        datafile.close()
    elif resolution == 'MEDS':
        directorydata = '/work/Zachary.Labe/Data/NClimGrid_MEDS/'
        filename = 'T2M_1895-2021.nc'
        datafile = Dataset(directorydata + filename)
        maskq = np.asarray(datafile.variables['T2M'][:])
        lats = datafile.variables['lat'][:]
        lons = datafile.variables['lon'][:]
        datafile.close()
    elif resolution == 'HIGHS':
        directorydata = '/work/Zachary.Labe/Data/NClimGrid_HIGHS/'
        filename = 'T2M_1895-2021.nc'
        datafile = Dataset(directorydata + filename)
        maskq = np.asarray(datafile.variables['T2M'][:])
        lats = datafile.variables['lat'][:]
        lons = datafile.variables['lon'][:]
        datafile.close()
    elif resolution == 'original':
        directorydata = '/work/Zachary.Labe/Data/NClimGrid/'
        filename = 'T2M_1895-2021.nc'
        datafile = Dataset(directorydata + filename)
        maskq = np.asarray(datafile.variables['T2M'][:])
        lats = datafile.variables['lat'][:]
        lons = datafile.variables['lon'][:]
        datafile.close()
    else:
        print(ValueError('WRONG RESOLUTION SELECTED FOR MASK!'))
        sys.exit()
        
    maskq,lats,lons = df.getRegion(maskq,lats,lons,lat_bounds,lon_bounds)
        
    ### Mask values 
    maskq[np.where(np.isnan(maskq))] = 0.
    maskq[np.where(maskq != 0.)] = 1
    datamask = data * maskq[0]
    data_obsmask = data_obs * maskq[0]
    
    ### Set to nans
    datamask[np.where(datamask == 0.)] = np.nan
    data_obsmask[np.where(data_obsmask == 0.)] = np.nan

    print('<<<<<< COMPLETED: mask_CONUS()')
    return datamask,data_obsmask

###############################################################################

def remove_land(data,data_obs,lat_bounds,lon_bounds,resolution):
    """
    Masks out the ocean for ocean_only == True
    """
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_dataFunctions as df
    import sys
    
    ### Read in ocean mask
    if resolution == 'LOWS':
        directorydata = '/work/Zachary.Labe/Data/masks/'
        filename = 'ocean_maskcoarse_SPEAR_LOW.nc'
        datafile = Dataset(directorydata + filename)
        maskq = datafile.variables['ocean_mask'][:]
        lats = datafile.variables['lat'][:]
        lons = datafile.variables['lon'][:]
        datafile.close()
    elif resolution == 'MEDS':
        directorydata = '/work/Zachary.Labe/Data/masks/'
        filename = 'ocean_maskcoarse_SPEAR_MED.nc'
        datafile = Dataset(directorydata + filename)
        maskq = datafile.variables['ocean_mask'][:]
        lats = datafile.variables['lat'][:]
        lons = datafile.variables['lon'][:]
        datafile.close()
    else:
        print(ValueError('WRONG RESOLUTION SELECTED FOR MASK!'))
        sys.exit()
    
    mask,lats,lons = df.getRegion(maskq,lats,lons,lat_bounds,lon_bounds)
    
    ### Mask out model and observations
    datamask = data * mask
    data_obsmask = data_obs * mask
    
    ### Check for floats
    datamask[np.where(datamask==0.)] = 0
    data_obsmask[np.where(data_obsmask==0.)] = 0
    
    return datamask, data_obsmask

###############################################################################

def standardize_data(Xtrain,Xtest):
    """
    Standardizes training and testing data
    """
    
    ### Import modulates
    import numpy as np

    Xmean = np.mean(Xtrain,axis=0)
    Xstd = np.std(Xtrain,axis=0)
    
    Xtrain = (Xtrain - Xmean)/Xstd
    Xtest = (Xtest - Xmean)/Xstd
    
    stdVals = (Xmean,Xstd)
    stdVals = stdVals[:]
    
    ### If there is a nan (like for land/ocean masks)
    if np.isnan(np.min(Xtrain)) == True:
        Xtrain[np.isnan(Xtrain)] = 0
        Xtest[np.isnan(Xtest)] = 0
        print('--THERE WAS A NAN IN THE STANDARDIZED DATA!--')
    
    return Xtrain,Xtest,stdVals

###############################################################################

def standardize_dataVal(Xtrain,Xtest,Xval,baselineSTD,years,yrminb,yrmaxb):
    """
    Standardizes training, testing, and validation data
    """
    
    ### Import modulates
    import numpy as np
    import sys
    print('\n<<<<< Starting to standardize data! >>>>>')

    if baselineSTD == False:
        Xmean = np.mean(Xtrain,axis=0)
        Xstd = np.std(Xtrain,axis=0)
        
    elif any([baselineSTD == True,baselineSTD == 'MMLEA']):
        yearq = np.where((years >= yrminb) & (years <= yrmaxb))[0]
        print(years[yearq])
        if len(years[yearq]) != 30:
            print(ValueError('SOMETHING IS WRONG WITH THE BASELINE LENGTH'))
            sys.exit()
         
        ### Slice only those years for each ensemble
        Xtrainn = Xtrain.reshape(Xtrain.shape[0]//years.shape[0],years.shape[0],Xtrain.shape[1])
        print('Reshaped standardize training data = ',[Xtrainn.shape])
        
        ### Average the training ensembles and baseline years
        Xmean = np.mean(np.mean(Xtrainn[:,yearq,:],axis=1),axis=0)
        Xstd = np.mean(np.std(Xtrainn[:,yearq,:],axis=1),axis=0)
        
    else:
        print(ValueError('BASELINE ARGUMENT IS NOT WORKING FOR STANDARDIZING!'))
        sys.exit()
    
    Xtrain = (Xtrain - Xmean)/Xstd
    Xtest = (Xtest - Xmean)/Xstd
    Xval = (Xval - Xmean)/Xstd
    
    stdVals = (Xmean,Xstd)
    stdVals = stdVals[:]
    
    ### If there is a nan (like for land/ocean masks)
    if np.isnan(np.min(Xtrain)) == True:
        Xtrain[np.isnan(Xtrain)] = 0
        Xtest[np.isnan(Xtest)] = 0
        Xval[np.isnan(Xval)] = 0
        print('--THERE WAS A NAN IN THE STANDARDIZED DATA!--')
    
    print('<<<<< Ending standardize data! >>>>>\n')
    return Xtrain,Xtest,Xval,stdVals

###############################################################################
    
def rm_standard_dev(var,window,ravelmodeltime,numOfEns):
    """
    Smoothed standard deviation
    """
    import pandas as pd
    import numpy as np
    
    print('\n\n-----------STARTED: Rolling std!\n\n')
    
    
    if var.ndim == 3:
        rollingstd = np.empty((var.shape))
        for i in range(var.shape[1]):
            for j in range(var.shape[2]):
                series = pd.Series(var[:,i,j])
                rollingstd[:,i,j] = series.rolling(window).std().to_numpy()
    elif var.ndim == 4:
        rollingstd = np.empty((var.shape))
        for ens in range(var.shape[0]):
            for i in range(var.shape[2]):
                for j in range(var.shape[3]):
                    series = pd.Series(var[ens,:,i,j])
                    rollingstd[ens,:,i,j] = series.rolling(window).std().to_numpy()
    elif var.ndim == 5:
        varn = np.reshape(var,(var.shape[0]*var.shape[1],var.shape[2],var.shape[3],var.shape[4]))
        rollingstd = np.empty((varn.shape))
        for ens in range(varn.shape[0]):
            for i in range(varn.shape[2]):
                for j in range(varn.shape[3]):
                    series = pd.Series(varn[ens,:,i,j])
                    rollingstd[ens,:,i,j] = series.rolling(window).std().to_numpy()
    
    newdataq = rollingstd[:,window:,:,:] 
    
    if ravelmodeltime == True:
        newdata = np.reshape(newdataq,(newdataq.shape[0]*newdataq.shape[1],
                                       newdataq.shape[2],newdataq.shape[3]))
    else:
        newdata = np.reshape(newdataq,(newdataq.shape[0]//numOfEns,numOfEns,newdataq.shape[1],
                                       newdataq.shape[2],newdataq.shape[3]))
    print('-----------COMPLETED: Rolling std!\n\n')     
    return newdata 

###############################################################################
    
def rm_variance_dev(var,window,ravelmodeltime):
    """
    Smoothed variance
    """
    import pandas as pd
    import numpy as np
    
    print('\n\n-----------STARTED: Rolling vari!\n\n')
    
    rollingvar = np.empty((var.shape))
    for ens in range(var.shape[0]):
        for i in range(var.shape[2]):
            for j in range(var.shape[3]):
                series = pd.Series(var[ens,:,i,j])
                rollingvar[ens,:,i,j] = series.rolling(window).var().to_numpy()
    
    newdataq = rollingvar[:,window:,:,:] 
    
    if ravelmodeltime == True:
        newdata = np.reshape(newdataq,(newdataq.shape[0]*newdataq.shape[1],
                                       newdataq.shape[2],newdataq.shape[3]))
    else:
        newdata = newdataq
    print('-----------COMPLETED: Rolling vari!\n\n')     
    return newdata 

###############################################################################

def convert_fuzzyDecade(data,startYear,classChunk,yearsall):
    import numpy as np
    import scipy.stats as sts
    
    years = np.arange(startYear-classChunk*2,yearsall.max()+classChunk*2)
    chunks = years[::int(classChunk)] + classChunk/2
    
    labels = np.zeros((np.shape(data)[0],len(chunks)))
    
    for iy,y in enumerate(data):
        norm = sts.uniform.pdf(years,loc=y-classChunk/2.,scale=classChunk)
        
        vec = []
        for sy in years[::classChunk]:
            j=np.logical_and(years>sy,years<sy+classChunk)
            vec.append(np.sum(norm[j]))
        vec = np.asarray(vec)
        vec[vec<.0001] = 0. # This should not matter

        vec = vec/np.sum(vec)
        
        labels[iy,:] = vec
    return labels, chunks

###############################################################################

def convert_fuzzyDecade_toYear(label,startYear,classChunk,yearsall):
    import numpy as np
    
    years = np.arange(startYear-classChunk*2,yearsall.max()+classChunk*2)
    chunks = years[::int(classChunk)] + classChunk/2
    
    return np.sum(label*chunks,axis=1)

#############################################################################

def read_InferenceLargeEnsemble(variq,dataset,dataset_obs,monthlychoice,scenario,resolution,lat_bounds,lon_bounds,rm_annual_mean,rm_merid_mean,land_only,ocean_only,CONUS_only,baselineSTD,yrminb,yrmaxb):
    import calc_dataFunctions as df
    import numpy as np
    import calc_Utilities as UT
    import sys
    
    ### Read in data
    def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
        data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
        datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
        print('\nOur dataset: ',dataset,' is shaped',data.shape)
        return datar,lats,lons  
    
    ### Obs doesn't do anything here, too lazy to rewrite functions
    data_all,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds)
    data_obs_all,lats_obs,lons_obs = read_primary_dataset(variq,dataset_obs,monthlychoice,scenario,lat_bounds,lon_bounds)
    
    ### Prepare data for preprocessing
    data, data_obs, = data_all, data_obs_all,
    if rm_annual_mean == True:        
        data, data_obs = remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs)
        print('*Removed annual mean*')
    if rm_merid_mean == True:
        data, data_obs = remove_merid_mean(data,data_obs,lats,lons,lats_obs,lons_obs)
        print('*Removed meridian mean*')  
    if land_only == True:
        data, data_obs = remove_ocean(data,data_obs,lat_bounds,lon_bounds) 
        print('*Removed ocean*')
    if ocean_only == True:
        data, data_obs = remove_land(data,data_obs,lat_bounds,lon_bounds) 
        print('*Removed land*')
    if CONUS_only == True:
        data, data_obs = mask_CONUS(data,data_obs,resolution,lat_bounds,lon_bounds)
        print('*Removed everything by CONUS*')
      
    ### Select inference ensemble member
    ensembleMember = 2
    yearsens = np.arange(1921,2100+1,1)
    inferenceMember = data[ensembleMember,:,:,:].squeeze()
    XinferenceMember = inferenceMember.reshape(inferenceMember.shape[0],inferenceMember.shape[1]*inferenceMember.shape[2])
    print('Using inference ensemble member number ----> %s!' % ensembleMember)
    
    ### Standardize testing ensemble member (does not work for MMLEA)
    if baselineSTD == True:
        yearsensq = np.where((yearsens >= yrminb) & (yearsens <= yrmaxb))[0]
        print(yearsens[yearsensq])
        if len(yearsens[yearsensq]) != 30:
            print(ValueError('SOMETHING IS WRONG WITH THE BASELINE LENGTH'))
            sys.exit()
            
        XmeaninferenceMember = np.nanmean(XinferenceMember[yearsensq,:],axis=0)
        XstdinferenceMember = np.nanstd(XinferenceMember[yearsensq,:],axis=0)  
        XinferenceMemberS = (XinferenceMember-XmeaninferenceMember)/XstdinferenceMember
        XinferenceMemberS[np.isnan(XinferenceMemberS)] = 0    
    elif baselineSTD == False:
        XmeaninferenceMember = np.nanmean(XinferenceMember,axis=0)
        XstdinferenceMember = np.nanstd(XinferenceMember,axis=0)  
        XinferenceMemberS = (XinferenceMember-XmeaninferenceMember)/XstdinferenceMember
        XinferenceMemberS[np.isnan(XinferenceMemberS)] = 0
    else:
        print(ValueError('BASELINE ARGUMENT IS NOT WORKING FOR STANDARDIZING!'))
        sys.exit()
        
    return XinferenceMemberS
                        
###############################################################################
###############################################################################
###############################################################################
