"""
Train ANN to predict the year for ToE in the CONUS

Reference   : Barnes et al. [2020, JAMES] & Labe and Barnes [2021, JAMES]
Author      : Zachary M. Labe
Date        : 29 September 2022
Environment : conda activate env-tf27
Tensorflow  : 2.7 (XAI for v2.0.1)
Version     : 4 (LRP positive and negative relevance)
"""

###############################################################################
###############################################################################
###############################################################################
### Import packages
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
import random
import scipy.stats as sts
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import calc_SegmentData as dSEG
import calc_LRPclass_tf27_v2 as LRP
import sys

###############################################################################
###############################################################################
###############################################################################
### For XAI
import innvestigate
tf.compat.v1.disable_eager_execution() # bug fix with innvestigate v2.0.1 - tf2

###############################################################################
###############################################################################
###############################################################################
### To remove some warnings I don't care about
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Remove GPU error message
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)

###############################################################################
###############################################################################
###############################################################################
### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
datasetsingle = ['SPEAR_MED_NATURAL']
yearsall = [np.arange(1921,2100+1,1)]
segment_data_factorq = [0.8]

dataset_obs = 'NClimGrid_MEDS'
resolution = 'MEDS'

variq = 'T2M'
monthlychoiceall = ['DJF','MAM','JJA','SON','annual']

reg_name = 'US'
lat_bounds,lon_bounds = UT.regions(reg_name) 

### Set counter
SAMPLEQ = 100

###############################################################################
###############################################################################
###############################################################################
### Decide whether to limit the number of ensembles for training/testing/val
sliceEns_only = False
sliceEns_only_N = np.nan

###############################################################################
###############################################################################
###############################################################################
### Select whether to standardize over a baseline
baselineSTD = True
yrminb = 1981
yrmaxb = 2010
baselineSTDyrs = np.arange(yrminb,yrmaxb+1,1)

###############################################################################
###############################################################################
###############################################################################
### Masking and preprocessing arguments
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
land_only = False
ocean_only = False
CONUS_only = True

### Parameters
if reg_name == 'US':
    debug = True
    NNType = 'ANN'
    annType = 'class'
    classChunkHalf = 5
    classChunk = 10
    biasBool = False
    if resolution == 'MEDS':
        hiddensList = [[10,10,10]]
    elif resolution == 'LOWS':
        hiddensList = [[20,20,20]]
    elif resolution == 'HIGHS':
        hiddensList = [[20,20,20]]
    else:
        print(ValueError('This is the wrong resolution!!!'))
        sys.exit()
    ridge_penalty = [0.001]
    actFun = 'relu'
    iterations = [500]
elif any([reg_name=='W_US',reg_name=='Ce_US',reg_name=='E_US']):
    debug = True
    NNType = 'ANN'
    annType = 'class'
    classChunkHalf = 5
    classChunk = 10
    biasBool = False
    hiddensList = [[100,100]]
    ridge_penalty = [0.001]
    actFun = 'relu'
    iterations = [500]
else:
    print(ValueError('This is the wrong region name selected!!!'))
    sys.exit()

###############################################################################
###############################################################################
###############################################################################
### XAI methods to try
lrpRule1 = 'z'
lrpRule2 = 'epsilon'
lrpRule3 = 'integratedgradient'
normLRP = True
numDim = 3

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Looping through each month/season and loop through simulation
for ms in range(len(monthlychoiceall)):
    monthlychoice = monthlychoiceall[ms]
    
    for sis,singlesimulation in enumerate(datasetsingle):
        
        lrpmapstime = []
        lrpmapstime_obs = []
        valslopes = [] 
        valrr = []
        YpredObs_all = []
        for isample in range(SAMPLEQ): 
            
            ### Select years of data and emissions scenario
            if datasetsingle[sis] == 'SPEAR_MED':
                scenario = 'SSP585'
            if datasetsingle[sis] == 'SPEAR_MED_FA':
                scenario = 'SSP585'
            elif datasetsingle[sis] == 'SPEAR_MED_NOAER':
                scenario = 'SSP585'
            elif datasetsingle[sis] == 'SPEAR_LOW':
                scenario = 'SSP585'
            elif datasetsingle[sis] == 'SPEAR_MED_NATURAL':
                scenario = 'NATURAL'
            elif datasetsingle[sis] == 'SPEAR_MED_Scenario':
                scenario = 'SSP245'
            if datasetsingle[sis] == 'SPEAR_HIGH':
                scenario = 'SSP585'
            elif datasetsingle[sis] == 'LENS2_LOWS':
                scenario = 'SSP370'
            elif datasetsingle[sis] == 'FLOR':
                scenario = 'RCP85'
            elif datasetsingle[sis] == 'FLOR_LOWS':
                scenario = 'RCP85'
            elif datasetsingle[sis] == 'LENS1_LOWS':
                scenario = 'RCP85'
            elif datasetsingle[sis] == 'MMLEA':
                scenario = None
                
            if dataset_obs == 'NClimGrid_MEDS':
                yearsobs = np.arange(1921,2022+1,1)
                obsyearstart = yearsobs[0]
            elif dataset_obs == 'NClimGrid_LOWS':
                yearsobs = np.arange(1921,2022+1,1)
                obsyearstart = yearsobs[0]
            elif dataset_obs == '20CRv3_LOWS':
                yearsobs = np.arange(1921,2015+1,1)
                obsyearstart = yearsobs[0]
            elif dataset_obs == 'ERA5_LOWS':
                yearsobs = np.arange(1979,2021+1,1)
                obsyearstart = yearsobs[0]
            elif dataset_obs == 'ERA5_MEDS':
                yearsobs = np.arange(1979,2021+1,1)
                obsyearstart = yearsobs[0]
                
            ### Calculate other seasons
            if monthlychoice == 'DJF':
                yearsall = [np.arange(1921,2100+1,1)[:-1]]
                yearsobs = yearsobs[:-1]
            else:
                yearsall = [np.arange(1921,2100+1,1)]
                yearsobs = yearsobs
        
    ###############################################################################
    ###############################################################################
    ###############################################################################
            ### ANN preliminaries
            directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/PredictionsYear/'
            directorymodel = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/savedModels/'
            directorylrp = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/LRP/Regions/'
            directorypredictions = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/Regions/'
            
            ### Define primary dataset to use
            dataset = singlesimulation
            modelType = 'TrainedOn' + dataset
                
            #######################################################################
            #######################################################################
            #######################################################################
            ### For the MMLEA to keep track of the models
            if dataset == 'MMLEA':
                mmleaName = np.concatenate([np.repeat('SPEAR_LOW',30),
                                            np.repeat('LENS2_LOWS',100),
                                            np.repeat('MPI_ESM12_HR_LOWS',10),
                                            np.repeat('MIROC6_LE_LOWS',50),
                                            np.repeat('LENS1_LOWS',40),
                                            np.repeat('FLOR_LOWS',40)],
                                            axis=0)
            
            ### Need to change proportion for fewer ensembles in SPEAR_MED_NOAER
            segment_data_factor = segment_data_factorq[sis]
            
    ###############################################################################
    ###############################################################################
    ###############################################################################
            ### Read in climate models      
            def read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds):
                data,lats,lons = df.readFiles(variq,dataset,monthlychoice,scenario)
                datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
                print('\nOur dataset: ',dataset,' is shaped',data.shape)
                return datar,lats,lons        
            
    ###############################################################################
    ###############################################################################
    ###############################################################################
            ### Neural Network Creation      
            class TimeHistory(tf.keras.callbacks.Callback):
                def on_train_begin(self, logs={}):
                    self.times = []
                def on_epoch_begin(self, epoch, logs={}):
                    self.epoch_time_start = time.time()
                def on_epoch_end(self, epoch, logs={}):
                    self.times.append(time.time() - self.epoch_time_start)
            
            def defineNN(hidden, input_shape, output_shape, ridgePenalty):        
               
                ### Begin Model
                tf.keras.backend.clear_session()
                model = Sequential()
                
                ###################################################################
                ###################################################################
                ###################################################################
                ### Initialize first layer
                if hidden[0]==0:
                    ### Model is linear
                    model.add(Dense(1,input_shape=(input_shape,),
                                    activation='linear',use_bias=True,
                                    kernel_regularizer=regularizers.l1_l2(l1=0.00,l2=ridgePenalty),
                                    bias_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed),
                                    kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed)))
                    print('\nTHIS IS A LINEAR NN!\n')
                    
                ###################################################################
                ###################################################################
                ###################################################################
                ### Model is a single node with activation function
                else:
                    model.add(Dense(hidden[0],input_shape=(input_shape,),
                                    activation=actFun, use_bias=True,
                                    kernel_regularizer=regularizers.l1_l2(l1=0.00,l2=ridgePenalty),
                                    bias_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed),
                                    kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed)))
            
                    ### Initialize other layers
                    for layer in hidden[1:]:
                        model.add(Dense(layer,activation=actFun,use_bias=True,
                                        kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.00),
                                        bias_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed),
                                        kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed)))
                        
                    print('\nTHIS IS AN ANN!\n')
            
                #### Initialize output layer
                model.add(Dense(output_shape,activation=None,use_bias=True,
                                kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.0),
                                bias_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed),
                                kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed)))
            
                ### Add softmax layer at the end
                model.add(Activation('softmax'))
                
                return model
            
    ###############################################################################
    ###############################################################################
    ###############################################################################
            ### Neural network training
            def trainNN(model, Xtrain, Ytrain, Xval, Yval, niter, verbose=True):
              
                ### Declare the relevant model parameters
                global lr_here, batch_size
                lr_here = .01
                batch_size = 32 
                
                ### Compile the model
                model.compile(optimizer=optimizers.SGD(learning_rate=lr_here,momentum=0.9,nesterov=True), 
                              loss = 'binary_crossentropy',metrics=[metrics.categorical_accuracy])
                print('----ANN Training: learning rate = '+str(lr_here)+'; activation = '+actFun+'; batch = '+str(batch_size) + '----')    
                
                ### Callbacks
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               patience=25,
                                                               verbose=1,
                                                               mode='auto',
                                                               restore_best_weights=True)
                time_callback = TimeHistory()
                
                ### Fit the model
                history = model.fit(Xtrain,Ytrain,batch_size=batch_size,epochs=niter,
                                    shuffle=True,verbose=verbose,
                                    callbacks=[time_callback,early_stopping],
                                    validation_data=(Xval,Yval))
                print('******** done training ***********')
            
                return model, history
            
            def test_train_loopClass(Xtrain,Ytrain,Xtest,Ytest,Xval,Yval,iterations,ridge_penalty,hiddens):
                """or loops to iterate through training iterations, ridge penalty, 
                and hidden layer list
                """
              
                ### Get parameters (not currently using the looping method)
                niter = iterations[0]
                penalty = ridge_penalty[0]
                hidden = hiddens[0]
                global random_network_seed
                            
                ### Check / use random seed
                if random_network_seed == None:
                    np.random.seed(None)
                    random_network_seed = int(np.random.randint(1, 100000))
                np.random.seed(random_network_seed)
                random.seed(random_network_seed)
                tf.random.set_seed(0)
    
                ### Standardize the data
                Xtrain,Xtest,Xval,stdVals = dSS.standardize_dataVal(Xtrain,Xtest,Xval,baselineSTD,yearsall[sis],yrminb,yrmaxb)
                Xmean,Xstd = stdVals
                
                ### Define the model function (see above)
                model = defineNN(hidden,
                                 input_shape=np.shape(Xtrain)[1],
                                 output_shape=np.shape(Ytrain)[1],
                                 ridgePenalty=penalty)  
               
                ### Train the model function (see above)
                model, history = trainNN(model,
                                         Xtrain,Ytrain,
                                         Xval,Yval,
                                         niter,verbose=1)
                
                ### 'unlock' the random seed
                np.random.seed(None)
                random.seed(None)
                tf.random.set_seed(None)
                
                return model
    
    ###############################################################################
    ###############################################################################
    ###############################################################################        
            ### Functions for fuzzy classification
            def convert_fuzzyDecade(data,startYear,classChunk,yearsall):
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
            
            def convert_fuzzyDecade_toYear(label,startYear,classChunk,yearsall):
                years = np.arange(startYear-classChunk*2,yearsall.max()+classChunk*2)
                chunks = years[::int(classChunk)] + classChunk/2
                
                return np.sum(label*chunks,axis=1)
            
    ###############################################################################
    ###############################################################################
    ###############################################################################         
            ### Get data
            lat_bounds,lon_bounds = UT.regions(reg_name)
            data_all,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,scenario,lat_bounds,lon_bounds)
            data_obs_all,lats_obs,lons_obs = read_primary_dataset(variq,dataset_obs,monthlychoice,scenario,lat_bounds,lon_bounds)
            
            ### Prepare data for preprocessing
            data, data_obs, = data_all, data_obs_all,
            if rm_annual_mean == True:        
                data, data_obs = dSS.remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs)
                print('*Removed annual mean*')
            if rm_merid_mean == True:
                data, data_obs = dSS.remove_merid_mean(data,data_obs,lats,lons,lats_obs,lons_obs)
                print('*Removed meridian mean*')  
            if land_only == True:
                data, data_obs = dSS.remove_ocean(data,data_obs,lat_bounds,lon_bounds) 
                print('*Removed ocean*')
            if ocean_only == True:
                data, data_obs = dSS.remove_land(data,data_obs,lat_bounds,lon_bounds) 
                print('*Removed land*')
            if CONUS_only == True:
                data, data_obs = dSS.mask_CONUS(data,data_obs,resolution,lat_bounds,lon_bounds)
                print('*Removed everything by CONUS*')
                
            ### Decide whether to only look at n number of ensembles
            if sliceEns_only == True:
                data = data[:sliceEns_only_N,:,:,:]
                print('*Only kept %s ensembles*' % sliceEns_only_N)
                
    ###############################################################################
    ###############################################################################
    ############################################################################### 
            ### Clear session and prepare to read in data
            tf.keras.backend.clear_session()
    
            ###########################
            random_segment_seed = None
            ###########################
            Xtrain,Ytrain,Xtest,Ytest,Xval,Yval,Xtest_shape,Xtrain_shape,Xval_shape,data_train_shape,data_test_shape,data_val_shape,trainIndices,testIndices,valIndices = dSEG.segment_data(data,yearsall[sis],segment_data_factor,random_segment_seed)
            
            ### Convert year into decadal class
            startYear = Ytrain[0] 
            YtrainClassMulti, decadeChunks = convert_fuzzyDecade(Ytrain,startYear,classChunk,yearsall[sis])  
            YtestClassMulti, __ = convert_fuzzyDecade(Ytest,startYear,classChunk,yearsall[sis])  
            YvalClassMulti, __ = convert_fuzzyDecade(Yval,startYear,classChunk,yearsall[sis])  
        
            ### For use later
            XtrainS,XtestS,XvalS,stdVals = dSS.standardize_dataVal(Xtrain,Xtest,Xval,baselineSTD,yearsall[sis],yrminb,yrmaxb)
            Xmean, Xstd = stdVals      
    
            ###########################
            random_network_seed = None
            ###########################
        
    ###############################################################################
    ###############################################################################
    ############################################################################### 
            ### Create and train network
            model = test_train_loopClass(Xtrain,YtrainClassMulti,
                                         Xtest,YtestClassMulti,
                                         Xval,YvalClassMulti,
                                         iterations,
                                         ridge_penalty,
                                         hiddensList)
            model.summary()  
    
    ###############################################################################
    ###############################################################################
    ###############################################################################         
            ################################################################################################################################################                
            ### Save models
            savename = modelType+'_Vari-'+variq+'_Obs-' + dataset_obs + '_Region-' + reg_name + '_Months-' + monthlychoice + '_L2-'+ str(ridge_penalty[0])+ '_LR-' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
            savenameModelTestTrain = modelType+'_'+variq+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)
        
            # model.save(directorymodel + savename + '.h5')
            # np.savez(directorymodel + savenameModelTestTrain + '.npz',trainModels=trainIndices,testModels=testIndices,Xtrain=Xtrain,Ytrain=Ytrain,Xtest=Xtest,Ytest=Ytest,Xmean=Xmean,Xstd=Xstd,lats=lats,lons=lons)
            print('Saving ------->' + savename)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################         
            ###############################################################
            ### Assess observations
            Xobs = data_obs.reshape(data_obs.shape[0],data_obs.shape[1]*data_obs.shape[2])
        
            startYear = yearsall[sis].min()
            endYear = yearsall[sis].max()
            years = np.arange(startYear,endYear+1,1)    
            
            ###############################################################
            ###############################################################
            ###############################################################
            ### Standardize testing 'obs'
            if baselineSTD == True:
                yearsobsq = np.where((yearsobs >= yrminb) & (yearsobs <= yrmaxb))[0]
                print(yearsobs[yearsobsq])
                if len(yearsobs[yearsobsq]) != 30:
                    print(ValueError('SOMETHING IS WRONG WITH THE BASELINE LENGTH'))
                    sys.exit()
                    
                Xmeanobs = np.nanmean(Xobs[yearsobsq,:],axis=0)
                Xstdobs = np.nanstd(Xobs[yearsobsq,:],axis=0)  
                XobsS = (Xobs-Xmeanobs)/Xstdobs
                XobsS[np.isnan(XobsS)] = 0    
            elif baselineSTD == False:
                Xmeanobs = np.nanmean(Xobs,axis=0)
                Xstdobs = np.nanstd(Xobs,axis=0)  
                XobsS = (Xobs-Xmeanobs)/Xstdobs
                XobsS[np.isnan(XobsS)] = 0
            elif baselineSTD == 'MMLEA':
                XobsS = (Xobs-Xmean)/Xstd
                XobsS[np.isnan(XobsS)] = 0
            else:
                print(ValueError('BASELINE ARGUMENT IS NOT WORKING FOR STANDARDIZING!'))
                sys.exit()
                
            ###############################################################
            ###############################################################
            ###############################################################
            ### Standardize training
            XtrainS = (Xtrain-Xmean)/Xstd
            XtrainS[np.isnan(XtrainS)] = 0
            
            ### Standardize testing
            XtestS = (Xtest-Xmean)/Xstd
            XtestS[np.isnan(XtestS)] = 0     
        
            ### Standardize testing
            XValS = (Xval-Xmean)/Xstd
            XValS[np.isnan(XValS)] = 0                                                                                       
            
            ### Chunk by individual year
            YpredObs = convert_fuzzyDecade_toYear(model.predict(XobsS),
                                                  startYear,
                                                  classChunk,
                                                  yearsall[sis])
            YpredTrain = convert_fuzzyDecade_toYear(model.predict(XtrainS),
                                                    startYear,
                                                    classChunk,
                                                    yearsall[sis])
            YpredTest = convert_fuzzyDecade_toYear(model.predict(XtestS),
                                                    startYear,
                                                    classChunk,
                                                    yearsall[sis])
            YpredVal = convert_fuzzyDecade_toYear(model.predict(XValS),
                                                    startYear,
                                                    classChunk,
                                                    yearsall[sis])
            
            ##############################################################################
            ##############################################################################
            ##############################################################################
            ### Visualizing through LRP
            numLats = lats.shape[0]
            numLons = lons.shape[0]  
            num_of_class = len(yearsall[sis])

            ### For training data only
            lrptestz = LRP.calc_LRPModel(model,XtestS,Ytest,biasBool,
                                                    annType,num_of_class,
                                                    yearsall,lrpRule1,normLRP,
                                                    numLats,numLons,numDim,
                                                    classChunk,startYear,yearsall[sis])
            
            
            ### For observations data only
            lrpobservationsz = LRP.calc_LRPObs(model,XobsS,biasBool,annType,
                                                num_of_class,yearsall,lrpRule1,
                                                normLRP,numLats,numLons,numDim,yearsall[sis])
         
        ############################################################################### 
        ############################################################################### 
        ###############################################################################        
            ### Define variable for analysis
            print('\n\n------------------------')
            print(variq,'= Variable!')
            print(monthlychoice,'= Time!')
            print(reg_name,'= Region!')
            print(lat_bounds,lon_bounds)
            print(dataset,'= Model!')
            print(dataset_obs,'= Observations!\n')
            print(rm_annual_mean,'= rm_annual_mean') 
            print(rm_merid_mean,'= rm_merid_mean') 
            print(rm_ensemble_mean,'= rm_ensemble_mean') 
            print(CONUS_only,'= CONUS_only') 
            print(land_only,'= land_only')
            print(ocean_only,'= ocean_only')
            print('------------------------\n')
            
            
            ### Select observations to save
            obsactual = yearsobs
            obspredic = YpredObs
            
            ### Regression
            slopeobs,interceptobs,r_valueobs,p_valueobs,std_errobs = sts.linregress(obsactual,obspredic)
            r_corr = sts.spearmanr(obsactual,obspredic)[0]
            
            ### Append slopes
            valslopes.append(slopeobs)
            valrr.append(r_corr)
            YpredObs_all.append(YpredObs)
            
            ### Append lrp averaged over all years
            lrpmapstime.append(lrptestz)
            lrpmapstime_obs.append(lrpobservationsz)
            
    ### See statistics for observations
    modelslopes = np.asarray(valslopes)
    modelr = np.asarray(valrr)
    lrpmapsallarray = np.asarray(lrpmapstime)
    lrpmapsallarray_obs = np.asarray(lrpmapstime_obs)
    YpredObs_allmodels = np.asarray(YpredObs_all)
    
    ### Save the arrays
    directoryscores = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/LoopFinal/Seasons/'
    np.savetxt(directoryscores + 'ANN_PredictTheYear_v3-LoopFinal-SLOPE_%s_%s_%s-%s_%s_SAMPLES-%s.txt' % (variq,monthlychoice,dataset,
                                                                                              dataset_obs,reg_name,SAMPLEQ),
               modelslopes)
    np.savetxt(directoryscores + 'ANN_PredictTheYear_v3-LoopFinal-R_%s_%s_%s-%s_%s_SAMPLES-%s.txt' % (variq,monthlychoice,dataset,
                                                                                              dataset_obs,reg_name,SAMPLEQ),
               modelr)
    np.savetxt(directoryscores + 'ANN_PredictTheYear_v3-LoopFinal-ObsPred_%s_%s_%s-%s_%s_SAMPLES-%s.txt' % (variq,monthlychoice,dataset,
                                                                                              dataset_obs,reg_name,SAMPLEQ),
               YpredObs_allmodels)
    ##############################################################################
    ##############################################################################
    ##############################################################################
    def netcdfLRPz(lats,lons,var,directory,typemodel,trainingdata,variq,savename):
        print('\n>>> Using netcdfLRP-z function!')
        
        from netCDF4 import Dataset
        import numpy as np
        
        name = 'LRPMap_LoopFinal_Z-neg_' + typemodel + '_' + variq + '_' + savename + '.nc'
        filename = directory + name
        ncfile = Dataset(filename,'w',format='NETCDF4')
        ncfile.description = 'LRP maps for using selected seed' 
        
        ### Dimensions
        ncfile.createDimension('samples',var.shape[0])
        ncfile.createDimension('years',var.shape[1])
        ncfile.createDimension('lat',var.shape[2])
        ncfile.createDimension('lon',var.shape[3])
        
        ### Variables
        samples = ncfile.createVariable('samples','f4',('samples'))
        years = ncfile.createVariable('years','f4',('years'))
        latitude = ncfile.createVariable('lat','f4',('lat'))
        longitude = ncfile.createVariable('lon','f4',('lon'))
        varns = ncfile.createVariable('LRP','f4',('samples','years','lat','lon'))
        
        ### Units
        varns.units = 'unitless relevance'
        ncfile.title = 'LRP relevance'
        ncfile.instituion = 'Colorado State University'
        ncfile.references = 'Barnes et al. [2020]'
        
        ### Data
        samples[:] = np.arange(var.shape[0])
        years[:] = np.arange(var.shape[1])
        latitude[:] = lats
        longitude[:] = lons
        varns[:] = var
        
        ncfile.close()
        print('*Completed: Created netCDF4 File!')
    
    directoryscoreslrp = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/LRP/LoopFinal/'
    netcdfLRPz(lats,lons,lrpmapsallarray,directoryscoreslrp,'Testing',dataset,variq,savename)
    netcdfLRPz(lats,lons,lrpmapsallarray_obs,directoryscoreslrp,'Obs',dataset,variq,savename)
