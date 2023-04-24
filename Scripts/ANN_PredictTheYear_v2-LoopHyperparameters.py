"""
Train ANN to predict the year for ToE in the CONUS

Reference   : Barnes et al. [2020, JAMES] & Labe and Barnes [2021, JAMES]
Author      : Zachary M. Labe
Date        : 29 September 2022
Environment : conda activate env-tf27
Tensorflow  : 2.7 (XAI for v2.0.1)
Version     : 2 (looping through L2 and architecture)
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
from sklearn.metrics import mean_absolute_error
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
monthlychoiceall = ['JJA']

reg_name = 'US'
lat_bounds,lon_bounds = UT.regions(reg_name)

###############################################################################
###############################################################################
###############################################################################
### Decide whether to limit the number of ensembles for training/testing/val
if any([datasetsingle[0]=='LENS1_LOWS',datasetsingle[0]=='LENS2_LOWS',
        datasetsingle[0]=='MIROC6_LE_LOWS']):
    sliceEns_only = True
    sliceEns_only_N = 30
else:
    sliceEns_only = False
    sliceEns_only_N = np.nan

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
### Looping hyperparameters
COUNTER = 20
hiddenall = [[10],[20],[10,10],[20,20],[50,50],[100,100],[10,10,10],[20,20,20]]
ridgePenaltyall = [0.001,0.01,0.1,0.5,1,5]

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Looping through each month/season and loop through simulation        
for ms in range(len(monthlychoiceall)):
    for sis,singlesimulation in enumerate(datasetsingle):
        meanslope_trainrp = []
        meanslope_testrp = []
        meanslope_valrp = []
        
        meanr_trainrp = []
        meanr_testrp = []
        meanr_valrp = []
        
        mae_testarp = []
        mae_testbrp = []
        mae_valarp = []
        mae_valbrp = []
        
        mae_testallrp = []
        mae_valallrp = []
        for hhh in range(len(hiddenall)):
            meanslope_trainhhh = []
            meanslope_testhhh = []
            meanslope_valhhh = []
            
            meanr_trainhhh = []
            meanr_testhhh = []
            meanr_valhhh = []
            
            mae_testahhh = []
            mae_testbhhh = []
            mae_valahhh = []
            mae_valbhhh = []
            
            mae_testallhhh = []
            mae_valallhhh = []
            for rp in range(len(ridgePenaltyall)):
                meanslope_trainCOUNT = []
                meanslope_testCOUNT = []
                meanslope_valCOUNT = []
                
                meanr_trainCOUNT = []
                meanr_testCOUNT = []
                meanr_valCOUNT = []
                
                mae_testaCOUNT = []
                mae_testbCOUNT = []
                mae_valaCOUNT = []
                mae_valbCOUNT = []
                
                mae_testallCOUNT = []
                mae_valallCOUNT = []
                for count in range(COUNTER):
        
                    #######################################################################
                    ### Parameters
                    debug = True
                    NNType = 'ANN'
                    annType = 'class'
                    classChunkHalf = 5
                    classChunk = 10
                    biasBool = False
                    hiddensList = [hiddenall[hhh]]
                    ridge_penalty = [ridgePenaltyall[rp]]
                    actFun = 'relu'
                    iterations = [1000]
                    
                    ### Select month
                    monthlychoice = monthlychoiceall[ms]
            
                    ### Select years of data and emissions scenario
                    if datasetsingle[sis] == 'SPEAR_MED':
                        scenario = 'SSP585'
                    elif datasetsingle[sis] == 'SPEAR_MED_FA':
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
                    elif datasetsingle[sis] == 'MIROC6_LE_LOWS':
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
                    elif dataset_obs == 'NClimGrid_HIGHS':
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
                    elif dataset_obs == 'ERA5_HIGHS':
                        yearsobs = np.arange(1979,2021+1,1)
                        obsyearstart = yearsobs[0]
                
            ###############################################################################
            ###############################################################################
            ###############################################################################
                    ### ANN preliminaries
                    directoryscores = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/Loop/'
                    
                    ### Define primary dataset to use
                    dataset = singlesimulation
                    modelType = 'TrainedOn' + dataset
                    
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
            
                        # ### Graph each epoc
                        # plt.figure(figsize=(15,5))
                        # plt.subplot(1,2,1)
                        # plt.plot(history.history['loss'],label = 'training')
                        # plt.plot(history.history['val_loss'], label = 'validation')
                        # plt.title('loss')
                        # plt.xlabel('epoch')
                        # plt.legend()
            
                        # plt.subplot(1,2,2)
                        # plt.plot(convert_fuzzyDecade_toYear(Ytrain,startYear,
                        #                                     classChunk,yearsall[sis]),
                        #           convert_fuzzyDecade_toYear(model.predict(Xtrain),
                        #                                     startYear,
                        #                                     classChunk,yearsall[sis]),'o',
                        #                                       color='gray')
                        # plt.plot(convert_fuzzyDecade_toYear(Ytest,startYear,
                        #                                     classChunk,yearsall[sis]),
                        #           convert_fuzzyDecade_toYear(model.predict(Xtest),
                        #                                     startYear,
                        #                                     classChunk,yearsall[sis]),'x', 
                        #                                     color='red')
                        # plt.plot([startYear,yearsall[sis].max()],[startYear,yearsall[sis].max()],'--k')
                        # plt.yticks(np.arange(yearsall[sis].min(),yearsall[sis].max(),10))
                        # plt.xticks(np.arange(yearsall[sis].min(),yearsall[sis].max(),10))
                        
                        # plt.grid(True)
                        # plt.show()
            
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
                        
            ###############################################################################            
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
                    savename = modelType+'_Vari-'+variq+'_Obs-' + dataset_obs + '_Months-' + monthlychoice + '_L2-'+ str(ridge_penalty[0])+ '_LR-' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
                    savenameModelTestTrain = modelType+'_'+variq+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)
            
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
            
                    #######################################################
                    #######################################################
                    #######################################################
                    ### Try considering order of years as parameter
                    train_output_rs = YpredTrain.reshape(len(trainIndices),len(yearsall[sis]))
                    test_output_rs = YpredTest.reshape(len(testIndices),len(yearsall[sis]))
                    val_output_rs = YpredVal.reshape(len(valIndices),len(yearsall[sis]))
                    xs_train = (np.arange(np.shape(train_output_rs)[1]) + yearsall[sis].min())
                    xs_test = (np.arange(np.shape(test_output_rs)[1]) + yearsall[sis].min())
                    xs_val = (np.arange(np.shape(val_output_rs)[1]) + yearsall[sis].min())
                    
                    slopetrain = np.empty((train_output_rs.shape[0]))
                    r_valuetrain = np.empty((train_output_rs.shape[0]))
                    for talin in range(train_output_rs.shape[0]):
                        slopetrain[talin],intercepttrain,r_valuetrain[talin],p_valuetrain,std_errtrain = sts.linregress(xs_train,train_output_rs[talin,:])
 
                    slopetest = np.empty((test_output_rs.shape[0]))
                    r_valuetest = np.empty((test_output_rs.shape[0]))
                    for telin in range(test_output_rs.shape[0]):
                        slopetest[telin],intercepttest,r_valuetest[telin],p_valuetest,std_errtest = sts.linregress(xs_test,test_output_rs[telin,:])
                        
                    slopeval = np.empty((val_output_rs.shape[0]))
                    r_valueval = np.empty((val_output_rs.shape[0]))
                    for tvlin in range(val_output_rs.shape[0]):
                        slopeval[tvlin],interceptval,r_valueval[tvlin],p_valueval,std_errval = sts.linregress(xs_val,val_output_rs[tvlin,:])

                    #######################################################
                    #######################################################
                    #######################################################
                    ### Try considering RMSE before/after 1990  
                    yrerror = 1990
                       
                    iyearstesta = np.where(Ytest<yrerror)[0]
                    rma_test = np.round(dSS.rmse(YpredTest[iyearstesta,],Ytest[iyearstesta,0]),decimals=1)
                    
                    iyearstestb = np.where(Ytest>=yrerror)[0]
                    rmb_test = np.round(dSS.rmse(YpredTest[iyearstestb,],Ytest[iyearstestb,0]),decimals=1)
                    
                    iyearsvala = np.where(Yval<yrerror)[0]
                    rma_val = np.round(dSS.rmse(YpredVal[iyearsvala,],Yval[iyearsvala,0]),decimals=1)
                    
                    iyearsvalb = np.where(Yval>=yrerror)[0]
                    rmb_val = np.round(dSS.rmse(YpredVal[iyearsvalb,],Yval[iyearsvalb,0]),decimals=1)
                    
                    #######################################################
                    #######################################################
                    #######################################################
                    ### Try considering RMSE for all years 
                    rmall_test = np.round(dSS.rmse(YpredTest[:],Ytest[:,0]),decimals=1)
                    rmall_val = np.round(dSS.rmse(YpredVal[:],Yval[:,0]),decimals=1)
                    
                    #######################################################
                    #######################################################
                    #######################################################
                    ### Try considering RMSE before/after 1990   
                    mae_testa = np.round(mean_absolute_error(YpredTest[iyearstesta],Ytest[iyearstesta,0]),1)
                    mae_testb = np.round(mean_absolute_error(YpredTest[iyearstestb],Ytest[iyearstestb,0]),1)
                    
                    mae_vala = np.round(mean_absolute_error(YpredVal[iyearsvala],Yval[iyearsvala,0]),1)
                    mae_valb = np.round(mean_absolute_error(YpredVal[iyearsvalb],Yval[iyearsvalb,0]),1)
                    
                    #######################################################
                    #######################################################
                    #######################################################
                    ### Try considering MAE for all years 
                    mae_testall = np.round(mean_absolute_error(YpredTest[:],Ytest[:,0]),1)
                    mae_valall = np.round(mean_absolute_error(YpredVal[:],Yval[:,0]),1)
                    
                    #######################################################
                    #######################################################
                    #######################################################
                    ### Stats on testing
                    meanslope_train = np.round(np.nanmean(slopetrain),3)
                    meanslope_test = np.round(np.nanmean(slopetest),3)
                    meanslope_val = np.round(np.nanmean(slopeval),3)
                    
                    meanr_train = np.round(np.nanmean(r_valuetrain),3)
                    meanr_test = np.round(np.nanmean(r_valuetest),3)
                    meanr_val = np.round(np.nanmean(r_valueval),3)
                    
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
                    
                    print('\n\n\n<<<<<<<<< [%s] COUNTER for hiddens=%s with l2=%s  >>>>>>>>>>\n\n\n' % (count+1,hiddensList[0],ridge_penalty))

                    #######################################################
                    #######################################################
                    #######################################################
                    ### Save everything
                    print(np.round(np.corrcoef(yearsobs,YpredObs)[0,1],2),'= pearson correlation for obs')
                    print(np.round(sts.spearmanr(yearsobs,YpredObs)[0],2),'= spearman correlation for obs')
                    
                    meanslope_trainCOUNT.append(meanslope_train)
                    meanslope_testCOUNT.append(meanslope_test)
                    meanslope_valCOUNT.append(meanslope_val)
                    
                    meanr_trainCOUNT.append(meanr_train)
                    meanr_testCOUNT.append(meanr_test)
                    meanr_valCOUNT.append(meanr_val)
                    
                    mae_testaCOUNT.append(mae_testa)
                    mae_testbCOUNT.append(mae_testb)
                    mae_valaCOUNT.append(mae_vala)
                    mae_valbCOUNT.append(mae_valb)
                    
                    mae_testallCOUNT.append(mae_testall)
                    mae_valallCOUNT.append(mae_valall)
                    
                #######################################################    
                meanslope_trainhhh.append(meanslope_trainCOUNT)
                meanslope_testhhh.append(meanslope_testCOUNT)
                meanslope_valhhh.append(meanslope_valCOUNT)
                
                meanr_trainhhh.append(meanr_trainCOUNT)
                meanr_testhhh.append(meanr_testCOUNT)
                meanr_valhhh.append(meanr_valCOUNT)
                
                mae_testahhh.append(mae_testaCOUNT)
                mae_testbhhh.append(mae_testbCOUNT)
                mae_valahhh.append(mae_valaCOUNT)
                mae_valbhhh.append(mae_valbCOUNT)
                
                mae_testallhhh.append(mae_testallCOUNT)
                mae_valallhhh.append(mae_valallCOUNT)
            #######################################################    
            meanslope_trainrp.append(meanslope_trainhhh)
            meanslope_testrp.append(meanslope_testhhh)
            meanslope_valrp.append(meanslope_valhhh)
            
            meanr_trainrp.append(meanr_trainhhh)
            meanr_testrp.append(meanr_testhhh)
            meanr_valrp.append(meanr_valhhh)
            
            mae_testarp.append(mae_testahhh)
            mae_testbrp.append(mae_testbhhh)
            mae_valarp.append(mae_valahhh)
            mae_valbrp.append(mae_valbhhh)
            
            mae_testallrp.append(mae_testallhhh)
            mae_valallrp.append(mae_valallhhh)

        #######################################################
        np.savez(directoryscores + 'ANN_PredictTheYear_v2-LoopHyperparameters_%s_%s_%s-%s_%s.npz' % (variq,monthlychoice,dataset,
                                                                                                  dataset_obs,reg_name),
                 lats = lats,
                 lons = lons,
                 savename = savename,
                 meanslope_train = meanslope_trainrp,
                 meanslope_test = meanslope_testrp,
                 meanslope_val = meanslope_valrp,
                 meanr_train = meanr_trainrp,
                 meanr_test = meanr_testrp,
                 meanr_val = meanr_valrp,
                 mae_testa = mae_testarp,
                 mae_testb = mae_testbrp,
                 mae_vala = mae_valarp,
                 mae_valb = mae_valbrp,
                 mae_testall = mae_testallrp,
                 mae_valall = mae_valallrp,
                 COUNTER = COUNTER,
                 hiddenall = hiddenall,
                 ridgePenaltyall = ridgePenaltyall,
                 yearsobs = yearsobs,
                 yearmodel = yearsall[sis],
                 monthlychoice = monthlychoice,
                 dataset = dataset,
                 dataset_obs = dataset_obs)
