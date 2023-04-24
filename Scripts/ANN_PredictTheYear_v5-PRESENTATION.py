"""
Train ANN to predict the year for ToE in the CONUS

Reference   : Barnes et al. [2020, JAMES] & Labe and Barnes [2021, JAMES]
Author      : Zachary M. Labe
Date        : 18 February 2023
Environment : conda activate env-tf27
Tensorflow  : 2.7 (XAI for v2.0.1)
Version     : 5 (shuffle maps of SPEAR_MED) - prediction figures for talks
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
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

###############################################################################
###############################################################################
###############################################################################
### Testing data for ANN --------------------------------------
datasetsingle = ['SPEAR_MED']
dataset_obs = 'ERA5_MEDS'
yearsall = [np.arange(1921,2100+1,1)]
monthlychoiceall = ['JFM','AMJ','JAS','OND','annual','JJA']
monthlychoiceall = ['JJA']
resolution = 'MEDS'
variq = 'RUNOFF'
reg_name = 'US'
segment_data_factorq = [0.8]
lat_bounds,lon_bounds = UT.regions(reg_name)

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
        
        ### Select years of data and emissions scenario
        if datasetsingle[sis] == 'SPEAR_MED':
            scenario = 'SSP585'
        elif any([datasetsingle[sis] == 'SPEAR_MED_shuffle_space',datasetsingle[sis] == 'SPEAR_MED_shuffle_time']):
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
        elif datasetsingle[sis] == 'SPEAR_HIGH':
            scenario = 'SSP585'
        elif datasetsingle[sis] == 'MIROC6_LE_LOWS':
            scenario = 'SSP585'
        elif datasetsingle[sis] == 'LENS2_LOWS':
            scenario = 'SSP370'
        elif datasetsingle[sis] == 'LENS2_cmip6bb_LOWS':
            scenario = 'SSP370'
        elif datasetsingle[sis] == 'LENS2_smoothed_LOWS':
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
         
        ### Calculate other seasons
        if monthlychoice == 'DJF':
            yearsall = [np.arange(1921,2100+1,1)[:-1]]
            yearsobs = yearsobs[:-1]
    
###############################################################################
###############################################################################
###############################################################################
        ### ANN preliminaries
        directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Dark_Figures/Predictions/'
        
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
        def read_primary_dataset(variq,datasetname,monthlychoice,scenario,lat_bounds,lon_bounds):
            
            if any([datasetname == 'SPEAR_MED_shuffle_space',datasetname == 'SPEAR_MED_shuffle_time']):
                dataset_pick = 'SPEAR_MED'
            else:
                dataset_pick = datasetname
            
            data,lats,lons = df.readFiles(variq,dataset_pick,monthlychoice,scenario)
            datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
            print('\nOur dataset: ',datasetname,' is shaped',data.shape)
            return datar,lats,lons 
        
###############################################################################
###############################################################################
###############################################################################
        ### Plotting functions
        def adjust_spines(ax, spines):
            for loc, spine in ax.spines.items():
                if loc in spines:
                    spine.set_position(('outward', 5))
                else:
                    spine.set_color('none')  
            if 'left' in spines:
                ax.yaxis.set_ticks_position('left')
            else:
                ax.yaxis.set_ticks([])
        
            if 'bottom' in spines:
                ax.xaxis.set_ticks_position('bottom')
            else:
                    ax.xaxis.set_ticks([]) 
            
        def beginFinalPlot(YpredTrain,YpredTest,Ytrain,Ytest,testIndices,years,yearsobs,YpredObs):
            """
            Plot prediction of year
            """
            
            plt.rc('text',usetex=True)
            plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
                    
            fig = plt.figure(figsize=(6,5))
            ax = plt.subplot(111)
            
            adjust_spines(ax, ['left', 'bottom'])
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('darkgrey')
            ax.spines['bottom'].set_color('darkgrey')
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.tick_params('both',length=4,width=2,which='major',color='darkgrey')
            
            plt.plot(np.arange(yearsall[sis].min(),yearsall[sis].max()+1,1),np.arange(yearsall[sis].min(),yearsall[sis].max()+1,1),'-',
                      color='darkgrey',linewidth=2,clip_on=False)
        
            train_output_rs = YpredTrain.reshape(len(trainIndices),
                                              len(years))
            test_output_rs = YpredTest.reshape(len(testIndices),
                                          len(years))
        
            xs_test = (np.arange(np.shape(test_output_rs)[1]) + yearsall[sis].min())
            print(xs_test)
            
            # yeartoeq = np.where((yearsobs >= 1921) & (yearsobs <= 1950))[0]
            # maxobsyr = np.max(YpredObs[yeartoeq])
            # maxobswhere = np.argmax(YpredObs[yeartoeq])
            # print(maxobsyr,maxobswhere,yearsobs[maxobswhere])
            
            # if sts.spearmanr(yearsobs,YpredObs)[1] < 0.05:
            #     plt.fill_between(xs_test,np.min(xs_test),maxobsyr,color='deepskyblue',
            #                      alpha=0.1,edgecolor='none',clip_on=False)

            for i in range(0,test_output_rs.shape[0]):
                if i == test_output_rs.shape[0]-1:
                    plt.plot(xs_test,test_output_rs[i,:],'o',
                            markersize=7,color='crimson',clip_on=False,alpha=0.7,
                            markeredgecolor='crimson',markeredgewidth=0.4,
                            label=r'\textbf{%s}' % singlesimulation)
            
            if rm_ensemble_mean == False:
                iy = np.where(yearsobs>=obsyearstart)[0]
                plt.plot(yearsobs[iy],YpredObs[iy],'x',color='deepskyblue',
                          label=r'\textbf{Observations}',clip_on=False,
                          markersize=7)
                # if sts.spearmanr(yearsobs,YpredObs)[1] < 0.05:
                #     plt.plot(yearsobs[maxobswhere],YpredObs[maxobswhere],'x',color='gold',clip_on=False,
                #               markersize=10)
            
            plt.xlabel(r'\textbf{ACTUAL YEAR -- %s -- %s -- %s}' % (monthlychoice,reg_name,variq),fontsize=12,color='w')
            plt.ylabel(r'\textbf{PREDICTED YEAR}',fontsize=12,color='w')
            
            plt.xticks(np.arange(yearsall[sis].min(),2102,20),map(str,np.arange(yearsall[sis].min(),2102,20)),size=6)
            plt.yticks(np.arange(yearsall[sis].min(),2102,20),map(str,np.arange(yearsall[sis].min(),2102,20)),size=6)
            plt.xlim([yearsall[sis].min(),yearsall[sis].max()+1])   
            plt.ylim([yearsall[sis].min(),yearsall[sis].max()+1])
            
            iyears = np.where(Ytest<1980)[0]
            plt.text(yearsall[sis].max(),yearsall[sis].min()+5, r'\textbf{RMSE before 1980 = %s}' % (np.round(dSS.rmse(YpredTest[iyears,],
                                                                                Ytest[iyears,0]),
                                                                          decimals=1)),fontsize=5,ha='right',color='crimson')
            
            iyears = np.where(Ytest>=1980)[0]
            plt.text(yearsall[sis].max(),yearsall[sis].min()+1, r'\textbf{RMSE after 1980 = %s}' % (np.round(dSS.rmse(YpredTest[iyears,],
                                                                                  Ytest[iyears,0]),
                                                                              decimals=1)),fontsize=5,ha='right',color='crimson')
            
            leg = plt.legend(shadow=False,fontsize=16,loc='upper left',
                          bbox_to_anchor=(-0.01,1),fancybox=True,ncol=1,frameon=False,
                          handlelength=1,handletextpad=0.5)
            for line,text in zip(leg.get_lines(), leg.get_texts()):
                text.set_color(line.get_color())
            
            ### Save the predict the year plot
            savefigName = modelType+'_'+variq+'_scatterPred_'+savename 
            plt.tight_layout()
            plt.savefig(directoryfigure+savefigName+'_%s_land%s_ocean%s_20ens.png' % (monthlychoice,land_only,ocean_only),
                        dpi=300)  
            
            ### Quick statistics for the observations
            print(np.round(np.corrcoef(yearsobs,YpredObs)[0,1],2))
            print(sts.spearmanr(yearsobs,YpredObs))
            return 
        
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
        
        data, data_obs, = data_all, data_obs_all,
        
        ### Possible to do shuffle
        if dataset == 'SPEAR_MED_shuffle_space':
            ### SHUFFLE data array along space dimensions
            datafinal = np.empty((data.shape))
            for ensem in range(data.shape[0]):
                for yyy in range(data.shape[1]):
                    temp = data[ensem,yyy,:,:].ravel()
                    np.random.shuffle(temp)
                    tempq = np.reshape(temp,(data.shape[2],data.shape[3]))
                    datafinal[ensem,yyy,:,:] = tempq
            
            data = datafinal
            print('\n\n<<<<<<< SHUFFLED ARRAY FOR TESTING STATS ON SPACE/MAP DIMENSION!!! >>>>>>\n\n') 
        elif dataset == 'SPEAR_MED_shuffle_time':
            ### SHUFFLE data array along space dimensions
            datafinal = np.empty((data.shape))
            for iii in range(data.shape[2]):
                for jjj in range(data.shape[3]):
                    temp = data[:,:,iii,jjj].ravel()
                    np.random.shuffle(temp)
                    tempq = np.reshape(temp,(data.shape[0],data.shape[1]))
                    datafinal[:,:,iii,jjj] = tempq
                    
            data = datafinal
            print('\n\n<<<<<<< SHUFFLED ARRAY FOR TESTING STATS ON SPACE/MAP DIMENSION!!! >>>>>>\n\n')  
        
        ### Prepare data for preprocessing
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
        random_segment_seed = 71541
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
        random_network_seed = 87750
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
            
        ### Create final plot
        beginFinalPlot(YpredTrain,YpredTest,Ytrain,Ytest,testIndices,years,yearsobs,YpredObs)
