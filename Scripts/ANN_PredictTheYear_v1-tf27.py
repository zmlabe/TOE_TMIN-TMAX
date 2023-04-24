"""
Train ANN to predict the year for ToE in the CONUS

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 7 September 2022
Version   : v1-tf27 -- this version is compatible with tensorflow v2.7
"""

### Import packages
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Remove GPU error message
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
import innvestigate
tf.compat.v1.disable_eager_execution() # bug fix with innvestigate v2.0.1
import random
import scipy.stats as sts
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import calc_SegmentData as dSEG
import calc_LRPclass_tf27 as LRP
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
datasetsingle = ['SPEAR_MED','SPEAR_MED_NATURAL','SPEAR_MED_Scenario','SPEAR_MED_NOAER']
yearsall = [np.arange(1921,2100+1,1),np.arange(1921,2100+1,1),np.arange(1921,2100+1,1),np.arange(1921,2020+1,1)]
dataset_obs = 'NClimGrid_MEDS'
resolution = 'MEDS'
variq = 'TMAX'
monthlychoiceall = ['JFM','AMJ','JAS','OND','annual','JJA']
reg_name = 'Ce_US'
lat_bounds,lon_bounds = UT.regions(reg_name)
segment_data_factorq = [0.8,0.8,0.8,0.7]

datasetsingle = ['SPEAR_MED']
yearsall = [np.arange(1921,2100+1,1)]
monthlychoiceall = ['JJA']

### Masking and preprocessing arguments
rm_annual_mean = False
rm_merid_mean = False
rm_ensemble_mean = False
land_only = False
ocean_only = False
CONUS_only = True

### Select whether to standardize over a baseline
baselineSTD = True
yrminb = 1981
yrmaxb = 2010
baselineSTDyrs = np.arange(yrminb,yrmaxb+1,1)

### Parameters
if reg_name == 'US':
    debug = True
    NNType = 'ANN'
    classChunkHalf = 5
    classChunk = 10
    biasBool = False
    hiddensList = [[20,20]]
    ridge_penalty = [0.01]
    actFun = 'relu'
    iterations = [500]
elif any([reg_name=='W_US',reg_name=='Ce_US',reg_name=='E_US']):
    debug = True
    NNType = 'ANN'
    classChunkHalf = 5
    classChunk = 10
    biasBool = False
    hiddensList = [[20,20]]
    ridge_penalty = [0.001]
    actFun = 'relu'
    iterations = [500]

###############################################################################
###############################################################################
lrpRule1 = 'z'
lrpRule2 = 'epsilon'
lrpRule3 = 'integratedgradient'
normLRP = True
###############################################################################
###############################################################################

for ms in range(len(monthlychoiceall)):
    monthlychoice = monthlychoiceall[ms]
    
    for sis,singlesimulation in enumerate(datasetsingle):
        
        ### Select years of data
        if datasetsingle[sis] == 'SPEAR_MED':
            scenario = 'SSP585'
        elif datasetsingle[sis] == 'SPEAR_MED_NOAER':
            scenario = 'SSP585'
        elif datasetsingle[sis] == 'SPEAR_LOW':
            scenario = 'SSP585'
        elif datasetsingle[sis] == 'SPEAR_MED_NATURAL':
            scenario = 'NATURAL'
        elif datasetsingle[sis] == 'SPEAR_MED_Scenario':
            scenario = 'SSP245'
            
        if dataset_obs == 'NClimGrid_MEDS':
            yearsobs = np.arange(1921,2021+1,1)
            obsyearstart = yearsobs[0]
        elif dataset_obs == 'NClimGrid_LOWS':
            yearsobs = np.arange(1921,2021+1,1)
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
            
        ### Need to change proportion for fewer ensembles in SPEAR_MED_NOAER
        segment_data_factor = segment_data_factorq[sis]
    
###############################################################################
###############################################################################
###############################################################################
        ### ANN preliminaries
        directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/Figures/PredictionsYear/'
        dirname = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/savedModels/'
        directoryoutput = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/LRP/Regions/'
        
        ### Define primary dataset to use
        dataset = singlesimulation
        modelType = 'TrainedOn' + dataset
        
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
                    
            fig = plt.figure()
            ax = plt.subplot(111)
            
            adjust_spines(ax, ['left', 'bottom'])
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('dimgrey')
            ax.spines['bottom'].set_color('dimgrey')
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
        
            train_output_rs = YpredTrain.reshape(len(trainIndices),
                                              len(years))
            test_output_rs = YpredTest.reshape(len(testIndices),
                                          len(years))
        
            xs_test = (np.arange(np.shape(test_output_rs)[1]) + yearsall[sis].min())
        
            for i in range(0,train_output_rs.shape[0]):
                if i == train_output_rs.shape[0]-1:
                    plt.plot(xs_test,train_output_rs[i,:],'o',
                                markersize=4,color='lightgray',clip_on=False,
                                alpha=0.4,markeredgecolor='k',markeredgewidth=0.4,
                                label=r'\textbf{%s - Training Data}' % singlesimulation)
                else:
                    plt.plot(xs_test,train_output_rs[i,:],'o',
                                markersize=4,color='lightgray',clip_on=False,
                                alpha=0.4,markeredgecolor='k',markeredgewidth=0.4)
            for i in range(0,test_output_rs.shape[0]):
                if i == test_output_rs.shape[0]-1:
                    plt.plot(xs_test,test_output_rs[i,:],'o',
                            markersize=4,color='crimson',clip_on=False,alpha=0.3,
                            markeredgecolor='crimson',markeredgewidth=0.4,
                            label=r'\textbf{%s - Testing Data}' % singlesimulation)
                else:
                    plt.plot(xs_test,test_output_rs[i,:],'o',
                            markersize=4,color='crimson',clip_on=False,alpha=0.3,
                            markeredgecolor='crimson',markeredgewidth=0.4)
            
            if rm_ensemble_mean == False:
                iy = np.where(yearsobs>=obsyearstart)[0]
                plt.plot(yearsobs[iy],YpredObs[iy],'x',color='deepskyblue',
                          label=r'\textbf{%s}' % dataset_obs,clip_on=False)
            
            plt.xlabel(r'\textbf{ACTUAL YEAR - %s - %s}' % (monthlychoice,reg_name),fontsize=10,color='dimgrey')
            plt.ylabel(r'\textbf{PREDICTED YEAR}',fontsize=10,color='dimgrey')
            plt.plot(np.arange(yearsall[sis].min(),yearsall[sis].max()+1,1),np.arange(yearsall[sis].min(),yearsall[sis].max()+1,1),'-',
                      color='black',linewidth=2,clip_on=False)
            
            plt.xticks(np.arange(yearsall[sis].min(),2101,20),map(str,np.arange(yearsall[sis].min(),2101,20)),size=6)
            plt.yticks(np.arange(yearsall[sis].min(),2101,20),map(str,np.arange(yearsall[sis].min(),2101,20)),size=6)
            plt.xlim([yearsall[sis].min(),yearsall[sis].max()+1])   
            plt.ylim([yearsall[sis].min(),yearsall[sis].max()+1])
            
            plt.title(r'\textbf{[ %s ] $\bf{\longrightarrow}$ RMSE Train = %s; RMSE Test = %s}' % (variq,np.round(dSS.rmse(YpredTrain[:,],
                                                                            Ytrain[:,0]),1),np.round(dSS.rmse(YpredTest[:,],
                                                                                                                  Ytest[:,0]),
                                                                                                                  decimals=1)),
                                                                                                              color='k',
                                                                                                              fontsize=15)
            
            iyears = np.where(Ytest<1960)[0]
            plt.text(yearsall[sis].max()+1,yearsall[sis].min()+5, r'\textbf{Test RMSE before 1960 = %s}' % (np.round(dSS.rmse(YpredTest[iyears,],
                                                                                Ytest[iyears,0]),
                                                                          decimals=1)),
                      fontsize=5,ha='right')
            
            iyears = np.where(Ytest>=1960)[0]
            plt.text(yearsall[sis].max()+1,yearsall[sis].min(), r'\textbf{Test RMSE after 1960 = %s}' % (np.round(dSS.rmse(YpredTest[iyears,],
                                                                                  Ytest[iyears,0]),
                                                                              decimals=1)),
                      fontsize=5,ha='right')
            
            plt.legend(shadow=False,fontsize=7,loc='upper left',
                          bbox_to_anchor=(-0.01,1),fancybox=True,ncol=1,frameon=False,
                          handlelength=1,handletextpad=0.5)
            savefigName = modelType+'_'+variq+'_scatterPred_'+savename 
    
            plt.savefig(directoryfigure+savefigName+'_%s_land%s_ocean%s_20ens.png' % (monthlychoice,land_only,ocean_only),
                        dpi=300)      
            print(np.round(np.corrcoef(yearsobs,YpredObs)[0,1],2))
            return 
        
###############################################################################
###############################################################################
###############################################################################
        ### Neural Network Creation & Training        
        class TimeHistory(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.times = []
            def on_epoch_begin(self, epoch, logs={}):
                self.epoch_time_start = time.time()
            def on_epoch_end(self, epoch, logs={}):
                self.times.append(time.time() - self.epoch_time_start)
        
        def defineNN(hidden, input_shape, output_shape, ridgePenalty):        
           
            ### Begin Model
            model = Sequential()
            
            ### Initialize first layer
            if hidden[0]==0:
                ### Model is linear
                model.add(Dense(1,input_shape=(input_shape,),
                                activation='linear',use_bias=True,
                                kernel_regularizer=regularizers.l1_l2(l1=0.00,l2=ridgePenalty),
                                bias_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed)))
                print('\nTHIS IS A LINEAR NN!\n')
            else:
                ### Model is a single node with activation function
                model.add(Dense(hidden[0],input_shape=(input_shape,),
                                activation=actFun, use_bias=True,
                                kernel_regularizer=regularizers.l1_l2(l1=0.00,l2=ridgePenalty),
                                bias_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed)))
        
                ### Initialize other layers
                for layer in hidden[1:]:
                    model.add(Dense(layer,activation=actFun,
                                    use_bias=True,
                                    kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.00),
                                    bias_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed)))
                    
                print('\nTHIS IS A ANN!\n')
        
            #### Initialize output layer
            model.add(Dense(output_shape,activation=None,use_bias=True,
                            kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.0),
                            bias_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_network_seed)))
        
            ### Add softmax layer at the end
            model.add(Activation('softmax'))
            
            return model
        
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
            global nnet,random_network_seed
          
            for niter in iterations:
                for penalty in ridge_penalty:
                    for hidden in hiddens:
                        
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
                        model, history = trainNN(model,Xtrain,Ytrain,Xval,Yval,niter,verbose=1)
        
                        ### Graph each epoc
                        plt.figure(figsize=(15,5))
                        plt.subplot(1,2,1)
                        plt.plot(history.history['loss'],label = 'training')
                        plt.plot(history.history['val_loss'], label = 'validation')
                        plt.title('loss')
                        plt.xlabel('epoch')
                        plt.legend()
    
                        plt.subplot(1,2,2)
                        
                        plt.plot(convert_fuzzyDecade_toYear(Ytrain,startYear,
                                                            classChunk,yearsall[sis]),
                                  convert_fuzzyDecade_toYear(model.predict(Xtrain),
                                                            startYear,
                                                            classChunk,yearsall[sis]),'o',
                                                              color='gray')
                        plt.plot(convert_fuzzyDecade_toYear(Ytest,startYear,
                                                            classChunk,yearsall[sis]),
                                  convert_fuzzyDecade_toYear(model.predict(Xtest),
                                                            startYear,
                                                            classChunk,yearsall[sis]),'x', 
                                                            color='red')
                        plt.plot([startYear,yearsall[sis].max()],[startYear,yearsall[sis].max()],'--k')
                        plt.yticks(np.arange(yearsall[sis].min(),yearsall[sis].max(),10))
                        plt.xticks(np.arange(yearsall[sis].min(),yearsall[sis].max(),10))
                        
                        plt.grid(True)
                        plt.show()
        
                        #'unlock' the random seed
                        np.random.seed(None)
                        random.seed(None)
                        tf.random.set_seed(None)
          
            return model
        
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
        ### Clear session and prepare to read in data
        tf.keras.backend.clear_session()
        
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
            data, data_obs = dSS.mask_CONUS(data,data_obs,'MEDS',lat_bounds,lon_bounds)
            print('*Removed everything by CONUS*')

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
    
        ### Create and train network
        model = test_train_loopClass(Xtrain,YtrainClassMulti,
                                     Xtest,YtestClassMulti,
                                     Xval,YvalClassMulti,
                                     iterations,
                                     ridge_penalty,
                                     hiddensList)
        model.summary()  
        
        ################################################################################################################################################                
        ### Save models
        savename = modelType+'_Vari-'+variq+'_Obs-' + dataset_obs + '_Months-' + monthlychoice + '_L2-'+ str(ridge_penalty[0])+ '_LR-' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
        savenameModelTestTrain = modelType+'_'+variq+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)
    
        # model.save(dirname + savename + '.h5')
        # np.savez(dirname + savenameModelTestTrain + '.npz',trainModels=trainIndices,testModels=testIndices,Xtrain=Xtrain,Ytrain=Ytrain,Xtest=Xtest,Ytest=Ytest,Xmean=Xmean,Xstd=Xstd,lats=lats,lons=lons)
        print('Saving ------->' + savename)
        
        ###############################################################
        ### Assess observations
        Xobs = data_obs.reshape(data_obs.shape[0],data_obs.shape[1]*data_obs.shape[2])
    
        annType = 'class'
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
            
        ### Create final plot
        beginFinalPlot(YpredTrain,YpredTest,Ytrain,Ytest,testIndices,years,yearsobs,YpredObs)

        ### Save prediction output for predict the year
        directorypredictions = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/Regions/'
        np.savetxt(directorypredictions + savename + 'OBSERVATION_PREDICTIONS.txt',YpredObs)
        np.savetxt(directorypredictions + savename + 'TRAIN_PREDICTIONS.txt',YpredTrain)
        np.savetxt(directorypredictions + savename + 'VAL_PREDICTIONS.txt',YpredVal)
        np.savetxt(directorypredictions + savename + 'TEST_PREDICTIONS.txt',YpredTest)
        
        np.savetxt(directorypredictions + savename + 'TrainIndices_ENSEMBLES.txt',trainIndices)
        np.savetxt(directorypredictions + savename + 'ValIndices_ENSEMBLES.txt',valIndices)
        np.savetxt(directorypredictions + savename + 'TestIndices_ENSEMBLES.txt',testIndices)
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        ### Visualizing through LRP
        numLats = lats.shape[0]
        numLons = lons.shape[0]  
        numDim = 3
        num_of_class = len(yearsall[sis])
      
        lrpallz = LRP.calc_LRPModel(model,np.append(XtrainS,XtestS,axis=0),
                                                np.append(Ytrain,Ytest,axis=0),
                                                biasBool,annType,num_of_class,
                                                yearsall,lrpRule1,normLRP,
                                                numLats,numLons,numDim,
                                                classChunk,startYear,yearsall[sis])
        meanlrpz = np.nanmean(lrpallz,axis=0)
        fig=plt.figure()
        plt.contourf(meanlrpz,np.arange(0,0.21,0.01),cmap=cmocean.cm.thermal)
        
        ### For training data only
        lrptrainz = LRP.calc_LRPModel(model,XtrainS,Ytrain,biasBool,
                                                annType,num_of_class,
                                                yearsall,lrpRule1,normLRP,
                                                numLats,numLons,numDim,
                                                classChunk,startYear,yearsall[sis])
        
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
    ### LRP epsilon rule
        lrpalle = LRP.calc_LRPModel(model,np.append(XtrainS,XtestS,axis=0),
                                                np.append(Ytrain,Ytest,axis=0),
                                                biasBool,annType,num_of_class,
                                                yearsall,lrpRule2,normLRP,
                                                numLats,numLons,numDim,
                                                classChunk,startYear,yearsall[sis])
        meanlrpe = np.nanmean(lrpalle,axis=0)
        fig=plt.figure()
        plt.contourf(meanlrpe,np.arange(0,0.21,0.01),cmap=cmocean.cm.thermal)
        
        ### For training data only
        lrptraine = LRP.calc_LRPModel(model,XtrainS,Ytrain,biasBool,
                                                annType,num_of_class,
                                                yearsall,lrpRule2,normLRP,
                                                numLats,numLons,numDim,
                                                classChunk,startYear,yearsall[sis])
        
        ### For training data only
        lrpteste = LRP.calc_LRPModel(model,XtestS,Ytest,biasBool,
                                                annType,num_of_class,
                                                yearsall,lrpRule2,normLRP,
                                                numLats,numLons,numDim,
                                                classChunk,startYear,yearsall[sis])
        
        
        ### For observations data only
        lrpobservationse = LRP.calc_LRPObs(model,XobsS,biasBool,annType,
                                            num_of_class,yearsall,lrpRule2,
                                            normLRP,numLats,numLons,numDim,yearsall[sis])
        
    ############################################################################### 
    ############################################################################### 
    ############################################################################### 
    ###  Integrated Gradient
        lrpallig = LRP.calc_LRPModel(model,np.append(XtrainS,XtestS,axis=0),
                                                np.append(Ytrain,Ytest,axis=0),
                                                biasBool,annType,num_of_class,
                                                yearsall,lrpRule3,normLRP,
                                                numLats,numLons,numDim,
                                                classChunk,startYear,yearsall[sis])
        meanlrpig = np.nanmean(lrpallig,axis=0)
        fig=plt.figure()
        plt.contourf(meanlrpig,np.arange(0,0.21,0.01),cmap=cmocean.cm.thermal)
        
        ### For training data only
        lrptrainig = LRP.calc_LRPModel(model,XtrainS,Ytrain,biasBool,
                                                annType,num_of_class,
                                                yearsall,lrpRule3,normLRP,
                                                numLats,numLons,numDim,
                                                classChunk,startYear,yearsall[sis])
        
        ### For training data only
        lrptestig = LRP.calc_LRPModel(model,XtestS,Ytest,biasBool,
                                                annType,num_of_class,
                                                yearsall,lrpRule3,normLRP,
                                                numLats,numLons,numDim,
                                                classChunk,startYear,yearsall[sis])
        
        
        ### For observations data only
        lrpobservationsig = LRP.calc_LRPObs(model,XobsS,biasBool,annType,
                                            num_of_class,yearsall,lrpRule3,
                                            normLRP,numLats,numLons,numDim,yearsall[sis])
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        def netcdfLRPz(lats,lons,var,directory,typemodel,trainingdata,variq,savename):
            print('\n>>> Using netcdfLRP-z function!')
            
            from netCDF4 import Dataset
            import numpy as np
            
            name = 'LRPMap_Z_' + typemodel + '_' + variq + '_' + savename + '.nc'
            filename = directory + name
            ncfile = Dataset(filename,'w',format='NETCDF4')
            ncfile.description = 'LRP maps for using selected seed' 
            
            ### Dimensions
            ncfile.createDimension('years',var.shape[0])
            ncfile.createDimension('lat',var.shape[1])
            ncfile.createDimension('lon',var.shape[2])
            
            ### Variables
            years = ncfile.createVariable('years','f4',('years'))
            latitude = ncfile.createVariable('lat','f4',('lat'))
            longitude = ncfile.createVariable('lon','f4',('lon'))
            varns = ncfile.createVariable('LRP','f4',('years','lat','lon'))
            
            ### Units
            varns.units = 'unitless relevance'
            ncfile.title = 'LRP relevance'
            ncfile.instituion = 'Colorado State University'
            ncfile.references = 'Barnes et al. [2020]'
            
            ### Data
            years[:] = np.arange(var.shape[0])
            latitude[:] = lats
            longitude[:] = lons
            varns[:] = var
            
            ncfile.close()
            print('*Completed: Created netCDF4 File!')
        
        netcdfLRPz(lats,lons,lrpallz,directoryoutput,'AllData',dataset,variq,savename)
        netcdfLRPz(lats,lons,lrptrainz,directoryoutput,'Training',dataset,variq,savename)
        netcdfLRPz(lats,lons,lrptestz,directoryoutput,'Testing',dataset,variq,savename)
        netcdfLRPz(lats,lons,lrpobservationsz,directoryoutput,'Obs',dataset,variq,savename)
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        def netcdfLRPe(lats,lons,var,directory,typemodel,trainingdata,variq,savename):
            print('\n>>> Using netcdfLRP-e function!')
            
            from netCDF4 import Dataset
            import numpy as np
            
            name = 'LRPMap_E_' + typemodel + '_' + variq + '_' + savename + '.nc'
            filename = directory + name
            ncfile = Dataset(filename,'w',format='NETCDF4')
            ncfile.description = 'LRP maps for using selected seed' 
            
            ### Dimensions
            ncfile.createDimension('years',var.shape[0])
            ncfile.createDimension('lat',var.shape[1])
            ncfile.createDimension('lon',var.shape[2])
            
            ### Variables
            years = ncfile.createVariable('years','f4',('years'))
            latitude = ncfile.createVariable('lat','f4',('lat'))
            longitude = ncfile.createVariable('lon','f4',('lon'))
            varns = ncfile.createVariable('LRP','f4',('years','lat','lon'))
            
            ### Units
            varns.units = 'unitless relevance'
            ncfile.title = 'LRP relevance'
            ncfile.instituion = 'Colorado State University'
            ncfile.references = 'Barnes et al. [2020]'
            
            ### Data
            years[:] = np.arange(var.shape[0])
            latitude[:] = lats
            longitude[:] = lons
            varns[:] = var
            
            ncfile.close()
            print('*Completed: Created netCDF4 File!')
        
        netcdfLRPe(lats,lons,lrpalle,directoryoutput,'AllData',dataset,variq,savename)
        netcdfLRPe(lats,lons,lrptraine,directoryoutput,'Training',dataset,variq,savename)
        netcdfLRPe(lats,lons,lrpteste,directoryoutput,'Testing',dataset,variq,savename)
        netcdfLRPe(lats,lons,lrpobservationse,directoryoutput,'Obs',dataset,variq,savename)
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        def netcdfLRPig(lats,lons,var,directory,typemodel,trainingdata,variq,savename):
            print('\n>>> Using netcdfLRP-e function!')
            
            from netCDF4 import Dataset
            import numpy as np
            
            name = 'LRPMap_IG_' + typemodel + '_' + variq + '_' + savename + '.nc'
            filename = directory + name
            ncfile = Dataset(filename,'w',format='NETCDF4')
            ncfile.description = 'LRP maps for using selected seed' 
            
            ### Dimensions
            ncfile.createDimension('years',var.shape[0])
            ncfile.createDimension('lat',var.shape[1])
            ncfile.createDimension('lon',var.shape[2])
            
            ### Variables
            years = ncfile.createVariable('years','f4',('years'))
            latitude = ncfile.createVariable('lat','f4',('lat'))
            longitude = ncfile.createVariable('lon','f4',('lon'))
            varns = ncfile.createVariable('LRP','f4',('years','lat','lon'))
            
            ### Units
            varns.units = 'unitless relevance'
            ncfile.title = 'LRP relevance'
            ncfile.instituion = 'Colorado State University'
            ncfile.references = 'Barnes et al. [2020]'
            
            ### Data
            years[:] = np.arange(var.shape[0])
            latitude[:] = lats
            longitude[:] = lons
            varns[:] = var
            
            ncfile.close()
            print('*Completed: Created netCDF4 File!')
    
        netcdfLRPig(lats,lons,lrpallig,directoryoutput,'AllData',dataset,variq,savename)
        netcdfLRPig(lats,lons,lrptrainig,directoryoutput,'Training',dataset,variq,savename)
        netcdfLRPig(lats,lons,lrptestig,directoryoutput,'Testing',dataset,variq,savename)
        netcdfLRPig(lats,lons,lrpobservationsig,directoryoutput,'Obs',dataset,variq,savename)
     
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
        
        ### Variables for plotting
        lons2,lats2 = np.meshgrid(lons,lats) 
        observations = data_obs
        modeldata = data
        modeldatamean = np.nanmean(modeldata,axis=0)
        
        spatialmean_obs = UT.calc_weightedAve(observations,lats2)
        spatialmean_mod = UT.calc_weightedAve(modeldata,lats2)
        spatialmean_modmean = np.nanmean(spatialmean_mod,axis=0)
        plt.figure()
        plt.plot(spatialmean_modmean.transpose())
        plt.plot(spatialmean_obs,color='k',linewidth=2)
