"""
Train ANN to predict the year for ToE in the CONUS

Reference   : Barnes et al. [2020, JAMES] & Labe and Barnes [2021, JAMES]
Author      : Zachary M. Labe
Date        : 28 March 2023
Environment : conda activate env-tf27
Tensorflow  : 2.7 (XAI for v2.0.1)
Version     : 5 (shuffle maps of SPEAR_MED)
"""

###############################################################################
###############################################################################
###############################################################################
### Import packages
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import calc_Utilities as UT
import sys
import scipy.stats as sts
import calc_Stats as dSS

###############################################################################
###############################################################################
###############################################################################
### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

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

letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]

###############################################################################
###############################################################################
###############################################################################
### Testing data for ANN --------------------------------------
resolutionall = ['LOWS','LOWS','LOWS','LOWS','LOWS','LOWS']
datasetsingleall = ['MIROC6_LE_LOWS','LENS1_LOWS','LENS2_LOWS','MIROC6_LE_LOWS30ens','LENS1_LOWS30ens','LENS2_LOWS30ens']
datasetnamelist = ['MIROC6-LE','CESM1-LE','CESM2-LE','MIROC6-LE','CESM1-LE','CESM2-LE']
dataobsname = 'NClimGrid'
yearsall = [np.arange(1921,2100+1,1)]
monthlychoiceall = ['JJA']
monthlychoice = monthlychoiceall[0]
variqall = ['T2M','T2M','T2M','T2M','T2M','T2M']
variqname = ['TAVG','TAVG','TAVG','TAVG','TAVG','TAVG']
reg_name = 'US'
segment_data_factorq = [0.8]
lat_bounds,lon_bounds = UT.regions(reg_name)

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

### Years
yearsmodel = yearsall[0]
yearsobs = np.arange(1921,2022+1,1)

fig = plt.figure(figsize=(10,7))
for lll in range(len(datasetsingleall)):
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Read in data
    directorypredictions = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/Regions/'
    directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/'
    dataset = datasetsingleall[lll]
    variq = variqall[lll]
    resolution = resolutionall[lll]
    
    ### Parameters
    if reg_name == 'US':
        debug = True
        NNType = 'ANN'
        annType = 'class'
        classChunkHalf = 5
        classChunk = 10
        biasBool = True
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
        biasBool = True
        hiddensList = [[100,100]]
        ridge_penalty = [0.001]
        actFun = 'relu'
        iterations = [500]
    else:
        print(ValueError('This is the wrong region name selected!!!'))
        sys.exit()
        
    lr_here = .01
    batch_size = 32 
    random_network_seed = 87750
    random_segment_seed = 71541
    
    dataset_obs = 'NClimGrid_%s' % resolution
    modelType = 'TrainedOn' + dataset
    savename = modelType+'_Vari-'+variq+'_Obs-' + dataset_obs + '_Region-' + reg_name + '_Months-' + monthlychoice + '_L2-'+ str(ridge_penalty[0])+ '_LR-' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
    savenameModelTestTrain = modelType+'_'+variq+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)
    
    testIndices = np.genfromtxt(directorypredictions + savename + 'TestIndices_ENSEMBLES.txt')
    lenens = len(testIndices)
    
    trainIndices = np.genfromtxt(directorypredictions + savename + 'TrainIndices_ENSEMBLES.txt')
    lenensTrain = len(trainIndices)
    
    Ypred = np.genfromtxt(directorypredictions + savename + 'TEST_PREDICTIONS.txt').reshape(lenens,yearsmodel.shape[0])
    Yactual = yearsmodel
    Ypredobs = np.genfromtxt(directorypredictions + savename + 'OBSERVATION_PREDICTIONS.txt')
    
    ### Calculate statistics on observations
    slope, intercept, r, p, se = sts.linregress(yearsobs,Ypredobs)
    trendline = slope*yearsobs + intercept
    print(r,p,se)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Plotting functions
    ax = plt.subplot(2,3,lll+1)
    
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    ax.yaxis.grid(color='darkgrey',linestyle='-',linewidth=0.5,clip_on=False,alpha=0.4)
    
    plt.plot(np.arange(yearsall[0].min(),yearsall[0].max()+1,1),np.arange(yearsall[0].min(),yearsall[0].max()+1,1),'-',
              color='dimgrey',linewidth=2,clip_on=False)
    
    for i in range(Ypred.shape[0]):
        if i == 0:
            plt.plot(Yactual,Ypred[i,:],'o',
                    markersize=4,color='teal',clip_on=False,alpha=0.4,
                    markeredgecolor='dimgrey',markeredgewidth=0.8,
                    label=r'\textbf{%s [%s ens]}' % (datasetnamelist[lll],lenens))
        else:
            plt.plot(Yactual,Ypred[i,:],'o',
                    markersize=4,color='teal',clip_on=False,alpha=0.4,
                    markeredgecolor='dimgrey',markeredgewidth=0.4)        
    
    plt.plot(yearsobs,Ypredobs,'X',color='maroon',markersize=4,
              label=r'\textbf{%s}' % dataobsname,clip_on=False,markeredgecolor='dimgrey',markeredgewidth=0.4)
    # plt.plot(yearsobs,trendline,linewidth=2,color='crimson',clip_on=False)
    # plt.text(yearsobs[-1]+5,trendline[-1]-1,r'\textbf{R$^{2}$=%s}' % np.round(r**2,2),color='crimson',fontsize=9)
    
    if any([lll == 0,lll==3]):
        plt.ylabel(r'\textbf{PREDICTED YEAR}',fontsize=15,color='dimgrey')
    
    if lll == 4:
        plt.xlabel(r'\textbf{ACTUAL YEAR}',fontsize=15,color='dimgrey')
        
    plt.text(1921,2088, r'\underline{\textbf{Trained on [%s] ensemble members}}' % (lenensTrain),
              fontsize=8,ha='left',color='teal')
        
    iyears = np.where(Yactual<1990)[0]
    plt.text(1921,2079, r'\textbf{RMSE before 1990} = %s years' % (np.round(dSS.rmse(Ypred[:,iyears],Yactual[iyears]),decimals=1)),
              fontsize=8,ha='left',color='teal')
    
    iyears = np.where(Yactual>=1990)[0]
    plt.text(1921,2072, r'\textbf{RMSE after 1990} = %s years' % (np.round(dSS.rmse(Ypred[:,iyears],Yactual[iyears]),decimals=1)),
              fontsize=8,ha='left',color='teal')
    
    plt.xticks(np.arange(yearsall[0].min()-1,2101,25),map(str,np.arange(yearsall[0].min()-1,2101,25)),size=8)
    plt.yticks(np.arange(yearsall[0].min()-1,2101,25),map(str,np.arange(yearsall[0].min()-1,2101,25)),size=8)
    plt.xlim([yearsall[0].min()-1,yearsall[0].max()])   
    plt.ylim([yearsall[0].min()-1,yearsall[0].max()])
    
    leg = plt.legend(shadow=False,fontsize=8,loc='upper center',
                  bbox_to_anchor=(0.78,0.135),fancybox=True,ncol=1,frameon=False,
                  handlelength=1,handletextpad=0.5)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
        
    plt.text(1920,2096.5,r'\textbf{[%s]}' % letters[lll],fontsize=10,color='k')

### Save the predict the year plot
plt.tight_layout()
plt.savefig(directoryfigure + 'SPEAR_MED_prediction_LargeEnsembles.png',dpi=600)  
