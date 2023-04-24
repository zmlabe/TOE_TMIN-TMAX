"""
Train ANN to predict the year for ToE in the CONUS

Reference   : Barnes et al. [2020, JAMES] & Labe and Barnes [2021, JAMES]
Author      : Zachary M. Labe
Date        : 6 April 2023
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
resolutionall = ['MEDS','LOWS']
datasetsingleall = ['SPEAR_MED','SPEAR_LOW']
datasetnamelist = ['SPEAR_MED_SSP585','SPEAR_LOW_SSP585']
dataobsname = ['ERA5 (MED)','20CRv3 (LOW)']
dataset_obsall = ['ERA5_MEDS','20CRv3_LOWS']
yearsall = [np.arange(1921,2100+1,1)]
monthlychoiceall = ['JJA']
monthlychoice = monthlychoiceall[0]
variqall = ['T2M','T2M']
variqname = ['TAVG','TAVG']
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
    
lr_here = .01
batch_size = 32 
random_network_seed = 87750
random_segment_seed = 71541

### Years
yearsmodel = yearsall[0]
yearsobsall = [np.arange(1940,2022+1,1),np.arange(1921,2015+1,1)]

fig = plt.figure(figsize=(10,4.5))
for lll in range(len(datasetsingleall)):
    resolution = resolutionall[lll]
    yearsobs = yearsobsall[lll]

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
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Read in data
    directorypredictions = '/work/Zachary.Labe/Research/TOE_TMIN-TMAX/Data/Predictions/Regions/'
    directoryfigure = '/home/Zachary.Labe/Research/TOE_TMIN-TMAX/MS_Figures/'
    dataset = datasetsingleall[lll]
    dataset_obs = dataset_obsall[lll]
    variq = variqall[lll]
    modelType = 'TrainedOn' + dataset
    savename = modelType+'_Vari-'+variq+'_Obs-' + dataset_obs + '_Region-' + reg_name + '_Months-' + monthlychoice + '_L2-'+ str(ridge_penalty[0])+ '_LR-' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + NNType + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
    savenameModelTestTrain = modelType+'_'+variq+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)
    
    testIndices = np.genfromtxt(directorypredictions + savename + 'TestIndices_ENSEMBLES.txt')
    lenens = len(testIndices)
    
    Ypred = np.genfromtxt(directorypredictions + savename + 'TEST_PREDICTIONS.txt').reshape(lenens,yearsmodel.shape[0])
    Yactual = yearsmodel
    Ypredobs = np.genfromtxt(directorypredictions + savename + 'OBSERVATION_PREDICTIONS.txt')
    
    ### Calculate statistics on observations
    slope, intercept, r, p, se = sts.linregress(yearsobs,Ypredobs) # 1979-2019
    trendline = slope*yearsobs + intercept
    print(r,p)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Plotting functions
    ax = plt.subplot(1,2,lll+1)
    
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
                    label=r'\textbf{%s}' % datasetnamelist[lll])
        else:
            plt.plot(Yactual,Ypred[i,:],'o',
                    markersize=4,color='teal',clip_on=False,alpha=0.4,
                    markeredgecolor='dimgrey',markeredgewidth=0.4)        
    
    plt.plot(yearsobs,Ypredobs,'X',color='maroon',markersize=4,
              label=r'\textbf{%s}' % dataobsname[lll],clip_on=False,markeredgecolor='dimgrey',markeredgewidth=0.4)
    plt.plot(yearsobs,trendline,linewidth=2,color='crimson',clip_on=False)
    plt.text(yearsobs[-1]+5,trendline[-1]-1,r'\textbf{R$^{2}$=%s}' % np.round(r**2,2),color='crimson',fontsize=9)

    plt.xlabel(r'\textbf{ACTUAL YEAR}',fontsize=15,color='dimgrey')
    
    if lll == 0:
        plt.ylabel(r'\textbf{PREDICTED YEAR}',fontsize=15,color='dimgrey')
    
    plt.xticks(np.arange(yearsall[0].min()-1,2101,25),map(str,np.arange(yearsall[0].min()-1,2101,25)),size=8)
    plt.yticks(np.arange(yearsall[0].min()-1,2101,25),map(str,np.arange(yearsall[0].min()-1,2101,25)),size=8)
    plt.xlim([yearsall[0].min()-1,yearsall[0].max()])   
    plt.ylim([yearsall[0].min()-1,yearsall[0].max()])
    
    leg = plt.legend(shadow=False,fontsize=11,loc='upper center',
                  bbox_to_anchor=(0.8,0.135),fancybox=True,ncol=1,frameon=False,
                  handlelength=1,handletextpad=0.5)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
        
    plt.text(1920,2096,r'\textbf{[%s]}' % letters[lll],fontsize=10,color='k')
    plt.text(1950,2074,r'\textbf{%s}' % variqname[lll],fontsize=25,color='dimgrey')

### Save the predict the year plot
plt.tight_layout()
plt.savefig(directoryfigure + 'SPEAR_MED_prediction_OtherObs.png',dpi=600)  
