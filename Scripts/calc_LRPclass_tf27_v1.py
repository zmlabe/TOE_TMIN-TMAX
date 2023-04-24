"""
Functions are useful untilities for interpretation of ANN
 
Notes
-----
    Author  : Zachary Labe
    Date    : 29 September 2022
    Version : Compatible with tensorflow v2.7
    
Usage
-----
    [1] calc_LRPModel(model,XXt,YYt,biasBool,annType,num_of_class,yearlabels,lrpRule,normLRP,numLats,numLons,numDim,classChunk,startYear,yearsall)
    [2] calc_LRPObs(model,XobsS,biasBool,annType,num_of_class,yearlabels,lrpRule,normLRP,numLats,numLons,numDim,yearsall)
"""

###############################################################################
###############################################################################
###############################################################################

def calc_LRPModel(model,XXt,YYt,biasBool,annType,num_of_class,yearlabels,lrpRule,normLRP,numLats,numLons,numDim,classChunk,startYear,yearsall):
    """
    Calculate Deep Taylor for LRP
    """
    print('\n\n<<<< Started LRP-Rules() >>>>')
    
    ### Import modules
    import numpy as np 
    import innvestigate
    import calc_Stats as SSS
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution() # bug fix with innvestigate v2.0.1
    
    ### Define useful functions
    def invert_year_output(ypred,startYear,yearsall):
        inverted_years = SSS.convert_fuzzyDecade_toYear(ypred,startYear,
                                                        classChunk,yearsall)    
        return inverted_years
    
    ### Define prediction error
    yearsUnique = np.unique(YYt)
    percCutoff = 90
    withinYearInc = 5.
    errTolerance = withinYearInc  
    if(annType=='class'):
        err = YYt[:,0] - invert_year_output(model.predict(XXt),startYear,yearsall)
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Create the innvestigate analyzer instance for each sample
    if(annType=='class'):
        model_nosoftmax = innvestigate.model_wo_softmax(model)
    
    ###########################################################################    
    if lrpRule == 'alphabeta':
        analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlphaBeta(
                                    model_nosoftmax,alpha=1,beta=0,bias=biasBool)
        print('LRP RULE === Alpha-Beta !!!')
        #######################################################################
    elif lrpRule == 'z':
        analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZ(model_nosoftmax)
        print('LRP RULE === Z !!!')
        #######################################################################
    elif lrpRule == 'epsilon':
        analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPEpsilon(model_nosoftmax, 
                                                                                        epsilon=1e10,
                                                                                        bias=biasBool)
        #######################################################################
    elif lrpRule == 'integratedgradient':
        analyzer = innvestigate.analyzer.gradient_based.IntegratedGradients(model_nosoftmax,steps=64)
        print('LRP RULE === Epsilon !!!')
        #######################################################################
    else:
        print(ValueError('Wrong LRP RULE!!!!!!!!!'))

    ###########################################################################

    deepTaylorMaps = np.empty(np.shape(XXt))
    deepTaylorMaps[:] = np.nan

    ### Analyze each input via the analyzer
    for i in np.arange(0,np.shape(XXt)[0]):
        ### ensure error is small, i.e. model was correct
        if(np.abs(err[i])<=errTolerance):
            sample = XXt[i]
            analyzer_output = analyzer.analyze(sample[np.newaxis,...])
            deepTaylorMaps[i] = analyzer_output/np.sum(analyzer_output.flatten())

    ### Save only the positive contributions
    if any([lrpRule=='z',lrpRule=='epsilon',lrpRule=='integratedgradient']):
        deepTaylorMaps[np.where(deepTaylorMaps < 0)] = 0.
        print('\nONLY POSITIVE CONTRIBUTIONS FOR LRP RULE ALLOWED!\n')    
    else:
        print('\nskip line for other rules, except alpha-beta\n')
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Compute the frequency of data at each point and the average relevance 
    ### normalized by the sum over the area and the frequency above the 90th 
    ### percentile of the map
    if numDim == 3:
        summaryDTq = np.reshape(deepTaylorMaps,(deepTaylorMaps.shape[0],deepTaylorMaps.shape[1]))
        
        ### Reshape into gridded maps
        lrpmaps = np.reshape(summaryDTq,(summaryDTq.shape[0],numLats,numLons))
        
        ### Normalize lrp to have maximum of 1
        if normLRP == True:
            lrpmaps = lrpmaps/np.nanmax(lrpmaps,axis=(-2,-1))[:,np.newaxis,np.newaxis]
            print('\n <<< Normalized LRP for max value of 1 >>> \n')
    ###########################################################################
    elif numDim == 4:
        summaryDTq = np.reshape(deepTaylorMaps,(deepTaylorMaps.shape[0]//len(yearlabels),
                                                len(yearlabels),deepTaylorMaps.shape[1]))
        
        ### Reshape into gridded maps
        lrpmaps = np.reshape(summaryDTq,(summaryDTq.shape[0],len(yearlabels),
                                         numLats,numLons))
        
        ### Normalize lrp to have maximum of 1
        if normLRP == True:
            lrpmaps = lrpmaps/np.nanmax(lrpmaps,axis=(-2,-1))[:,:,np.newaxis,np.newaxis]
            print('\n <<< Normalized LRP for max value of 1 >>> \n')
    else:
        print(ValueError('ISSUE WITH LRP FILE SIZE - CHECK!'))
    ###########################################################################        
    
    print('<<<< Completed LRP-Rules() >>>>')    
    return lrpmaps

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

def calc_LRPObs(model,XobsS,biasBool,annType,num_of_class,yearlabels,lrpRule,normLRP,numLats,numLons,numDim,yearsall):
    """
    Calculate Deep Taylor for LRP observations
    """
    print('\n\n<<<< Started LRP-Rules() OBSERVATIONS >>>>')
    
    ### Import modules
    import numpy as np 
    import innvestigate
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution() # bug fix with innvestigate v2.0.1
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Create the innvestigate analyzer instance for each sample
    if(annType=='class'):
        model_nosoftmax = innvestigate.model_wo_softmax(model)
    
    ###########################################################################    
    if lrpRule == 'alphabeta':
        analyzerobs = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlphaBeta(
                                    model_nosoftmax,alpha=1,beta=0,bias=biasBool)
        print('LRP RULE === Alpha-Beta !!!')
        #######################################################################
    elif lrpRule == 'z':
        analyzerobs = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZ(model_nosoftmax)
        print('LRP RULE === Z !!!')
        #######################################################################
    elif lrpRule == 'epsilon':
        analyzerobs = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPEpsilon(model_nosoftmax, 
                                                                                        epsilon=1e10,
                                                                                        bias=biasBool)
        print('LRP RULE === Epsilon !!!')
        #######################################################################
    elif lrpRule == 'integratedgradient':
        analyzerobs = innvestigate.analyzer.gradient_based.IntegratedGradients(model_nosoftmax,steps=64)
        print('LRP RULE === Integrated Gradient !!!')
        #######################################################################
    else:
        print(ValueError('Wrong LRP RULE!!!!!!!!!'))

    ########################################################################### 
    ### Analyze each input via the analyzer                                                                                  
    analyzer_output = analyzerobs.analyze(XobsS)
    analyzer_output = analyzer_output/np.nansum(analyzer_output,axis=1)[:,np.newaxis]  
    
    ### Save only the positive contributions
    if any([lrpRule=='z',lrpRule=='epsilon',lrpRule=='integratedgradient']):
        analyzer_output[np.where(analyzer_output < 0)] = 0.
        print('\nONLY POSITIVE CONTRIBUTIONS FOR LRP RULE ALLOWED!\n')    
    else:
        print('\nskip line for other rules, except alpha-beta\n')    

    ### Turn into LRP maps
    lrpmaps_obs = np.reshape(analyzer_output,(XobsS.shape[0],numLats,numLons))
    
    ### Normalize lrp to have maximum of 1
    if normLRP == True:
        lrpmaps_obs = lrpmaps_obs/np.nanmax(lrpmaps_obs,axis=(-2,-1))[:,np.newaxis,np.newaxis]
        print('\n <<< Normalized LRP for max value of 1 >>> \n')
    
    print('<<<< Completed LRP-Rules() OBSERVATIONS >>>>')
    return lrpmaps_obs
