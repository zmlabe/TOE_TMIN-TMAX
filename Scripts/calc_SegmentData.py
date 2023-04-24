"""
Function to slice trainging/testing/validation data
 
Notes
-----
    Author  : Zachary Labe
    Date    : 9 September 2022
    Version : 1  
    
Usage
-----
    [1] segment_data(data,fac,random_segment_seed)
"""
def segment_data(data,yearsall,fac,random_segment_seed):
    """
    Function to segment data based on ensemble members
    
    Usage
    -----
    segment_data(data,fac,random_segment_seed)
    """
    print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Using segment_data function!\n')
    
    ### Import modules
    import numpy as np
    import sys
  
    if random_segment_seed == None:
        random_segment_seed = int(int(np.random.randint(1, 100000)))
    np.random.seed(random_segment_seed)
    
    if fac < 1 :
        nrows = data.shape[0]
        segment_train = int(np.round(nrows * fac))
        segment_test = int(np.round(nrows * ((1-fac)/2)) - 1)
        segment_val = nrows - segment_train - segment_test
        print('--------------------------------------------------------------------')
        print('Training on',segment_train,'ensembles, Testing on',segment_test,'ensembles, Validation on',segment_val,'ensembles')
        print('--------------------------------------------------------------------')

        ### Picking out random ensembles
        i = 0
        trainIndices = list()
        while i < segment_train:
            line = np.random.randint(0, nrows)
            if line not in trainIndices:
                trainIndices.append(line)
                i += 1
            else:
                pass
    
        i = 0
        testIndices = list()
        while i < segment_test:
            line = np.random.randint(0, nrows)
            if line not in trainIndices:
                if line not in testIndices:
                    testIndices.append(line)
                    i += 1
            else:
                pass
            
        i = 0
        valIndices = list()
        while i < segment_val:
            line = np.random.randint(0, nrows)
            if line not in trainIndices:
                if line not in testIndices:
                    if line not in valIndices:
                        valIndices.append(line)
                        i += 1
            else:
                pass

###############################################################################  
###############################################################################  
###############################################################################        
        ### Training segment----------
        data_train = ''
        for ensemble in trainIndices:
            this_row = data[ensemble, :, :, :]
            this_row = this_row.reshape(-1,data.shape[1],data.shape[2],
                                        data.shape[3])
            if data_train == '':
                data_train = np.empty_like(this_row)
            data_train = np.vstack((data_train,this_row))
        data_train = data_train[1:, :, :, :]
    
        ### Reshape into X and Y
        Xtrain = data_train.reshape((data_train.shape[0] * data_train.shape[1]),
                                    (data_train.shape[2] * data_train.shape[3]))
        Ytrain = np.tile((np.arange(data_train.shape[1]) + yearsall.min()).reshape(data_train.shape[1],1),(data_train.shape[0],1))
        Xtrain_shape = (data_train.shape[0],data_train.shape[1])
        data_train_shape = data_train.shape[1]
        
        ### Random ensembles are picked
        print('\n----------------------------------------')
        print('Training on ensembles: ',trainIndices)
        print('Testing on ensembles: ',testIndices)
        print('Validation on ensembles: ',valIndices)
        print('----------------------------------------')
        print('\n----------------------------------------')
        print('org data - shape', data.shape)
        print('training data - shape', data_train.shape)
        
###############################################################################  
###############################################################################  
###############################################################################  
        ### Testing segment----------
        data_test = ''
        for ensemble in testIndices:
            this_row = data[ensemble, :, :, :]
            this_row = this_row.reshape(-1,data.shape[1],data.shape[2],
                                        data.shape[3])
            if data_test == '':
                data_test = np.empty_like(this_row)
            data_test = np.vstack((data_test, this_row))
        data_test = data_test[1:, :, :, :]
          
        ### Reshape into X and Y
        Xtest = data_test.reshape((data_test.shape[0] * data_test.shape[1]),
                                  (data_test.shape[2] * data_test.shape[3]))
        Ytest = np.tile((np.arange(data_test.shape[1]) + yearsall.min()).reshape(data_test.shape[1],1),(data_test.shape[0], 1))   
        Xtest_shape = (data_test.shape[0], data_test.shape[1])
        data_test_shape = data_test.shape[1]
        
        ### Random ensembles are picked
        print('----------------------------------------\n')
        print('----------------------------------------')
        print('Training on ensembles: count %s' % len(trainIndices))
        print('Testing on ensembles: count %s' % len(testIndices))
        print('Validation on ensembles: count %s' % len(valIndices))
        print('----------------------------------------\n')
        
        print('----------------------------------------')
        print('org data - shape', data.shape)
        print('testing data - shape', data_test.shape)
        print('----------------------------------------')
        
###############################################################################  
###############################################################################  
###############################################################################  
        ### Validation segment----------
        data_val = ''
        for ensemble in valIndices:
            this_row = data[ensemble, :, :, :]
            this_row = this_row.reshape(-1,data.shape[1],data.shape[2],
                                        data.shape[3])
            if data_val == '':
                data_val= np.empty_like(this_row)
            data_val = np.vstack((data_val, this_row))
        data_val = data_val[1:, :, :, :]
          
        ### Reshape into X and Y
        Xval = data_val.reshape((data_val.shape[0] * data_val.shape[1]),
                                  (data_val.shape[2] * data_val.shape[3]))
        Yval = np.tile((np.arange(data_val.shape[1]) + yearsall.min()).reshape(data_val.shape[1],1),(data_val.shape[0], 1))   
        Xval_shape = (data_val.shape[0], data_val.shape[1])
        data_val_shape = data_val.shape[1]
        
        ### Random ensembles are picked
        print('----------------------------------------\n')
        print('----------------------------------------')
        print('Training on ensembles: count %s' % len(trainIndices))
        print('Testing on ensembles: count %s' % len(testIndices))
        print('Validation on ensembles: count %s' % len(valIndices))
        print('----------------------------------------\n')
        
        print('----------------------------------------')
        print('org data - shape', data.shape)
        print('validation data - shape', data_val.shape)
        print('----------------------------------------')
  
    ### 'unlock' the random seed
    np.random.seed(None)
  
    return Xtrain,Ytrain,Xtest,Ytest,Xval,Yval,Xtest_shape,Xtrain_shape,Xval_shape,data_train_shape,data_test_shape,data_val_shape,trainIndices,testIndices,valIndices
