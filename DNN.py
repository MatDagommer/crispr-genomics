# Copy right (c) Xue Zhang and Weijia Xiao 2020. All rights reserved.
#
#

import os
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import logging
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
#import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle


logging.basicConfig(format='%(levelname)s: %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger('DNN')


# regression parameter dictionary
regParamDict = {
    'epoch': 200,
    'batchSize': 32,
    'dropOut': 0.2,
    'loss': 'mse',
    'metrics': ['mse'],
    'activation1': 'relu',
    'activation2': 'sigmoid',
    'monitor': 'loss',
    'save_best_only': True,
    'mode': 'max'
}

# classification parameter dictionary
classParamDict = {
    'epoch': 200,
    'batchSize': 32,
    'dropOut': 0.2,
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy'],
    'activation1': 'relu',
    'activation2': 'sigmoid',
    'monitor': 'val_accuracy',
    'save_best_only': True,
    'mode': 'max'
}

class_weight = {0: 1.0, 1: 4.0}

optimizerDict = {
    'adam': Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
}

hl = [128, 256, 512, 1024, 1024, 1024,1024, 1024, 1024, 1024];

def make_folder(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)
        logger.info('{} is created.'.format(folder))
    else:
        logger.info('{} is already there.'.format(folder))


class DNN(object):

    def __init__(self, pdata, f_tp, f_fp, f_th, expName, iteration, numHidden=3, result_dir='node2vec/results/', reg = False):
        super(DNN, self).__init__()
        self.pdata = pdata
        self.expName = expName
        self.model_dir = os.path.join(result_dir, 'model')
        self.numHidden = numHidden
        #Added by Matthieu
        #----------------------
        self.reg = reg
        #----------------------
        make_folder(self.model_dir)

        self.evaluationInfo = dict()

        self.trainingData, self.validationData, self.testingData = pdata.partitionDataset()

        X_train, Y_train = separateDataAndClassLabel(self.trainingData)
        X_valid, Y_valid = separateDataAndClassLabel(self.validationData)
        X_test, Y_test = separateDataAndClassLabel(self.testingData)

        #X_train = pdata.getScaledData(X_train)
        #X_valid = pdata.getScaledData(X_valid)
        #X_test = pdata.getScaledData(X_test)

        if not reg:
            self.numberOfClasses = encodeClassLabel(Y_train)
            self.numberOfFeature = X_train.shape[1]
            
            print("NUMBER OF CLASSES: ", self.numberOfClasses)
            # reshaping class labels
            Y_train_reshaped = np_utils.to_categorical(Y_train, self.numberOfClasses)
            Y_valid_reshaped = np_utils.to_categorical(Y_valid, self.numberOfClasses)
            Y_test_reshaped = np_utils.to_categorical(Y_test, self.numberOfClasses)

        #Added by Matthieu
        #---------------------------------------
        else:
            self.numberOfClasses = 1
            self.numberOfFeature = X_train.shape[1]

            Y_train_reshaped = Y_train.copy()
            Y_valid_reshaped = Y_valid.copy()
            Y_test_reshaped = Y_test.copy()
        #---------------------------------------

        self.dataDict = {
            'train': X_train,
            'trainLabel': Y_train_reshaped,
            'valid': X_valid,
            'validLabel': Y_valid_reshaped,
            'test': X_test,
            'testLabel': Y_test_reshaped
        }

        self.evaluationInfo = buildModel(self.dataDict, self.numberOfFeature, self.numberOfClasses,
                                               f_tp, f_fp, f_th, expName, iteration, self.model_dir, result_dir, self.numHidden, self.reg)
        self.evaluationInfo['numTrain'] = X_train.shape[0]
        self.evaluationInfo['numTest'] = X_test.shape[0]
        self.evaluationInfo['numValidation'] = X_valid.shape[0]
        self.evaluationInfo['numFeature'] = self.numberOfFeature

    def getEvaluationStat(self):
        return self.evaluationInfo

# returns the TP, TN, FP and FN values
def getTPTNValues(test, testPred):
    TP, TN, FP, FN = 0, 0, 0, 0
    #for i in range(len(testPred)):
    print("testPred shape: ", testPred.shape)
    for i in range(testPred.shape[0]):
        if test[i] == testPred[i] == 1:
            TP += 1
        elif testPred[i] == 1 and test[i] != testPred[i]:
            FP += 1
        elif test[i] == testPred[i] == 0:
            TN += 1
        elif testPred[i] == 0 and test[i] != testPred[i]:
            FN += 1

    return TP, TN, FP, FN


# separating feature matrix and class label
def separateDataAndClassLabel(dataMatrix):
    featureMatrix = dataMatrix[:, :(dataMatrix.shape[1] - 1)]
    classLabelMatrix = dataMatrix[:, -1]

    return featureMatrix, classLabelMatrix


# returns the number of classes and encode it
def encodeClassLabel(classLabel):
    labelEncoder = LabelEncoder().fit(classLabel)
    labels = labelEncoder.transform(classLabel)
    classes = list(labelEncoder.classes_)
    print("THIS IS LEN(CLASSES): ", len(classes))
    return len(classes)

# building the DNN model and run with the data, returns a list of metrics
def buildModel(dataDict, numFeat, numberOfClasses, f_tp, f_fp, f_th, expName, iteration, model_dir, result_dir, numHidden, reg):
    trainData = dataDict['train']
    trainLabel = dataDict['trainLabel']
    validData = dataDict['valid']
    validLabel = dataDict['validLabel']
    testData = dataDict['test']
    testLabel = dataDict['testLabel']

    # Added by Matthieu
    #---------------------------------
    if reg:
        paramDict = regParamDict
    else:
        paramDict = classParamDict
    #---------------------------------
    
    # building NN model
    model = Sequential()
    model.add(Dense(hl[0], activation = paramDict['activation1'], input_shape = (numFeat, )))
    model.add(Dropout(paramDict['dropOut']))
    for i in range(1, numHidden):
        if i < len(hl):
            model.add(Dense(hl[i], activation = paramDict['activation1']))
            model.add(Dropout(paramDict['dropOut']))
        else:
            model.add(Dense(1024, activation = paramDict['activation1']))
            model.add(Dropout(paramDict['dropOut']))

    if not reg:
        model.add(Dense(numberOfClasses, activation=paramDict['activation2']))
    else:
        model.add(Dense(numberOfClasses))

    model.compile(optimizer=optimizerDict['adam'],
                  loss=paramDict['loss'],
                  metrics=paramDict['metrics'])
    

    # saving best model by validation accuracy
    
    filePath = os.path.join(model_dir, expName + str(iteration) + '_weights.best.hdf5')
    checkpointer = ModelCheckpoint(filepath=filePath, verbose=0, monitor=paramDict['monitor'], save_best_only=True)
    earlystopper = EarlyStopping(paramDict['monitor'], patience=15, verbose=1)

    print(trainData.shape)
    print(trainLabel.shape)
    #print(class_weight[0].shape)
    print(validData.shape)
    print(validLabel.shape)
    # fit the model to the training data and verify with validation data

    if not reg:
        weight = class_weight
    else:
        weight = None

    model.fit(trainData, trainLabel,
              epochs=paramDict['epoch'],
              callbacks=[checkpointer, earlystopper],
              batch_size=paramDict['batchSize'],
              shuffle=True,
              verbose=1,
              validation_data=(validData, validLabel), class_weight = weight)

    # load best model and compile
    model.load_weights(filePath)
    model.compile(optimizer=optimizerDict['adam'],
                  loss=paramDict['loss'],
                  metrics=paramDict['metrics'])
    
    # serialize model to JSON (save the model structure in order to use the saved weights)
    #one time save
    fn = os.path.join(model_dir, 'model3.json')
    if not os.path.isfile(fn):
        model_json = model.to_json()
        with open(fn, 'w') as json_file:
            json_file.write(model_json)
            
    #save model for later use (including model structure and weights)
    model_file = os.path.join(model_dir, expName + str(iteration) + '_model.h5')
    model.save(model_file)
    

    if not reg:
        # evaluation scores
        roc_auc = metrics.roc_auc_score(testLabel, model.predict(testData))
        
        #precision here is the auc of precision-recall curve
        precision = metrics.average_precision_score(testLabel, model.predict(testData))

        # get predicted class label
        #probs = model.predict_proba(testData)
        #testPredLabel = model.predict(testData)
        testPredLabel = model.predict(testData)
        true_y = list()
        for y_i in range(len(testLabel)):
            true_y.append(testLabel[y_i][1])
        probs = testPredLabel[:, 1]

        fpr, tpr, threshold = metrics.roc_curve(true_y, probs)

        for i in range(len(fpr)):
            f_fp.write(str(fpr[i]) + '\t')
        f_fp.write('\n')
        
        for i in range(len(tpr)):
            f_tp.write(str(tpr[i]) + '\t')
        f_tp.write('\n')
        
        for i in range(len(threshold)):
            f_th.write(str(threshold[i]) + '\t')
        f_th.write('\n')


        # Added by Matthieu
        #-------------------------------------------------------
        with open(result_dir + '/' + '_True_Positives_it%d'%iteration, 'wb') as f:
            pickle.dump(fpr, f)
        with open(result_dir + '/' + '_False_positives_it%d'%iteration, 'wb') as f:
            pickle.dump(tpr, f)
        with open(result_dir + '/' + '_Thresholds_it%d'%iteration, 'wb') as f:
            pickle.dump(threshold, f)
        with open(result_dir + '/' + '_true_y_it%d'%iteration, 'wb') as f:
            pickle.dump(true_y, f)
        with open(result_dir + '/' + '_probs_it%d'%iteration, 'wb') as f:
            pickle.dump(probs, f)
        #-------------------------------------------------------

        # save precision, recall, and thresholds for PR curve plot
        p0, r0, t0 = metrics.precision_recall_curve(true_y, probs)
        fnp0 = os.path.join(result_dir, expName + '_precision.txt')
        fnr0 = os.path.join(result_dir, expName + '_recall.txt')
        fnt0 = os.path.join(result_dir, expName + '_PR_threshold.txt')

        with open(fnp0, 'a') as f0:
            for i in range(len(p0)):
                f0.write(str(p0[i]) + '\t')
            f0.write('\n')
                
        with open(fnr0, 'a') as f0:
            for i in range(len(r0)):
                f0.write(str(r0[i]) + '\t')
            f0.write('\n')
        
        with open(fnt0, 'a') as f0:
            for i in range(len(t0)):
                f0.write(str(t0[i]) + '\t')
            f0.write('\n')
        

        # Added by Matthieu 
        #--------------------------------------------------------------
        p0, r0, t0 = metrics.precision_recall_curve(true_y, probs)
        fnp0 = os.path.join(result_dir, expName + '_precision')
        fnr0 = os.path.join(result_dir, expName + '_recall')
        fnt0 = os.path.join(result_dir, expName + '_PR_threshold')

        with open(fnp0, 'wb') as f:
            pickle.dump(p0, f)
                
        with open(fnr0, 'wb') as f:
            pickle.dump(r0, f)
        
        with open(fnt0, 'wb') as f:
            pickle.dump(t0, f)
        #--------------------------------------------------------------

        print("test label : ", testLabel.shape)
        print("test pred label : ", testPredLabel.shape)
        # convert back class label from categorical to integer label
        testLabelRev = np.argmax(testLabel, axis=1)
        testPredLabelRev = np.argmax(testPredLabel, axis=1)
        #testPredLabelRev = np.argmax(probs)#, axis=1)

        print("labelrev : ", testLabelRev.shape)
        print("predlabelrev : ", testPredLabelRev.shape)
        # get TP, TN, FP, FN to calculate sensitivity, specificity, PPV and accuracy
        TP, TN, FP, FN = getTPTNValues(testLabelRev, testPredLabelRev)
        print("TP .... ", TP, TN, FP, FN)
        sensitivity = float(TP) / float(TP + FN + 1e-15)
        specificity = float(TN) / float(TN + FP + 1e-15)
        PPV = float(TP) / float(TP + FP + 1e-15)
        accuracy = float(TP + TN) / float(TP + FP + FN + TN + 1e-15)

        # dictionary to store evaluation stat
        evaluationInfo = {
            'roc_auc': roc_auc,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'PPV': PPV,
            'accuracy': accuracy,
            'batch_size': paramDict['batchSize'],
            'activation': paramDict['activation2'],
            'dropout': paramDict['dropOut']
        }

    else:
        # get predicted class label
        #probs = model.predict_proba(testData)
        #testPredLabel = model.predict(testData)
        testPredLabel = model.predict(testData)
        true_y = list()
        for y_i in range(len(testLabel)):
            true_y.append(testLabel[y_i])
        probs = testPredLabel


        with open(result_dir + '/' + '_true_y_it%d'%iteration, 'wb') as f:
            pickle.dump(true_y, f)
        with open(result_dir + '/' + '_probs_it%d'%iteration, 'wb') as f:
            pickle.dump(probs, f)


        # dictionary to store evaluation stat
        evaluationInfo = {
            'mse': mse( true_y, probs ),
            'batch_size': paramDict['batchSize'],
            'activation': paramDict['activation2'],
            'dropout': paramDict['dropOut']
        }

    return evaluationInfo
    
def mse(true_y, probs):
    true_y = np.array(true_y)
    probs = np.array(probs)
    return np.sum( np.square( true_y - probs ) ) / true_y.shape[0]