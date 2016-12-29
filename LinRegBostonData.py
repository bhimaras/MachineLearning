# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 22:05:31 2016

@author: Aravind Bhimarasetty
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import copy
import itertools
import random
import matplotlib.pyplot as plt

def normalizeTrainData(dataSet, features):
    '''
    Normalize all feature data such that they are centered at 0 and std. dev. is 1.
    Returns a copy of dataset with normalized data and the means and std. devs
    '''
    dfCopy= dataSet.copy()
    meanFeatures=[]
    stdFeatures=[]
    
    for feature in features:
        meanFeature= dataSet[feature].mean()
        meanFeatures.append(meanFeature)
        
        stdFeature= (dataSet[feature].var())**.5
        if (stdFeature==0):
            raise ValueError('Standard dev. value is zero!')
        stdFeatures.append(stdFeature)
        
        dfCopy[feature]= (dfCopy[feature]- meanFeature)/stdFeature
    
    mu= pd.Series(meanFeatures, features)
    sigma= pd.Series(stdFeatures, features)
    
    return dfCopy, mu, sigma
    
def normalizeTestData(dataSet, features, mus, sigmas):
    '''
    Normalize dataset using the given features and their means and std. devs.
    Returns a copy of the dataset with normalized feature values
    '''
    dfCopy= dataSet.copy()
    
    for feature in features:
         dfCopy[feature]= (dfCopy[feature]- mus[feature])/sigmas[feature]
    
    return dfCopy

def doLinReg(dataSet, features, clsClmName):
    '''
    Perform linear regression- computes w by solving (X'X) w = X'y
    Returns the array, w
    '''
    dfMatX= dataSet.as_matrix(features)
    y= dataSet.as_matrix([clsClmName])
    
    numPts= dataSet.shape[0]
    
    X= np.c_[np.ones(numPts), dfMatX]
    LHS= np.matmul(X.T, X)
    RHS= np.matmul(X.T, y)
    
    w= np.linalg.solve(LHS, RHS)
    
    return w
    
def doLinRegReg(dataSet, features, clsClmName, lam):
    '''
    Perform linear regression with regularization- computes w by solving (X'X+ lam*I) w = X'y
    Returns the array, w    
    '''
    dfMatX= dataSet.as_matrix(features)
    y= dataSet.as_matrix([clsClmName])
    
    numPts= dataSet.shape[0]
    
    X= np.c_[np.ones(numPts), dfMatX]
    LHS= np.matmul(X.T, X)
    lamMat= np.eye(LHS.shape[0])
    lamMat[0][0]=0
    lamMat= lam*lamMat
    LHS= LHS+ lamMat
    
    RHS= np.matmul(X.T, y)
    
    w= np.linalg.solve(LHS, RHS)
    
    return w

def MSE(w, dataSet, features, clsClmName):
    '''
    Compute residue= y- X*w and mean squared error= (||residue||^2)/ #of pts.
    Returns mean squared error and residue array
    '''
    numPts= dataSet.shape[0]
    
    dfMatX= dataSet.as_matrix(features)
    y= dataSet.as_matrix([clsClmName])
    res= y- w[0][0]- np.matmul(dfMatX, w[1:])
    MSE= (np.sum( res**2))/numPts
    
    return MSE, res
    
###Load boston data and create pandas dataframe with the data
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

features= dataset.feature_names.tolist()

###Split 506 points into training and test set where the latter consists of points 7*i (where i=0,1,2,...,72)
testDataInds=[]
for i in range(73):
    testDataInds.append(7*i)
    
dataTest= df.loc[testDataInds, :]
dataTrain= df.drop(df.index[testDataInds])

###Plot histograms of the features
plt.figure()
dataTrain.hist(features)

###Normalize training and test data
[dataTrainNorm, mu, sigma]= normalizeTrainData(dataTrain, features)
dataTestNorm= normalizeTestData(dataTest, features, mu, sigma)

###Do linear regression using training data
w= doLinReg(dataTrainNorm, features, 'target')

###Compute mean squared errors
[MSETrain, res]= MSE(w, dataTrainNorm, features, 'target')
[MSETest, res]= MSE(w, dataTestNorm, features, 'target')

print "Linear Regression"
print "MSE (Non-reg) for training set: %0.3f" %MSETrain
print "MSE (Non-reg) for test set: %0.3f" %MSETest
print

###Ridge Regression
lams= [0.01, 0.1, 1] #range of regularization coefficients

print "Ridge Regression"
###Perform ridge regression (linear regression with regularization) for different reg. coeffs.
for lam in lams:
    wReg= doLinRegReg(dataTrainNorm, features, 'target', lam)
    [MSETrain, res]= MSE(wReg, dataTrainNorm, features, 'target')
    [MSETest, res]= MSE(wReg, dataTestNorm, features, 'target')
    
    print "MSE (Reg. with lambda= %0.3f) for training set: %0.3f" %(lam, MSETrain)
    print "MSE (Reg. with lambda= %0.3f) for test set: %0.3f" %(lam, MSETest)

print
    
###Ridge Regression with 10-fold CV
dataSets= [[] for x in xrange(10)]

for i in dataTrainNorm.index:
    setNum= random.randint(0,9)
    dataSets[setNum].append(i)

lams= [0.0001, 0.001, 0.01, 0.1, 1, 10]

print "Ridge Regression with Cross Validation"

for lam in lams:
    MSETraink=0
    for k in range(10):
        trainDataIndsk= copy.deepcopy(dataSets)
        testDataInds= trainDataIndsk.pop(k)
        trainDataIndsk= [item for sublist in trainDataIndsk for item in sublist]
        dataTraink= dataTrainNorm.loc[trainDataIndsk, :]
        wk= doLinRegReg(dataTraink, features, 'target', lam)
        dataTestk= dataTrainNorm.loc[testDataInds, :]
        [MSEtemp, res]=  MSE(wk, dataTestk, features, 'target')
        MSETraink+= MSEtemp
        
    MSETraink= MSETraink/10.
    
    print "Avg. MSE (Reg. with lambda=%0.4f) with 10-fold CV: %0.3f"%(lam, MSETraink)

print

###Feature Selection
sortedCorr= dataTrainNorm.corr().iloc[:-1, -1].abs().sort_values(ascending=False)
corrTop4Lbls= sortedCorr.iloc[:4].index.tolist()

wTop4= doLinReg(dataTrainNorm, corrTop4Lbls, 'target')

[MSETrainTop4, res]= MSE(wTop4, dataTrainNorm, corrTop4Lbls, 'target')
[MSETestTop4, res]= MSE(wTop4, dataTestNorm, corrTop4Lbls, 'target')

print "Feature Selection part a"
print "Top 4 features based on correlation: ", sorted(corrTop4Lbls)
print "MSE (training set) using top 4 correlated attr.: %0.3f" %MSETrainTop4
print "MSE (test set) using top 4 correlated attr.: %0.3f" %MSETestTop4
print

###Feature Selection with residues
corrTop1Lbls=[]
corrTop1Lbls.append(corrTop4Lbls[0])

wTop1=  doLinReg(dataTrainNorm, corrTop1Lbls, 'target')
[MSETop1, res]= MSE(wTop1, dataTrainNorm, corrTop1Lbls, 'target')
dataTrainNormRes= dataTrainNorm.copy()
dataTrainNormRes['Res']= res

for i in range(3):
    sortedCorr= dataTrainNormRes.corr().iloc[:-2, -1].abs().sort_values(ascending=False)
    sortedCorrTop4Lbls= sortedCorr.iloc[:4].index.tolist()
    corrTop1Lbls.append(sortedCorrTop4Lbls[0])
    wTop1=  doLinReg(dataTrainNormRes, corrTop1Lbls, 'target')
    [MSETop1, res]= MSE(wTop1, dataTrainNormRes, corrTop1Lbls, 'target')
    dataTrainNormRes['Res']= res


[MSETestComb, res]= MSE(wTop1, dataTestNorm, corrTop1Lbls, 'target')

print "Feature Selection part b"
print "Top 4 features based on correlation: ", sorted(corrTop1Lbls)
print "MSE (training set) using top 4 correlated attr.: %0.3f" %MSETop1
print "MSE (test set) using top 4 correlated attr.: %0.3f" %MSETestComb
print

###Feature Selection with brute-force
MSETrainCombBest=1000000
combBest=[]

for comb in itertools.combinations(features, 4):
    
    wComb= doLinReg(dataTrainNorm, comb, 'target')
    
    [MSETrainComb, res]= MSE(wComb, dataTrainNorm, comb, 'target')
    
    if (MSETrainComb< MSETrainCombBest):
        MSETrainCombBest= MSETrainComb
        combBest= copy.deepcopy(list(comb))

wCombBest= doLinReg(dataTrainNorm, combBest, 'target')
[MSETestCombBest, res]= MSE(wCombBest, dataTestNorm, combBest, 'target')

print "Feature selection using Brute force"
print "Top 4 features based on brute-force search: ", sorted(combBest)
print "MSE (training set) using top 4 correlated attr.: %0.3f" %MSETrainCombBest
print "MSE (test set) using top 4 correlated attr.: %0.3f" %MSETestCombBest
print
    
###Polynomial feature expansion
dataTrainNormExp= dataTrainNorm.copy()
dataTestNormExp= dataTestNorm.copy()
newFeatures=[]

for xixj in itertools.combinations_with_replacement(features, 2):
    dataTrainNormExp[xixj]= dataTrainNorm[xixj[0]]*dataTrainNorm[xixj[1]]
    dataTestNormExp[xixj]= dataTestNorm[xixj[0]]*dataTestNorm[xixj[1]]
    newFeatures.append(xixj)

[dataTrainExpNorm, muNew, sigmaNew]= normalizeTrainData(dataTrainNormExp, newFeatures)
dataTestExpNorm= normalizeTestData(dataTestNormExp, newFeatures,  muNew, sigmaNew)

allFeatures= features+newFeatures
w= doLinReg(dataTrainExpNorm, allFeatures, 'target')

[MSETrain, res]= MSE(w, dataTrainExpNorm, allFeatures, 'target')
[MSETest, res]= MSE(w, dataTestExpNorm, allFeatures, 'target')

print "Polynomial Feature Expansion"
print "MSE for training set: %0.3f" %MSETrain
print "MSE for test set: %0.3f" %MSETest
print