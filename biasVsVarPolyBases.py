# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 19:44:01 2016

@author: Aravind Bhimarasetty

This program illustrates the concept of bias-variance in ML.
100 datasets are generated randomly. Each dataset contains 10/100 samples where xi is uniformly sampled from [-1, 1] and yi= f(xi),
where f(xi)= 2*xi^2 + noise
Six different hypotheses are considered:
    g1(x)= 1    
    g2(x)= w0
    g3(x)= w0 + w1*x
    .
    .
    g6(x)= w0 + w1*x + w2*x^2 + w3*x^3 + w4*x^4
    
The bias and variance are computed for the above hypothesis using 10/ 100 samples per dataset and with regularization.

Plots similar to those given in lec. 8 of Learning from Data (http://work.caltech.edu/telecourse.html) are produced.
"""

import numpy as np
import matplotlib.pyplot as plt

def mse(y1, y2):
    return ((np.linalg.norm(y1-y2, 2))**2)/len(y1)
    
def phiMat(xData, n):
    '''
    Compute phi matrix ([[1, x1,..., x1^n], [1, x2,..., x2^n],...])
    Returns the phi matrix
    '''
    phi = np.array([])
    
    for x in xData:
        row=[]
        for i in range(n+1):
            row.append(x**i)
        phi= np.concatenate((phi, np.asarray(row)))
    
    phi= phi.reshape((len(xData), n+1 ) )
    
    return phi

def calcYData(xData, w):
    '''
    Compute y1= w0 + w1*x1 + w2*x1^2 +..., 
            y2= w0 + w1*x2 + w2*x2^2 +...,
            .
            .
    Returns the array of y= [y1, y2,...]
    '''
    yData=[]
    for x in xData:
        yVal=0
        for i in range(len(w)):
            yVal+= (x**i)*w[i]
        yData.append(yVal)
    
    return np.asarray(yData)

def genDataset(ND, NS, noiseMu, noiseStd):
    '''
    Generate ND number of datasets each containing NS samples where y= 2x^2 + noise and xs are generated randomly
    The noise is a gaussian process with mean= noiseMu and std. deviation= noiseStd
    Returns x and y matrices (ND by NS matrices)
    '''
    xDataFull= np.array([])
    yDataFull= np.array([])

    for dSetNum in range(ND):
        xD= np.random.random([NS])*2-1
        xD= np.sort(xD)
        xDataFull= np.concatenate((xDataFull, xD))
        yD= 2*(xD**2) + np.random.normal(noiseMu, noiseStd,NS)
        yDataFull= np.concatenate((yDataFull,yD))
    
    xDataFull= xDataFull.reshape([ND, NS])
    yDataFull= yDataFull.reshape([ND, NS])
    
    return xDataFull, yDataFull
    
def calcBiasVar(xDataFull, yDataFull, x, y, plotSet=None):
    '''
    Compute the bias and variance and plot the results for the different polynomial basis functions (g2-g6):
    g2(x)= w0
    g3(x)= w0 + w1*x
    .
    .
    g6(x)= w0 + w1*x + w2*x^2 + w3*x^3 + w4*x^4
    '''
    if plotSet==None:
        plotSet=0
        
    g1MSEs= []
    g2MSEs= []
    g3MSEs= []
    g4MSEs= []
    g5MSEs= []
    g6MSEs= []
    
    w0Avg= np.zeros(1)
    w1Avg= np.zeros(2)
    w2Avg= np.zeros(3)
    w3Avg= np.zeros(4)
    w4Avg= np.zeros(5)
    
    w0s=[]
    w1s=[]
    w2s=[]
    w3s=[]
    w4s=[]
    
    ND= xDataFull.shape[0]
    NS= xDataFull.shape[1]
    
    for dSetNum in range(ND):
        
        xData= xDataFull[dSetNum]
        yData= yDataFull[dSetNum]
            
        
        ### Case 1
        yG1= np.ones(NS)
        
        mseG1= mse(yG1, yData)
        g1MSEs.append(mseG1)
        
        ### Case 2
        phi0= phiMat(xData, 0)
        w0= np.linalg.solve(np.matmul(phi0.T, phi0), np.matmul(phi0.T,yData))
        y0= calcYData(xData, w0)
        y0Plot= calcYData(x, w0)
        
        mseG2= mse(y0, yData)
        g2MSEs.append(mseG2)
        
        w0s.append(w0)
        w0Avg+= w0
        
        ### Case 3
        phi1= phiMat(xData, 1)
        w1= np.linalg.solve(np.matmul(phi1.T, phi1), np.matmul(phi1.T,yData))
        y1= calcYData(xData, w1)
        y1Plot= calcYData(x, w1)
        
        mseG3= mse(y1, yData)
        g3MSEs.append(mseG3)
        
        w1s.append(w1)
        w1Avg+= w1
        
        ### Case 4
        phi2= phiMat(xData, 2)
        w2= np.linalg.solve(np.matmul(phi2.T, phi2), np.matmul(phi2.T,yData))
        y2= calcYData(xData, w2)
        y2Plot= calcYData(x, w2)
        
        mseG4= mse(y2, yData)
        g4MSEs.append(mseG4)
        
        w2s.append(w2)
        w2Avg+= w2
        
        ### Case 5
        phi3= phiMat(xData, 3)
        w3= np.linalg.solve(np.matmul(phi3.T, phi3), np.matmul(phi3.T,yData))
        y3= calcYData(xData, w3)
        y3Plot= calcYData(x, w3)
        
        mseG5= mse(y3, yData)
        g5MSEs.append(mseG5)
        
        w3s.append(w3)
        w3Avg+= w3
        
        ### Case 5
        phi4= phiMat(xData, 4)
        w4= np.linalg.solve(np.matmul(phi4.T, phi4), np.matmul(phi4.T,yData))
        y4= calcYData(xData, w4)
        y4Plot= calcYData(x, w4)
        
        mseG6= mse(y4, yData)
        g6MSEs.append(mseG6)
        
        w4s.append(w4)
        w4Avg+= w4
        
        plt.figure(1+5*plotSet)
#        plt.plot(xData, yData, 'r+', label="data" if dSetNum==0 else "", alpha=0.5)
        plt.plot(x, y, 'b-', label="f(x)=true fn." if dSetNum==0 else "", alpha=0.5)
        plt.plot(x, y0Plot, 'y-', label="g2(x)" if dSetNum==0 else "", alpha=0.5)
        plt.xlabel("x")
        plt.title("Plot of g2(x)s with sample size= "+str(NS))
        
        plt.figure(2+5*plotSet)
#        plt.plot(xData, yData, 'r+', label="data" if dSetNum==0 else "", alpha=0.5)
        plt.plot(x, y, 'b-', label="f(x)=true fn." if dSetNum==0 else "", alpha=0.5)
        plt.plot(x, y1Plot, 'c-', label="g3(x)" if dSetNum==0 else "", alpha=0.5)
        plt.xlabel("x")
        plt.title("Plot of g3(x)s with sample size= "+str(NS))
        
        plt.figure(3+5*plotSet)
#        plt.plot(xData, yData, 'r+',label="data" if dSetNum==0 else "", alpha=0.5)
        plt.plot(x, y, 'b-', label="f(x)=true fn." if dSetNum==0 else "", alpha=0.5)
        plt.plot( x, y2Plot, color="brown", label="g4(x)" if dSetNum==0 else "", alpha=0.5)
        plt.xlabel("x")
        plt.title("Plot of g4(x)s with sample size= "+str(NS))
        
        plt.figure(4+5*plotSet)
#        plt.plot(xData, yData, 'r+',  label="data" if dSetNum==0 else "", alpha=0.5)
        plt.plot(x, y, 'b-', label="f(x)=true fn." if dSetNum==0 else "", alpha=0.5)
        plt.plot( x, y3Plot, 'g-', label="g5(x)" if dSetNum==0 else "", alpha=0.5)
        plt.xlabel("x")
        plt.title("Plot of g5(x)s with sample size= "+str(NS))
        
        plt.figure(5+5*plotSet)
#        plt.plot(xData, yData, 'r+', label="data" if dSetNum==0 else "", alpha=0.5)
        plt.plot(x, y, 'b-', label="f(x)=true fn." if dSetNum==0 else "", alpha=0.5)
        plt.plot(x, y4Plot, 'm-', label="g6(x)" if dSetNum==0 else "", alpha=0.5)
        plt.xlabel("x")
        plt.title("Plot of g6(x)s with sample size= "+str(NS))
    
    
    w0Avg= w0Avg/ND
    w1Avg= w1Avg/ND
    w2Avg= w2Avg/ND
    w3Avg= w3Avg/ND
    w4Avg= w4Avg/ND
    
    y0PlotAvg= calcYData(x, w0Avg)
    y1PlotAvg= calcYData(x, w1Avg)
    y2PlotAvg= calcYData(x, w2Avg)
    y3PlotAvg= calcYData(x, w3Avg)
    y4PlotAvg= calcYData(x, w4Avg)
    
    #### Variance Calculation
    g1Var=0
    g2Var=0
    g3Var=0
    g4Var=0
    g5Var=0
    g6Var=0
    
    for dSetNum in range(ND):
        y0D= calcYData(x, w0s[dSetNum])
        temp= (y0D- y0PlotAvg)**2
        g2VarD= sum(temp)/ len(x)
        g2Var+= g2VarD
        
        y1D= calcYData(x, w1s[dSetNum])
        temp= (y1D- y1PlotAvg)**2
        g3VarD= sum(temp)/ len(x)
        g3Var+= g3VarD
        
        y2D= calcYData(x, w2s[dSetNum])
        temp= (y2D- y2PlotAvg)**2
        g4VarD= sum(temp)/ len(x)
        g4Var+= g4VarD
        
        y3D= calcYData(x, w3s[dSetNum])
        temp= (y3D- y3PlotAvg)**2
        g5VarD= sum(temp)/ len(x)
        g5Var+= g5VarD
        
        y4D= calcYData(x, w4s[dSetNum])
        temp= (y4D- y4PlotAvg)**2
        g6VarD= sum(temp)/ len(x)
        g6Var+= g6VarD
    
    g2Var=g2Var/ND
    g3Var=g3Var/ND
    g4Var=g4Var/ND
    g5Var=g5Var/ND
    g6Var=g6Var/ND
    
    ### Bias Calculation
    g1Bias= sum( (np.ones(len(x))- y)**2 )/ len(x)
    g2Bias= sum((y0PlotAvg- y)**2)/ len(x)
    g3Bias= sum((y1PlotAvg- y)**2)/ len(x)
    g4Bias= sum((y2PlotAvg- y)**2)/ len(x)
    g5Bias= sum((y3PlotAvg- y)**2)/ len(x)
    g6Bias= sum((y4PlotAvg- y)**2)/ len(x)
    
    ### Plotting
    plt.figure(1+5*plotSet)
    plt.plot(x, y0PlotAvg, 'k--', linewidth=2.5, label="g2_Avg")
    plt.legend()
    plt.figure(2+5*plotSet)
    plt.plot(x, y1PlotAvg, 'k--', linewidth=2.5,  label="g3_Avg")
    plt.legend()
    plt.figure(3+5*plotSet)
    plt.plot(x, y2PlotAvg, 'k--', linewidth=2.5,  label="g4_Avg")
    plt.legend()
    plt.figure(4+5*plotSet)
    plt.plot(x, y3PlotAvg, 'k--', linewidth=2.5,  label="g5_Avg")
    plt.legend()
    plt.figure(5+5*plotSet)
    plt.plot(x, y4PlotAvg, 'k--', linewidth=2.5,  label="g6_Avg")
    plt.legend()
    
    
    print "g1 (bias, var): %0.5f, %0.5f" %(g1Bias, g1Var)
    print "g1 bias+ var: %0.5f" %(g1Bias+ g1Var)
    print "g2 (bias, var): %0.5f, %0.5f" %(g2Bias, g2Var)
    print "g2 bias+ var: %0.5f" %(g2Bias+ g2Var)
    print "g3 (bias, var): %0.5f, %0.5f" %(g3Bias, g3Var)
    print "g3 bias+ var: %0.5f" %(g3Bias+ g3Var)
    print "g4 (bias, var): %0.5f, %0.5f" %(g4Bias, g4Var)
    print "g4 bias+ var: %0.5f" %(g4Bias+ g4Var)
    print "g5 (bias, var): %0.5f, %0.5f" %(g5Bias, g5Var)
    print "g5 bias+ var: %0.5f" %(g5Bias+ g5Var)
    print "g6 (bias, var): %0.5f, %0.5f" %(g6Bias, g6Var)
    print "g6 bias+ var: %0.5f" %(g6Bias+ g6Var)


def calcBiasVarReg(xDataFull, yDataFull, lambdas,x, y):
    '''
    Compute the bias and variance using regularization coefficients (lambdas) for h(x)= w0 + w1*x + w2*x^2
    '''
    
    ND= xDataFull.shape[0]
    
    for lam in lambdas:
    
        g4MSEs= []
        
        w2Avg= np.zeros(3)
    
        w2s=[]
        
        for dSetNum in range(ND):
    
            
            xData= xDataFull[dSetNum]
            yData= yDataFull[dSetNum]
                    
            ### Case 4
            phi2= phiMat(xData, 2)
                
            lhs= np.matmul(phi2.T, phi2)+ lam*np.eye(3)
            
            w2= np.linalg.solve(lhs, np.matmul(phi2.T,yData))
            y2= calcYData(xData, w2)
            y2Plot= calcYData(x, w2)
        
            mseG4= mse(y2, yData)
            g4MSEs.append(mseG4)
            
            w2s.append(w2)
            w2Avg+= w2
            
    
        w2Avg= w2Avg/ND
    
        y2PlotAvg= calcYData(x, w2Avg)
#        plt.plot(x, y2PlotAvg, label="lambda= {0}".format(lam))
        ###Variance Calculation
        g4Var=0
        
        for dSetNum in range(ND):
            
            y2D= calcYData(x, w2s[dSetNum])
            temp= (y2D- y2PlotAvg)**2
            g4VarD= sum(temp)/ len(x)
            g4Var+= g4VarD
        
        g4Var=g4Var/ND
        
        ###Bias Calculation
        g4Bias= sum((y2PlotAvg- y)**2)/ len(x)
        
        print "h(x) (bias, var) for lambda= %0.05f: %0.5f, %0.5f" %(lam, g4Bias, g4Var)
        print "Bias+ Var for lambda= %0.05f: %0.5f"%(lam, g4Bias+ g4Var)
    plt.legend()
    
print("--------------------------------")
print("|Bias Variance Tradeoff Problem|")
print("--------------------------------")
print

NOISE_MEAN=0.
NOISE_STD_DEV= 0.1**0.5
NUM_DATASETS=500

x= np.random.random([1000])*2-1
x= np.sort(x)
y= 2*(x**2) 

[xData10, yData10]= genDataset(NUM_DATASETS, 10, NOISE_MEAN, NOISE_STD_DEV)
[xData100, yData100]= genDataset(NUM_DATASETS, 100, NOISE_MEAN, NOISE_STD_DEV)

print("No regularization & Sample size=10")
print("----------------------------------")
calcBiasVar(xData10, yData10, x, y, 0)

print
print("No regularization & Sample size=100")
print("-----------------------------------")
calcBiasVar(xData100, yData100, x, y, 1)

lambdas= [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

print
print("With regularization & Sample size=100")
print("-------------------------------------")
calcBiasVarReg(xData100, yData100, lambdas,x, y)