# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:06:01 2017

@author: georg
"""





import numpy as np
import numpy.linalg as la
import scipy.linalg as lalg

import physicalSystems as ps
import heom2 as heom
import myAlgebra as ma
#import ordinaryDEq as ode





    
#==============================================================================
    
    
    
def heomTimeIndependentMatrix(ham,coupOpLs,bathDicLs,nMax):
    
    sysSize=ham.shape[0]
    
    numbMat , heomDiss =  heom.heomDiss (coupOpLs,bathDicLs,nMax)
    hamC=heom.hamSparser(ham,numbMat)
    hamD=[]
    
    sizeHeom = sysSize*sysSize*numbMat
   
    heomSpinBoson = heom.HeomFkt2( sizeHeom, heomDiss ,hamC,hamD, None )
    
    
    heomMatrix = heomSpinBoson.heomConstMatrix()
        
    return heomMatrix

def heomMatrixExp(heomMatrix,dt):
    hM = heomMatrix*dt
    return lalg.expm(hM)


def myInitialState(dic,length):
    initState = np.zeros( (length,) )
    initState[0]=1
    return initState
    
def lowestEigenstate(dic,length):
    
    ham = dic['systemHamiltonian']
    sysSize=ham.shape[0]
    

    eigV,eigS = la.eig(ham)
    eigV,eigS = ma.sortEigensystem(eigV,eigS) 
    
    initState= eigS.T[0]
    
    densityMatrixEigen= ma.vectorsToTensor(initState, np.conjugate( initState )  )
    densityMatrixEigen=densityMatrixEigen.reshape( (sysSize*sysSize,) )
    
    initState = np.zeros( (length) )
    initState[:sysSize*sysSize]=densityMatrixEigen
    
    return initState


def heomSpinBosonPreparation(dic):
    
    kT=dic['kT']
    gamma=dic['gamma']
    wC=dic['wC']
    matsMax=dic['matsMax']
    
    hz0 = dic['hz0']
    hx0 = dic['hx0']
    
    sZ=ps.sigmaZ()
    sX=ps.sigmaX()
    #sY=ps.sigmaY()
    
    coupOpLs=[sX]
    
    nMax= dic['heomNMax']
    
    dtTimeEvolution = dic['dtTimeEvolution']
    
    
    bath0Dic={
    'eta':gamma,
    'wC':wC,
    'kT':kT,
    'mMax':matsMax,

    } 
    bathDicLs=[bath0Dic]
    ham = np.add( -hz0/2*sZ , -hx0/2 * sX)
    
    heomMatrix = heomTimeIndependentMatrix(ham,coupOpLs,bathDicLs,nMax)
    
    heomExp = heomMatrixExp(heomMatrix,dtTimeEvolution)
    
    dicPrep={
    'heomExp' : heomExp,
    'systemHamiltonian':ham
    }
    
    print('gamma',gamma)
    return dicPrep
    
        
class currentStateStore:
    def __init__(self):
        
        self.state=np.array([-1])
        
    def currentState(self):
        return self.state
        
    def setState(self,v):
        self.state=v    
        
        
def heomSpinBosonObs(dic):
    
    if 'heomExp' in dic:
        heomExp =  dic['heomExp']
    else: 
        dicPrep = heomSpinBosonPreparation(dic)
        heomExp= dicPrep['heomExp']
    
    #current state
    length =  heomExp.shape[0]  
    if 'storeObject' in dic:
        state0= dic['storeObject'].currentState()
    else:
        state0=np.zeros( (length),dtype=complex)
        state0[0]=1.
    
    
    #initial state if not defined   or simulation just started
    if state0.shape[0]==1:
        if dic['initState'] == 'lowestEigenstate':
            state0 = lowestEigenstate(dic,length)
        else:
            state0 = myInitialState(dic,length)
       
    

    
    rho = heom.heomVecToDensityMatrix(state0,sysSize=2)
    
    sX=ps.sigmaX()
    sY=ps.sigmaY()
    sZ=ps.sigmaZ()
    
    
    expValSx = np.trace( np.dot(sX,rho) )
    expValSy = np.trace( np.dot(sY,rho) )
    expValSz = np.trace( np.dot(sZ,rho) )
    
    #Output
    dicOut={
    'expValSx':expValSx,
    'expValSy':expValSy,
    'expValSz':expValSz
    }    
    
    state=np.dot(heomExp,state0)
    
    

    if 'storeObject' in dic:
       dic['storeObject'].setState( state )
    
    return dicOut
        
        
        

                
    
###############################################################################
#Testing the program


param={

'hx0': 1.,
'hy0':0. ,
'hz0':0,

#
'gamma':0.05 ,'wC': 10 ,'kT':3, 
#


'nMax':2,
'heomNMax': 3,
'matsMax':3,
'dtTimeEvolution': 0.1

}


