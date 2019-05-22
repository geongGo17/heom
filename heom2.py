# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:00:14 2018

@author: georg
"""
import numpy as np
import copy
from matplotlib import pyplot as plt
import numpy.linalg as la

import myAlgebra as ma
import physicalSystems as ps
import ordinaryDEq as odeq


def matsubaraTerms(eta,wC,beta,mMax):
    matFreqArTemp = []
    matCoefArTemp= []
    
    for k in range (1,mMax+1):
        vk=2*np.pi* k /beta
        fk= eta/beta*2 *wC*wC / (vk*vk - wC*wC)
        
        matFreqArTemp.append( vk)
        matCoefArTemp.append( fk)
        
    return matFreqArTemp , matCoefArTemp


def heomParameters(paramDicLs):
    coupOpNum = len( paramDicLs)
    matFreqAr,matCoefAr=[],[]
    thetaLs=[]
    g1Ls=[]
    
    mMaxLs=[]
    
    for i in range(coupOpNum):

        dic=paramDicLs[i]
        
        eta=dic['eta']
        wC=dic['wC']
        beta=1/dic['kT']
        mMax= dic['mMax']
        
        
        
        theta=eta*wC/2
        phi= theta/np.tan(beta*wC/2)
        matFreqArTemp,matCoefArTemp= matsubaraTerms(eta,wC,beta,mMax)
        
        matFreqArTemp.insert(0,wC)
        matCoefArTemp.insert(0,phi)
        
        g1= eta/beta - sum(matCoefArTemp)
        
        
        
        
        
        
    
        mMaxLs.append(mMax)
        thetaLs.append(theta)
        g1Ls.append(g1)
        matFreqAr.append(matFreqArTemp)
        matCoefAr.append(matCoefArTemp)
        

        
    return  np.array(mMaxLs) , np.array(thetaLs) , np.array(g1Ls) , np.array( matFreqAr) , np.array(matCoefAr)
    
    
def heomSuperoperators(coupOpLs):
    coupOpNum=len (coupOpLs)
    comAr= []
    antiAr= []
    comComAr=[]
    
    dim  =  coupOpLs[0].shape[0]
    iden = np.identity( dim)
    
    for i in range(coupOpNum):
            supOpLeft= ma.opToSupOp(coupOpLs[i],iden)
            supOpRight= ma.opToSupOp(iden,coupOpLs[i])
            comCoupOp= supOpLeft - supOpRight
            antCoupOp= supOpLeft + supOpRight
            comComCoupOp=np.dot(comCoupOp,comCoupOp)
            
            comAr.append(comCoupOp)
            antiAr.append(antCoupOp)
            comComAr.append(comComCoupOp)
            
    return -1.j*np.array(comAr) , - np.array(antiAr), np.array(comComAr)
            


class IndexGenerator:
    def __init__(self,numbOpIn,mMaxArIn,KIn):

        self.k = 0
        self.l = 0
        self.mMaxAr=mMaxArIn
        self.K=KIn
        self.nOp=numbOpIn
        self.numbEntry = (numbOpIn * (max( mMaxArIn) +1)  )  
        self.ar=np.zeros( (numbOpIn * (max( mMaxArIn) +1)  ) , dtype= int  )
        self.denMatrInd= np.zeros( (numbOpIn , max( mMaxArIn) +1  ))
        
        self.ar[0]=KIn



    def nextIndex (self):
        #print(self.ar)
        l=0
        while self.ar[l]==0:
            l+=1
            if l == self.numbEntry -1:
                return 1
                
        tmp=self.ar[l]-1
        
        self.ar[l]=0
        self.ar[0]=tmp
        self.ar[l+1]+=1
        
        return 0


        
    def getIndexMatrix(self):
        #print(self.ar)
        return self.ar.reshape( self.nOp , max( self.mMaxAr) +1   )

        
        
    


def auxMatrixIndex(numbOp,mMaxAr,K):
     
    
    denMatrIndLs=[]
    
    idxGen=IndexGenerator(numbOp,mMaxAr,K)
    
    step=0
    while step ==0 :
        
        
        
        idxMat = copy.deepcopy( idxGen.getIndexMatrix() )
        
        denMatrIndLs.append( idxMat)

        
        step = idxGen.nextIndex()
        
        
    return np.array(denMatrIndLs)



    
    
def returnIndex(idxMat1,idxMat2):
    tmpAr= abs ( np.add( idxMat1,-idxMat2) )
    if np.sum( tmpAr)>1:
        return -1,-1
    for k in range( len( tmpAr) ):
        for l in range( len( tmpAr[k] ) ):
            if tmpAr[k,l] == 1:
                return k, l


def matrixSparser(m,ix,iy):
    l=[]
    
    dX=m.shape[0]
    dY=m.shape[1]
    
    for x in range(dX):
        for y in range(dY):
            ele=[ix+x,iy+y,m[x,y] ]
            l.append(ele)
    
    return l #np.array(l)


def heomDiss (coupOpLs,paramDicLs,nMax):
    
    dim=coupOpLs[0].shape[0]
    dimQ=dim*dim
    numbCoupOp=len(coupOpLs)
    
    #Calculation of the parameters entering the HEOM
    mMaxAr , thetaAr, g1Ar , matFreqAr ,matCoefAr = heomParameters(paramDicLs)

    #Constructing the superoperators entering the HEOM
    comAr,antiAr, comComAr = heomSuperoperators(coupOpLs)
    sumCom=np.sum(comAr,axis = 0)
    sumComCom=-comComAr[0]*g1Ar[0]
    for i in range(1, numbCoupOp):
                sumComCom = - np.add (sumComCom,  comComAr[i]*g1Ar[i])
    
    
    
    #==========================================================================
    iu=0
    
    heomLs=[]
    
    #idxLs= [ np.zeros( (numbCoupOp , max( mMaxAr) +1  )) ]
    idxLs= [ ]
    numbIdx=0
    
    # loop for the order K of the heom matrices
    for K in range(nMax+1):
        idxLs0=copy.deepcopy(idxLs)         
        numbIdx0=copy.deepcopy ( numbIdx )
        
        #print('heom order:',K)
        idxLs = auxMatrixIndex(numbCoupOp,mMaxAr,K)    
        #print('number of matrices in this order: ',idxLs.shape )


        numbIdx=len(idxLs)
                
        uK = copy.deepcopy(iu)
        
        for u in range(numbIdx):
            
            #Diagonal elements K to K
            
            tmpAr=sumComCom
            if ( K !=nMax or nMax ==0 ):
                tmpAr1= -np.vdot(matFreqAr,idxLs[u] ) *np.identity(dimQ) #*np.ones((dimQ,dimQ))
                tmpAr=np.add(tmpAr,tmpAr1)
            
            sparseLs = matrixSparser(tmpAr , iu*dimQ , iu*dimQ )

            heomLs.extend(sparseLs)
            #print('mat index:', iu,iu )   
            #print(tmpAr)
            
           
            for v in range(numbIdx0):
                
                #Coupling of higher order K to K-1
                al,k = returnIndex( idxLs[u], idxLs0[v])    
                    
                if (al,k) == (-1,-1):
                        continue
                sparseLs = matrixSparser(sumCom , (uK- numbIdx0 +v) *dimQ , (uK+u) *dimQ )
                heomLs.extend(sparseLs)
                #print('mat index:', (uK- numbIdx0 +v), (uK+u))   
                #print(sumCom)
                
                #Coupling to lower orders, K-1 to K
                if K!= nMax:
                    #al,k = returnIndex( idxLs[u], idxLs0[v])    
                    
                    if (al,k) == (-1,-1):
                        continue
                    
                    tmpAr= matFreqAr[al,k] * matCoefAr[al,k]* idxLs[u,al,k]*comAr[al]
                    if k ==0:
                        tmpAr1= matFreqAr[al,0] * thetaAr[al]*idxLs[u,al,0]*antiAr[al]
                        tmpAr=np.add(tmpAr,tmpAr1)
                    
                    #print('mat index:', (uK+u) , (uK- numbIdx0 +v),'alpha k',al,k)   
                    #print(tmpAr)
                    
                    
                    sparseLs = matrixSparser(tmpAr ,(uK+u) *dimQ, (uK- numbIdx0 +v) *dimQ  )
                    heomLs.extend(sparseLs)
                    #heomLs=heomLs+sparseLs
                    

            iu+=1
            
    return iu, heomLs# np.array( heomLs )
        

def hamSparser(ham,numbMat):
    dim=ham.shape[0]
    dimQ=dim*dim
    
    iden = np.identity( dim)
    supOpLeft= ma.opToSupOp(ham,iden)
    supOpRight= ma.opToSupOp(iden,ham)
    comHam= -1.j * (supOpLeft - supOpRight)
    
    heomLs=[]
    for iu in range(numbMat):
        sparseLs = matrixSparser(comHam , iu*dimQ , iu*dimQ )
        heomLs.extend(sparseLs)
        
    return heomLs
        

        

class  HeomFkt2:
    def __init__(self , length, heomDiss, heomHamConst, heomHamDriv, drivFkt=None):
        self.heomD = np.zeros((length,length),dtype=complex)
        self.heomHc= np.zeros((length,length),dtype=complex)
        self.heomHd= np.zeros((length,length),dtype=complex)
        
        if drivFkt is None:
            self.dFkt= lambda arg1: 1 
        else:
            self.dFkt=drivFkt 
        #self.l = length
        #self.w = np.zeros( (self.l,) )
        
        for ele in heomDiss:
            i,j,val=ele[0],ele[1],ele[2]
            self.heomD[i,j]+=val
             
        for ele in heomHamConst:
            i,j,val=ele[0],ele[1],ele[2]
            self.heomHc[i,j]+=val
             
        for ele in heomHamDriv:
            i,j,val=ele[0],ele[1],ele[2]
            self.heomHd[i,j]+=val
        
        self.heomConst= np.add(self.heomD , self.heomHc )
                

    def grad(self,t,v):
        return np.dot(  np.add(self.heomConst , self.dFkt(t)* self.heomHd ),v)
        
    def heomConstMatrix(self):
        return self.heomConst
        
        
def sparseToMatrix(sparse,length):

    m=np.zeros((length,length),dtype=complex)
    
    for ele in sparse:
            i,j=int( ele[0] ) ,int(ele[1]) 
            m[i,j]+=ele[2]
            
    return m

        
def heomStationaryState(heom,sysSize):
    
    eigVal,eigStat = la.eig(heom)
    eigStat=eigStat.T

    order= np.argsort(-np.real( eigVal) ) 
    
    ev=eigVal[order[0  ] ] 
 
    print('heom EV:')
    print(eigVal[order[0 ]  ])
    print(eigVal[order[:10 ]])
    print(eigVal[order[-1 ] ])
    

    
    if abs(ev)> 0.000001:
        print('Warning! ', ev)

    ss0=eigStat[order[0  ] ] 
    
    
    ss=ss0[:sysSize*sysSize]/ss0[0]
    ss=ss.reshape(  (sysSize,sysSize) )  
    


    #print(ss0[:4*sysSize*sysSize])
    
    

    norm=0
    for i in range(sysSize):
        norm+=ss[i,i]
    
    
    ss = ss/norm

    
    return ss #,ss1      

def heomVecToDensityMatrix(heomVec,sysSize):
    rho=heomVec[:sysSize*sysSize]
    rho=rho.reshape(  (sysSize,sysSize) )  
    return rho

def matrixEntryGrap(mat,a,b,c,d,dim):
    i= a*dim+b
    j= c*dim+ d 
    return mat[i,j]    

def symmetryCheck(mat,dim):
    test=0
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                for d in range(dim):
                    tmp1=matrixEntryGrap(mat,a,b,c,d,dim)
                    tmp2=matrixEntryGrap(mat,b,a,d,c,dim)
                    tmp2=np.conjugate(tmp2)
                    if abs(tmp1-tmp2)>0.0001:
                        test+=1
    return test
    
    
def heomSymmetryCheck(heom,dimQ):
    dimH = heom.shape[0]
    dim=int( np.sqrt(dimQ) )
    
    dimH=int(dimH/dimQ/dimQ)
    print('Hallo',dimH)
    
    for i in range(dimH):
        for j in range(dimH):
            mat=heom[i*dimQ:(i+1)*dimQ,j*dimQ:(j+1)*dimQ]

            check = symmetryCheck(mat,dim)

            if check != 0 :
                print('not herm at ', i,j)
                print(mat)

def matrixEntryView(mat,a,b,dimQ):
    m=mat[a*dimQ :(a+1)*dimQ , b*dimQ :(b+1)*dimQ   ]
    print(m)
    
    
    
    
    
###############################################################################        
###############################################################################
###############################################################################
#Testing the program  

"""
a=1
kT=3.1 #20./2/np.pi


bath0Dic={
'eta':0.0001/a/a,
'wC':10,
'kT':kT,
'mMax':4
}    



V=1*a
hx=-1



hz=0

tMin,tMax=0,300
tNumb=200
dt= 0.002



##########################################################3
sZ=ps.sigmaZ()
sX=ps.sigmaX()
sY=ps.sigmaY()
def driveFkt(t):
    return np.cos(t)



#bathsLs=[bath0Dic,bath0Dic]
#coupOpLs=[sZ,sZ]
bathsLs=[bath0Dic]
coupOpLs=[V*sZ]


numbMat , heomDissAr = heomDiss (coupOpLs,bathsLs,nMax=4)
hamC = hamSparser(hx*sX,numbMat)
hamD = hamSparser(hz*sZ,numbMat)

length=numbMat*4
print('size of heom:', length)


heomSpinBoson = HeomFkt2(length,heomDissAr ,hamC,hamD,  driveFkt  )
#





#------------------------------------------------------------------------------



heomSpinBosonSparse = np.concatenate((heomDissAr, hamC), axis=0)
print('shape: ',heomSpinBosonSparse.shape )
heomSpinBosonSqM= sparseToMatrix(heomSpinBosonSparse ,length)

heomSymmetryCheck(heomSpinBosonSqM,dimQ=4)


print('\n matrix entries:')
matrixEntryView(heomSpinBosonSqM,6,18,dimQ=4)

ss= heomStationaryState(heomSpinBosonSqM,sysSize=2 )

print('\n stationary state:')
print(ss)



eigVal,eigStat = la.eig(ss)

a=np.real( eigVal[0]/eigVal[1] )
b=np.exp(2*hx/kT)
print('\n rations: ', (a-b) /(a+b)*2 , (a-1/b ) /(a+1/b)*2 )





#------------------------------------------------------------------------------




def func(v):
    return v[:4]
    
initState = np.zeros([length] )    
initState[0]=1

tSeries = odeq.timeSeries(heomSpinBoson , initState,func,tMin,tMax,tNumb,dt)


print('\n final state:')
fs=tSeries[-1,1:5].reshape(2,2)
print(fs)

eigVal,eigStat = la.eig(fs)

a=np.real( eigVal[0]/eigVal[1] )
b=np.exp(2*hx/kT)
print('\n rations: ', a, 1/a ,b,1/b)



plt.plot(tSeries[:,0],tSeries[:,1]) 
#plt.plot(tSeries[:,0],tSeries[:,4]) 
plt.show()

plt.plot(tSeries[:,0],tSeries[:,2]-np.conjugate( tSeries[:,3] ) )  
plt.show()

"""