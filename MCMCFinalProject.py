#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:28:39 2024

@author: leon
"""

import numpy as np
import camb
from time import perf_counter

def TrimModel(ell,model,minell=2,maxell=2508):
    minindex=np.where(ell==minell)[0]
    maxindex=int(np.max(ell)-maxell)
    newell=np.delete(ell,np.arange(0,minindex))[:-maxindex]
    newmodel=np.delete(model,np.arange(0,minindex))[:-maxindex]
    return newell,newmodel

def LogLikelihoodDiag(data,model,sigma):
    return -np.sum(((data-model)/sigma)**2)/2

def MCMC(elldata,data,sigma,xstart,stepbasis,loglikelihood,chainlength,PrintProgress=True):
    nparams=len(xstart)
    chain=np.zeros((chainlength,nparams))
    
    ntrials=np.zeros(nparams)
    naccept=np.zeros(nparams)
    
    x=xstart.astype(float)
    pars=camb.set_params(H0=x[0],ombh2=x[1], omch2=x[2], mnu=0.06, omk=0, tau=x[3], As=x[4], 
                         ns=x[5], halofit_version='mead', lmax=2508)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        
    totCL = powers['total']
    ell1 = np.arange(totCL.shape[0])
    TT1 = totCL[:,0]
        
    ell,TT = TrimModel(ell1,TT1,minell=np.min(elldata),maxell=np.max(elldata))
    
    loglike = loglikelihood(data,TT,sigma)
    
    for i in range(chainlength):
        par_to_vary = np.random.randint(nparams)
        newx = x.copy() + np.random.normal()*stepbasis[par_to_vary]
        ntrials[par_to_vary] += 1
        
        accept=False
        if 50<newx[0]<100 and 0<newx[1]<0.1 and 0<newx[2]<0.3 and 0<newx[3]<0.1 and 0<newx[4]<4e-9 and 0.8<newx[5]<=1.1:     
            pars=camb.set_params(H0=newx[0],ombh2=newx[1],omch2=newx[2],mnu=0.06,omk=0,
                                 tau=newx[3], As=newx[4], ns=newx[5], halofit_version='mead', 
                                 lmax=2508)
            results = camb.get_results(pars)
            powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
            
            totCL = powers['total']
            ell1 = np.arange(totCL.shape[0])
            TT1 = totCL[:,0]
        
            ellmodel,TTnew=TrimModel(ell1,TT1,minell=np.min(elldata),maxell=np.max(elldata))
            
            newloglike = loglikelihood(data,TTnew,sigma)

            accept = np.random.rand() < np.exp(newloglike-loglike)
            
        if PrintProgress==True:
            print(np.round((i+1)/chainlength*100,2), '% done')
        
        if accept==True:
            naccept[par_to_vary]+=1
            x = newx
            loglike = newloglike
            
        chain[i] = x
        
    return chain, naccept/ntrials

def MCMC_timed(elldata,data,sigma,xstart,stepbasis,loglikelihood,runtime,PrintProgress=True):
    starttime = perf_counter()
    
    nparams=len(xstart)
    chain=np.zeros((0,nparams))
    
    ntrials=np.zeros(nparams)
    naccept=np.zeros(nparams)
    
    x=xstart.astype(float)
    pars=camb.set_params(H0=x[0],ombh2=x[1], omch2=x[2], mnu=0.06, omk=0, tau=x[3], As=x[4], 
                         ns=x[5], halofit_version='mead', lmax=2508)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        
    totCL = powers['total']
    ell1 = np.arange(totCL.shape[0])
    TT1 = totCL[:,0]
        
    ell,TT = TrimModel(ell1,TT1,minell=np.min(elldata),maxell=np.max(elldata))
    
    loglike = loglikelihood(data,TT,sigma)
    
    chainlength=1
    
    while perf_counter()-starttime<runtime:
        
        if PrintProgress==True:
            print('Chain length:', chainlength, '   Time remaining:', 
                  round(runtime - (perf_counter() - starttime)), 's')
        
        par_to_vary = np.random.randint(nparams)
        newx = x.copy() + np.random.normal()*stepbasis[par_to_vary]
        ntrials[par_to_vary] += 1
        
        newloglike=loglike
        
        accept=False
        if 50<newx[0]<100 and 0<newx[1]<0.1 and 0<newx[2]<0.3 and 0<newx[3]<0.1 and 0<newx[4]<4e-9 and 0.8<newx[5]<=1.1:     
            pars=camb.set_params(H0=newx[0],ombh2=newx[1],omch2=newx[2],mnu=0.06,omk=0,
                                 tau=newx[3], As=newx[4], ns=newx[5], halofit_version='mead', 
                                 lmax=2508)
            results = camb.get_results(pars)
            powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
            
            totCL = powers['total']
            ell1 = np.arange(totCL.shape[0])
            TT1 = totCL[:,0]
        
            ellmodel,TTnew=TrimModel(ell1,TT1,minell=np.min(elldata),maxell=np.max(elldata))
            
            newloglike = loglikelihood(data,TTnew,sigma)

            accept = np.random.rand() < np.exp(newloglike-loglike)
            
        if accept==True:
            naccept[par_to_vary]+=1
            x = newx
            loglike = newloglike
        
        chain=np.append(chain,[x],axis=0)
        chainlength+=1
        
    return chain, naccept/ntrials

def gelmanrubin(chains):
    
    nchains = np.shape(chains)[0]    
    chain_lengths = np.zeros(nchains)
    
    for i in range(nchains):
        chain_lengths[i] = np.shape(chains[i])[0]
        
    L = int(np.min(chain_lengths))
    for i in range(nchains):
        chains[i] = chains[i][-L:]
        
    mean_of_chains = np.sum(chains,axis=1)/L
    mean_of_params = np.sum(mean_of_chains,axis=0)/nchains
    
    B = np.sum((mean_of_chains-mean_of_params*np.ones((np.shape(mean_of_chains))))**2,axis=0)*L/(nchains-1)
    
    W = np.sum(np.sum((chains-np.repeat(mean_of_chains,L,axis=0).reshape(np.shape(chains)))**2,axis=1),axis=0)/(nchains*(L-1))
    
    R = ((L-1)*W + B)/(L*W)
    
    return(R)
    
def autocorr(n, chain):
    X = np.zeros(len(chain) - n)
    for i in range(len(X)):
        X[i] = chain[i + n] - chain[i]
    return np.sum(X**2) / ((len(X)) * (np.mean(chain**2) - np.mean(chain)**2)) 