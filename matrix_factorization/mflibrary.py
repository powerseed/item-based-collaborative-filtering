import numpy as np# -*- coding: utf-8 -*-

def matrix_fatorization(R,I,U,K, step=5000, alpha = 0.0005, beta = 0.02):
    assert I.shape[1] == K and U.shape[0] == K
    for s in range(step+1):##add one since there is one step is for check for the origin input
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if R[i,j] > 0:
                    eij = R[i,j] - np.sum(I[i,:] * U[:,j])
                    for k in range(K):
                        I[i,k] = I[i,k] + alpha*((2.0*eij*U[k,j]) - beta*I[i,k])
                        U[k,j] = U[k,j] + alpha*((2.0*eij*I[i,k]) - beta*U[k,j])
                        
                    
        e = 0
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if R[i,j] > 0:
                    e = e + pow(R[i,j] - np.sum(I[i,:] * U[:,j]),2)
                    e = e + beta/2*(np.sum(np.power(I[i,:],2)) + np.sum(np.power(U[:,j],2)))
        print("error: {}, step: {}".format(e,s))
        if e < 500000:
            break
    return I,U