import time
import numpy as np
from scipy.special import logsumexp
from scipy.special import softmax
# import tensorflow as tf
# from util import  py_softmax, MovingAverage
# from multigpu import gpu_mul_Ax, gpu_mul_xA, aggreg_multi_gpu, gpu_mul_AB
def py_softmax(x, axis=None):
    """stable softmax"""
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

def optimize_L_sk(L, PS, num_clusters, lamb):
      
    DTYPE = np.float64
    PS = PS.detach().cpu().numpy()
    #print (np.max(PS), np.min(PS))
    #PS = softmax(PS,1)
    #print (np.max(PS), np.min(PS))
    #print ("--------")
    PS = DTYPE(PS)
    N = L.shape[1]
   
    tt = time.time()
    PS = np.transpose(PS) # now it is K x N

    r = np.ones((num_clusters, 1), dtype=DTYPE) / num_clusters
    c = np.ones((N, 1), dtype=DTYPE) / N

    PS **= lamb  # K x N
    inv_K = DTYPE(1./num_clusters)
    inv_N = DTYPE(1./N)

    err = 1e6
    _counter = 0
    while err > 1e-1:
        r = inv_K / (PS @ c)          # (KxN)@(N,1) = K x 1
        c_new = inv_N / np.transpose(np.transpose(r) @ PS)  # ((1,K)@(KxN)).t() = N x 1
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    # print("error: ", err, 'step ', _counter, flush=True)  # " nonneg: ", sum(I), flush=True)
    # inplace calculations.
    PS *= np.squeeze(c)
    PS = np.transpose(PS)
    PS *= np.squeeze(r)
    PS = np.transpose(PS)
    argmaxes = np.nanargmax(PS, 0) # size N
    # print (np.min(argmaxes), np.max(argmaxes))
    # newL = torch.LongTensor(argmaxes)
    # self.L[nh] = newL.to(self.dev)
    # print('opt took {0:.2f}min, {1:4d}iters'.format(((time.time() - tt) / 60.), _counter), flush=True)
    return argmaxes
