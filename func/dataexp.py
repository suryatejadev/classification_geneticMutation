import numpy as np
from tqdm import tqdm

def k_nearest_neighbors(data,data_i,k):
    n = data.shape[0]
    dist = np.zeros((n))
    for i in range(n):
        dist[i] = np.dot(data_i,data[i])/(np.linalg.norm(data[i])*np.linalg.norm(data_i))
    knn = dist.argsort()[::-1][:k]
    return data[knn]

def expand_data(data,factor,method):
    (m,n) = data.shape
    data_exp = np.zeros((m*factor,n))
    if factor>1:
        if method == 'oversample':
            for i in range(m):
                data_exp[i*factor:(i+1)*factor] = data[i]
        elif method=='smote':
            for i in range(m):
                data_i = data[i]
                data_exp[0] = data_i
                data_nn = k_nearest_neighbors(data,data_i,factor-1)
                for j in range(1,factor):
                    alpha = np.random.rand()
                    data_exp[i*factor+j,:] = data_i*alpha+data_nn[j-1]*(1-alpha)
    else:
        return data
    return data_exp

def dataexp(data,label):
    (m,n) = data.shape
    class_len = np.zeros((9))
    method = 'smote'
    for i in range(9):
        class_len[i] = np.where(label==i)[0].size
    maxlen = class_len.max()
    for i in tqdm(range(9)):
        data_class = data[np.where(label==i)[0]]
        m_class = data_class.shape[0]
        factor = np.floor(maxlen*1./m_class).astype(int)
        if factor>m_class:
            factor = m_class-1
        data_class_exp = expand_data(data_class,factor,method)
        if i==0:
            data_exp = data_class_exp
            label_exp = np.zeros((m_class*factor))
        else:
            data_exp = np.concatenate((data_exp,data_class_exp),axis=0)
            label_exp = np.concatenate((label_exp,np.zeros((m_class*factor))+i),axis=0)
    return data_exp,label_exp

