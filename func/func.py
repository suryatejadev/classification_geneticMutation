from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
from time import time
from multiprocessing import Pool

maxlen = 10000

def select_class(data,label):
    req_class = [0,1,3,6]
    (n_train,n_dim) = data.shape
    for i in range(len(req_class)):
        loc = np.where(label==req_class[i])[0]
        if i==0:
            x = data[loc]
            y = np.zeros((loc.size))
        else:
            x = np.concatenate((x,data[loc]),axis=0)
            y = np.concatenate((y,np.zeros((loc.size))+i),axis=0)
    return x,y

def load_feat(file_name):
    f = open(file_name,'rb')
    feat = []
    for line in f:
        a = line.replace('[','').replace(']','').split()
        for k in a:
            feat.append(float(k))
    feat = np.array(feat)
    return feat.reshape(int(len(feat)/100),100)

def get_feat(n):
	doc_path = '../outputs/'+'w2v_'+str(n+1)+'.csv'
	print doc_path
	feat = load_feat(doc_path)
	if feat.shape[0]>maxlen:
		return feat[:maxlen,:]
	else:
		feat_pad = np.zeros((maxlen,100))
		feat_pad[:feat.shape[0],:] = feat
		return feat_pad

def import_data(label):
	req_class = [0,1,3,6]
	N = 3321
	t = time()
	pool = Pool()
	
	for i in range(len(req_class)):
		feat_train=[]
		loc = np.where(label==req_class[i])[0]
		feat_train.append(pool.map(get_feat,loc))
		feat_train = np.array(feat_train[0])
		if i==0:
			y = np.zeros((loc.size))
			x = feat_train
		else:
			y = np.concatenate((y,np.zeros((loc.size))+i),axis=0)
			x = np.concatenate((x,feat_train),axis=0)
			
	#feat_train.append(pool.map(get_feat,range(N)))
	#feat_train = feat_train[0]
	print time()-t
	return x,y#np.array(feat_train)
		
def stratified_split3(data,label):
    data3 = []
    label3 = []
    skf = StratifiedKFold(n_splits=3)
    skf.get_n_splits(data,label)
    for a in skf.split(data,label):
        (x,y)=a
        break
    data3.append(data[y])
    label3.append(label[y])

    temp_data = data[x]
    temp_label = label[x]
    skf = StratifiedKFold(n_splits=2)
    skf.get_n_splits(temp_data,temp_label)
    for a in skf.split(temp_data,temp_label):
        (x,y)=a
        break
    data3.append(temp_data[x])
    label3.append(temp_label[x])
    data3.append(temp_data[y])
    label3.append(temp_label[y])
    return data3,label3

def prepare_data(x):
    x = x.astype('float32')
    #x = x-np.mean(x,axis=0)
    #x=x/np.std(x,axis=0)
    return x


