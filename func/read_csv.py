import numpy as np
from time import time

def load_feat(file_name):
    f = open(file_name,'rb')
    feat = []
    for line in f:
        a = line.replace('[','').replace(']','').split()
        for k in a:
            feat.append(float(k))
    feat = np.array(feat)
    return feat.reshape(int(len(feat)/100),100)

if __name__=='__main__':
    file_name = 'file_0.csv'
    feat = load_feat(file_name)
