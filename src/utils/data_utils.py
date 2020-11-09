import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob 
from scipy import io
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import make_moons

class CWRUDataset(Dataset):
    def __init__(self, raw_path, split, p=0.5, split_val=False, snr=None,seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.snr=snr
        self.split=split
        self.split_val=split_val
        self.DATA = []
        self.p = p
        self.idcs = None
        self.labels = None
        self.mask = None
        self.label_dict={
            (0,0) : 9,  
            (1,1) : 3,  
            (1,2) : 4,
            (1,3) : 5,
            (2,1) : 0,
            (2,2) : 1,
            (2,3) : 2,
            (3,1) : 6,
            (3,2) : 7,
            (3,3) : 8
        }
        self.load_cwru(raw_path)
    def __len__(self):
        return self.idcs.shape[0]
    def load_cwru(self, raw_path):
        dirnames = glob('%s/*'%raw_path)
        dirnames.sort()
        cnt=0
        
        idcs_all=[]
        labels_all=[]
        DATA_all=[]
        for dir in  dirnames:
            l = int(os.path.basename(dir)[0])
            fnames =  glob(dir+'/*.mat')
            fnames.sort()
            for f in fnames:
                f_id = int(os.path.basename(f).split('.')[0])
                fault = self._find_diameter(f_id)
                if fault == 4:  
                    continue
                mat = io.loadmat(f)
                r=0
                s=0
                label = (min(l,3),fault)
                for k in mat:
                    if k[-3:] == 'RPM':
                        r=mat[k][0,0]
                    if k[-4:] == 'time' and k[:4] == 'X098' and 'X%03d'%f_id=='X099':
                        continue
                    if k[-4:] == 'time':
                        if k[5:7] != 'DE':
                            continue
                        DATA_all.append(mat[k])
                        s=mat[k].shape[0]
                idx = np.arange(cnt, cnt+s-2048, 256)
                cnt+=s
                
                train_idx, test_idx = np.split(idx, [(idx.shape[0]*4)//5])
                if self.split_val:
                    train_idx, val_idx = np.split(train_idx, [(train_idx.shape[0]*4)//5])
                if self.split == 'train':
                    idcs = train_idx
                elif self.split == 'val':
                    idcs = val_idx
                elif self.split == 'test':
                    idcs=test_idx
                idcs_all.append(idcs)
                labels_all.append(np.ones_like(idcs)*self.label_dict[label])
        self.DATA = np.concatenate(DATA_all, axis=0)
        self.idcs = np.concatenate(idcs_all)
        self.labels = np.concatenate(labels_all)
        
        idcs=[]
        labels=[]
        
        
        n = min([self.idcs[self.labels==i].shape[0] for i in range(10)])
        for i in range(10):
            idcs.append(np.random.choice(self.idcs[self.labels==i],n, replace=False))
            
        self.idcs=np.concatenate(idcs)
        self.labels=np.arange(10).repeat(n)
        
        SEED=0
        np.random.seed(SEED)
        if self.split in ['train','val']:
            if self.p == 0:
                n_l = 10
            else:
                n_l = int(self.p*self.idcs.shape[0])
            while True:
                self.mask = np.random.choice(self.idcs.shape[0], self.idcs.shape[0], replace=False) < n_l
                if set([0,1,2,3,4,5,6,7,8,9]).issubset(set(self.labels[self.mask])):
                    break
        elif self.split == 'test':
            self.mask= np.array([True]*self.idcs.shape[0])
        print('%s dataset size:%d, labeled:%d, unlabeled:%d'%(self.split, self.idcs.shape[0], self.idcs[self.mask].shape[0], self.idcs[~self.mask].shape[0]))
        print('masks : [0:%d, 1:%d, 2:%d, 3:%d, 4:%d, 5:%d, 6:%d, 7:%d, 8:%d, 9:%d]'%tuple([self.idcs[self.mask & (self.labels==i)].shape[0] for i in range(10)]))
    def _find_diameter(self, n):
        if n<=100:
            return 0
        elif n<169:
            return 1
        elif n<209:
            return 2
        elif n<3000:
            return 3
        else:
            return 4
    def __getitem__(self,index):
        
        X = self.DATA[self.idcs[index]:self.idcs[index]+2048]
        Y = self.labels[index]
        M = self.mask[index]
        X= X.reshape(1,-1)
        if self.snr is not None:
            X = self._add_noise(X, self.snr)
            if False:
                
                ax = plt.subplot(611)
                ax.plot(X[0])
                snrs = [10,1,0.1,0.01,0.001]
                for i in range(5):
                    X = self._add_noise(X, snrs[i])
                    ax = plt.subplot(6,1,6-i)
                    ax.plot(X[0])
                
                plt.show()
        return X,Y,M
    def _add_noise(self,x, snr):
        snr1 = 10**(snr/10.0)
        xpower = np.sum(x**2, axis=-1) / x.shape[-1]
        npower = xpower/snr1
        noise = np.random.normal(0, np.sqrt(npower),x.shape)
        noise_data = x+noise
        return noise_data

class TwoMoonDataset(Dataset):
    def __init__(self, N, noise=None, ratio=1.0, n_label=None,seed=0):
        self.X, self.y, self.M = twomoon(N,noise,ratio,n_label, seed)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index],self.y[index],self.M[index]
        
def twomoon(N, noise=0.1, ratio=1.0, n_label=None,seed=0):
    data,label = make_moons(N,shuffle=False,noise=noise,random_state=seed)
    data = (data - data.mean(0,keepdims=True))/data.std(0,keepdims=True)
    l0_idx = (label==0)
    l1_idx = (label==1)
    
    np.random.seed(seed)
    m = np.array([False]*label.shape[0])
    l1 = np.array([False]*l0_idx.sum())
    l2 = np.array([False]*l1_idx.sum())
    
    
    if n_label is not None:
        if type(n_label) is tuple:
            l1[:n_label[0]]=True
            l2[:n_label[1]]=True
            #print(l1[n_label[0]:])
        elif type(n_label) in [int,float]:
            if n_label <1:
                l1[:int(n1*n_label)]=True
                l2[:int(n2*n_label)]=True
            else:
                l1[:n_label]=True
                l2[:n_label]=True
    np.random.shuffle(l1)
    np.random.shuffle(l2)
    m[l0_idx] = l1
    m[l1_idx] = l2
    return data, label, m