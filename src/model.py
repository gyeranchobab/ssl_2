from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import pickle
class BaseModel(nn.Module):
    def __init__(self,model_name='model',p=None):
        '''
            [Input]
                - model_name : model name
        '''
        super(BaseModel, self).__init__()
        self.name = model_name
        if p is not None:
            self.name += '_p%04d'%(10000*p)
        print('Model Name : %s'%self.name)
        
        self.e=0        ## self.e : save current epoch
        
        ## Save training logs
        self.total_train_loss=[]
        self.total_test_loss=[]
        self.total_train_acc=[]
        self.total_test_acc=[]
        
    def train_model(self, train_dataloader, test_dataloader, epoch, optimizer,**kwargs):
        '''
            [Input]
                - train_dataloader : training dataloader to sample batch from
                - test_dataloader : test dataloader to sample batch from
                - epoch : # of epochs to run
                - optimizer : optimizer
        '''
        best_eval=float('inf')
        
        while True:
            if self.e >= epoch:
                break
            ts = time.time()
            
            ## Training epoch step
            self.train()
            train_loss, train_acc = self.epoch_step(train_dataloader, optimizer=optimizer,**kwargs)
            
            ## Test epoch step
            self.eval()
            with torch.no_grad():
                test_loss, test_acc = self.epoch_step(test_dataloader,**kwargs)
            
            print('(%.2fs) [Epoch %d]'%(time.time()-ts,self.e))
            print('\tTrain Loss : %.5f,\tTrain Acc : %.5f'%(train_loss, train_acc))
            print('\tEval Loss : %.5f,\tEval Acc : %.5f'%(test_loss, test_acc))
            
            self.total_train_loss.append(train_loss)
            self.total_test_loss.append(test_loss)
            self.total_train_acc.append(train_acc)
            self.total_test_acc.append(test_acc)
            
            if self.e%10 == 9:
                torch.save(self.state_dict(), './saved_models/%s_e%d.pth'%(self.name, self.e+1))
                pickle.dump((self.total_train_loss, self.total_test_loss, self.total_train_acc, self.total_test_acc), open('./train_logs/%s.bin'%(self.name),'wb'))
            self.e += 1
    def epoch_step(self, dataloader, optimizer=None,return_result=False,**kwargs):
        '''
            [Input]
                - dataloader : training dataloader to sample batch from
                - optimizer : optimizer
                - return_result : whether to return prediction result or not (for evaluation only)
            [Output]
                - avg_loss : average loss of current epoch
                - accuracy : prediction accuracy of current epoch
        '''
        avg_loss = 0.0
        n = 0
        total_correct = 0
        total_z = []
        total_pred = []
        
        ## For all data
        for batch, label, mask in dataloader:
            ## Upload to GPU
            batch=batch.cuda().to(dtype=torch.float)
            label=label.cuda().to(dtype=torch.long)
            mask = mask.cuda().to(dtype=torch.bool)
            
            ## Screen labels of unlabeled data
            masked_label = label.clone()
            masked_label[~mask] = -1
            
            ## Batch step
            loss, p, z = self.batch_step(batch, masked_label, mask, optimizer=optimizer,**kwargs)
            
            ## Get predicted labels from predicted probabilities
            pred = torch.argmax(p, dim=1)
            
            ## Update moving average of loss
            avg_loss = (n*avg_loss + loss.item()*batch.size(0)) / (n+batch.shape[0])
            n = n+batch.shape[0]
            
            correct = (label[label==pred]).shape[0]
            total_correct += correct
            total_z.append(z.detach().cpu())
            total_pred.append(pred.detach().cpu())
            
        if return_result:
            return avg_loss, total_correct / n, torch.cat(total_z), torch.cat(total_pred)
        return avg_loss, total_correct / n
    
    def batch_step(self, batch, label, mask, optimizer=None):
        '''
            [Input]
                - batch : batch of input data
                - label : ground-truth label
                - mask : mask_ij is True if x_ij is labeled, else False
                - optimizer : optimizer
            [Output]
                - loss : loss
                - p : batch of predicted probabilites for each categories
                - z : latent vector
        '''
        ## Forward pass
        p, loss, z = self.forward(batch, label, mask)
        
        ## Update model weight if training and loss is computed
        if self.training and (loss is not None):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        if loss is None:
            loss = torch.tensor(0.0)
            
        return loss, p, z
    
    
    def predict(self, dataloader, **kwargs):
        '''
            [Input]
                - dataloader : dataloader to sample batches from
            [Output]
                - z : latent vectors
                - pred : predictions
        '''
        self.eval()
        with torch.no_grad():
            loss,acc,z,pred = self.epoch_step(dataloader,return_result=True,**kwargs)
        return acc, z, pred
        
        
class WDCNN(BaseModel):
    def __init__(self, in_dim=1, h_dim=64,out_dim=10,model_name='wdcnn',*args,**kwargs):
        super(WDCNN, self).__init__(model_name=model_name, *args,**kwargs)
        self.conv1 = self._conv_block(in_dim, 16, 64, 16, 24)
        self.conv2 = self._conv_block(16, 32, 3, 1, 1)
        self.conv3 = self._conv_block(32,64,3,1,1)
        self.conv4 = self._conv_block(64,h_dim,3,1,1)
        self.conv5 = self._conv_block(h_dim,h_dim,3,1,0)
        self.fc = nn.Linear(h_dim, out_dim)
        
    def _conv_block(self, in_dim, out_dim, k,s,p):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=k,stride=s,padding=p, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )
    def forward(self, x, y, mask):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        z = x.mean(-1)
        
        x = self.fc(z)
        if mask.any():
            loss = nn.CrossEntropyLoss()(x[mask],y[mask])
        else:
            loss=None
        return x, loss, z