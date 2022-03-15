from __future__ import division
import sys
sys.path.append('../LightNDF')
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import time
from torchsummary import summary
import numpy as np 
import GPUtil

"""
This implementation is adopted from NDF- Neural Unsigned Distance Fields. 
"""

class Trainer(object):

    def __init__(self, model, device, train_dataset, val_dataset, exp_name,gpu_index, optimizer='Adam', lr = 1e-4, threshold = 0.1):
        self.device = device
        if (not gpu_index == 'all'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)
        self.model = model 
        self.model = torch.nn.DataParallel(self.model).to(device)
        if torch.cuda.is_available():
            self.model.cuda()
        
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr= lr)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters())
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.exp_path = os.path.dirname(__file__) + '/experiments/{}/'.format( exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format( exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None
        self.max_dist = threshold


    def train_step(self,batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self,batch):
        device = self.device

        p = batch.get('grid_coords').to(device)
        df_gt = batch.get('df').to(device) 
        inputs = batch.get('inputs').to(device) 
        df_pred = self.model(p,inputs) 
        #GPUtil.showUtilization(all=True)

        loss_i = torch.nn.L1Loss(reduction='none')(torch.clamp(df_pred, max=self.max_dist),torch.clamp(df_gt, max=self.max_dist))
        loss = loss_i.sum(-1).mean() 

        return loss

    def train_model(self, epochs):
        loss = 0
        train_data_loader = self.train_dataset.get_loader()
        start, training_time = self.load_checkpoint()
        iteration_start_time = time.time()
        Loss_list = []
        for epoch in range(start, epochs):
            sum_loss = 0
            print('**********************************')
            print('\n')
            print('Start epoch {}'.format(epoch))
            print('\n')
            print('**********************************')


            for batch in train_data_loader:
               
                iteration_duration = time.time() - iteration_start_time
                if iteration_duration > 30 * 30: 
                    training_time += iteration_duration
                    iteration_start_time = time.time()

                    self.save_checkpoint(epoch, training_time)
                    val_loss = self.compute_val_loss()

                    if self.val_min is None:
                        self.val_min = val_loss

                    if val_loss < self.val_min:
                        self.val_min = val_loss
                        for path in glob(self.exp_path + 'val_min=*'):
                            os.remove(path)
                        np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, val_loss])

                    self.writer.add_scalar('val loss batch avg', val_loss, epoch)

                #optimize model
                loss = self.train_step(batch)
                print("Current loss: {}".format(loss / self.train_dataset.num_sample_points))
                sum_loss += loss
                ccloss = loss / self.train_dataset.num_sample_points
                Loss_list.append(ccloss)



            self.writer.add_scalar('training loss last batch', loss, epoch)
            self.writer.add_scalar('training loss batch avg', sum_loss / len(train_data_loader), epoch)
        np.save('/media/digiaires/8TBHDD1/LightNDF_V3/ndf-master/Training_Losses/LightNDF_V3_loss_Curve_cars_extraTraining.npy', Loss_list)



    def save_checkpoint(self, epoch, training_time):
        path = self.checkpoint_path + 'checkpoint_{}h_{}m_{}s_{}.tar'.format(*[*convertSecs(training_time),training_time])
        if not os.path.exists(path):
            torch.save({ #'state': torch.cuda.get_rng_state_all(),
                        'training_time': training_time ,'epoch':epoch,
                        'model_state_dict': self.model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path)



    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0,0

        checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=float)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + 'checkpoint_{}h_{}m_{}s_{}.tar'.format(*[*convertSecs(checkpoints[-1]),checkpoints[-1]])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        training_time = checkpoint['training_time']
        return epoch, training_time

    def compute_val_loss(self):
        self.model.eval()

        sum_val_loss = 0
        num_batches = 15
        for _ in range(num_batches):
            try:
                val_batch = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_dataset.get_loader().__iter__()
                val_batch = self.val_data_iterator.next()

            sum_val_loss += self.compute_loss( val_batch).item()

        return sum_val_loss / num_batches

def convertMillis(millis):
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (1000 * 60)) % 60)
    hours = int((millis / (1000 * 60 * 60)))
    return hours, minutes, seconds

def convertSecs(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)))
    return hours, minutes, seconds
