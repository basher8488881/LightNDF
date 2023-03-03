import sys
sys.path.append('../LightNDF')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

"""
This implementation is adopted from following study. 
Chibane, Julian, and Gerard Pons-Moll. 
"Neural unsigned distance fields for implicit function learning." 
Advances in Neural Information Processing Systems 33 (2020): 21638-21652.

"""

class LightNDF(nn.Module):


    def __init__(self, hidden_dim=128):
        super(LightNDF, self).__init__()

        self.conv_in = nn.Conv3d(1, 16, 3, padding=1)  
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1)  
        self.conv_0_1 = nn.Conv3d(32, 64, 3, padding=1)  
        self.conv_1 = nn.Conv3d(64, 64, 3, padding=1)  
        self.conv_1_1 = nn.Conv3d(64, 128, 3, padding=1)  
        displacments_num  = 1
        feature_size = (1 +  16 + 64 + 128 ) * displacments_num  + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim * 2, 1)
        self.fc_out = nn.Conv1d(hidden_dim * 2, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(64)
        self.conv1_1_bn = nn.BatchNorm3d(128)
        
        displacments = []
        displacments.append([0, 0, 0])

        assert len(displacments) == displacments_num
     
        self.displacments = torch.Tensor(displacments).cuda()

    def encoder(self,x):
        x = x.unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        return f_0, f_1, f_2, f_3 

    def decoder(self, p, f_0, f_1, f_2, f_3):
        device = torch.device("cuda")
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d.to(device) for d in self.displacments], dim=2)

        feature_0 = F.grid_sample(f_0, p, align_corners=True)
        feature_1 = F.grid_sample(f_1, p, align_corners=True)
        feature_2 = F.grid_sample(f_2, p, align_corners=True)
        feature_3 = F.grid_sample(f_3, p, align_corners=True)
        features = torch.cat((feature_0, feature_1, feature_2, feature_3),
                             dim=1)  

        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  
        features = torch.cat((features, p_features), dim=1)
        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_out(net))
        out = net.squeeze(1)
        return  out

    def forward(self, p, x):
        out = self.decoder(p, *self.encoder(x))
        return out


