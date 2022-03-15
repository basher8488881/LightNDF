import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models import training
import torch
import configs.config_loader as cfg_loader
"""
This implementation is adopted from NDF- Neural Unsigned Distance Fields. 
"""
cfg = cfg_loader.get_config()
net = model.LightNDF(device=torch.device("cuda"))
if cfg.gpu == "auto":
    deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                    excludeUUID=[])
    gpu = deviceIDs[0]
else:
    gpu = cfg.gpu

train_dataset = voxelized_data.VoxelizedDataset('train',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=cfg.split_file,
                                          batch_size=cfg.batch_size,
                                          num_sample_points=cfg.num_sample_points_training,
                                          num_workers=8,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)
val_dataset = voxelized_data.VoxelizedDataset('val',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=cfg.split_file,
                                          batch_size=cfg.batch_size,
                                          num_sample_points=cfg.num_sample_points_training,
                                          num_workers=8,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)



trainer = training.Trainer(net,
                           torch.device("cuda"),
                           train_dataset,
                           val_dataset,
                           cfg.exp_name,
                           gpu_index= gpu,
                           optimizer=cfg.optimizer,
                           lr=cfg.lr)

trainer.train_model(cfg.num_epochs)
