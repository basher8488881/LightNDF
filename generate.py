import sys
sys.path.append('../ndf-master')
import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models.generation import Generator
import torch
import configs.config_loader as cfg_loader
import os
import trimesh
import numpy as np
from tqdm import tqdm
"""
This implementation is adopted from following study. 
Chibane, Julian, and Gerard Pons-Moll. 
"Neural unsigned distance fields for implicit function learning." 
Advances in Neural Information Processing Systems 33 (2020): 21638-21652.

"""
cfg = cfg_loader.get_config()

#device = torch.device("cpu")
device = torch.device("cuda")
net = model.LightNDF(device)

dataset = voxelized_data.VoxelizedDataset('test',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=cfg.split_file,
                                          batch_size=1,
                                          num_sample_points=cfg.num_sample_points_generation,
                                          num_workers=8,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)

gen = Generator(net, cfg.exp_name, device=device)

out_path = 'experiments/{}/evaluation/'.format(cfg.exp_name)


def gen_iterator(out_path, dataset, gen_p):
    global gen
    gen = gen_p

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)

    # can be run on multiple machines: dataset is shuffled and already generated objects are skipped.
    loader = dataset.get_loader(shuffle=True)

    for i, data in tqdm(enumerate(loader)):

        path = os.path.normpath(data['path'][0])
        export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])

        if os.path.exists(export_path):
            print('Path exists - skip! {}'.format(export_path))
            continue
        else:
            os.makedirs(export_path)

        for num_steps in [5]:
            point_cloud, duration = gen.generate_point_cloud(data, num_steps)
            ## Here we are reversing the rotation and scaling 
            point_cloudx =point_cloud[:,0]
            point_cloudx = point_cloudx[:,np.newaxis]
            point_cloudy =point_cloud[:,1]
            point_cloudy = point_cloudy[:,np.newaxis]
            point_cloudz =point_cloud[:,2]
            point_cloudz = point_cloudz[:,np.newaxis]
            point_cloud_r = np.concatenate((point_cloudz, point_cloudy, point_cloudx), axis=1)
            point_cloud_r = point_cloud_r / 2
            ## Save the generated dense point cloud ###
            np.savez(export_path + 'dense_point_cloud_{}'.format(num_steps), point_cloud=point_cloud_r, duration=duration)
            print('num_steps', num_steps, 'duration', duration)
            trimesh.Trimesh(vertices=point_cloud_r, faces=[]).export(
                export_path + 'dense_point_cloud_{}.off'.format(num_steps))


gen_iterator(out_path, dataset, gen)
