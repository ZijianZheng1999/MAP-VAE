import torch

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from tqdm import tqdm
from model.vae.lidar_vae import LidarVAE
from dataset.kitti_point_cloud import KittiPointCloud
from torch.utils.data import DataLoader

sample_ratio = 0.01
total_epoch = 5
dataset_root = 'E:/Dataset/'
batch_size = 1
latent_dim_list = [4, 8, 16, 32, 64, 128, 256, 512]
device = 'cuda'
seed = 19990929

def train(latent_dim):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.deterministic = True
    # define datasets and dataloader
    kitti_lidar_dataset = KittiPointCloud(dataset_root, transform=None)

    # generate the index for training and testing
    max_idx = len(kitti_lidar_dataset)
    train_num = int(max_idx * 0.8)
    test_num = max_idx - train_num
    train_dataset, test_dataset = torch.utils.data.random_split(kitti_lidar_dataset, [train_num, test_num])

    kitti_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=KittiPointCloud.collate_fn_tensor,
                              num_workers=8,
                              pin_memory=True)
    # the small dataset used to do visualization, including cloud points from 0 to 99
    lidar_test_loader = DataLoader(test_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   collate_fn=KittiPointCloud.collate_fn_tensor,
                                   num_workers=8,
                                   pin_memory=True)

    # define model
    ladir_vae = LidarVAE(latent_dim, point_dim=4, batch_parallel=True).to(device)

    # define optimizer
    optimizer = torch.optim.Adam(ladir_vae.parameters())

    # train the model
    with tqdm(range(total_epoch), desc='training') as tbar:
        for epoch in tbar:
            loss_rcd = []
            epoch_rcd = []
            ladir_vae.train()
            for i, data in enumerate(kitti_loader):
                # when batch_parallel is disabled, the data is a list contains [tensor(F, N1), tensor(F, N2), ...]
                if len(data) == 1:
                    point_cloud_batch = data[0]
                    point_cloud_batch = [point_cloud.to(device, non_blocking=True) for point_cloud in point_cloud_batch]

                # when batch_parallel is enabled, the data is: point_cloud [B, F, N], padding_mask [B, F, N]
                else:
                    point_cloud_batch = data[0]
                    mask_batch = data[1]
                    point_cloud_batch = point_cloud_batch.to(device, non_blocking=True)
                    mask_batch = mask_batch.to(device, non_blocking=True)

                recon_points, mu, log_var = ladir_vae(point_cloud_batch)

                # remove the padding points using mask
                point_cloud_batch = point_cloud_batch * mask_batch.unsqueeze(1)

                optimizer.zero_grad()
                loss = ladir_vae.loss_function(recon_points, point_cloud_batch, mu, log_var)
                loss.backward()
                optimizer.step()
                tbar.write(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')
                # save data
                loss_rcd.append(loss.item())
                epoch_rcd.append(i / len(kitti_loader) + epoch)
                # plot picture
                if i % 1000 == 0:
                    # covert the tensor to numpy array
                    # points: [B, F, N]
                    point_cloud_batch = point_cloud_batch.cpu().detach().numpy()
                    recon_points = recon_points.cpu().detach().numpy()
                    # plot the reconstructed images via matplotlib
                    plt.figure()
                    point_size = min(point_cloud_batch.shape[2], recon_points.shape[2])
                    sample_indices = np.random.choice(point_size, int(point_size * sample_ratio), replace=False)
                    fig = plt.figure()
                    ax1 = fig.add_subplot(121, projection='3d')
                    ax1.scatter(point_cloud_batch[0, 0, sample_indices], point_cloud_batch[0, 1, sample_indices], point_cloud_batch[0, 2, sample_indices],
                                c='r', marker='.', s=1)
                    ax1.set_title('Original Point Cloud')
                    ax1.view_init(elev=90, azim=-90)
                    ax1.set_xlim(-10, 10)
                    ax1.set_ylim(-10, 10)
                    ax1.set_zlim(-10, 10)

                    ax2 = fig.add_subplot(122, projection='3d')
                    ax2.scatter(recon_points[0, 0, sample_indices], recon_points[0, 1, sample_indices], recon_points[0, 2, sample_indices], c='b', marker='.',
                                s=1)
                    ax2.set_title('Reconstructed Point Cloud')
                    ax2.view_init(elev=90, azim=-90)
                    ax2.set_xlim(-10, 10)
                    ax2.set_ylim(-10, 10)
                    ax2.set_zlim(-10, 10)

                    plt.title(f'Epoch: {epoch}, loss: {loss.item()}')
                    plt.show()
                    plt.close()

                # save the model with epoch number as name
                torch.save(ladir_vae.state_dict(), f'records/lidar_vae_single_modal/checkpoints-{latent_dim}.pth')

            ladir_vae.eval()
            test_loss = 0
            with torch.no_grad():
                for i, data in enumerate(lidar_test_loader):
                    # when batch_parallel is disabled, the data is a list contains [tensor(F, N1), tensor(F, N2), ...]
                    if len(data) == 1:
                        point_cloud_batch = data[0]
                        point_cloud_batch = [point_cloud.to(device, non_blocking=True) for point_cloud in
                                             point_cloud_batch]
                    # when batch_parallel is enabled, the data is: point_cloud [B, F, N], padding_mask [B, F, N]
                    else:
                        point_cloud_batch = data[0]
                        mask_batch = data[1]
                        point_cloud_batch = point_cloud_batch.to(device, non_blocking=True)
                        mask_batch = mask_batch.to(device, non_blocking=True)
                    # test the model
                    recon_points, mu, log_var = ladir_vae(point_cloud_batch)
                    # remove the padding points using mask
                    point_cloud_batch = point_cloud_batch * mask_batch.unsqueeze(1)
                    # get the test loss
                    test_loss += ladir_vae.loss_function(recon_points, point_cloud_batch, mu, log_var).item() * point_cloud_batch.size(0)
            test_loss /= len(test_dataset)
            tbar.write(f'Epoch: {epoch}, Test Loss: {test_loss}')
            # save the test loss record
            dataframe = pd.DataFrame({'epoch': [epoch], 'train_loss': [test_loss]})
            if epoch == 0:
                dataframe.to_csv(f'records/lidar_vae_single_modal/test_plot-{latent_dim}.csv', index=False)
            else:
                dataframe.to_csv(f'records/lidar_vae_single_modal/test_plot-{latent_dim}.csv', index=False, mode='a',
                                 header=False)
            # save the train loss record
            dataframe = pd.DataFrame({'epoch': epoch_rcd, 'loss': loss_rcd})
            if epoch == 0:
                dataframe.to_csv(f'records/lidar_vae_single_modal/train_plot-{latent_dim}.csv', index=False)
            else:
                dataframe.to_csv(f'records/lidar_vae_single_modal/train_plot-{latent_dim}.csv', index=False, mode='a',
                                 header=False)
        # save the model
        torch.save(ladir_vae.state_dict(), f'records/lidar_vae_single_modal/final-{latent_dim}.pth')


if __name__ == '__main__':
    for latent_dim in latent_dim_list:
        train(latent_dim)
