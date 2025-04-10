import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from tqdm import tqdm

from model.vae.camera_vae import CameraVAE

total_epoch = 40
dataset_root = '../'
# dataset_root = 'E:/Dataset/'
batch_size = 1
latent_dim_list = [32, 64, 128, 256, 512]
device = 'cuda'
seed = 19990929


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets


def train(latent_dim):
    # set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.deterministic = True    # define datasets
    kitti_camera_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((375, 1224)),
        torchvision.transforms.CenterCrop((375, 1224)),
        torchvision.transforms.ToTensor()
    ])

    kitti_img_dataset = torchvision.datasets.Kitti(dataset_root, download=False, transform=kitti_camera_transform)

    # generate the index for training and testing
    max_idx = len(kitti_img_dataset)
    train_num = int(max_idx * 0.8)
    test_num = max_idx - train_num
    train_dataset, test_dataset = torch.utils.data.random_split(kitti_img_dataset, [train_num, test_num])

    kitti_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=collate_fn,
                                               num_workers=4,
                                               pin_memory=True)
    img_test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  num_workers=4,
                                                  pin_memory=True)

    # define model
    camera_vae = CameraVAE(latent_dim)
    camera_vae.to(device)

    # define optimizer
    optimizer = torch.optim.Adam(camera_vae.parameters())

    # train the model
    with tqdm(range(total_epoch), desc='training') as tbar:
        for epoch in tbar:
            loss_rcd = []
            epoch_rcd = []
            camera_vae.train()
            for i, (img, _) in enumerate(kitti_loader):
                img = img.to(device)
                optimizer.zero_grad()
                recon_img, mu, log_var = camera_vae(img)
                loss = camera_vae.loss_function(recon_img, img, mu, log_var)
                loss.backward()
                optimizer.step()
                tbar.write(f'Epoch: {epoch}, Batch: {i}, Train Loss: {loss.item()}')
                # save data
                loss_rcd.append(loss.item())
                epoch_rcd.append(i / len(kitti_loader) + epoch)
                # plot picture
                if i % 500 == 0:
                    # plot the reconstructed images via matplotlib
                    plt.figure()
                    img_show_recon = recon_img[0, :, :, :].cpu().detach().numpy()
                    img_show = img[0, :, :, :].cpu().detach().numpy()
                    plt.subplot(1, 2, 1)
                    plt.imshow(np.transpose(img_show_recon, (1, 2, 0)))
                    plt.subplot(1, 2, 2)
                    plt.imshow(np.transpose(img_show, (1, 2, 0)))
                    plt.show()
                    plt.close()
            # save the model with epoch number as name
            torch.save(camera_vae.state_dict(), f'records/camera_vae_single_modal/checkpoints-{latent_dim}.pth')
            # # save the constructed images from display subset
            camera_vae.eval()
            test_loss = 0
            with torch.no_grad():
                for i, (img, _) in enumerate(img_test_loader):
                    img = img.to(device)
                    recon_img, mu, log_var = camera_vae(img)
                    test_loss += camera_vae.loss_function(recon_img, img, mu, log_var).item() * img.size(0)
                    # plt.figure()
                    # img_show_recon = recon_img.cpu().detach().numpy()
                    # for j in range(img_show_recon.shape[0]):
                    #     plt.imshow(np.transpose(img_show_recon[j, :, :, :], (1, 2, 0)))
                    #     plt.axis('off')
                    #     plt.savefig(
                    #         f'records/camera_vae_single_modal/pic/{i * img_show_recon.shape[0] + j}_{epoch}.png',
                    #         bbox_inches='tight', dpi=300)
                    # plt.close()
            test_loss /= len(test_dataset)
            tbar.write(f'Epoch: {epoch}, Test Loss: {test_loss}')
            # save the loss record
            dataframe = pd.DataFrame({'epoch': [epoch], 'train_loss': [test_loss]})
            if epoch == 0:
                dataframe.to_csv(f'records/camera_vae_single_modal/test_plot-{latent_dim}.csv', index=False)
            else:
                dataframe.to_csv(f'records/camera_vae_single_modal/test_plot-{latent_dim}.csv', index=False, mode='a', header=False)
            # tbar.update()
            # save the loss record
            dataframe = pd.DataFrame({'epoch': epoch_rcd, 'train_loss': loss_rcd})
            if epoch == 0:
                dataframe.to_csv(f'records/camera_vae_single_modal/train_plot-{latent_dim}.csv', index=False)
            else:
                dataframe.to_csv(f'records/camera_vae_single_modal/train_plot-{latent_dim}.csv', index=False, mode='a', header=False)
    # save the model
    torch.save(camera_vae.state_dict(), f'records/camera_vae_single_modal/final-{latent_dim}.pth')

if __name__ == '__main__':
    for latent_set in latent_dim_list:
        train(latent_set)