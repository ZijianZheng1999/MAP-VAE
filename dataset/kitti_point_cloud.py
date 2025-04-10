import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class KittiPointCloud(Dataset):
    def __init__(self, root, transform=None):
        super(KittiPointCloud, self).__init__()
        self.root = root
        self.transform = transform
        self.root = os.path.join(root, 'Kitti', 'raw', 'training', 'velodyne')
        self.point_files = [f for f in os.listdir(self.root) if f.endswith('.bin')]

    def __len__(self):
        return len(self.point_files)

    # single data shape: [F, N]
    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.point_files[idx])
        # point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(4, -1, order='F')
        point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        point_cloud = torch.from_numpy(point_cloud).float().permute(1, 0)
        if self.transform:
            point_cloud = self.transform(point_cloud)
        return point_cloud,

    # do not collate the batch into tenser, return [(tensor(N1, 4), ), (tensor(N2, 4), ), ...]
    @staticmethod
    def collate_fn_list(batch):
        return batch

    # padding to utilize 'batch_parallel' in PointNetPlusEnc and PointNetPlusDec
    @staticmethod
    def collate_fn_tensor(batch):

        max_points = max([data[0].shape[1] for data in batch])

        padded_list = []
        mask_list = []

        for data in batch:

            point_cloud = data[0]
            # object_label = data[1]

            num_points = point_cloud.shape[1]

            # original shape: [F, N]
            # pad max_points - num_points in last dim in right
            pad_size = (0, max_points - num_points)
            padded_cloud = torch.nn.functional.pad(point_cloud, pad_size, 'constant', 0)

            # mask for padding (True for real data, False for padding)
            mask = torch.zeros(max_points, dtype=torch.bool)
            mask[:num_points] = True

            padded_list.append(padded_cloud)
            mask_list.append(mask)

        padded_tensor = torch.stack(padded_list, dim=0)
        mask_tensor = torch.stack(mask_list, dim=0)

        return padded_tensor, mask_tensor
