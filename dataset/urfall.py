import numpy as np
import os
import torch
import torchvision
import pandas as pd
import PIL

from torch.utils.data.dataset import Dataset
from torchvision.datapoints import Image
from torchvision import transforms, datasets


# return size:
# acc: [batch(ADL40+Fall30) * tensor(length(approx 250), 3)]: length indicates samples across time, 3 indicates x, y, z axis of accelerometer
# img: [batch(ADL40+Fall30) * list[length(approx 250)]]: length indicates samples across time, ch indicates channels of image (1/3), others are size
# tag: [batch(ADL40+Fall30), 2 (One-Hot) ]: indicates the tag of each sample, 0 for ADL, 1 for Fall

class UrFall(Dataset):
    def __init__(self, path, key='acc'):
        super(UrFall, self).__init__()
        adl_seq_path = os.path.join(path, 'URFall', 'ADL_sequences')
        fall_seq_path = os.path.join(path, 'URFall', 'Fall_sequences')

        self.key = key

        # self.data = torch[batch][seq_len(max400)][3(xyz)], [[cam0_d_path(max400)] * batch], [[cam0_rgb_path(max400)] * batch]
        self.data = [], [], []
        self.tag = []
        self.len = 0
        # used to pad the sequence (max=400)
        self.max_seq_len = 0

        # walk in ADL_sequences file and Fall_sequences file
        for type_idx in range(2):
            for dir_idx in range(1, 41):
                if type_idx == 0:
                    sample_path = os.path.join(adl_seq_path, str(dir_idx))
                    prefix = 'adl-' + "{:02d}".format(dir_idx) + '-'
                else:
                    if dir_idx == 31:
                        break
                    sample_path = os.path.join(fall_seq_path, str(dir_idx))
                    prefix = 'fall-' + "{:02d}".format(dir_idx) + '-'
                # import acc data
                time_stp = pd.read_csv(os.path.join(sample_path, prefix + 'data.csv'), header=None)
                time_stp = torch.tensor(time_stp.values[:, 1])
                acc_sample = pd.read_csv(os.path.join(sample_path, prefix + 'acc.csv'), header=None)
                acc_sample = torch.tensor(acc_sample.values, dtype=torch.float32)
                sel_idx = [int(torch.argmin(torch.abs(acc_sample[:, 0] - t))) for t in time_stp]
                acc_sample = acc_sample[sel_idx, 2:5]
                # import depth image data
                dp_path = [os.path.join(sample_path, prefix + 'cam0-d', f)
                           for f in os.listdir(os.path.join(sample_path, prefix + 'cam0-d'))]
                # import rgb image data
                egb_path = [os.path.join(sample_path, prefix + 'cam0-rgb', f)
                            for f in os.listdir(os.path.join(sample_path, prefix + 'cam0-rgb'))]
                # load data
                data_acc, data_dp, data_rgb = self.data
                data_acc.append(acc_sample)
                data_dp.append(dp_path)
                data_rgb.append(egb_path)
                self.data = data_acc, data_dp, data_rgb
                self.tag.append(type_idx)

        self.tag = torch.stack([torch.tensor(t) for t in self.tag], dim=0)
        #self.tag = torch.nn.functional.one_hot(self.tag, num_classes=2).float()

        data_acc, path_dp, path_rgb = self.data
        for acc_sample in data_acc:
            self.max_seq_len = max(self.max_seq_len, acc_sample.shape[0])
        # self.max_seq_len = 50

        self.to_tensor = transforms.ToTensor()

        # derive the mean amd std of data

        # read the calculated mean and std of raw data
        if os.path.exists(os.path.join(path, 'URFall', 'normalization.pth')):
            print('load normalization data')
            mean_acc, std_acc, mean_dp, std_dp, mean_rgb, std_rgb = torch.load(os.path.join(path, 'URFall', 'normalization.pth'))
        else:
            print('fail in data load, calculating mean and std...')

            # acc data
            data_acc = torch.cat(data_acc, dim=0)
            mean_acc = torch.mean(data_acc, dim=0)
            std_acc = torch.std(data_acc, dim=0)
            print('mean_acc = {}, std_acc = {}'.format(mean_acc, std_acc))
            # mean_acc = torch.tensor([-0.0065,  0.6310,  0.0779])
            # std_acc = torch.tensor([0.5312, 0.5485, 0.5891])

            # depth img
            n = 0
            mean_dp = torch.tensor([0.0])
            mean_sq_dp = torch.tensor([0.0])
            for sub_path in path_dp:
                for p in sub_path:
                    # img[1(sample), 1(channel), 480(height), 640(width)], 0-65535 int
                    img = torch.stack([Image(self.to_tensor(PIL.Image.open(p)))]) / 65535.0
                    n += img.numel()
                    delta = img - mean_dp.view([1, 1, 1, 1])
                    mean_dp += torch.sum(delta, dim=[0, 2, 3]) / n
                    delta2 = img - mean_dp.view([1, 1, 1, 1])
                    mean_sq_dp += torch.sum(delta * delta2, dim=[0, 2, 3])
            std_dp = torch.sqrt(mean_sq_dp / (n - 1))
            print('mean_dp = {}, std_dp = {}'.format(mean_dp, std_dp))
            # mean_dp = torch.tensor([0.4969])
            # std_dp = torch.tensor([0.2238])

            # rbg img
            n = 0
            mean_rgb = torch.tensor([0.0, 0.0, 0.0])
            mean_sq_rgb = torch.tensor([0.0, 0.0, 0.0])
            for sub_path in path_rgb:
                for p in sub_path:
                    # img[1(sample), 3(channel), 480(height), 640(width)], 0-1 float
                    img = torch.stack([Image(self.to_tensor(PIL.Image.open(p)))])
                    n += img.numel() / 3
                    delta = img - mean_rgb.view([1, 3, 1, 1])
                    mean_rgb += torch.sum(delta, dim=[0, 2, 3]) / n
                    delta2 = img - mean_rgb.view([1, 3, 1, 1])
                    mean_sq_rgb += torch.sum(delta * delta2, dim=[0, 2, 3])
            std_rgb = torch.sqrt(mean_sq_rgb / (n - 1))
            print('mean_rgb = {}, std_rgb = {}'.format(mean_rgb, std_rgb))
            # mean_rgb = torch.tensor([0.4555, 0.3935, 0.3859])
            # std_rgb = torch.tensor([0.2786, 0.2792, 0.2831])
            torch.save([mean_acc, std_acc, mean_dp, std_dp, mean_rgb, std_rgb],
                       os.path.join(path, 'URFall', 'normalization.pth'))
            print('finish calculating, save normalization data')

        self.mean_acc = mean_acc
        self.std_acc = std_acc

        self.mean_imgbp = mean_dp
        self.std_imgbp = std_dp

        self.mean_imgrgb = mean_rgb
        self.std_imgrgb = std_rgb

        self.transform_imgdp = transforms.Compose([
            transforms.Normalize(mean=mean_dp, std=std_dp)
        ])
        self.transform_imgrgb = transforms.Compose([
            transforms.Normalize(mean=mean_rgb, std=std_rgb)
        ])

    def __len__(self):
        return int(self.tag.shape[0])

    # acc: [length, 3]: length indicates samples across time, 3 indicates x, y, z axis of accelerometer
    # img: [length, ch, 480, 640]: length indicates samples across time, ch indicates channels of image (1/3), others are size
    # all: (acc[length, 3], img[length, rgb+depth (4), 480, 640], tag)
    def __getitem__(self, index):
        # fetch data and tag
        tag = self.tag[index]
        sensor_data_3d, image_dp_path, image_rgb_path = self.data
        raw_seq_len = sensor_data_3d[index].shape[0]
        # if dataset is acc mono-modal
        if self.key == 'acc' or self.key == 'all':
            acc_data = sensor_data_3d[index]
            # acc_data = self.transform_acc(sensor_data_3d[index])
            # normalize the mean of the acc data
            acc_data -= torch.mean(acc_data, dim=0, keepdim=True)
            # normalize the power of acc data
            max_power, _ = torch.max(torch.sum(torch.square(sensor_data_3d[index]), dim=1), dim=0)
            acc_data /= torch.sqrt(max_power.view(1, 1))
            # padding
            if acc_data.shape[0] < self.max_seq_len:
                acc_data = torch.cat([acc_data, torch.zeros([self.max_seq_len - acc_data.shape[0], 3])], dim=0)
            #acc_data = acc_data[0:self.max_seq_len, :]
        # if dataset is image mono-modal
        if 'cam0' in self.key or self.key == 'all':
            if '_d' in self.key:
                image = torch.stack([Image(self.to_tensor(PIL.Image.open(path)) / 65535.0) for path in image_dp_path[index]])
                image = self.transform_imgdp(image)
            elif '_rgb' in self.key:
                image = torch.stack([Image(self.to_tensor(PIL.Image.open(path))) for path in image_rgb_path[index]])
                image = self.transform_imgrgb(image)
            elif self.key == 'all':
                image = torch.cat([
                    self.transform_imgdp(torch.stack([Image(self.to_tensor(PIL.Image.open(path)) / 65535.0) for path in image_dp_path[index]])),
                    self.transform_imgrgb(torch.stack([Image(self.to_tensor(PIL.Image.open(path))) for path in image_rgb_path[index]])),
                ], dim=1)
            # padding
            if image.shape[0] < self.max_seq_len:
                image = torch.cat([image, torch.zeros([self.max_seq_len - image.shape[0], image.shape[1], image.shape[2], image.shape[3]])], dim=0)
        # return data
        if self.key == 'acc':
            return acc_data, tag, raw_seq_len
        elif self.key == 'cam0_d':
            return image, tag, raw_seq_len
        elif self.key == 'cam0_rgb':
            return image, tag, raw_seq_len
        elif self.key == 'all':
            return acc_data, image, tag, raw_seq_len

    @staticmethod
    def get_all_mono_modal_data(path):
        return {
            'acc': UrFall(path,'acc'),
            'cam0_d': UrFall(path,'cam0_d'),
            'cam0_rgb': UrFall(path,'cam0_rgb'),
            'all': UrFall(path,'all')
        }

    def transform_acc(self, x):
        return (x - self.mean_acc.view(1, 3)) / self.std_acc.view(1, 3)
