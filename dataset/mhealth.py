import os
import torch
import pandas as pd

from torch.utils.data.dataset import Dataset


class MHealth(Dataset):
    def __init__(self, path, key='all'):
        super(MHealth, self).__init__()
        # get the file path to read raw data
        sub_path = 'MHEALTHDATASET'
        data_path = os.path.join(path, sub_path)
        file_name = [
            'mHealth_subject1.log', 'mHealth_subject2.log', 'mHealth_subject3.log', 'mHealth_subject4.log',
            'mHealth_subject5.log', 'mHealth_subject6.log', 'mHealth_subject7.log', 'mHealth_subject8.log',
            'mHealth_subject9.log', 'mHealth_subject10.log'
        ]
        # get the dataframe
        df = pd.concat([pd.read_csv(os.path.join(data_path, fn), header=None, sep='\t') for fn in file_name],
                       ignore_index=True)

        self.modal_idx = {
            'acc_chest': [0, 1, 2],
            'elc_cardiogram': [3, 4],
            'acc_left_ankle': [5, 6, 7],
            'gyro_left_ankle': [8, 9, 10],
            'mag_left_ankle': [11, 12, 13],
            'acc_right_ankle': [14, 15, 16],
            'gyro_right_ankle': [17, 18, 19],
            'mag_right_ankle': [20, 21, 22],
            'tag': 23,
            'all': [i for i in range(23)]
        }
        self.data = torch.tensor(df.values[:, self.modal_idx[key]])
        self.tag = torch.tensor(df.values[:, self.modal_idx['tag']])
        self.class_num = len(torch.unique(self.tag))

    def __getitem__(self, index):
        return self.data[index, :], self.tag[index]

    def __len__(self):
        return len(self.tag)

    @staticmethod
    def get_all_mono_modal_data(path):
        return {
            'acc_chest': MHealth(path, 'acc_chest'),
            'elc_cardiogram': MHealth(path, 'elc_cardiogram'),
            'acc_left_ankle': MHealth(path, 'acc_left_ankle'),
            'gyro_left_ankle': MHealth(path, 'gyro_left_ankle'),
            'mag_left_ankle': MHealth(path, 'mag_left_ankle'),
            'acc_right_ankle': MHealth(path, 'acc_right_ankle'),
            'gyro_right_ankle': MHealth(path, 'gyro_right_ankle'),
            'mag_right_ankle': MHealth(path, 'mag_right_ankle'),
            'all': MHealth(path, 'all')
        }

