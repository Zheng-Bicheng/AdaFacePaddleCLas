from PIL import Image
import numpy as np
from ppcls.data.preprocess import transform as transform_func
import os
from .common_dataset import create_operators
import pickle
from io import BytesIO

import paddle
import paddle.vision.datasets as datasets
from paddle.io import Dataset
from paddle.vision.transforms import transforms as trans
import json


class AdaFaceDataset(datasets.folder.ImageFolder):
    def __init__(self,
                 root,
                 transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None):
        super(AdaFaceDataset, self).__init__(root,
                                             loader=loader,
                                             transform=transform,
                                             is_valid_file=is_valid_file)
        self.root = root
        self.transform = create_operators(transform)

    def __getitem__(self, index):
        path = self.samples[index]
        target = int(path.split("/")[-2])
        with open(path, 'rb') as f:
            img = Image.open(f)
            sample = img.convert('RGB')
        if self.transform is not None:
            sample = transform_func(sample, self.transform)
        return sample, target


class FiveValidationDataset(Dataset):
    def __init__(self, val_data_path):
        super(FiveValidationDataset, self).__init__()
        self.val_data_path = val_data_path

        data_memmap_path = os.path.join(self.val_data_path, "concat_data.dat")
        issame_list_memmap_path = os.path.join(self.val_data_path, "concat_issame_list.dat")
        all_dataname_memmap_path = os.path.join(self.val_data_path, "concat_all_dataname.dat")
        if not os.path.exists(data_memmap_path):
            # create memfiles
            self.val_targets = ["agedb_30", "cfp_fp", "lfw", "cplfw", "calfw"]
            self.val_data_dict = {}
            self.get_val_data(image_size=(112, 112))

            self.dataname_to_idx = {
                "agedb_30": 0,
                "cfp_fp": 1,
                "lfw": 2,
                "cplfw": 3,
                "calfw": 4
            }
            # create concat memfiles
            concat_dataname = []
            key_orders = []
            concat_data = []
            concat_issame_list = []
            for key, (np_array, issame) in self.val_data_dict.items():
                key_orders.append(key)
                concat_dataname.append([self.dataname_to_idx[key]] * len(np_array))
                concat_data.append(np_array)
                dup_issame = []  # hacky way to make the issame length same as imgs. [1, 1, 0, 0, ...]
                for same in issame:
                    dup_issame.append(same)
                    dup_issame.append(same)
                concat_issame_list.append(dup_issame)
            assert key_orders == ['agedb_30', 'cfp_fp', 'lfw', 'cplfw', 'calfw']
            concat_data = np.concatenate(concat_data)
            make_memmap(data_memmap_path, concat_data)
            del concat_data

            concat_issame_list = np.concatenate(concat_issame_list)
            make_memmap(issame_list_memmap_path, concat_issame_list)
            del concat_issame_list

            concat_dataname = np.concatenate(concat_dataname)
            make_memmap(all_dataname_memmap_path, concat_dataname)
            del concat_dataname
        self.all_imgs = read_memmap(data_memmap_path)
        self.all_issame = read_memmap(issame_list_memmap_path)
        self.all_dataname = read_memmap(all_dataname_memmap_path)

    def __getitem__(self, index):
        x_np = self.all_imgs[index].copy()
        x = paddle.to_tensor(x_np)
        y = self.all_issame[index]
        dataname = self.all_dataname[index]
        return x, y, dataname, index

    def __len__(self):
        return len(self.all_imgs)

    def get_val_data(self, image_size):
        for val_name in self.val_targets:
            load_path = os.path.join(self.val_data_path, val_name + ".bin")
            data_memmap_path = os.path.join(self.val_data_path, val_name + "_data.dat")
            issame_list_memmap_path = os.path.join(self.val_data_path, val_name + "_issame_list.dat")
            if not os.path.exists(data_memmap_path):
                print("正在读取文件[{}]".format(load_path))
                data, issame_list = load_bin(load_path, image_size)
                make_memmap(data_memmap_path, data.astype('float32'))
                make_memmap(issame_list_memmap_path, issame_list)
                del data, issame_list
            np_array = read_memmap(data_memmap_path)
            issame_list = read_memmap(issame_list_memmap_path)
            self.val_data_dict[val_name] = (np_array, issame_list)


def load_bin(path, image_size):
    # test_transform = trans.Compose([
    #     trans.ToTensor(),
    #     trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ])
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = np.empty((len(bins), 1, 3, image_size[0], image_size[1]))
    for i in range(len(bins)):
        _bin_1 = bins[i]
        img_ori = Image.open(BytesIO(_bin_1))
        if img_ori.mode != 'RGB':
            img_ori = img_ori.convert('RGB')
        img_ori = np.array(img_ori).astype('float32').transpose((2, 0, 1))
        img_ori = (img_ori - 127.5) * 0.00784313725
        # data[i, ...] = test_transform(img_ori)
        data[i, ...] = img_ori
    return data.reshape((len(bins), 3, image_size[0], image_size[1])), np.array(issame_list)


def make_memmap(mem_file_name, np_to_copy):
    memmap_configs = dict()
    memmap_configs['shape'] = shape = tuple(np_to_copy.shape)
    memmap_configs['dtype'] = dtype = str(np_to_copy.dtype)
    json.dump(memmap_configs, open(mem_file_name + '.conf', 'w'))
    # w+ mode: Create or overwrite existing file for reading and writing
    mm = np.memmap(mem_file_name, mode='w+', shape=shape, dtype=dtype)
    mm[:] = np_to_copy[:]
    mm.flush()  # memmap data flush
    return mm


def read_memmap(mem_file_name):
    # r+ mode: Open existing file for reading and writing
    with open(mem_file_name + '.conf', 'r') as file:
        memmap_configs = json.load(file)
        return np.memmap(mem_file_name, mode='r+',
                         shape=tuple(memmap_configs['shape']),
                         dtype=memmap_configs['dtype'])
