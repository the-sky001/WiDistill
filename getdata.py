import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import random



##################xrf55_normal_downsample 500   

# class XRFBertDatasetNewMix(Dataset):
#     def __init__(self, file_path='/home/zhumengdie/XRFDataset/new_data/', text_path='/home/wangtiantian/code/XRF55-repo-main/dataset/XRF_dataset/',is_train=True, scene='dml'):
#         super(XRFBertDatasetNewMix, self).__init__()
#         self.file_path = file_path
#         self.text_path = text_path
#         self.is_train = is_train
#         self.scene = scene
#         if self.is_train:
#             self.text = self.text_path + self.scene + '_train.txt'
#         else:
#             self.text = self.text_path + self.scene + '_val.txt'
#         text = open(self.text)
#         val_list = text.readlines()
#         self.data = {
#             'file_name': list(),
#             'label': list()
#         }
#         self.path = self.file_path 
#         for string in val_list:
#             self.data['file_name'].append(string.split(',')[0])
#             self.data['label'].append(int(string.split(',')[2]) - 1)
#         # log.info("load XRF dataset")
#         self.mean = 9.6302
#         self.std = 3.8425
#     def __len__(self):
#         return len(self.data['label'])

#     def __getitem__(self, idx):
#         file_name = self.data['file_name'][idx]
#         label = self.data['label'][idx]
#         wifi_data = load_wifi(file_name, self.is_train, path=self.path)
#         # 归一化数据
#         normalized_wifi_data = (wifi_data - self.mean) / self.std
#         # print(normalized_wifi_data.shape)
#         # normalized_wifi_data_new = torch.zeros_like(normalized_wifi_data[:, :100])  # 假设降采样到10个特征
#         normalized_wifi_data_new = torch.zeros_like(normalized_wifi_data[:, :500])  # 假设降采样到10个特征

#         # for i in range(100):
#         #     start = i * 10
#         #     end = start + 10
#         #     normalized_wifi_data_new[:, i] = torch.mean(normalized_wifi_data[:, start:end], axis=1)
#         for i in range(500):
#             start = i * 2
#             end = start + 2
#             normalized_wifi_data_new[:, i] = torch.mean(normalized_wifi_data[:, start:end], axis=1)
#         # print(normalized_wifi_data_new.shape)
#         return normalized_wifi_data_new, label
# def load_wifi(filename, is_train, path='./dataset/XRFDataset/'):
#     if is_train:
#         path = path + 'train_data/'
#     else:
#         path = path + 'test_data/'
#     record = np.load(path + 'WiFi/' + filename + ".npy")
#     return torch.from_numpy(record).float()


##################xrf55 IPC 50 downsample (270,500)

# import random

# class XRFBertDatasetNewMix(Dataset):
#     def __init__(self, file_path='/home/zhumengdie/XRFDataset/new_data/', text_path='/home/wangtiantian/code/XRF55-repo-main/dataset/XRF_dataset/', is_train=True, scene='dml', samples_per_class=50):
#         super(XRFBertDatasetNewMix, self).__init__()
#         self.file_path = file_path
#         self.text_path = text_path
#         self.is_train = is_train
#         self.scene = scene
#         self.samples_per_class = samples_per_class
        
#         # 设置数据文件路径
#         self.text = self.text_path + self.scene + ('_train.txt' if is_train else '_val.txt')
        
#         self.data = {
#             'file_name': [],
#             'label': []
#         }
        
#         with open(self.text, 'r') as f:
#             val_list = f.readlines()
        
#         # 创建列表存储每个类别的索引
#         class_indices = [[] for _ in range(55)]
#         for string in val_list:
#             file_name = string.split(',')[0]
#             label = int(string.split(',')[2]) - 1
#             class_indices[label].append(file_name)
        
#         # 随机抽取每个类别的指定数量的样本
#         for label, indices in enumerate(class_indices):
#             selected_samples = random.sample(indices, min(len(indices), self.samples_per_class))
#             self.data['file_name'].extend(selected_samples)
#             self.data['label'].extend([label] * len(selected_samples))

#     def __len__(self):
#         return len(self.data['label'])

#     def __getitem__(self, idx):
#         file_name = self.data['file_name'][idx]
#         label = self.data['label'][idx]
#         wifi_data = self.load_wifi(file_name, self.is_train, path=self.file_path)

#         # 归一化数据
#         mean = 9.6302
#         std = 3.8425
#         normalized_wifi_data = (wifi_data - mean) / std

#         # 降采样到500个特征
#         normalized_wifi_data_new = torch.zeros_like(normalized_wifi_data[:, :500])  # 初始化为500个特征的新数据
#         for i in range(500):
#             start = i * 2
#             end = start + 2
#             normalized_wifi_data_new[:, i] = torch.mean(normalized_wifi_data[:, start:end], axis=1)
        
#         return normalized_wifi_data_new, label

#     def load_wifi(self, filename, is_train, path):
#         if is_train:
#             path = path + 'train_data/'
#         else:
#             path = path + 'test_data/'
#         record = np.load(path + 'WiFi/' + filename + ".npy")
#         return torch.from_numpy(record).float()


############  xrf (30,100) 

import torch
import numpy as np
from torch.utils.data import Dataset

# class XRFBertDatasetNewMix(Dataset):
#     def __init__(self, file_path='/home/zhumengdie/XRFDataset/new_data/', text_path='/home/wangtiantian/code/XRF55-repo-main/dataset/XRF_dataset/', is_train=True, scene='dml'):
#         super(XRFBertDatasetNewMix, self).__init__()
#         self.file_path = file_path
#         self.text_path = text_path
#         self.is_train = is_train
#         self.scene = scene
#         if self.is_train:
#             self.text = self.text_path + self.scene + '_train.txt'
#         else:
#             self.text = self.text_path + self.scene + '_val.txt'
#         with open(self.text) as text:
#             val_list = text.readlines()
#         self.data = {
#             'file_name': [],
#             'label': []
#         }
#         self.path = self.file_path
#         for string in val_list:
#             self.data['file_name'].append(string.split(',')[0])
#             self.data['label'].append(int(string.split(',')[2]) - 1)
#         self.mean = 9.6302
#         self.std = 3.8425

#     def __len__(self):
#         return len(self.data['label'])

#     def __getitem__(self, idx):
#         file_name = self.data['file_name'][idx]
#         label = self.data['label'][idx]
#         wifi_data = load_wifi(file_name, self.is_train, path=self.path)
#         normalized_wifi_data = (wifi_data - self.mean) / self.std

#         # 第一阶段降采样：从 (270, 1000) 到 (270, 100)
#         normalized_wifi_data_stage1 = torch.zeros((270, 100))
#         for i in range(100):
#             start = i * 10
#             end = start + 10
#             normalized_wifi_data_stage1[:, i] = torch.mean(normalized_wifi_data[:, start:end], dim=1)

#         # 第二阶段降采样：从 (270, 100) 到 (30, 100)
#         normalized_wifi_data_new = torch.zeros((30, 100))
#         for i in range(30):
#             start_t = i * 9
#             end_t = start_t + 9
#             normalized_wifi_data_new[i, :] = torch.mean(normalized_wifi_data_stage1[start_t:end_t, :], dim=0)

#         return normalized_wifi_data_new, label

def load_wifi(filename, is_train, path='./dataset/XRFDataset/'):
    if is_train:
        path = path + 'train_data/'
    else:
        path = path + 'test_data/'
    record = np.load(path + 'WiFi/' + filename + ".npy")
    return torch.from_numpy(record).float()

###################### xrf (90,100)

# class XRFBertDatasetNewMix(Dataset):
#     def __init__(self, file_path='/home/zhumengdie/XRFDataset/new_data/', text_path='/home/wangtiantian/code/XRF55-repo-main/dataset/XRF_dataset/', is_train=True, scene='dml'):
#         super(XRFBertDatasetNewMix, self).__init__()
#         self.file_path = file_path
#         self.text_path = text_path
#         self.is_train = is_train
#         self.scene = scene
#         if self.is_train:
#             self.text = self.text_path + self.scene + '_train.txt'
#         else:
#             self.text = self.text_path + self.scene + '_val.txt'
#         with open(self.text) as text:
#             val_list = text.readlines()
#         self.data = {
#             'file_name': [],
#             'label': []
#         }
#         self.path = self.file_path
#         for string in val_list:
#             self.data['file_name'].append(string.split(',')[0])
#             self.data['label'].append(int(string.split(',')[2]) - 1)
#         self.mean = 9.6302
#         self.std = 3.8425

#     def __len__(self):
#         return len(self.data['label'])

#     def __getitem__(self, idx):
#         file_name = self.data['file_name'][idx]
#         label = self.data['label'][idx]
#         wifi_data = load_wifi(file_name, self.is_train, path=self.path)
#         normalized_wifi_data = (wifi_data - self.mean) / self.std

#         # 第一阶段降采样：从 (270, 1000) 到 (90, 1000)
#         normalized_wifi_data_stage1 = torch.zeros((90, 1000))
#         for i in range(90):
#             start = i * 3
#             end = start + 3
#             normalized_wifi_data_stage1[i, :] = torch.mean(normalized_wifi_data[start:end, :], dim=0)

#         # 第二阶段降采样：从 (90, 1000) 到 (90, 100)
#         normalized_wifi_data_new = torch.zeros((90, 100))
#         for i in range(100):
#             start_t = i * 10
#             end_t = start_t + 10
#             normalized_wifi_data_new[:, i] = torch.mean(normalized_wifi_data_stage1[:, start_t:end_t], dim=1)
#         # print(normalized_wifi_data_new.shape)             #  (90,100)
#         return normalized_wifi_data_new, label

################# xrf (270,1000)

class XRFBertDatasetNewMix(Dataset):
    def __init__(self, file_path='/home/dataset/XRFDataset/new_data/', text_path='/home/wangtiantian/code/XRF55-repo-main/dataset/XRF_dataset/', is_train=True, scene='dml'):
        super(XRFBertDatasetNewMix, self).__init__()
        self.file_path = file_path
        self.text_path = text_path
        self.is_train = is_train
        self.scene = scene
        if self.is_train:
            self.text = self.text_path + self.scene + '_train.txt'
        else:
            self.text = self.text_path + self.scene + '_val.txt'
        with open(self.text) as text:
            val_list = text.readlines()
        self.data = {
            'file_name': [],
            'label': []
        }
        self.path = self.file_path
        for string in val_list:
            self.data['file_name'].append(string.split(',')[0])
            self.data['label'].append(int(string.split(',')[2]) - 1)
        self.mean = 9.6302
        self.std = 3.8425

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        file_name = self.data['file_name'][idx]
        label = self.data['label'][idx]
        wifi_data = load_wifi(file_name, self.is_train, path=self.path)
        normalized_wifi_data = (wifi_data - self.mean) / self.std

        # 不进行任何降采样，直接返回标准化后的数据
        return normalized_wifi_data, label

        
##############     xrf55 IPC50 (90,500)
# import random
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# import random
# import torch
# import numpy as np
# from torch.utils.data import Dataset

# class XRFBertDatasetNewMix(Dataset):
#     def __init__(self, file_path='/home/zhumengdie/XRFDataset/new_data/', text_path='/home/wangtiantian/code/XRF55-repo-main/dataset/XRF_dataset/', is_train=True, scene='dml', samples_per_class=5):
#         super(XRFBertDatasetNewMix, self).__init__()
#         self.file_path = file_path
#         self.text_path = text_path
#         self.is_train = is_train
#         self.scene = scene
#         self.samples_per_class = samples_per_class
        
#         # 设置数据文件路径
#         self.text = self.text_path + self.scene + ('_train.txt' if is_train else '_val.txt')
        
#         self.data = {
#             'file_name': [],
#             'label': []
#         }
        
#         with open(self.text, 'r') as f:
#             val_list = f.readlines()
        
#         # 创建列表存储每个类别的索引
#         class_indices = [[] for _ in range(55)]
#         for string in val_list:
#             file_name = string.split(',')[0]
#             label = int(string.split(',')[2]) - 1
#             class_indices[label].append(file_name)
        
#         # 随机抽取每个类别的指定数量的样本
#         for label, indices in enumerate(class_indices):
#             selected_samples = random.sample(indices, min(len(indices), self.samples_per_class))
#             self.data['file_name'].extend(selected_samples)
#             self.data['label'].extend([label] * len(selected_samples))

#     def __len__(self):
#         return len(self.data['label'])

#     def __getitem__(self, idx):
#         file_name = self.data['file_name'][idx]
#         label = self.data['label'][idx]
#         wifi_data = self.load_wifi(file_name, self.is_train, path=self.file_path)

#         # 归一化数据
#         mean = 9.6302
#         std = 3.8425
#         normalized_wifi_data = (wifi_data - mean) / std

#         # 第一阶段降采样：从 (270, 1000) 到 (270, 500)
#         normalized_wifi_data_stage1 = torch.zeros((270, 500))
#         for i in range(500):
#             start = i * 2
#             end = start + 2
#             normalized_wifi_data_stage1[:, i] = torch.mean(normalized_wifi_data[:, start:end], dim=1)

#         # 第二阶段降采样：从 (270, 500) 到 (90, 500)
#         normalized_wifi_data_new = torch.zeros((90, 500))
#         for i in range(90):
#             start_t = i * 3
#             end_t = start_t + 3
#             normalized_wifi_data_new[i, :] = torch.mean(normalized_wifi_data_stage1[start_t:end_t, :], dim=0)
#         # print(normalized_wifi_data_new.shape)
#         return normalized_wifi_data_new, label

#     def load_wifi(self, filename, is_train, path):
#         if is_train:
#             path = path + 'train_data/'
#         else:
#             path = path + 'test_data/'
#         record = np.load(path + 'WiFi/' + filename + ".npy")
#         return torch.from_numpy(record).float()



##############     xrf55 IPC50 (30,100)


import random
import torch
import numpy as np
from torch.utils.data import Dataset

# class XRFBertDatasetNewMix(Dataset):
#     def __init__(self, file_path='/home/zhumengdie/XRFDataset/new_data/', text_path='/home/wangtiantian/code/XRF55-repo-main/dataset/XRF_dataset/', is_train=True, scene='dml', samples_per_class=5):
#         super(XRFBertDatasetNewMix, self).__init__()
#         self.file_path = file_path
#         self.text_path = text_path
#         self.is_train = is_train
#         self.scene = scene
#         self.samples_per_class = samples_per_class
        
#         # 设置数据文件路径
#         self.text = self.text_path + self.scene + ('_train.txt' if is_train else '_val.txt')
        
#         self.data = {
#             'file_name': [],
#             'label': []
#         }
        
#         with open(self.text, 'r') as f:
#             val_list = f.readlines()
        
#         # 创建列表存储每个类别的索引
#         class_indices = [[] for _ in range(55)]
#         for string in val_list:
#             file_name = string.split(',')[0]
#             label = int(string.split(',')[2]) - 1
#             class_indices[label].append(file_name)
        
#         # 随机抽取每个类别的指定数量的样本
#         for label, indices in enumerate(class_indices):
#             selected_samples = random.sample(indices, min(len(indices), self.samples_per_class))
#             self.data['file_name'].extend(selected_samples)
#             self.data['label'].extend([label] * len(selected_samples))

#     def __len__(self):
#         return len(self.data['label'])

#     def __getitem__(self, idx):
#         file_name = self.data['file_name'][idx]
#         label = self.data['label'][idx]
#         wifi_data = self.load_wifi(file_name, self.is_train, path=self.file_path)

#         # 归一化数据
#         mean = 9.6302
#         std = 3.8425
#         normalized_wifi_data = (wifi_data - mean) / std

#         # 第一阶段降采样：从 (270, 1000) 到 (270, 100)
#         normalized_wifi_data_stage1 = torch.zeros((270, 100))
#         for i in range(100):
#             start = i * 10
#             end = start + 10
#             normalized_wifi_data_stage1[:, i] = torch.mean(normalized_wifi_data[:, start:end], dim=1)

#         # 第二阶段降采样：从 (270, 100) 到 (30, 100)
#         normalized_wifi_data_new = torch.zeros((30, 100))
#         for i in range(30):
#             start_t = i * 9
#             end_t = start_t + 9
#             normalized_wifi_data_new[i, :] = torch.mean(normalized_wifi_data_stage1[start_t:end_t, :], dim=0)
#         # print(normalized_wifi_data_new.shape)
#         return normalized_wifi_data_new, label

#     def load_wifi(self, filename, is_train, path):
#         if is_train:
#             path = path + 'train_data/'
#         else:
#             path = path + 'test_data/'
#         record = np.load(path + 'WiFi/' + filename + ".npy")
#         return torch.from_numpy(record).float()

    

##################widar 
class Widar_Dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_list = glob.glob(root_dir+'/*/*.csv')
        # self.folder = glob.glob(root_dir+'/*/')
        self.folder = sorted(glob.glob(root_dir+'/*/'))

        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}
        print(self.category)
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        # print("y",y)
        x = np.genfromtxt(sample_dir, delimiter=',')
        # index_to_classname = {v: k for k, v in self.category.items()}
        # print(index_to_classname)
        # normalize
        x = (x - 0.0025)/0.0119
        
        # reshape: 22,400 -> 22,20,20
        x = x.reshape(22,20,20)
        # interpolate from 20x20 to 32x32
        # x = self.reshape(x)
        x = torch.FloatTensor(x)

        return x,y






##################widar IPC 50 

# class Widar_Dataset(Dataset):
#     def __init__(self, root_dir, samples_per_class=50):
#         self.root_dir = root_dir
#         self.samples_per_class = samples_per_class
#         self.data_list = glob.glob(root_dir + '/*/*.csv')
#         self.folder = sorted(glob.glob(root_dir + '/*/'))

#         # Create a dictionary for class categories
#         self.category = {self.folder[i].split('/')[-2]: i for i in range(len(self.folder))}

#         # Initialize a dictionary to store file paths for each category
#         self.class_files = {folder.split('/')[-2]: [] for folder in self.folder}
#         for file_path in self.data_list:
#             class_name = file_path.split('/')[-2]
#             self.class_files[class_name].append(file_path)

#         # Reduce the files in each class to the specified number per class
#         self.reduced_data_list = []
#         for class_name, files in self.class_files.items():
#             if len(files) > self.samples_per_class:
#                 self.reduced_data_list.extend(random.sample(files, self.samples_per_class))
#             else:
#                 self.reduced_data_list.extend(files)

#         print(self.category)  # Optional: print categories

#     def __len__(self):
#         return len(self.reduced_data_list)

#     def __getitem__(self, idx):
#         sample_dir = self.reduced_data_list[idx]
#         y = self.category[sample_dir.split('/')[-2]]
        
#         x = np.genfromtxt(sample_dir, delimiter=',')
#         x = (x - 0.0025) / 0.0119
#         x = x.reshape(22, 20, 20)
#         x = torch.FloatTensor(x)

#         return x, y






