import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import random





def load_wifi(filename, is_train, path='./dataset/XRFDataset/'):
    if is_train:
        path = path + 'train_data/'
    else:
        path = path + 'test_data/'
    record = np.load(path + 'WiFi/' + filename + ".npy")
    return torch.from_numpy(record).float()


################# xrf (270,1000)

class XRFBertDatasetNewMix(Dataset):
    def __init__(self, file_path='/home/dataset/XRFDataset/new_data/', text_path='/home/xxx/code/XRF55-repo-main/dataset/XRF_dataset/', is_train=True, scene='dml'):
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













