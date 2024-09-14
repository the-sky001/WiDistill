# adapted from
# https://github.com/VICO-UoE/DatasetCondensation

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import kornia as K
import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
# from networks import MLP, ConvNet, LeNet, AlexNet, VGG11BN, VGG11, ResNet18, ResNet18BN_AP, ResNet18_AP, resnet18_widar, Widar_MLP
from networks import *
from getdata import *
from getdata_mmfi import *
import yaml
import imp

import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import random
class Config:
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    # ["australian_terrier", "border_terrier", "samoyed", "beagle", "shih-tzu", "english_foxhound", "rhodesian_ridgeback", "dingo", "golden_retriever", "english_sheepdog"]
    imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]

    # ["tabby_cat", "bengal_cat", "persian_cat", "siamese_cat", "egyptian_cat", "lion", "tiger", "jaguar", "snow_leopard", "lynx"]
    imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]

    # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

    # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
    imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

    # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
    imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

    dict = {
        "imagenette" : imagenette,
        "imagewoof" : imagewoof,
        "imagefruit": imagefruit,
        "imageyellow": imageyellow,
        "imagemeow": imagemeow,
        "imagesquawk": imagesquawk,
    }

config = Config()
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

#引用本地cifar10
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    


class LocalWidar(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train  # 指示是训练集还是测试集

        # 加载数据
        if train:
            self.data, self.labels = self.load_data('train')
        else:
            self.data, self.labels = self.load_data('test')

    def load_data(self, mode):
        # 确定文件路径
        folder = 'train' if mode == 'train' else 'test'
        file_path = os.path.join(self.root, folder)
        data_files = glob.glob(file_path + '/*/*.csv')

        # 加载数据和标签
        data_list = []
        labels = []
        category_dict = {os.path.basename(os.path.dirname(fp)): i for i, fp in enumerate(sorted(glob.glob(file_path + '/*')))}
        
        for file in data_files:
            data = np.genfromtxt(file, delimiter=',')
            # data = (data - 0.0025) / 0.0119  # 归一化处理
            data = data.reshape(22, 20, 20)  # 调整形状
            data = np.stack([Image.fromarray(plane) for plane in data], axis=0)  # 转换每一层为图像
            data_list.append(data)
            labels.append(category_dict[os.path.basename(os.path.dirname(file))])
        
        return data_list, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        if self.transform:
            img = [self.transform(plane) for plane in img]  # 对每层图像应用转换
        return torch.stack(img), label  # 返回堆叠后的图像张量和标签

def get_dataset(dataset, data_path, batch_size=1, subset="imagenette", args=None):

    class_map = None
    loader_train_dict = None
    class_map_inv = None
    class_names = None
    if dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        # 创建数据集实例
        data_path = '/home/xxx/cifar10/cifar-10-batches-py'
        dst_train = LocalCIFAR10(data_path, train=True, transform=transform)
        dst_test = LocalCIFAR10(data_path, train=False, transform=transform)
        
        # dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        # dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}
    elif dataset == 'mmfi':#(3,114,10)TOTAL
        num_classes = 27
        channel = 1
        im_size = (342,350)   
        mean = [0.6887]
        std = [0.1696]

        dataset_root = '/home/xxx/mmfi_new2'

        config_file = '/home/xxx/config.yaml'
        # dataset_root = args.dataset_root
        with open(config_file, 'r') as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)

        dst_train, dst_test = make_dataset(dataset_root, config)
        rng_generator = torch.manual_seed(config['init_rand_seed'])

   
        train_loader = make_dataloader(dst_train, is_training=True, generator=rng_generator, **config['train_loader'])
        val_loader = make_dataloader(dst_test, is_training=False, generator=rng_generator, **config['validation_loader'])
        class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        # class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        
        class_map = {i: i for i in range(num_classes)}
    elif dataset == 'widar':
        num_classes = 6
        channel = 22
        im_size = (20,20)
        root = '/home/xxx'
        dst_train = Widar_Dataset(root + 'Widardata2/train/')
        dst_test = Widar_Dataset(root + 'Widardata2/test/')
        # train_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata2/train/', samples_per_class=50), batch_size=64, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata2/test/', samples_per_class=50), batch_size=128, shuffle=False)
        train_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata2/train/'), batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata2/test/'), batch_size=128, shuffle=False)        
        class_names = [
            'Push&Pull', 'Sweep', 'Clap', 'Slide',
            'Draw-O(H)', 
            'Draw-Zigzag(H)'
        ]

        # class_map = {class_names[i]: i+1 for i in range(num_classes)}
        class_map = {i: i for i in range(num_classes)}


    elif dataset == 'xrf55':
        num_classes = 55
        channel = 1
        im_size = (270,1000)   
        mean = [9.6302]
        std = [3.8425]
        # mean = [
        #     0.004, 0.0093, 0.0088, 0.005, 0.0027, 0.0013, 0.0008, 0.0004,
        #     0.0005, 0.0008, 0.0013, 0.0013, 0.001, 0.0007, 0.0008, 0.0007,
        #     0.0008, 0.0014, 0.0022, 0.0029, 0.0034, 0.0036
        # ]

        # std = [
        #     1.0574, 1.1838, 1.2135, 1.0918, 1.024, 0.9766, 0.9493, 0.9368,
        #     0.9222, 0.9261, 0.9221, 0.9143, 0.9078, 0.8973, 0.9054, 0.8994,
        #     0.9037, 0.927, 0.9515, 0.9784, 1.0013, 1.1113
        # ]

        dst_train = XRFBertDatasetNewMix(file_path='/home/xxx/xrf/new_data/', text_path='/home/xxx/XRF_dataset/',is_train=True, scene='dml')
        dst_test = XRFBertDatasetNewMix(file_path='/home/xxx/xrf/new_data/', text_path='/home/xxx/XRF_dataset/',is_train=False, scene='dml')
        train_loader = torch.utils.data.DataLoader(dataset=XRFBertDatasetNewMix(file_path='/home/xxx/xrf/new_data/', text_path='/home/xxx/XRF_dataset/',is_train=True, scene='dml'), batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=XRFBertDatasetNewMix(file_path='/home/xxx/xrf/new_data/', text_path='/home/xxx/XRF_dataset/',is_train=False, scene='dml'), batch_size=128, shuffle=False)        
        class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
        class_map = {i: i for i in range(num_classes)}

    else:
        exit('unknown dataset: %s'%dataset)



    testloader = torch.utils.data.DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=2)
    return num_classes, class_names, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv



class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(model, channel, num_classes=6, im_size=(20, 20), dist=True):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()



    if model == 'MLP':
        net = MLP(num_classes=num_classes)
    elif model == 'mmfi_resnet18':
        print("using model: mmfi_resnet18")
        net = mmfi_resnet18(num_classes)
    elif model == 'xrf_resnet':
        print("using model: xrf_resnet")
        net = resnet18()
    elif model == 'xrf_resnet50':
        print("using model: xrf_resnet50")
        net = xrf_resnet50()
    elif model == 'xrf_mlp':
        print("using model: xrf_mlp")
        net = xrf_MLP(num_classes)
    elif model == 'xrf_resnet18':
        print("using model: xrf_resnet18")
        net = xrf_resnet18(num_classes)
    elif model == 'xrf_CNN':
        print("using model: xrf_CNN")
        net = xrf_CNN(num_classes)
    elif model == 'xrf_UNet':
        print("using model: xrf_UNet")
        net = UNet1D(num_classes)

    elif model == 'widar_mlp':
        print("using model: widar_mlp")
        net = Widar_MLP(num_classes)    #Test Acc: 0.8534293141371726
    elif model == 'Widar_ViT':
        # num_classes = 6
        # print(num_classes)
        print(f" Channel: {channel}, Classes: {num_classes}, Image Size: {im_size}")
        print("using model: Widar_ViT")
        net = Widar_ViT(num_classes = num_classes)  #Test Acc: 0.763247350529894
    elif model == 'resnet18_widar':
        print("using model: resnet18_widar")
        net = resnet18_widar(num_classes)  #laji
        # train_epoch = 100
    elif model == 'mmfi_CNN':
        print("using model: mmfi_CNN")
        net = mmfi_CNN(num_classes)  
    elif model == 'mmfi_mlp':
        print("using model: mmfi_mlp")
        net = mmfi_MLP(num_classes)  

    elif model == 'BiLSTM':
        print("using model: BiLSTM")
        net = Widar_BiLSTM(num_classes)  #Test Acc: 0.8032393521295741
    elif model == 'widar_resnet18':
        print("using model: ResNet18")
        net = widar_resnet18(num_classes)   #Test Acc: 0.8264347130573885
    elif model == 'widar_CNN':
        print("using model: Widar_CNN")
        net = Widar_CNN(num_classes)



        # train_epoch = 100
    elif model == 'Widar_ResNet50':
        print("using model: ResNet50")
        net = Widar_ResNet50(num_classes)
        # train_epoch = 100 #40
    elif model == 'Widar_ResNet101':
        print("using model: ResNet101")
        net = Widar_ResNet101(num_classes)  #Test Acc: 0.17176564687062587
        # train_epoch = 100
    else:
        net = None
        exit('DC error: unknown model')

    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num>0:
            device = 'cuda'
            if gpu_num>1:
                net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

    return net



def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def epoch(mode, dataloader, net, optimizer, criterion, args, aug, texture=False):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)

    if args.dataset == "ImageNet":
        class_map = {x: i for i, x in enumerate(config.img_net_classes)}

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)

        if mode == "train" and texture:
            img = torch.cat([torch.stack([torch.roll(im, (torch.randint(args.im_size[0]*args.canvas_size, (1,)), torch.randint(args.im_size[0]*args.canvas_size, (1,))), (1,2))[:,:args.im_size[0],:args.im_size[1]] for im in img]) for _ in range(args.canvas_samples)])
            lab = torch.cat([lab for _ in range(args.canvas_samples)])

        # if aug:
        #     if args.dsa:
        #         img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
        #     else:
        #         img = augment(img, args.dc_aug_param, device=args.device)

        if args.dataset == "ImageNet" and mode != "train":
            lab = torch.tensor([class_map[x.item()] for x in lab]).to(args.device)

        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)

        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg



def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, return_loss=False, texture=False):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    args.lr_net = 0.01
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=True, texture=texture)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)


    time_train = time.time() - start

    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test


def augment(images, dc_aug_param, device):
    # # This can be sped up in the future.

    # if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
    #     scale = dc_aug_param['scale']
    #     crop = dc_aug_param['crop']
    #     rotate = dc_aug_param['rotate']
    #     noise = dc_aug_param['noise']
    #     strategy = dc_aug_param['strategy']

    #     shape = images.shape
    #     mean = []
    #     for c in range(shape[1]):
    #         mean.append(float(torch.mean(images[:,c])))

    #     def cropfun(i):
    #         im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
    #         for c in range(shape[1]):
    #             im_[c] = mean[c]
    #         im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
    #         r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
    #         images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

    #     def scalefun(i):
    #         h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
    #         w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
    #         tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
    #         mhw = max(h, w, shape[2], shape[3])
    #         im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
    #         r = int((mhw - h) / 2)
    #         c = int((mhw - w) / 2)
    #         im_[:, r:r + h, c:c + w] = tmp
    #         r = int((mhw - shape[2]) / 2)
    #         c = int((mhw - shape[3]) / 2)
    #         images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

    #     def rotatefun(i):
    #         im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
    #         r = int((im_.shape[-2] - shape[-2]) / 2)
    #         c = int((im_.shape[-1] - shape[-1]) / 2)
    #         images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

    #     def noisefun(i):
    #         images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


    # #     augs = strategy.split('_')

    #     for i in range(shape[0]):
    #         choice = np.random.permutation(augs)[0] # randomly implement one augmentation
    # #         if choice == 'crop':
    # #             cropfun(i)
    # #         elif choice == 'scale':
    # #             scalefun(i)
    # #         elif choice == 'rotate':
    # #             rotatefun(i)
    #         # elif choice == 'noise':
    #         if choice == 'noise':
    #             noisefun(i)
    return images



def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']:  # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M_widar': # multiple architectures
        # model_eval_pool = ['MLP', 'ConvNet', 'AlexNet', 'VGG11', 'ResNet18', 'LeNet']
        # model_eval_pool = ['ConvNet', 'AlexNet', 'VGG11', 'ResNet18_AP', 'ResNet18']
        model_eval_pool = ['widar_MLP']
    elif eval_mode == 'M_xrf': # ablation study on network width
        model_eval_pool = ['xrf_mlp']   
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'C':
        model_eval_pool = [model, 'ConvNet']
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('Error ZH: unknown augmentation mode.')
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode: # batch-wise:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}
