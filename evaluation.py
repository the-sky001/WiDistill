import os
import sys
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, ParamDiffAug
import copy
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# def setup_logging(log_file):
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )

def main(args):
    # setup_logging(args.log_file)
    logging.info("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    logging.info(f"Data directory: {args.data_dir}")
    
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 获取数据集
    logging.info("Loading dataset...")
    num_classes, class_names, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    if args.dataset == 'xrf55':
        im_size = (270, 1000)
        num_classes = 55
        channel = 1
    elif args.dataset == 'widar':
        im_size = (20, 20)
        num_classes = 6
        channel = 22
    elif args.dataset == 'mmfi':
        im_size = (342, 350)
        num_classes = 27
        channel = 1
    print(im_size, num_classes, channel)
    data_save = []

    if args.dsa:
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    args.distributed = torch.cuda.device_count() > 1

    logging.info('Hyper-parameters: \n{}'.format(args.__dict__))
    logging.info('Evaluation model pool: {}'.format(model_eval_pool))
    
    def SoftCrossEntropy(inputs, target, reduction='average'):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        return loss

    soft_cri = SoftCrossEntropy

    image_syn_eval = torch.load(args.data_dir)
    label_syn_eval = torch.load(args.label_dir)

    acc_test_values = []

    # 进行10次循环评估
    for _ in range(3):
        for model_eval in model_eval_pool:
            logging.info('Evaluating: ' + model_eval)
            network = get_network(model_eval, channel, num_classes, im_size, dist=False).to(args.device)
            _, acc_train, acc_test = evaluate_synset(0, copy.deepcopy(network), image_syn_eval, label_syn_eval, testloader, args, texture=False)
            acc_test_values.append(acc_test)
            logging.info('Accuracy on synthetic data: {:.2f}%'.format(acc_test * 100))
    max_acc_test = max(acc_test_values)
    avg_acc_test = sum(acc_test_values) / len(acc_test_values)	
    print('Average acc_test over 10 runs: {:.2f}%'.format(avg_acc_test * 100))
    print('Maximum acc_test over 10 runs: {:.2f}%'.format(max_acc_test * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')
    parser.add_argument('--epoch_eval_train', type=int, default=3000, help='epochs to train a model with synthetic data')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'], help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='/home/wangtiantian/Widardata2', help='dataset path')
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')
    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--parall_eva', type=bool, default=True, help='dataset')
    parser.add_argument('--data_dir', type=str, default='path', help='dataset')
    parser.add_argument('--label_dir', type=str, default='path', help='dataset')
    # parser.add_argument('--log_file', type=str, default='/home/qianbo/wangtiantian/mtt2/logged_file/mtt2/herding/xrf/herding.log', help='path to log file')

    args = parser.parse_args()
    main(args)
