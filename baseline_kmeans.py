import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, TensorDataset
import copy
import numpy as np
from sklearn.cluster import KMeans
import time
import logging

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def setup_logging(log_file):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def kmeans_selection(data, labels, num_samples_per_class):
    selected_indices = []
    unique_classes = np.unique(labels)
    for cls in unique_classes:
        class_indices = np.where(labels == cls)[0]
        class_data = data[class_indices]
        kmeans = KMeans(n_clusters=num_samples_per_class, random_state=0).fit(class_data.reshape(len(class_data), -1))
        cluster_centers = kmeans.cluster_centers_
        
        for center in cluster_centers:
            distances = np.linalg.norm(class_data.reshape(len(class_data), -1) - center, axis=1)
            selected_index = class_indices[np.argmin(distances)]
            selected_indices.append(selected_index)
    
    return selected_indices

def main(args):
    setup_logging(args.log_file)  # 设置日志文件路径
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cpu' if args.dataset == 'xrf55' else 'cuda'

    # 获取数据集
    logging.info("Loading dataset...")
    start_time = time.time()
    num_classes, class_names, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    logging.info(f"Dataset loaded in {time.time() - start_time:.2f} seconds")

    # 组织训练数据
    images_all = []
    labels_all = []
    indices_class = [[] for _ in range(num_classes)]
    logging.info("Building dataset...")
    start_time = time.time()
    for i in tqdm(range(len(dst_train)), desc="Processing training data"):
        sample = dst_train[i]
        tensor_sample = torch.from_numpy(sample[0]) if isinstance(sample[0], np.ndarray) else sample[0]
        images_all.append(torch.unsqueeze(tensor_sample, dim=0))
        if args.dataset == 'mmfi':
            labels_all.append(class_map[sample[1].clone().detach().item()])
        else:
            labels_all.append(class_map[torch.tensor(sample[1]).item()])
    logging.info(f"Training data processed in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    for i, lab in tqdm(enumerate(labels_all), desc="Organizing labels"):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
    logging.info(f"Labels organized in {time.time() - start_time:.2f} seconds")

    logging.info(f"images_all.shape: {images_all.shape}")
    for c in range(num_classes):
        logging.info(f'class c = {c}: {len(indices_class[c])} real images')

    channel = images_all.shape[1]  # 获取通道数
    for ch in range(channel):
        logging.info(f'real images channel {ch}, mean = {torch.mean(images_all[:, ch]):.4f}, std = {torch.std(images_all[:, ch]):.4f}')

    criterion = nn.CrossEntropyLoss().to(args.device)

    # 使用 K-means 方法选择每类的代表性样本
    num_samples_per_class = 50  # 设置每一类希望选择的样本数
    print(f"num_samples_per_class: {num_samples_per_class}")

    logging.info("Applying K-means selection...")
    start_time = time.time()
    selected_indices = kmeans_selection(images_all.numpy(), labels_all.numpy(), num_samples_per_class)
    logging.info(f"K-means selection applied in {time.time() - start_time:.2f} seconds")

    selected_features = images_all[selected_indices]
    selected_labels = labels_all[selected_indices]

    # 保存选择出的数据子集
    subset_save_path = os.path.join(args.buffer_path, 'kmeans_subset', args.dataset)
    os.makedirs(subset_save_path, exist_ok=True)
    torch.save(selected_features, os.path.join(subset_save_path, f'selected_features_ipc{num_samples_per_class}.pt'))
    torch.save(selected_labels, os.path.join(subset_save_path, f'selected_labels_ipc{num_samples_per_class}.pt'))
    logging.info(f"K-means subset saved to {subset_save_path}")

    # 打印最终生成子集的形状
    logging.info(f"Final subset shape: features {selected_features.shape}, labels {selected_labels.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='K-means Selection')
    parser.add_argument('--dataset', type=str, default='your_dataset', help='dataset name')
    parser.add_argument('--data_path', type=str, default='./data', help='path to dataset')
    parser.add_argument('--batch_real', type=int, default=64, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=64, help='batch size for training')
    parser.add_argument('--subset', type=str, default='subset', help='subset of dataset')
    parser.add_argument('--dsa', type=str, default='False', help='use DSA')
    parser.add_argument('--buffer_path', type=str, default='./buffer', help='path to buffer')
    parser.add_argument('--project', type=str, default='mtt_baseline', help='path to buffer')
    parser.add_argument('--log_file', type=str, default='./kmeans.log', help='path to log file')

    args = parser.parse_args()
    os.makedirs(args.buffer_path, exist_ok=True)
    main(args)
