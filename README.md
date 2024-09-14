# WiDistill
Official implementation of ''WiDistill: Distilling Large-scale Wi-Fi Datasets with Matching Trajectory''

WiDistill: Distilling Large-scale Wi-Fi Datasets with Matching Trajectory

Tiantian Wang,  [Fei Wang](https://scholar.google.com/citations?user=LKPpmXQAAAAJ&hl=en) 

Xi'an Jiaotong University

The task of "WiDistill" is to reduce the size of a large Wi-Fi dataset using trajectory matching-based distillation, creating a much smaller dataset that maintains similar performance to the original.



# Getting Started
First, download our repo:
```python
git clone https://github.com/the-sky001/WiDistill.git
cd WiDistill
```

For an express instillation, we include .yaml files.
```python
conda env create -f environment.yaml
 ```

You can then activate your conda environment with
```python
conda activate widistill
 ```

# Generating Expert Trajectories
Before doing any distillation, you'll need to generate some expert trajectories using buffer.py
```python
python buffer.py --dataset=xrf55 --model=xrf_CNN  --save_interval 1 --lr_teacher 0.01 --train_epochs=150 --num_experts=10 --buffer_path=/home/xxx/buffer/ --data_path=/home/xxx/xrf/new_data/
 ```

# Distillation by Matching Training Trajectories
The following command will then use the buffers we just generated to distill.
```python
python distill.py --dataset=xrf55 --ipc=50 --syn_steps=1 --expert_epochs=2 --max_start_epoch=15   --lr_img=100 --lr_lr=1e-05 --model=xrf_resnet18 --lr_teacher=0.01 --buffer_path=/home/xxx/buffer --data_path=/home/xxx/xrf/new_data/
 ```














