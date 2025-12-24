import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset, Dataset
from torchvision.datasets import CIFAR100
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from collections import defaultdict
import os

from src.helpers import _visualize_samples, _get_FSL_split, _get_stratified_split

"""
OPTIMAL IMPLEMENTATIONS
"""

class myFinalDataset:

    def __init__(self, config, logger):
        
        self.config = config
        self.logger = logger


    def _get_dataset(self, **kwargs):
        
        if self.config.apply_transform and kwargs.get("transform") != None:
            self.config.dataset_transform = kwargs.get("transform")
            self.logger.info(kwargs.get("transform"))

        self.dataset_dict = {"CIFAR100": {"TRAIN":CIFAR100(root="./data",
                                                            train=True,
                                                            download=True,
                                                            transform=self.config.dataset_transform),
                                           "TEST":CIFAR100(root="./data",
                                                            train=False,
                                                            download=True,
                                                            transform=self.config.dataset_transform)}}
        
        train_data = self.dataset_dict[self.config.dataset_name]["TRAIN"]
        test_data = self.dataset_dict[self.config.dataset_name]["TEST"]
        
        # Debug Line
        if self.config.visualize_samples:
            _visualize_samples(train_data,
                            test_data,
                            config=self.config,
                            logger=self.logger)
        
        
        train_data, val_data, test_data = _get_FSL_split(train_dataset=train_data,
                                                         test_dataset=test_data,
                                                         config=self.config,
                                                         logger=self.logger)
        
        train_data = DistanceDataset(base_dataset=train_data)
        val_data = DistanceDataset(base_dataset=val_data)
        test_data = DistanceDataset(base_dataset=test_data)
        
        self.logger.info(f"Training with full dataset: {len(train_data)} samples")

        return train_data, val_data, test_data

    
##################
# Prototypes
##################

class myProtoDataset:

    def __init__(self, config, logger):
        
        self.config = config
        self.logger = logger


    def _get_dataset(self, **kwargs):
        if self.config.apply_transform and kwargs.get("transform") != None:
            self.config.dataset_transform = kwargs.get("transform")
            self.logger.info(kwargs.get("transform"))

        self.dataset_dict = {"CIFAR100": {"TRAIN":CIFAR100(root="./data",
                                                            train=True,
                                                            download=True,
                                                            transform=self.config.dataset_transform),
                                           "TEST":CIFAR100(root="./data",
                                                            train=False,
                                                            download=True,
                                                            transform=self.config.dataset_transform)}}
        
        train_data = self.dataset_dict[self.config.dataset_name]["TRAIN"]
        test_data = self.dataset_dict[self.config.dataset_name]["TEST"]
        
        # Debug Line
        if self.config.visualize_samples:
            _visualize_samples(train_data,
                            test_data,
                            config=self.config,
                            logger=self.logger)
        
        
        train_data, val_data, test_data = _get_FSL_split(train_dataset=train_data,
                                                         test_dataset=test_data,
                                                         config=self.config,
                                                         logger=self.logger)
        
        self.logger.info(f"Training with full dataset: {len(train_data)} samples")

        train_data = PrototypicalDataset(base_dataset=train_data)
        val_data = PrototypicalDataset(base_dataset=val_data)
        test_data = PrototypicalDataset(base_dataset=test_data)

        self.logger.info(f"Completed FSL Train/Val/Test Set Configurations!")

        return train_data, val_data, test_data
    

    

####################
# Auxiliary
####################

class PrototypicalDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.labels = []
        
        # Determine labels depending on if it's a raw CIFAR dataset or a Subset
        if hasattr(base_dataset, 'targets'):
            self.labels = base_dataset.targets
        elif hasattr(base_dataset, 'dataset'): # If it's a Subset
            # self.labels = [base_dataset.dataset.targets[i] for i in base_dataset.indices]
            self.labels = [base_dataset.dataset.targets[i] for i in base_dataset.indices]
        
        # Map each class to the indices of samples belonging to it
        self.classes = np.unique(self.labels)
        self.class_to_indices = {c: np.where(np.array(self.labels) == c)[0] for c in self.classes}

    def __getitem__(self, index):
        # Simply returns the image and label from the base CIFAR100
        return self.base_dataset[index]

    def __len__(self):
        return len(self.base_dataset)
    

class DistanceDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.labels = []
        
        # Determine labels depending on if it's a raw CIFAR dataset or a Subset
        if hasattr(base_dataset, 'targets'):
            self.labels = base_dataset.targets
        elif hasattr(base_dataset, 'dataset'): # If it's a Subset
            # self.labels = [base_dataset.dataset.targets[i] for i in base_dataset.indices]
            self.labels = [base_dataset.dataset.targets[i] for i in base_dataset.indices]
        
        # Map each class to the indices of samples belonging to it
        self.classes = np.unique(self.labels)
        self.class_to_indices = {c: np.where(np.array(self.labels) == c)[0] for c in self.classes}

    def __getitem__(self, index):
        # Simply returns the image and label from the base CIFAR100
        return self.base_dataset[index]

    def __len__(self):
        return len(self.base_dataset)
    
