import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import numpy as np

class myConfig:

    def __init__(self):
        
        # Data Loading Params
        self.dataset_name = "CIFAR100"
        self.batch_size = 256
        self.img_size = 224#224#256
            # Statistical Normalization Values for ImageNet pretrained models
        self.norm_mean =  [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.dataset_transform = transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=self.norm_mean, std=self.norm_std)])
        self.train_shuffle = False
        self.apply_transform = True
        self.visualize_samples = False
            # Validation Set Percentage out of Training Set
        self.val_split = 0.2
        
        # Few Shot Learning Params
        self.num_classes_train = 50
        self.num_classes_val = 20
        self.num_classes_test = 30

        self.num_val_episodes_per_epoch = 5

        self.conf_interval_coeff = 1.96 # For Confidence Interval computations.


        # Model Loading Params
            # Flag to decide whether we train a model or just use a personally pretrained one
        self.train_new_model = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # ResNet model name for model loading (check model.py for models available)
        self.model_name = "resnet50"
            # Model pretrained weights to use
        self.weights = "IMAGENET1K_V1"
            # Number of output classes (for our prototypical implementation does not make a difference)
        self.num_classes = 100
            # Path to pretrained model to avoid retraining.
        self.my_pretrained_model_path = f"/home/jsg2/Desktop/rhome/jsg2/BDX_AT_IPCV/models/{self.model_name}/model_ep10.pt"

        # Model Training Params
            # Optimizer
        self.lr = 5e-3
        self.weight_decay = 3e-4
            # Epochs
        self.epochs = 10

        self.output_model_path = "./models"
        self.plot_results_path = "./plots"
        self.specific_experiment = "REFINING_DISTANCE"#"REFINING_REPO"
        self.specific_experiment_val = None

        # Distance Learning Params
            #  Contrastive Loss Parameters
        self.negative_margin = 1.0
        self.positive_margin = 0.8 # For Cosine Similarity loss
            # Output Embedding Dimension
        self.final_layer_dim = 256
            # k-NN Classifier
        self.K_support_points = 5
        self.K_knn = 11
            # Batch Structure
        self.num_classes_per_batch = 10
        self.n_samples_per_class = 10

        # Log File
        self.logs_path = "