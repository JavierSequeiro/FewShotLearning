import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

class myConfig:

    def __init__(self):
        
        self.root_path = "/autofs/unitytravail/travail/jsequeirogon/Advanced_Topics_IPCV/P3_Semi_Supervised"
        
        # Data Loading Params
        self.semisupervised_setting = False
        self.dataset_name = "CIFAR100"
        self.batch_size = 16
        self.img_size = 256
        self.train_shuffle = False
        self.apply_transform = True
        self.use_subsample = 1.0
        self.val_split = 0.2
        
        # Model Loading Params
        self.train_new_model = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "resnet34"
        self.weights = "IMAGENET1K_V1"
        self.num_classes = 100
        self.my_pretrained_model_path = f"/autofs/unitytravail/travail/jsequeirogon/Advanced_Topics_IPCV/P3_Semi_Supervised/models/{self.model_name}/model_ep10.pt"

        # Model Training Params
        self.train_criterion = nn.CrossEntropyLoss()
        self.val_criterion = nn.CrossEntropyLoss()
        self.lr = 5e-3
        self.weight_decay = 3e-4
        self.epochs = 20

        self.output_model_path = "./models"
        self.plot_results_path = "./plots"
        self.specific_experiment = "EX_0_SUPERVISED"
        self.specific_experiment_val = None


        # Log File
        self.logs_path = "./logs"
        self.log_filename = f"experiments_{self.model_name}_{self.specific_experiment}" 
            
        # Random Seed
        self.rand_seed = 14