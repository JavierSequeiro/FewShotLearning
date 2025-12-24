import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F

from src.myLogger import myLogger 

class ResNet:
  
  def __init__(self, config, logger):
      
      self.config = config
      self.logger = logger
      
  def load_pretrained_resnet(self):
      
      models_dict = {"resnet18":models.resnet18(weights=self.config.weights),
                    "resnet34":models.resnet34(weights=self.config.weights),
                    "resnet50":models.resnet50(weights=self.config.weights),
                    "resnet101":models.resnet101(weights=self.config.weights),
                    "resnet152":models.resnet152(weights=self.config.weights),
                    }
    
      try:
        pretrained_model = models_dict[self.config.model_name]
        architecture = self.config.model_name
      except KeyError:
        self.logger.info(f"{self.config.model_name} does not exist...")
        self.logger.info(f"Please be aware that your results will be saved to models/resnet18")
        pretrained_model = models.resnet18(weights=self.config.weights)
        architecture = "resnet18"

      # Modify architecture head to match our dataset's expected output
      pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, self.config.num_classes, bias=True)

      if not self.config.train_new_model:     
         state_dict = torch.load(self.config.my_pretrained_model_path, map_location='cpu')
         pretrained_model.load_state_dict(state_dict=state_dict)
        
      self.logger.info(f"Loading {architecture}")
      
      del self.logger
      return pretrained_model, architecture
  
  def load_pretrained_resnet_distance(self):
      
      models_dict = {"resnet18":models.resnet18(weights=self.config.weights),
                    "resnet34":models.resnet34(weights=self.config.weights),
                    "resnet50":models.resnet50(weights=self.config.weights),
                    "resnet101":models.resnet101(weights=self.config.weights),
                    "resnet152":models.resnet152(weights=self.config.weights),
                    }
    
      try:
        pretrained_model = models_dict[self.config.model_name]
        architecture = self.config.model_name
      except KeyError:
        self.logger.info(f"{self.config.model_name} does not exist...")
        self.logger.info(f"Please be aware that your results will be saved to models/resnet18")
        pretrained_model = models.resnet18(weights=self.config.weights)
        architecture = "resnet18"

      # Modify architecture head to match our dataset's expected output
      pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, self.config.final_layer_dim, bias=True)

      if not self.config.train_new_model:     
         state_dict = torch.load(self.config.my_pretrained_model_path, map_location='cpu')
         pretrained_model.load_state_dict(state_dict=state_dict)
        
      self.logger.info(f"Loading Distance {architecture}")
      distance_pretrained_model = DistanceResNet(model=pretrained_model)

      del self.logger
      return distance_pretrained_model, architecture
  
  def load_pretrained_resnet_prototypes(self):
      
      models_dict = {"resnet18":models.resnet18(weights=self.config.weights),
                    "resnet34":models.resnet34(weights=self.config.weights),
                    "resnet50":models.resnet50(weights=self.config.weights),
                    "resnet101":models.resnet101(weights=self.config.weights),
                    "resnet152":models.resnet152(weights=self.config.weights),
                    }
    
      try:
        pretrained_model = models_dict[self.config.model_name]
        architecture = self.config.model_name
      except KeyError:
        self.logger.info(f"{self.config.model_name} does not exist...")
        self.logger.info(f"Please be aware that your results will be saved to models/resnet18")
        pretrained_model = models.resnet18(weights=self.config.weights)
        architecture = "resnet18"

      # Modify architecture head to match our dataset's expected output
      pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, self.config.final_layer_dim, bias=True)

      if not self.config.train_new_model:     
         state_dict = torch.load(self.config.my_pretrained_model_path, map_location='cpu')
         pretrained_model.load_state_dict(state_dict=state_dict)
        
      self.logger.info(f"Loading Prototypical {architecture}")
      proto_pretrained_model = ProtoResNet(model=pretrained_model)

      del self.logger
      return proto_pretrained_model, architecture
      
      

class DistanceResNet(nn.Module):
    def __init__(self, model, **kwargs) -> None:
      super(DistanceResNet, self).__init__()
      self.model = model

    def forward(self, x):
      x = self.model(x)

      x = F.normalize(x, p=2, dim=1)
      return x
    
class ProtoResNet(nn.Module):
    def __init__(self, model, **kwargs) -> None:
      super(ProtoResNet, self).__init__()
      self.model = model

    def forward(self, x, proto_x):
      x = self.model(x)
      x = F.normalize(x, p=2, dim=1)
      
      proto_x = self.model(proto_x)
      proto_x = F.normalize(proto_x, p=2, dim=1)

      return x, proto_x
          