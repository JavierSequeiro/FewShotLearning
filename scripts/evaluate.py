import torch
import os
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import numpy as np
import torch.nn.functional as F

from src.myLogger import myLogger
from src.samplers import ProtoBatchSampler
from src.helpers import get_CI_stats, get_support_points
from src.losses import PrototypeCELoss

class Evaluator:

    def __init__(self,model, test_set, config, logger):
        self.config = config
        self.model = model.to(self.config.device).eval()
        self.logger = logger
        self.test_set = test_set
        # self.logger = myLogger(logger_name=__name__, config=self.config).get_logger()

    def evaluate_model(self):
        
        correct = 0
        total = 0
        test_loader = DataLoader(self.test_set, batch_size=self.config.batch_size, shuffle=False)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                # self.logger.info(labels.size(0))
                total += labels.size(0)

        test_acc = correct / total
        self.logger.info(f"Test Accuracy = {test_acc:.4f}")
        del self.logger


    def evaluate_distance_model(self):
    
        self.logger.info("Test Phase starts...")
        self.model.eval()

        test_episode_acc = []

        num_episodes_per_epoch = self.config.num_val_episodes_per_epoch
        
        with torch.no_grad():
            for ep in range(num_episodes_per_epoch):
                print(f"Test Episode {ep}")
                test_ma_acc = 0.
                try:
                    supp_imgs, supp_labels, clean_test_set = get_support_points(self.test_set,config=self.config)
                except ValueError as e:
                    self.logger.error(f"Test failed setup: {e}")
                    return 0.0, 0.0

                # Move support data to device
                supp_imgs = supp_imgs.to(self.config.device)
                supp_labels = supp_labels.to(self.config.device)

                # --- PRE-COMPUTE SUPPORT EMBEDDINGS ---
                
                supp_embeddings = self.model(supp_imgs)

                # --- ITERATE OVER CLEAN VALIDATION SET ---
                test_loader = DataLoader(clean_test_set, batch_size=self.config.batch_size, shuffle=False)
                
                for step, (inp, gt) in enumerate(test_loader):
                    inp, gt = inp.to(self.config.device), gt.to(self.config.device)
                    
                    # print(f"GT CLASSES: {gt}")
                    with torch.no_grad():
                        pred_embeddings = self.model(inp)

                        batch_acc = self.compute_metrics(pred_embeddings=pred_embeddings,
                                                                    supp_embeddings=supp_embeddings,
                                                                    gt=gt,
                                                                    supp_labels=supp_labels)

                        test_ma_acc += batch_acc
                        
                        # print(f"\r{step+1:03d}/{len(test_loader):03d}: loss = {val_ma_loss:.4f} - acc ({self.config.K_knn}) = {val_ma_acc:.4f}", end="")

                test_avg_acc = test_ma_acc / len(test_loader)

                test_episode_acc.append(test_avg_acc)

        CI_err = get_CI_stats(acc_list=test_episode_acc, config=self.config)
        final_test_avg_acc = sum(test_episode_acc)/num_episodes_per_epoch
        self.logger.info(f"Distance Test Accuracy = {final_test_avg_acc:.4f} || CI Err. = {CI_err}")

    def compute_metrics(self, pred_embeddings, supp_embeddings, gt, supp_labels):
        # Contrastive Loss on val set
        # loss = self.contrastive_loss(pred_embeddings, gt) 

        # --- k-NN Calculation ---
        # Calculate distance between Batch and Support
        dists = torch.cdist(pred_embeddings, supp_embeddings, p=2)
        
        min_dists, indices = torch.topk(input=dists,k=self.config.K_knn, dim=1, largest=False)
        predicted_classes = supp_labels[indices]
        
        # Most represented class
        majority_label, _ = torch.mode(predicted_classes, dim=1)
        
        # Accuracy
        correct = torch.sum(majority_label == gt).item()
        batch_acc = correct / pred_embeddings.size(0)

        return batch_acc
    
    def evaluate_proto_model(self, **kwargs):
        
        M = self.config.num_classes_per_batch
        K = self.config.K_support_points
        Q = self.config.Q_points

        num_episodes_per_epoch = self.config.num_val_episodes_per_epoch
        
        test_avg_loss = 0.
        test_avg_acc = 0.
        
        test_episode_acc = []
        # CI_bounds = []
        
        self.logger.info("Validation starts...")
        self.model.eval()
        
        with torch.no_grad():
            for ep in range(num_episodes_per_epoch):
                print(f"Test Episode {ep}")
                test_avg_acc = 0.
                myTestSampler = ProtoBatchSampler(dataset=self.test_set,
                                                  n_classes=M, n_samples=K+Q,
                                                  config=self.config)
                test_loader = DataLoader(self.test_set, batch_sampler=myTestSampler)
                for step, (inp, gt) in enumerate(test_loader):

                    inp, gt = inp.to(self.config.device), gt.to(self.config.device)

                    inp = inp.view(M, K + Q, *inp.shape[1:])

                    # support_inp: [M, K, C, H, W]
                    # query_inp:   [M, Q, C, H, W]
                    support_inp = inp[:, :self.config.K_support_points, :, :, :]
                    query_inp   = inp[:, self.config.K_support_points:, :, :, :]

                    support_inp = support_inp.reshape(-1, *inp.shape[2:])
                    query_inp = query_inp.reshape(-1,*inp.shape[2:])

                    with torch.no_grad():
                        pred, pred_proto = self.model(query_inp, support_inp)
                    pred_proto = pred_proto.view(M,K,-1)
                    centroid_pred_proto = torch.mean(pred_proto, dim=1)
                    
                    _, acc = PrototypeCELoss(M=M, Q=Q)(embeddings=pred,
                                            centroid_proto_embeddings=centroid_pred_proto)
                    
                    test_avg_acc += acc.item()

                test_avg_acc = test_avg_acc/len(test_loader)
                test_episode_acc.append(test_avg_acc)

            CI_err = get_CI_stats(acc_list=test_episode_acc, config=self.config)

        final_test_avg_acc = sum(test_episode_acc)/num_episodes_per_epoch
        self.logger.info(f"Prototype Test Accuracy = {final_test_avg_acc:.4f} || CI Err. = {CI_err}")

        