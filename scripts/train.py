import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from src.samplers import ProtoBatchSampler, DistanceBatchSampler

from src.myLogger import myLogger
from src.helpers import get_CI_stats, get_support_points
from src.losses import myEuclideanContrastiveLoss, PrototypeCELoss, CosineSimilarityContrastiveLossV2


class myTrainer:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.ss_iter = None

    def train_model(self,model,
                    train_loader,
                    val_loader,
                    show_training_curves=True,
                    semi_supervised_loader=None,
                    return_acc=False):

        train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []
        self.best_train_acc = 0.
        self.best_val_acc = 0.

        optimizer = optim.SGD(params=model.parameters(), lr=self.config.lr,weight_decay=self.config.weight_decay)
        # optimizer = optim.AdamW(params=model.parameters(), lr=self.config.lr,weight_decay=self.config.weight_decay, betas=(0.9, 0.999))
        model = model.to(self.config.device)

        self.experiment_path = f"{self.config.model_name}_{self.config.specific_experiment}_{self.config.specific_experiment_val}"
        if not self.config.semisupervised_setting:
            if self.config.use_subsample is not None:
                self.experiment_path = f"{self.experiment_path}_{str(self.config.use_subsample).split('.')[0]}_{str(self.config.use_subsample).split('.')[1]}"
                self.model_output_path = osp.join(self.config.output_model_path, self.experiment_path)
            else:
                self.model_output_path = osp.join(self.config.output_model_path, f"{self.experiment_path}") 
                
        else:
            if self.config.perform_online_mixmatch:
                self.experiment_path = f"{self.experiment_path}_ONLINE"
            self.model_output_path = osp.join(self.config.output_model_path, f"{self.experiment_path}_SS_0_{str(self.config.perc_semisupervised).split('.')[1]}")
        os.makedirs(self.model_output_path, exist_ok=True)

        # if semi_supervised_loader is not None:
        #     semi_supervised_loader = iter(semi_supervised_loader,)

        for epoch in range(1, self.config.epochs + 1):
            
            self.epoch = epoch
            self.logger.info(f'Epoch {epoch:02d}/{self.config.epochs:02d}')
            self.logger.info('-' * 10)
            if self.config.semisupervised_setting:
                model, train_acc, train_loss = self.train_fn_ss(model=model, train_loader=train_loader, optimizer=optimizer, semi_supervised_loader=semi_supervised_loader)
            else:
                model, train_acc, train_loss = self.train_fn(model=model, train_loader=train_loader, optimizer=optimizer, semi_supervised_loader=semi_supervised_loader)
            
            val_acc, val_loss = self.validate_fn(model=model, val_loader=val_loader)

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            
            if self.best_val_acc < val_acc or epoch == self.config.epochs:
                
                torch.save(model.state_dict(), osp.join(self.model_output_path, f"model_ep{epoch}.pt"))
                self.logger.info(f"Saving model with better perfs ({self.best_val_acc} -> {val_acc}")
                self.best_val_acc = val_acc

        if show_training_curves:
            self.show_training_curves({"train_acc":train_acc_list,
                                       "val_acc":val_acc_list,
                                       "train_loss":train_loss_list,
                                       "val_loss":val_loss_list})
            
        del self.logger

        if return_acc:
            return model, train_acc_list, val_acc_list
        else:
            return model

    def train_fn(self, model, train_loader, optimizer, semi_supervised_loader=None):

        model.train()
        train_ma_loss = 0.
        train_ma_acc = 0.
 
        self.logger.info("Training starts...")
        for step, (inp, gt) in enumerate(train_loader):
            
            inp, gt = inp.to(self.config.device), gt.to(self.config.device)
            gt_oh = F.one_hot(gt, num_classes=self.config.num_classes).float()
                
            optimizer.zero_grad()

            pred = model(inp)
            loss = self.config.train_criterion(pred, gt_oh)
            
            loss.backward()
            optimizer.step()
            train_ma_loss  = 0.9*train_ma_loss + 0.1*loss.item()
            
            # Supervised Evaluation
            _, pred_y_num = torch.max(pred, 1)
            _,gt_num = torch.max(gt_oh, 1)
            batch_acc = torch.sum(pred_y_num == gt_num).item()/self.config.batch_size

            train_ma_acc = 0.9*train_ma_acc + 0.1*batch_acc
            
            print(f"\r{step+1:03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f} - acc (train) = {train_ma_acc:.4f}", end="")
         
        # Keep just final epoch values
        # self.logger.info(f"\r{step+1:03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f} - acc (train) = {train_ma_acc:.4f}")
        self.logger.info(f"{step+1:03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f} - acc (train) = {train_ma_acc:.4f}")
        self.logger.info("\n")  # new oline

        if self.best_train_acc < train_ma_acc:
            self.best_train_acc = train_ma_acc

        return model, train_ma_acc, train_ma_loss

    def train_fn_ss(self, model, train_loader, optimizer, semi_supervised_loader=None):

        model.train()
        train_ma_loss = 0.
        train_ma_acc = 0.
 
        self.logger.info("Training starts...")
        for step, (inp, gt) in enumerate(train_loader):
            
            inp, gt = inp.to(self.config.device), gt.to(self.config.device)
            gt_oh = F.one_hot(gt, num_classes=self.config.num_classes).float()

            # Augment Labelled Input
            inp_aug = self.config.ss_possible_augmentations(inp)
            # inp_aug = inp
            
            # Generate Soft Labels for Unlabelled Set
            shuffled_idx = torch.randperm(n=self.config.batch_size*2).to(self.config.device)

            if self.ss_iter is None:
                    self.ss_iter = iter(semi_supervised_loader)
            # 2. Get Unlabeled Input (U)
            try:
                inp_ss, label_ss = next(self.ss_iter)
            except StopIteration:
                # Reset the iterator when it runs out of data
                self.logger.debug("Resetting semi_supervised_loader iterator.")
                self.ss_iter = iter(semi_supervised_loader)
                inp_ss, label_ss= next(self.ss_iter)
            inp_ss, label_ss = inp_ss.to(self.config.device), label_ss.to(self.config.device)

            if self.config.perform_online_mixmatch:
                with torch.no_grad():
                    preds_sum = 0
                    for j in range(self.config.K):
                        aug = self.config.ss_possible_augmentations(inp_ss)
                        y_aug = F.softmax(model(aug), dim=1)
                        
                        if j == 0:
                            preds_sum = y_aug
                        else:
                            preds_sum += y_aug

                    avg_pred = preds_sum / self.config.K
                    # Reduce Entropy with sharpen function
                    label_ss = self.sharpen(p=avg_pred, T=self.config.T)
                    
            # Concatenate Labelled and Unlabelled
            concat_inp = torch.cat([inp_aug, inp_ss], dim=0)
            concat_labels = torch.cat([gt_oh, label_ss], dim=0)
                # Concatenate Labelled and Unlabelled
            shuffled_concat_inp = concat_inp[shuffled_idx]
            shuffled_concat_labels = concat_labels[shuffled_idx]

            # MixUp Labelled and Unlabelled
            lambda_sampled = self._sample_mixup_lambda()
            labelled_mixup_inp = lambda_sampled*inp_aug + (1 - lambda_sampled)*shuffled_concat_inp[:self.config.batch_size]
            labelled_mixup_labels = lambda_sampled*gt_oh + (1 - lambda_sampled)*shuffled_concat_labels[:self.config.batch_size]

            unlabelled_mixup_inp = lambda_sampled*inp_ss + (1 - lambda_sampled)*shuffled_concat_inp[self.config.batch_size:]
            unlabelled_mixup_labels = lambda_sampled*label_ss + (1 - lambda_sampled)*shuffled_concat_labels[self.config.batch_size:]
                
                
            optimizer.zero_grad()
            # Predict "Labelled"
            labelled_pred = model(labelled_mixup_inp)
            # Predict "Unlabelled"
            unlabelled_pred = model(unlabelled_mixup_inp)            

            # Compute Loss
            probs_u = torch.softmax(unlabelled_pred, dim=1)

            Loss_labelled = -torch.mean(torch.sum(F.log_softmax(labelled_pred, dim=1) * labelled_mixup_labels, dim=1))
            Loss_unlabelled = torch.mean((probs_u - unlabelled_mixup_labels)**2)
            
            # Total MixMatch Loss
            loss = Loss_labelled + (self.config.lambda_loss*self.linear_rampup())*Loss_unlabelled 

            loss.backward()
            optimizer.step()
            train_ma_loss  = 0.9*train_ma_loss + 0.1*loss.item()
            
            # Supervised Evaluation
            # _, pred_y_num = torch.max(pred, 1)
            # _,gt_num = torch.max(gt_oh, 1)
            # batch_acc = torch.sum(pred_y_num == gt_num).item()/self.config.batch_size

            # Semi Supervised Evaluation
            _, labelled_pred_y_num = torch.max(labelled_pred, 1)
            _, gt_labelled = torch.max(labelled_mixup_labels, 1)
            
            _, unlabelled_pred_y_num = torch.max(unlabelled_pred, 1)
            _, gt_unlabelled = torch.max(unlabelled_mixup_labels, 1)

            batch_acc = (torch.sum(labelled_pred_y_num == gt_labelled).item() + torch.sum(unlabelled_pred_y_num == gt_unlabelled).item())/2
            batch_acc = batch_acc / (self.config.batch_size)

            train_ma_acc = 0.9*train_ma_acc + 0.1*batch_acc
                
            print(f"\r{step+1:03d}/{len(train_loader):03d} lambda={round(lambda_sampled,3)}: loss = {train_ma_loss:.4f} - acc (train) = {train_ma_acc:.4f}", end="")
            
        # Keep just final epoch values
        # self.logger.info(f"\r{step+1:03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f} - acc (train) = {train_ma_acc:.4f}")
        self.logger.info(f"{step+1:03d}/{len(train_loader):03d} lambda={round(lambda_sampled,3)}: loss = {train_ma_loss:.4f} - acc (train) = {train_ma_acc:.4f}")
        self.logger.info("\n")  # new oline

        if self.best_train_acc < train_ma_acc:
            self.best_train_acc = train_ma_acc

        return model, train_ma_acc, train_ma_loss

    def train_distance(self, model, train_loader, optimizer):

        model.train()
        train_ma_loss = 0.
        train_ma_acc = 0.
 
        self.logger.info("Training starts...")
        for step, (inp, gt) in enumerate(train_loader):
            
            inp, gt = inp.to(self.config.device), gt.to(self.config.device)
            gt_oh = F.one_hot(gt, num_classes=self.config.num_classes).float()
                
            optimizer.zero_grad()

            pred = model(inp)
            # loss = self.config.train_criterion(pred, gt_oh)
            loss = self.contrastive_loss(embeddings=pred, labels=gt)
            
            loss.backward()
            optimizer.step()
            train_ma_loss  = 0.9*train_ma_loss + 0.1*loss.item()
            
            # Supervised Evaluation (after kNN classifier)

            # _, pred_y_num = torch.max(pred, 1)
            # _,gt_num = torch.max(gt_oh, 1)
            # batch_acc = torch.sum(pred_y_num == gt_num).item()/self.config.batch_size

            # train_ma_acc = 0.9*train_ma_acc + 0.1*batch_acc
            
            print(f"\r{step+1:03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f}", end="")
         
        # Keep just final epoch values
        # self.logger.info(f"\r{step+1:03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f} - acc (train) = {train_ma_acc:.4f}")
        self.logger.info(f"{step+1:03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f}")
        self.logger.info("\n")  # new oline

        if self.best_train_acc < train_ma_acc:
            self.best_train_acc = train_ma_acc

        return model, train_ma_acc, train_ma_loss

    def validate_fn(self,model, val_loader):
        val_ma_loss = 0.
        val_ma_acc = 0.
        self.logger.info("Validation starts...")
        model.eval()
        for step, (inp, gt) in enumerate(val_loader):
            # TODO: put data to the correct device
            inp, gt = inp.to(self.config.device), gt.to(self.config.device)
            gt_oh = F.one_hot(gt,num_classes=self.config.num_classes).float()
            
            pred = model(inp)
            
            loss = self.config.val_criterion(pred, gt_oh)
            
            val_ma_loss  = 0.9*val_ma_loss + 0.1*loss.item()

            _, pred_y_num = torch.max(pred, 1)
            _,gt_num = torch.max(gt_oh, 1)
            batch_acc = torch.sum(pred_y_num == gt_num).item()/self.config.batch_size
            val_ma_acc = 0.9*val_ma_acc + 0.1*batch_acc
            print(f"\r{step+1:03d}/{len(val_loader):03d}: loss = {val_ma_loss:.4f} - acc (val) = {val_ma_acc:.4f}", end="")

        # self.logger.info(f"\r{step+1:03d}/{len(val_loader):03d}: loss = {val_ma_loss:.4f} - acc (val) = {val_ma_acc:.4f}")
        self.logger.info(f"{step+1:03d}/{len(val_loader):03d}: loss = {val_ma_loss:.4f} - acc (val) = {val_ma_acc:.4f}")
        self.logger.info("\n")  # new line

        return val_ma_acc, val_ma_loss
    
    def validate_fn_distance(self, model, val_set):
        val_ma_loss = 0.
        val_ma_acc = 0.
        
        self.logger.info("Validation starts...")
        model.eval()

        
        try:
            supp_imgs, supp_labels, clean_val_set = self.get_support_points(val_set)
        except ValueError as e:
            self.logger.error(f"Validation failed setup: {e}")
            return 0.0, 0.0

        # Move support data to device
        supp_imgs = supp_imgs.to(self.config.device)
        supp_labels = supp_labels.to(self.config.device)

        # --- PRE-COMPUTE SUPPORT EMBEDDINGS ---
        with torch.no_grad():
            supp_embeddings = model(supp_imgs)

        # --- ITERATE OVER CLEAN VALIDATION SET ---
        val_loader = DataLoader(clean_val_set, batch_size=self.config.batch_size, shuffle=False)
        
        for step, (inp, gt) in enumerate(val_loader):
            inp, gt = inp.to(self.config.device), gt.to(self.config.device)
            
            with torch.no_grad():
                pred_embeddings = model(inp)

                # Optional: Contrastive Loss on val set
                loss = self.contrastive_loss(pred_embeddings, gt) # Or your specific val criterion
                val_ma_loss = 0.9 * val_ma_loss + 0.1 * loss.item()

                # --- k-NN Calculation ---
                # Calculate distance between Batch and Support
                dists = torch.cdist(pred_embeddings, supp_embeddings, p=2)
                
                # Find closest neighbor (1-NN)
                min_dists, indices = torch.min(dists, dim=1)
                predicted_classes = supp_labels[indices]
                
                # Accuracy
                correct = torch.sum(predicted_classes == gt).item()
                batch_acc = correct / inp.size(0)

                val_ma_acc = 0.9 * val_ma_acc + 0.1 * batch_acc
                
                print(f"\r{step+1:03d}/{len(val_loader):03d}: loss = {val_ma_loss:.4f} - acc (1NN) = {val_ma_acc:.4f}", end="")

        self.logger.info(f"{step+1:03d}/{len(val_loader):03d}: loss = {val_ma_loss:.4f} - acc (1NN) = {val_ma_acc:.4f}\n")

        return val_ma_acc, val_ma_loss
    

    def show_training_curves(self,values_dict):
        self.logger.info("SHOW TRAINING CURVES UP AND RUNNING CONSTRUCTION!")

        plt.figure(figsize=(14,10))
        plt.subplot(1,2,1)
        plt.plot(values_dict["train_acc"],label="Train Acc.", color="blue")
        plt.plot(values_dict["val_acc"],label="Valid. Acc.", color="orange")
        plt.xlabel("Epochs")
        plt.title("Accuracy")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1)
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(values_dict["train_loss"],label="Train Loss", color="red")
        plt.plot(values_dict["val_loss"],label="Valid. Loss.", color="green")
        plt.xlabel("Epochs")
        plt.title("Loss")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(osp.join(self.model_output_path,"training_metrics_plot.png"))


    #####
    # Auxiliary
    #####
    def sharpen(self, p, T):
        """Sharpening function (MixMatch Q function)."""
        # Raises each element of p to the power of 1/T
        p_sharpened = p ** (1.0 / T)
        # Divides by the sum of elements along dim=1 to re-normalize
        return p_sharpened / p_sharpened.sum(dim=1, keepdim=True)
    
    def linear_rampup(self):

        if self.epoch == 0:
            return 1.0
        else:
            coeff = self.epoch/self.config.epochs
            coeff = np.clip(coeff, 0.0, 1.0)
            return float(coeff)

    def _sample_mixup_lambda(self):
        alpha = self.config.lambda_
        lambda_sampled = np.random.beta(alpha, alpha)

        lambda_sampled = max(lambda_sampled, 1-lambda_sampled)
        return lambda_sampled


    def contrastive_loss(self,embeddings, labels):

        # Euclidean Distance between embeddings
        # euclidean_distance = F.pairwise_distance(embeddings_1, embeddings_2)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float()
        mask_neg = torch.ne(labels, labels.T).float()

        # Loss for positive pairs (similar images)
        loss_poss = mask_pos*torch.pow(dist_matrix, 2)
        # loss_poss = labels*torch.pow(euclidean_distance, 2)

        # Loss for negative pairs (different images)
        # loss_neg = (1-labels)*torch.pow(torch.clamp(self.config.negative_margin - euclidean_distance, min=0.0),2)
        loss_neg = mask_neg*torch.pow(torch.clamp(self.config.negative_margin - dist_matrix, min=0.0),2)
        
        # total_loss = 0.5 *(labels*loss_poss + (1-labels)*loss_neg)
        # mean_loss = total_loss.mean()
        total_loss = 0.5*(loss_poss + loss_neg)
        return total_loss.mean()
    

    def get_support_points(self, dataset):

        k_per_class = self.config.K_knn
        indices = range(len(dataset))
        all_targets = [dataset[i][1] for i in indices]

        # 2. Group indices by class
        # class_map: { class_label : [global_dataset_index, ...] }
        class_map = defaultdict(list)
        for idx, label in zip(indices, all_targets):
            # Handle tensor labels if necessary
            if torch.is_tensor(label):
                label = label.item()
            class_map[label].append(idx)

        # 3. Select K indices per class
        support_indices = []
        val_remainder_indices = []

        for label, idx_list in class_map.items():
            # Check if we have enough samples
            if len(idx_list) < k_per_class:
                raise ValueError(f"Class {label} has {len(idx_list)} samples, but you requested K={k_per_class}.")
            
            # Shuffle indices for this class to ensure random selection
            np.random.shuffle(idx_list)
            
            # Pick K for support, rest for validation
            support_indices.extend(idx_list[:k_per_class])
            val_remainder_indices.extend(idx_list[k_per_class:])

        # 4. Create the final objects
        # Create the 'clean' validation subset (everything NOT in support)
        clean_val_set = Subset(dataset.dataset if isinstance(dataset, Subset) else dataset, 
                            val_remainder_indices)

        # Create a temporary subset for support to extract tensors
        support_subset = Subset(dataset.dataset if isinstance(dataset, Subset) else dataset, 
                                support_indices)
        
        # Load support data into tensors
        # batch_size = len(support_indices) ensures we get it all in one go
        temp_loader = DataLoader(support_subset, batch_size=len(support_indices), shuffle=False)
        support_images, support_labels = next(iter(temp_loader))

        return support_images, support_labels, clean_val_set
    
###############
# EX2
###############

class myTrainerDistance:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def train_model(self,model,
                    train_set,
                    val_set,
                    show_training_curves=True,
                    return_acc=False):

        train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []
        CI_err_list = []
        self.best_train_acc = 0.
        self.best_val_acc = 0.

        optimizer = optim.SGD(params=model.parameters(), lr=self.config.lr,weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer=optimizer,total_iters=self.config.epochs, power=0.9)
        model = model.to(self.config.device)

        self.experiment_path = f"{self.config.model_name}_{self.config.specific_experiment}_{self.config.specific_experiment_val}"
        self.model_output_path = osp.join(self.config.output_model_path, f"{self.experiment_path}") 
                
        os.makedirs(self.model_output_path, exist_ok=True)

        # Get Train Dataloader
        myTrainSampler = DistanceBatchSampler(dataset=train_set,
                                              n_classes=self.config.num_classes_per_batch,
                                              n_samples=self.config.n_samples_per_class)
        train_loader = DataLoader(train_set,batch_sampler=myTrainSampler)

        for epoch in range(1, self.config.epochs + 1):
            
            self.epoch = epoch
            self.logger.info(f'Epoch {epoch:02d}/{self.config.epochs:02d}')
            self.logger.info('-' * 10)
            
            model, train_acc, train_loss = self.train_fn_distance(model=model,
                                                                  train_loader=train_loader,
                                                                  optimizer=optimizer)
            
            val_acc, val_loss, CI_err = self.validate_fn_distance(model=model,
                                                                  val_set=val_set)
            
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            CI_err_list.append(CI_err)

            
            scheduler.step()
            
            if self.best_val_acc < val_acc or epoch == self.config.epochs:
                
                torch.save(model.state_dict(), osp.join(self.model_output_path, f"model_ep{epoch}.pt"))
                self.logger.info(f"Saving model with better perfs ({self.best_val_acc} -> {val_acc}")
                self.best_val_acc = val_acc

        if show_training_curves:
            self.show_training_curves({"train_acc":train_acc_list,
                                       "val_acc":val_acc_list,
                                       "train_loss":train_loss_list,
                                       "val_loss":val_loss_list})
        del self.logger

        if return_acc:
            return model, train_acc_list, val_acc_list, CI_err_list
        else:
            return model

    def train_fn_distance(self, model, train_loader, optimizer):

        model.train()
        train_ma_loss = 0.
        train_ma_acc = 0.
 
        self.logger.info("Training starts...")
        for step, (inp, gt) in enumerate(train_loader):
            
            inp, gt = inp.to(self.config.device), gt.to(self.config.device)
                
            optimizer.zero_grad()

            pred = model(inp)
            if self.config.distance_metric == "cosine_similarity":
                loss = CosineSimilarityContrastiveLossV2(neg_margin=self.config.negative_margin, pos_margin=self.config.positive_margin)(x=pred, labels=gt)
            else:
                loss = myEuclideanContrastiveLoss(neg_margin=self.config.negative_margin)(embeddings=pred, labels=gt)
            
            loss.backward()
            optimizer.step()
            train_ma_loss  = 0.9*train_ma_loss + 0.1*loss.item()
            
            print(f"\r{step+1:03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f}", end="")
         
        # Keep just final epoch values
        self.logger.info(f"{len(train_loader):03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f}")
        self.logger.info("\n")  # new oline

        if self.best_train_acc < train_ma_acc:
            self.best_train_acc = train_ma_acc

        return model, train_ma_acc, train_ma_loss

    def validate_fn_distance(self, model, val_set):
        val_episode_loss = []
        val_episode_acc = []

        num_episodes_per_epoch = self.config.num_val_episodes_per_epoch
        
        self.logger.info("Validation starts...")
        model.eval()
        with torch.no_grad():
            for ep in range(num_episodes_per_epoch):
                print(f"Validation Episode {ep}")
                val_ma_loss = 0.
                val_ma_acc = 0.
                try:
                    supp_imgs, supp_labels, clean_val_set = get_support_points(val_set, self.config)
                except ValueError as e:
                    self.logger.error(f"Validation failed setup: {e}")
                    return 0.0, 0.0

                # Move support data to device
                supp_imgs = supp_imgs.to(self.config.device)
                supp_labels = supp_labels.to(self.config.device)

                # --- PRE-COMPUTE SUPPORT EMBEDDINGS ---
                
                supp_embeddings = model(supp_imgs)

                # --- ITERATE OVER CLEAN VALIDATION SET ---
                val_loader = DataLoader(clean_val_set, batch_size=self.config.batch_size, shuffle=False)
                
                for step, (inp, gt) in enumerate(val_loader):
                    inp, gt = inp.to(self.config.device), gt.to(self.config.device)
                    
                    # print(f"GT CLASSES: {gt}")
                    with torch.no_grad():
                        pred_embeddings = model(inp)

                        batch_loss, batch_acc = self.compute_metrics(pred_embeddings=pred_embeddings,
                                                                    supp_embeddings=supp_embeddings,
                                                                    gt=gt,
                                                                    supp_labels=supp_labels)

                        val_ma_loss += batch_loss
                        val_ma_acc += batch_acc
                        
                        # print(f"\r{step+1:03d}/{len(val_loader):03d}: loss = {val_ma_loss:.4f} - acc ({self.config.K_knn}) = {val_ma_acc:.4f}", end="")

                val_avg_acc = val_ma_acc / len(val_loader)
                val_avg_loss = val_ma_loss / len(val_loader)

                val_episode_loss.append(val_avg_loss)
                val_episode_acc.append(val_avg_acc)

        CI_err = get_CI_stats(acc_list=val_episode_acc, config=self.config)
        final_val_avg_acc = sum(val_episode_acc)/num_episodes_per_epoch
        final_val_avg_loss = sum(val_episode_loss)/num_episodes_per_epoch
    
        self.logger.info(f"Validation: Avg Loss = {final_val_avg_loss:.4f} - Avg Acc. (K={self.config.K_knn}) = {final_val_avg_acc:.4f} (CI. Error ({CI_err})\n")
        
        return final_val_avg_acc, final_val_avg_loss, CI_err
    

    def show_training_curves(self,values_dict):
        self.logger.info("SHOW TRAINING CURVES UP AND RUNNING CONSTRUCTION!")

        plt.figure(figsize=(14,10))
        plt.subplot(1,2,1)
        plt.plot(values_dict["train_acc"],label="Train Acc.", color="blue")
        plt.plot(values_dict["val_acc"],label="Valid. Acc.", color="orange")
        plt.xlabel("Epochs")
        plt.title("Accuracy")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1)
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(values_dict["train_loss"],label="Train Loss", color="red")
        plt.plot(values_dict["val_loss"],label="Valid. Loss.", color="green")
        plt.xlabel("Epochs")
        plt.title("Loss")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(osp.join(self.model_output_path,"training_metrics_plot.png"))


    #####
    # Auxiliary
    #####
    
    def compute_metrics(self, pred_embeddings, supp_embeddings, gt, supp_labels):
        # Contrastive Loss on val set
        if self.config.distance_metric == "cosine_similarity":
            loss = CosineSimilarityContrastiveLossV2(neg_margin=self.config.negative_margin, pos_margin=self.config.positive_margin)(x=pred_embeddings, labels=gt)
        else:
            loss = myEuclideanContrastiveLoss(neg_margin=self.config.negative_margin)(embeddings=pred_embeddings, labels=gt)

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

        return loss.item(), batch_acc

###############
# EX3 PROTOTYPICAL LEARNING
###############

class myTrainerProto:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def train_model(self,model,
                    train_set,
                    val_set,
                    show_training_curves=True,
                    return_acc=False,
                    **kwargs):

        train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []
        CI_err_list = []
        self.best_train_acc = 0.
        self.best_val_acc = 0.

        optimizer = optim.SGD(params=model.parameters(), lr=self.config.lr,weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer=optimizer,total_iters=self.config.epochs, power=0.9)
        model = model.to(self.config.device)

        self.experiment_path = f"{self.config.model_name}_{self.config.specific_experiment}_{self.config.specific_experiment_val}"
        
        self.model_output_path = osp.join(self.config.output_model_path, f"{self.experiment_path}") 
                
        os.makedirs(self.model_output_path, exist_ok=True)

        myTrainSampler = ProtoBatchSampler(dataset=train_set,
                                           n_classes=self.config.num_classes_per_batch,
                                           n_samples=self.config.K_support_points + self.config.Q_points,
                                           config=self.config)
        
        myValSampler = ProtoBatchSampler(dataset=val_set,
                                           n_classes=self.config.num_classes_per_batch,
                                           n_samples=self.config.K_support_points + self.config.Q_points,
                                           config=self.config)
        
        train_loader = DataLoader(train_set,batch_sampler=myTrainSampler)
        val_loader = DataLoader(val_set,batch_sampler=myValSampler)

        self.train_iter = iter(train_loader)

        for epoch in range(1, self.config.epochs + 1):
            
            self.epoch = epoch
            self.logger.info(f'Epoch {epoch:02d}/{self.config.epochs:02d}')
            self.logger.info('-' * 10)
            
            model, train_acc, train_loss = self.train_fn_proto(model=model, train_loader=train_loader, optimizer=optimizer)
            
            val_acc, val_loss, CI_err = self.validate_fn_proto(model=model, val_set=val_set)

            scheduler.step()

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            CI_err_list.append(CI_err)

            
            if self.best_val_acc < val_acc or epoch == self.config.epochs:
                
                torch.save(model.state_dict(), osp.join(self.model_output_path, f"model_ep{epoch}.pt"))
                self.logger.info(f"Saving model with better perfs ({self.best_val_acc} -> {val_acc}")
                self.best_val_acc = val_acc

        if show_training_curves:
            self.show_training_curves({"train_acc":train_acc_list,
                                       "val_acc":val_acc_list,
                                       "train_loss":train_loss_list,
                                       "val_loss":val_loss_list})
        del self.logger

        if return_acc:
            return model, train_acc_list, val_acc_list, CI_err_list
        else:
            return model

    def train_fn_proto(self, model, train_loader, optimizer):

        M = self.config.num_classes_per_batch
        K = self.config.K_support_points
        Q = self.config.Q_points

        model.train()
        train_ma_loss = 0.
        train_ma_acc = 0.
        train_avg_loss = 0.
        train_avg_acc = 0.

        episodes = self.config.num_training_episodes_per_epoch
 
        self.logger.info("Training starts...")
        for step in range(episodes):#enumerate(train_loader):
            
            try:
                inp, gt = next(self.train_iter)
            except StopIteration:
                # Reset the iterator when it runs out of data
                self.logger.debug("Resetting semi_supervised_loader iterator.")
                self.train_iter = iter(train_loader)
                inp, gt = next(self.train_iter)
            
            inp, gt = inp.to(self.config.device), gt.to(self.config.device)
                
            optimizer.zero_grad()
            inp = inp.view(M, K + Q, *inp.shape[1:])

            # support_inp: [M, K, C, H, W]
            # query_inp:   [M, Q, C, H, W]
            support_inp = inp[:, :self.config.K_support_points, :, :, :]
            query_inp   = inp[:, self.config.K_support_points:, :, :, :]

            support_inp = support_inp.reshape(-1, *inp.shape[2:])
            query_inp = query_inp.reshape(-1,*inp.shape[2:])

            pred, pred_proto = model(query_inp, support_inp)
            pred_proto = pred_proto.view(M,K,-1)
            centroid_pred_proto = torch.mean(pred_proto, dim=1)

            loss, acc = PrototypeCELoss(M=M, Q=Q)(embeddings=pred,
                                       centroid_proto_embeddings=centroid_pred_proto)
            
            # gt = gt.view(M, Q + K, *gt.shape[1:])

            loss.backward()
            optimizer.step()
            train_ma_loss  = 0.9*train_ma_loss + 0.1*loss.item()
            train_avg_loss += loss.item()
            train_avg_acc += acc.item()
            
            
            print(f"\r{step+1:03d}/{episodes:03d}: loss = {train_ma_loss:.4f}", end="")
         
        # Keep just final epoch values
        train_avg_loss = train_avg_loss / episodes
        train_avg_acc = train_avg_acc / episodes

        self.logger.info(f"TRAIN: Avg. Loss = {train_avg_loss:.4f} | Avg. Acc = {train_avg_acc:.4f}")
        self.logger.info("\n")

        if self.best_train_acc < train_ma_acc:
            self.best_train_acc = train_ma_acc

        return model, train_avg_acc, train_avg_loss

    
    def validate_fn_proto(self, model, val_set, **kwargs):
        
        M = self.config.num_classes_per_batch
        K = self.config.K_support_points
        Q = self.config.Q_points
        num_episodes_per_epoch = self.config.num_val_episodes_per_epoch
        
        val_avg_loss = 0.
        val_avg_acc = 0.
        
        val_episode_acc = []
        val_episode_loss = []
        # CI_bounds = []
        
        self.logger.info("Validation starts...")
        model.eval()

        # val_set = kwargs.get("val_set")
        
        with torch.no_grad():
            for ep in range(num_episodes_per_epoch):
                print(f"Val Episode {ep}")
                val_avg_loss = 0.
                val_avg_acc = 0.
                myValSampler = ProtoBatchSampler(dataset=val_set, n_classes=M, n_samples=K+Q, config=self.config)
                val_loader = DataLoader(val_set, batch_sampler=myValSampler)
                for step, (inp, gt) in enumerate(val_loader):

                    inp, gt = inp.to(self.config.device), gt.to(self.config.device)
                    # gt_oh = F.one_hot(gt, num_classes=self.config.num_classes).float()

                    inp = inp.view(M, K + Q, *inp.shape[1:])

                    # support_inp: [M, K, C, H, W]
                    # query_inp:   [M, Q, C, H, W]
                    support_inp = inp[:, :self.config.K_support_points, :, :, :]
                    query_inp   = inp[:, self.config.K_support_points:, :, :, :]

                    support_inp = support_inp.reshape(-1, *inp.shape[2:])
                    query_inp = query_inp.reshape(-1,*inp.shape[2:])
                    
                    with torch.no_grad():
                        pred, pred_proto = model(query_inp, support_inp)
                    pred_proto = pred_proto.view(M,K,-1)
                    centroid_pred_proto = torch.mean(pred_proto, dim=1)
                    # loss = self.config.train_criterion(pred, gt_oh)
                    loss, acc = PrototypeCELoss(M=M, Q=Q)(embeddings=pred,
                                            centroid_proto_embeddings=centroid_pred_proto)
                    
                    val_avg_loss += loss.item()
                    val_avg_acc += acc.item()

                val_avg_loss = val_avg_loss/len(val_loader)
                val_avg_acc = val_avg_acc/len(val_loader)

                val_episode_loss.append(val_avg_loss)
                val_episode_acc.append(val_avg_acc)

           
            CI_err = get_CI_stats(acc_list=val_episode_acc, config=self.config)

        final_val_avg_acc = sum(val_episode_acc)/num_episodes_per_epoch
        final_val_avg_loss = sum(val_episode_loss)/num_episodes_per_epoch
        self.logger.info(f"Validation: Avg Loss = {final_val_avg_loss:.4f} - Avg Acc. = {final_val_avg_acc:.4f} (CI. Error ({CI_err})\n")

        return final_val_avg_acc, final_val_avg_loss, CI_err
    

    def show_training_curves(self,values_dict):
        self.logger.info("SHOW TRAINING CURVES UP AND RUNNING CONSTRUCTION!")

        plt.figure(figsize=(14,10))
        plt.subplot(1,2,1)
        plt.plot(values_dict["train_acc"],label="Train Acc.", color="blue")
        plt.plot(values_dict["val_acc"],label="Valid. Acc.", color="orange")
        plt.xlabel("Epochs")
        plt.title("Accuracy")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1)
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(values_dict["train_loss"],label="Train Loss", color="red")
        plt.plot(values_dict["val_loss"],label="Valid. Loss.", color="green")
        plt.xlabel("Epochs")
        plt.title("Loss")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(osp.join(self.model_output_path,"training_metrics_plot.png"))
    
    
class myTrainerDistanceSS:

    def __init__(self, config, logger):
        self.config = config
        # self.logger = logger
        self.logger = myLogger(logger_name=__name__, config=self.config).get_logger()
        self.ss_iter = None

    def train_model(self,model,
                    train_loader,
                    train_set,
                    val_loader,
                    show_training_curves=True,
                    semi_supervised_loader=None,
                    return_acc=False):

        train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []
        self.best_train_acc = 0.
        self.best_val_acc = 0.

        optimizer = optim.SGD(params=model.parameters(), lr=self.config.lr,weight_decay=self.config.weight_decay)

        model = model.to(self.config.device)
        # specific_experiment = "K_CROSS_VAL_EX1"
        self.experiment_path = f"{self.config.model_name}_{self.config.specific_experiment}_{self.config.specific_experiment_val}"
        if not self.config.semisupervised_setting:
            if self.config.use_subsample is not None:
                self.experiment_path = f"{self.experiment_path}_{str(self.config.use_subsample).split('.')[0]}_{str(self.config.use_subsample).split('.')[1]}"
                self.model_output_path = osp.join(self.config.output_model_path, self.experiment_path)
            else:
                self.model_output_path = osp.join(self.config.output_model_path, f"{self.experiment_path}") 
                
        else:
            if self.config.perform_online_mixmatch:
                self.experiment_path = f"{self.experiment_path}_ONLINE"
            self.model_output_path = osp.join(self.config.output_model_path, f"{self.experiment_path}_SS_0_{str(self.config.perc_semisupervised).split('.')[1]}")
        os.makedirs(self.model_output_path, exist_ok=True)


        for epoch in range(1, self.config.epochs + 1):
            
            self.epoch = epoch
            self.logger.info(f'Epoch {epoch:02d}/{self.config.epochs:02d}')
            self.logger.info('-' * 10)

            self.logger.info(f"Obtaining reference embeddings...")
            ref_embeddings, ref_labels = self.get_distance_embeddings_dataloader(train_loader, model)
            self.logger.info(f"Reference embeddings have been extracted...")
            
            model, train_acc, train_loss = self.train_fn_distance_ss(model=model,
                                                                     train_loader=train_loader,
                                                                     optimizer=optimizer,
                                                                     ref_embeddings=ref_embeddings,
                                                                     ref_labels=ref_labels,
                                                                     semi_supervised_loader=semi_supervised_loader)
        
            val_acc, val_loss = self.validate_fn(model=model, val_loader=val_loader)

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            
            if self.best_val_acc < val_acc or epoch == self.config.epochs:
                
                torch.save(model.state_dict(), osp.join(self.model_output_path, f"model_ep{epoch}.pt"))
                self.logger.info(f"Saving model with better perfs ({self.best_val_acc} -> {val_acc}")
                self.best_val_acc = val_acc

        if show_training_curves:
            self.show_training_curves({"train_acc":train_acc_list,
                                       "val_acc":val_acc_list,
                                       "train_loss":train_loss_list,
                                       "val_loss":val_loss_list})
        del self.logger

        if return_acc:
            return model, train_acc_list, val_acc_list
        else:
            return model


    def train_fn_distance_ss(self, model, train_loader, optimizer, ref_embeddings, ref_labels, semi_supervised_loader=None):

        model.train()
        train_ma_loss = 0.
        train_ma_acc = 0.
        total_unlabelled = 0
 
        self.logger.info("Training starts...")
        for step, (inp, gt) in enumerate(train_loader):
            
            inp, gt = inp.to(self.config.device), gt.to(self.config.device)
            gt_oh = F.one_hot(gt, num_classes=self.config.num_classes).float()

            if self.ss_iter is None:
                    self.ss_iter = iter(semi_supervised_loader)
            # 2. Get Unlabeled Input (U)
            try:
                inp_ss, label_ss = next(self.ss_iter)
            except StopIteration:
                # Reset the iterator when it runs out of data
                self.logger.debug("Resetting semi_supervised_loader iterator.")
                self.ss_iter = iter(semi_supervised_loader)
                inp_ss, label_ss= next(self.ss_iter)
            inp_ss, label_ss = inp_ss.to(self.config.device), label_ss.to(self.config.device)
                
            optimizer.zero_grad()   
            # Predict "Labelled"
            labeled_pred = model(inp)
            # Predict "Unlabelled"
            pred_ss = model(inp_ss)

            # Get unlabelled embeddings
            ss_embeddings = F.normalize(pred_ss, p=2, dim=1)

            # Compute distance with labelled set
            dists = torch.cdist(ss_embeddings, ref_embeddings)
            

            # Minimum distance for each unlabelled sample with labelled set
            min_dist, closest_idx = dists.min(dim=1)
            # print(min_dist)
            if step % 100 == 0:
                print(f"\n[DEBUG] Min Dist: {min_dist.min().item():.4f}, Mean Dist: {min_dist.mean().item():.4f}, Max Dist: {min_dist.max().item():.4f}")
            # Assign supervised label to unlabeled sample
            label_ss = ref_labels[closest_idx]    

            # Check whether the sample is below the max margin set
            check = min_dist < self.config.negative_margin
            total_unlabelled += check.sum().item()

            # Supervised
            loss_supervised = self.config.train_criterion(labeled_pred,gt_oh)
            
            # SemiSupervised
            loss_semisupervised = torch.tensor(0.).to(self.config.device)
            if check.sum() > 0:
                masked_preds_ss = pred_ss[check]
                masked_labels_ss = label_ss[check]
                masked_labels_ss_oh = F.one_hot(masked_labels_ss, num_classes=self.config.num_classes).float()             
                loss_semisupervised = self.config.train_criterion(masked_preds_ss, masked_labels_ss_oh)
            
            # Total Loss
            # print(f"\r{step+1:03d}/{len(train_loader):03d}: loss (supervised) = {loss_supervised:.4f} (semisupervised) = {loss_semisupervised:.4f}", end="")
            loss = loss_supervised + self.config.lambda_semisupervised*loss_semisupervised 

            loss.backward()
            optimizer.step()
            train_ma_loss  = 0.9*train_ma_loss + 0.1*loss.item()
            
            # Supervised Evaluation
            _, pred_y_num = torch.max(labeled_pred, 1)
            # _,gt_num = torch.max(gt_oh, 1)
            acc_labelled = torch.sum(pred_y_num == gt).item()/self.config.batch_size
            batch_acc = torch.sum(pred_y_num == gt).item()/self.config.batch_size

            # Semi Supervised Evaluation
            acc_unlabelled = 0.0
            if check.sum() > 0:
                _, ss_pred_y_num = torch.max(masked_preds_ss, 1)
                acc_unlabelled = torch.sum(ss_pred_y_num == masked_labels_ss).item()/check.sum().item()
                batch_acc = (batch_acc + acc_unlabelled)/2

            train_ma_acc = 0.9*train_ma_acc + 0.1*batch_acc
                
            print(f"\r{step+1:03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f} - acc (train) = {train_ma_acc:.4f} (labelled) {acc_labelled:.3f} (unlabelled) {acc_unlabelled:.3f} Num Unlabelled Samples Used: {total_unlabelled}", end="")
            
        # Keep just final epoch values
        # self.logger.info(f"\r{step+1:03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f} - acc (train) = {train_ma_acc:.4f}")
        self.logger.info(f"{step+1:03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f} - acc (train) = {train_ma_acc:.4f} Num Unlabelled Samples Used: {total_unlabelled}")
        self.logger.info("\n")  # new oline

        if self.best_train_acc < train_ma_acc:
            self.best_train_acc = train_ma_acc

        return model, train_ma_acc, train_ma_loss

    def validate_fn(self,model, val_loader):
        val_ma_loss = 0.
        val_ma_acc = 0.
        self.logger.info("Validation starts...")
        model.eval()
        for step, (inp, gt) in enumerate(val_loader):
            # TODO: put data to the correct device
            inp, gt = inp.to(self.config.device), gt.to(self.config.device)
            gt_oh = F.one_hot(gt,num_classes=self.config.num_classes).float()
            
            pred = model(inp)
            
            loss = self.config.val_criterion(pred, gt_oh)
            
            val_ma_loss  = 0.9*val_ma_loss + 0.1*loss.item()

            _, pred_y_num = torch.max(pred, 1)
            _,gt_num = torch.max(gt_oh, 1)
            batch_acc = torch.sum(pred_y_num == gt_num).item()/self.config.batch_size
            val_ma_acc = 0.9*val_ma_acc + 0.1*batch_acc
            print(f"\r{step+1:03d}/{len(val_loader):03d}: loss = {val_ma_loss:.4f} - acc (val) = {val_ma_acc:.4f}", end="")

        # self.logger.info(f"\r{step+1:03d}/{len(val_loader):03d}: loss = {val_ma_loss:.4f} - acc (val) = {val_ma_acc:.4f}")
        self.logger.info(f"{step+1:03d}/{len(val_loader):03d}: loss = {val_ma_loss:.4f} - acc (val) = {val_ma_acc:.4f}")
        self.logger.info("\n")  # new line

        return val_ma_acc, val_ma_loss

    def show_training_curves(self,values_dict):
        self.logger.info("SHOW TRAINING CURVES UP AND RUNNING CONSTRUCTION!")

        plt.figure(figsize=(14,10))
        plt.subplot(1,2,1)
        plt.plot(values_dict["train_acc"],label="Train Acc.", color="blue")
        plt.plot(values_dict["val_acc"],label="Valid. Acc.", color="orange")
        plt.xlabel("Epochs")
        plt.title("Accuracy")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1)
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(values_dict["train_loss"],label="Train Loss", color="red")
        plt.plot(values_dict["val_loss"],label="Valid. Loss.", color="green")
        plt.xlabel("Epochs")
        plt.title("Loss")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(osp.join(self.model_output_path,"training_metrics_plot.png"))


    #####
    # Auxiliary
    #####

    def get_distance_embeddings_dataloader(self, loader, model):
        
        embeddings = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (x,y) in enumerate(loader):
                x, y = x.to(self.config.device), y.to(self.config.device)

                preds = model(x)
                embedds = F.normalize(preds, p=2, dim=1)
                embeddings.append(embedds)
                labels.append(y)

        model.train()

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        return embeddings, labels
    
    def get_distance_embeddings_batch(self, batch, model):

        model.eval()
        with torch.no_grad():
            preds = model(batch.to(self.config.device))
            embedds = F.normalize(preds, p=2, dim=1)

        model.train()
        return embedds


###############
# EX4
###############


class myTrainerClusteringSS:

    def __init__(self, config, logger):
        self.config = config
        # self.logger = logger
        self.logger = myLogger(logger_name=__name__, config=self.config).get_logger()
        self.ss_iter = None

    def train_model(self,model,
                    train_loader,
                    val_loader,
                    show_training_curves=True,
                    semi_supervised_loader=None,
                    return_acc=False):

        train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []
        self.best_train_acc = 0.
        self.best_val_acc = 0.

        optimizer = optim.SGD(params=model.parameters(), lr=self.config.lr,weight_decay=self.config.weight_decay)

        model = model.to(self.config.device)
        # specific_experiment = "K_CROSS_VAL_EX1"
        self.experiment_path = f"{self.config.model_name}_{self.config.specific_experiment}_{self.config.specific_experiment_val}"
        if not self.config.semisupervised_setting:
            if self.config.use_subsample is not None:
                self.experiment_path = f"{self.experiment_path}_{str(self.config.use_subsample).split('.')[0]}_{str(self.config.use_subsample).split('.')[1]}"
                self.model_output_path = osp.join(self.config.output_model_path, self.experiment_path)
            else:
                self.model_output_path = osp.join(self.config.output_model_path, f"{self.experiment_path}") 
                
        else:
            if self.config.perform_online_mixmatch:
                self.experiment_path = f"{self.experiment_path}_ONLINE"
            self.model_output_path = osp.join(self.config.output_model_path, f"{self.experiment_path}_SS_0_{str(self.config.perc_semisupervised).split('.')[1]}")
        os.makedirs(self.model_output_path, exist_ok=True)


        for epoch in range(1, self.config.epochs + 1):
            
            self.epoch = epoch
            self.logger.info(f'Epoch {epoch:02d}/{self.config.epochs:02d}')
            self.logger.info('-' * 10)

            self.logger.info(f"Obtaining reference embeddings...")
            ref_embeddings, ref_labels = self.get_distance_embeddings_dataloader(train_loader, model)
            ss_embeddings, ss_labels = self.get_distance_embeddings_dataloader(semi_supervised_loader, model)
            clustering_ss_labels = self.perform_clustering_labeling(ref_embeddings.to(self.config.device),
                                                                    ss_embeddings.to(self.config.device),
                                                                    ref_labels.to(self.config.device))
            semi_supervised_loader.dataset.update_labels(clustering_ss_labels)
            self.logger.info(f"Reference embeddings have been extracted...")
            
            model, train_acc, train_loss = self.train_fn_clustering_ss(model=model,
                                                                     train_loader=train_loader,
                                                                     optimizer=optimizer,
                                                                     semi_supervised_loader=semi_supervised_loader)
        
            val_acc, val_loss = self.validate_fn(model=model, val_loader=val_loader)

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            
            if self.best_val_acc < val_acc or epoch == self.config.epochs:
                
                torch.save(model.state_dict(), osp.join(self.model_output_path, f"model_ep{epoch}.pt"))
                self.logger.info(f"Saving model with better perfs ({self.best_val_acc} -> {val_acc}")
                self.best_val_acc = val_acc

        if show_training_curves:
            self.show_training_curves({"train_acc":train_acc_list,
                                       "val_acc":val_acc_list,
                                       "train_loss":train_loss_list,
                                       "val_loss":val_loss_list})
        del self.logger

        if return_acc:
            return model, train_acc_list, val_acc_list
        else:
            return model


    def train_fn_clustering_ss(self, model, train_loader, optimizer, semi_supervised_loader=None):

        model.train()
        lab_train_ma_loss = 0.
        lab_train_ma_acc = 0.

        unlab_train_ma_loss = 0.
        unlab_train_ma_acc = 0.
 
        self.logger.info("Training starts...")
        
        if len(semi_supervised_loader) > len(train_loader):
            labeled_iter = iter(train_loader)
            for step, (inp_unlab, gt_unlab) in enumerate(semi_supervised_loader):

                try:
                    inp_lab, gt_lab = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(train_loader)
                    inp_lab, gt_lab = next(labeled_iter)

                inp_lab, gt_lab = inp_lab.to(self.config.device), gt_lab.to(self.config.device)
                inp_unlab, gt_unlab = inp_unlab.to(self.config.device), gt_unlab.to(self.config.device)

                optimizer.zero_grad()

                pred_lab = model(inp_lab)
                loss_lab = self.config.train_criterion(pred_lab, gt_lab)

                pred_unlab = model(inp_unlab)
                loss_unlab = self.config.train_criterion(pred_unlab, gt_unlab)

                loss = loss_lab + self.config.lambda_semisupervised*loss_unlab
                loss.backward()
                optimizer.step()

                # Labeled Accuracy
                _, pred_lab_idx = torch.max(pred_lab, 1)
                acc_lab = torch.sum(pred_lab_idx == gt_lab).item() / inp_lab.size(0)
                lab_train_ma_acc = 0.9 * lab_train_ma_acc + 0.1 * acc_lab

                # Unlabeled Accuracy (vs Pseudo Labels)
                _, pred_unlab_idx = torch.max(pred_unlab, 1)
                acc_unlab = torch.sum(pred_unlab_idx == gt_unlab).item() / inp_unlab.size(0)
                unlab_train_ma_acc = 0.9 * unlab_train_ma_acc + 0.1 * acc_unlab

                print(f"\r {step+1:03d}/{len(semi_supervised_loader):03d}: loss = {loss:.4f} | Labeled Acc (train) = {lab_train_ma_acc:.4f} | Unlabeled Acc (train) = {unlab_train_ma_acc:.4f}", end="")
        
        else:
            unlabeled_iter = iter(semi_supervised_loader)
            for step, (inp_lab, gt_lab) in enumerate(train_loader):

                try:
                    inp_unlab, gt_unlab = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(semi_supervised_loader)
                    inp_unlab, gt_unlab = next(unlabeled_iter)

                inp_lab, gt_lab = inp_lab.to(self.config.device), gt_lab.to(self.config.device)
                inp_unlab, gt_unlab = inp_unlab.to(self.config.device), gt_unlab.to(self.config.device)

                optimizer.zero_grad()

                pred_lab = model(inp_lab)
                loss_lab = self.config.train_criterion(pred_lab, gt_lab)

                pred_unlab = model(inp_unlab)
                loss_unlab = self.config.train_criterion(pred_unlab, gt_unlab)

                loss = loss_lab + self.config.lambda_semisupervised*loss_unlab
                loss.backward()
                optimizer.step()

                # Labeled Accuracy
                _, pred_lab_idx = torch.max(pred_lab, 1)
                acc_lab = torch.sum(pred_lab_idx == gt_lab).item() / inp_lab.size(0)
                lab_train_ma_acc = 0.9 * lab_train_ma_acc + 0.1 * acc_lab

                # Unlabeled Accuracy (vs Pseudo Labels)
                _, pred_unlab_idx = torch.max(pred_unlab, 1)
                acc_unlab = torch.sum(pred_unlab_idx == gt_unlab).item() / inp_unlab.size(0)
                unlab_train_ma_acc = 0.9 * unlab_train_ma_acc + 0.1 * acc_unlab

                print(f"\r {step+1:03d}/{len(train_loader):03d}: loss = {loss:.4f} | Labeled Acc (train) = {lab_train_ma_acc:.4f} | Unlabeled Acc (train) = {unlab_train_ma_acc:.4f}", end="")
        
        # Keep just final epoch values
        # self.logger.info(f"\r{step+1:03d}/{len(train_loader):03d}: loss = {train_ma_loss:.4f} - acc (train) = {train_ma_acc:.4f}")
        overall_train_ma_acc = 0.5*(lab_train_ma_acc + unlab_train_ma_acc)
        overall_train_ma_loss = 0.5*(lab_train_ma_loss + unlab_train_ma_loss)

        self.logger.info(f"OVERALL Epoch {self.epoch}:\n LABELLED: loss = {lab_train_ma_loss:.4f} - acc (train) = {lab_train_ma_acc:.4f} \n UNLABELLED: loss = {unlab_train_ma_loss:.4f} - acc (train) = {unlab_train_ma_acc:.4f} \n MERGED: loss = {overall_train_ma_loss:.4f} - acc (train) = {overall_train_ma_acc:.4f} ")
        self.logger.info("\n")  # new oline

        if self.best_train_acc < overall_train_ma_acc:
            self.best_train_acc = overall_train_ma_acc

        return model, overall_train_ma_acc, overall_train_ma_loss

    def validate_fn(self,model, val_loader):
        val_ma_loss = 0.
        val_ma_acc = 0.
        self.logger.info("Validation starts...")
        model.eval()
        for step, (inp, gt) in enumerate(val_loader):
            # TODO: put data to the correct device
            inp, gt = inp.to(self.config.device), gt.to(self.config.device)
            gt_oh = F.one_hot(gt,num_classes=self.config.num_classes).float()
            
            pred = model(inp)
            
            loss = self.config.val_criterion(pred, gt_oh)
            
            val_ma_loss  = 0.9*val_ma_loss + 0.1*loss.item()

            _, pred_y_num = torch.max(pred, 1)
            _,gt_num = torch.max(gt_oh, 1)
            batch_acc = torch.sum(pred_y_num == gt_num).item()/self.config.batch_size
            val_ma_acc = 0.9*val_ma_acc + 0.1*batch_acc
            print(f"\r{step+1:03d}/{len(val_loader):03d}: loss = {val_ma_loss:.4f} - acc (val) = {val_ma_acc:.4f}", end="")

        # self.logger.info(f"\r{step+1:03d}/{len(val_loader):03d}: loss = {val_ma_loss:.4f} - acc (val) = {val_ma_acc:.4f}")
        self.logger.info(f"{step+1:03d}/{len(val_loader):03d}: loss = {val_ma_loss:.4f} - acc (val) = {val_ma_acc:.4f}")
        self.logger.info("\n")  # new line

        return val_ma_acc, val_ma_loss

    def show_training_curves(self,values_dict):
        self.logger.info("SHOW TRAINING CURVES UP AND RUNNING CONSTRUCTION!")

        plt.figure(figsize=(14,10))
        plt.subplot(1,2,1)
        plt.plot(values_dict["train_acc"],label="Train Acc.", color="blue")
        plt.plot(values_dict["val_acc"],label="Valid. Acc.", color="orange")
        plt.xlabel("Epochs")
        plt.title("Accuracy")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1)
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(values_dict["train_loss"],label="Train Loss", color="red")
        plt.plot(values_dict["val_loss"],label="Valid. Loss.", color="green")
        plt.xlabel("Epochs")
        plt.title("Loss")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(osp.join(self.model_output_path,"training_metrics_plot.png"))


    #####
    # Auxiliary
    #####

    def get_distance_embeddings_dataloader(self, loader, model):
        
        embeddings = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (x,y) in enumerate(loader):
                x, y = x.to(self.config.device), y.to(self.config.device)

                preds = model(x)
                embedds = F.normalize(preds, p=2, dim=1)
                embeddings.append(embedds)
                labels.append(y)

        model.train()

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        return embeddings, labels

    
    def perform_clustering_labeling(self, sup_embeddings, unsup_embeddings, sup_labels):

        dists = torch.cdist(unsup_embeddings, sup_embeddings, p=2)
                
        # Find closest neighbor (1-NN)
        # min_dists, indices = torch.min(dists, dim=1)
        min_dists, indices = torch.topk(input=dists,k=self.config.K_knn, dim=1, largest=False)
        predicted_classes = sup_labels[indices]
        
        # Most represented class
        clustering_labels, _ = torch.mode(predicted_classes, dim=1)

        return clustering_labels

