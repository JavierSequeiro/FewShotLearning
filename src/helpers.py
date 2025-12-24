import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from collections import defaultdict
# Visualize Data Samples (Save them for visualization)

def _visualize_samples(train_data, test_data, config, logger, num_samples=2, folder_name="sample_imgs"):
    
        # Create the output folder if it doesn't exist
        os.makedirs(folder_name, exist_ok=True)
        
        # NOTE: These values must match the ones used for normalization
        mean = torch.tensor(config.norm_mean).view(3, 1, 1)
        std = torch.tensor(config.norm_std).view(3, 1, 1)

        def save_sample_images(dataset, set_name):
            for i in range(num_samples):
                # 1. Get the tensor (image, label) from the dataset
                # We don't need to manually seed here if we are just taking the first N elements.
                # If you wanted random samples, you would use torch.randperm.
                image_tensor, _ = dataset[i] 
                
                # 2. Reverse Normalization (De-normalize)
                # D = T * S + M  (where D=De-normalized, T=Tensor, S=Std, M=Mean)
                de_normalized_tensor = image_tensor * std + mean
                
                # 3. Convert to a PIL Image
                # (H, W, C) for PIL and (C, H, W) for PyTorch Tensor, so permute
                image_pil = transforms.ToPILImage()(de_normalized_tensor)

                # 4. Save the image
                filename = os.path.join(folder_name, f"{set_name}_sample_{i+1}.png")
                image_pil.save(filename)
                logger.info(f"Saved sample image to: {filename}")

        # Process Train and Test sets
        logger.info("\nSaving train set samples...")
        save_sample_images(train_data, "train")
        
        logger.info("\nSaving test set samples...")
        save_sample_images(test_data, "test")

        logger.info(f"\nSuccessfully saved {num_samples*2} samples to the '{folder_name}' folder.")


def _get_FSL_split(train_dataset, test_dataset, config, logger):
        all_train_indices = list(range(len(train_dataset)))
        all_test_indices = list(range(len(test_dataset)))

        try:
            all_train_labels = train_dataset.targets
            all_test_labels = test_dataset.targets
        except AttributeError:
            # Less efficient way, but works
            logger.info("Dataset does not contain .targets attribute. Will extract labels manually.")
            all_train_labels = [train_dataset[i][1] for i in all_train_indices]
            all_test_labels = [test_dataset[i][1] for i in all_test_indices]

        train_lim = config.num_classes_train - 1
        val_lim = config.num_classes_train + config.num_classes_val - 1
        # test_lim = config.num_classes_train + config.num_classes_val + config.num_classes_test - 1

        train_idx, val_idx, test_idx = [], [], []
        for i, lab in enumerate(all_train_labels):
            if lab <= train_lim:
                train_idx.append(all_train_indices[i])
            elif (lab >= train_lim) and (lab <= val_lim):
                val_idx.append(all_train_indices[i])

        for i, lab in enumerate(all_test_labels):
            if lab > val_lim:
                test_idx.append(all_test_indices[i])


        train_set = Subset(train_dataset, train_idx)
        val_set = Subset(train_dataset, val_idx)
        test_set = Subset(test_dataset, test_idx)

        return train_set, val_set, test_set


def _get_stratified_split(torch_dataset, set2_perc, config, logger):
        all_indices = list(range(len(torch_dataset)))

        try:
            all_labels = torch_dataset.targets
        except AttributeError:
            # Less efficient way, but works
            logger.info("Dataset does not contain .targets attribute. Will extract labels manually.")
            all_labels = [torch_dataset[i][1] for i in all_indices]

        strat_shuf_split = StratifiedShuffleSplit(n_splits=1, test_size=set2_perc, random_state=config.rand_seed)

        set1_indices, set2_indices = None, None
        for set1_idx, set2_idx in strat_shuf_split.split(np.array(all_indices), np.array(all_labels)):
            set1_indices = set1_idx
            set2_indices = set2_idx
        if (set1_indices != None) and (set2_indices != None):
            set1_dataset = Subset(torch_dataset, set1_indices)
            set2_dataset = Subset(torch_dataset, set2_indices)

            return set1_dataset, set2_dataset
        else:
             raise ValueError("SEQUE Unable to iterate through indices.")


def get_CI_stats(acc_list, config):

        val_acc_np = np.array(acc_list)

        # mean_acc = np.mean(val_acc_np)
        std_acc = np.std(val_acc_np, ddof=1)
        std_err = config.conf_interval_coeff*std_acc/np.sqrt(np.size(val_acc_np))
        # lower_bound = mean_acc - std_err
        # upper_bound = mean_acc + std_err

        return std_err


def get_support_points(dataset, config):

        k_per_class = config.K_knn
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

        # rng = np.random.RandomState(14)

        for label, idx_list in class_map.items():
            # Check if we have enough samples
            if len(idx_list) < k_per_class:
                raise ValueError(f"Class {label} has {len(idx_list)} samples, but you requested K={k_per_class}.")
            
            # Shuffle indices for this class to ensure random selection
            # rng.shuffle(idx_list)
            np.random.shuffle(x=idx_list)
            
            # Pick K for support, rest for validation
            support_indices.extend(idx_list[:k_per_class])
            val_remainder_indices.extend(idx_list[k_per_class:])

        # 4. Create the final objects
        # Create the 'clean' validation subset (everything NOT in support)
        clean_val_set = Subset(dataset, 
                            val_remainder_indices)

        # Create a temporary subset for support to extract tensors
        # support_subset = Subset(dataset.dataset if isinstance(dataset, Subset) else dataset, 
        #                         support_indices)
        
        support_subset = Subset(dataset, 
                                support_indices)
        
        # Load support data into tensors
        # batch_size = len(support_indices) ensures we get it all in one go
        temp_loader = DataLoader(support_subset, batch_size=len(support_indices), shuffle=False)
        support_images, support_labels = next(iter(temp_loader))

        return support_images, support_labels, clean_val_set