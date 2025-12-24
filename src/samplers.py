import numpy as np
import torch
from torch.utils.data import Sampler, Subset
from collections import defaultdict

# class DistanceBatchSampler(Sampler):
#     def __init__(self, dataset, n_classes, n_samples):
#         """
#         Args:
#             dataset: The training dataset (can be a Subset).
#             n_classes (int): How many distinct classes per batch (P).
#             n_samples (int): How many samples per class (K).
            
#         Batch Size will be n_classes * n_samples.
#         """
#         self.dataset = dataset
#         self.n_classes = n_classes
#         self.n_samples = n_samples
#         self.batch_size = n_classes * n_samples
        
#         # 1. Extract Labels (Optimized for Subsets)
#         if isinstance(dataset, Subset):
#         # Access the original dataset's targets using the subset's indices
#         # check if original dataset has targets (CIFAR/MNIST do, ImageFolder uses .targets too)
#             orig_dataset = dataset.dataset
#             if hasattr(orig_dataset, "targets"):
#                 _targets = [orig_dataset.targets[i] for i in dataset.indices]
#             else:
#                 # Fallback (slow) only if absolutely necessary
#                 _targets = [dataset[i][1] for i in range(len(dataset))]
#         else:
#             if hasattr(dataset, "targets"):
#                 _targets = dataset.targets
#             else:
#                 _targets = [dataset[i][1] for i in range(len(dataset))]
            
#         # 2. Build the Class Map: {class_label: [list_of_indices]}
#         self.labels_to_indices = defaultdict(list)
#         for idx, label in enumerate(_targets):
#             self.labels_to_indices[int(label)].append(idx)
            
#         self.classes = list(self.labels_to_indices.keys())
#         self.length_of_single_pass = len(dataset) // self.batch_size

#     def __iter__(self):
#         # At the start of every epoch, we shuffle the order of classes
#         # and shuffle the indices within each class
#         curr_labels_to_indices = {k: np.random.permutation(v).tolist() for k, v in self.labels_to_indices.items()}
        
#         # We perform this loop enough times to cover the full dataset size
#         for _ in range(self.length_of_single_pass):
            
#             # 1. Randomly pick P classes (n_classes)
#             # We ensure we don't crash if we ask for more classes than exist
#             selected_classes = np.random.choice(self.classes, self.n_classes, replace=False)
            
#             batch_indices = []
            
#             for c in selected_classes:
#                 # 2. For each class, pick K samples (n_samples)
#                 indices_of_class = curr_labels_to_indices[c]
                
#                 # If we run out of samples for this class, refill it (or you could skip)
#                 if len(indices_of_class) < self.n_samples:
#                      # Reshuffle and refill to ensure we can always pick n_samples
#                     indices_of_class = np.random.permutation(self.labels_to_indices[c]).tolist()
#                     curr_labels_to_indices[c] = indices_of_class
                
#                 # Pop the indices so they aren't repeated immediately
#                 for _ in range(self.n_samples):
#                     batch_indices.append(indices_of_class.pop())
            
#             yield batch_indices

#     def __len__(self):
#         return self.length_of_single_pass

class DistanceBatchSampler(Sampler):
    def __init__(self, dataset, n_classes, n_samples):
        """
        Args:
            dataset: The training dataset (can be a Subset).
            n_classes (int): How many distinct classes per batch (P).
            n_samples (int): How many samples per class (K).
            
        Batch Size will be n_classes * n_samples.
        """
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        
        self.classes = dataset.classes
        self.iterations = len(dataset) //(self.n_samples*n_classes)
        self.class_indices = dataset.class_to_indices

    def __iter__(self):

        for _ in range(self.iterations):
            # 1. Randomly select M classes
            selected_classes = np.random.choice(self.classes, self.n_classes, replace=False)
            batch = []
            
            # 2. For each class, randomly sample K + Q data points
            for c in selected_classes:
                indices = self.class_indices[c]
                # Sample K+Q indices for this class
                sampled_indices = np.random.choice(indices, self.n_samples, replace=False)
                batch.append(torch.from_numpy(sampled_indices))
            
            # Stack and yield (M * (K+Q)) indices
            yield torch.stack(batch).reshape(-1)

    def __len__(self):
        return self.iterations
    


class ProtoBatchSampler(Sampler):

    def __init__(self, dataset, n_classes, n_samples, config):
        """
        Args:
            dataset: dataset
            n_classes: M (number of classes per episode)
            n_samples: K + Q (support samples + query samples per class)
            iterations: Number of episodes per epoch
        """
        
        self.classes = dataset.classes

        self.n_classes = n_classes
        self.n_samples = n_samples
        self.iterations = len(dataset) //(self.n_samples*n_classes)

        self.class_indices = dataset.class_to_indices

        

    def __iter__(self):
        for _ in range(self.iterations):
            # 1. Randomly select M classes
            selected_classes = np.random.choice(self.classes, self.n_classes, replace=False)
            batch = []
            
            # 2. For each class, randomly sample K + Q data points
            for c in selected_classes:
                indices = self.class_indices[c]
                # Sample K+Q indices for this class
                sampled_indices = np.random.choice(indices, self.n_samples, replace=False)
                batch.append(torch.from_numpy(sampled_indices))
            
            # Stack and yield (M * (K+Q)) indices
            yield torch.stack(batch).reshape(-1)

    def __len__(self):
        return self.iterations