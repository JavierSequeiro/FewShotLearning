import torch
import torch.nn as nn
import torch.nn.functional as F

class myEuclideanContrastiveLoss:
    
    def __init__(self, neg_margin) -> None:
        self.negative_margin = neg_margin

    def __call__(self, embeddings:torch.Tensor, labels:torch.Tensor, **kwargs):

        # Euclidean Distance between embeddings
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        upper_triang_mat = torch.triu(torch.ones_like(dist_matrix, dtype=torch.bool), diagonal=1)

        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T) & upper_triang_mat    
        mask_neg = torch.ne(labels, labels.T) & upper_triang_mat

        # Loss for positive pairs (similar images)
        loss_pos = torch.pow(dist_matrix, 2)
        loss_pos = loss_pos[mask_pos].sum()

        # Loss for negative pairs (different images)
        loss_neg = torch.pow(torch.clamp(self.negative_margin - dist_matrix, min=0.0),2)
        loss_neg = loss_neg[mask_neg].sum()
        # active_negatives = torch.count_nonzero(loss_neg)
        num_pos_samples = torch.sum(mask_pos).float()
        num_neg_samples = torch.sum(mask_neg).float()
        
        total_loss = (loss_pos/(num_pos_samples + 1e-8) + loss_neg/(num_neg_samples + 1e-8))
        
        return total_loss
    

class CosineSimilarityContrastiveLossV2:
    """
    Variant of the contrastive loss using cosine similarity instead of the Euclidean distance, and two margin for
    matching and mismatching samples. Note that this loss requires that the embedding space is L2-normalised. Also,
    the cosine similarity is not a distance, so instead of minimizing it, we have to maximize it for matching samples.

    If vectors have a norm of 1, then the cosine similarity can only vary from -1 to +1. We must have
    `pos_margin` > `neg_margin`. Practically, a positive margin above 0.6 and a negative margin below 0.4 are usual
    values. Not that we don't want the margin to be negative (albeit the cosine similarity might), because it would make
    the embedding space impossible de learn.
    """
    def __init__(self, pos_margin=0.8, neg_margin=0.3):
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin

    def __call__(self, x: torch.Tensor, labels: torch.Tensor, **kwargs):
        x = torch.nn.functional.normalize(x, dim=1)
        sim = x @ x.T

        if labels.ndim == 1:
            labels = labels[:, None]

        pos_mask = labels == labels.T
        neg_mask = labels != labels.T

        pos_loss = torch.sum(pos_mask * torch.relu(self.pos_margin - sim)) / torch.sum(pos_mask)
        neg_loss = torch.sum(torch.relu(sim - self.neg_margin) * neg_mask) / torch.sum(neg_mask)

        return pos_loss + neg_loss
    

class myCrossEntropyLoss:

    def __init__(self) -> None:
        self.criterion = nn.CrossEntropyLoss()
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs):
        return self.criterion(preds, targets) # Loss, apply loss.backward() in training step
    

class PrototypeCELoss:

    def __init__(self, M:int, Q:int):
        
        self.M = M
        self.Q = Q
        
    def __call__(self, embeddings: torch.Tensor, centroid_proto_embeddings:torch.Tensor, **kwargs):
        
        dist = torch.cdist(embeddings, centroid_proto_embeddings)**2
        neg_dist = -dist

        # Indices assuming samples from each class are clustered together in the tensor
        targets = torch.arange(self.M).repeat_interleave(self.Q).to(embeddings.device)

        loss = F.cross_entropy(neg_dist, targets)

        preds = neg_dist.argmax(dim=1)
        acc = (preds == targets).float().mean()

        return loss, acc
