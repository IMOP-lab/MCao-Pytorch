import torch  
import torch.nn as nn  
import torch.nn.functional as F  


class AMPLoss(nn.Module):
    def __init__(self, alpha_t=1, gamma=2, lambda_fn=1, lambda_fp=1, reduction='mean'):
        """
        Initialize the AMPLoss function, an adaptive misclassification penalty loss.
        Parameters:
        - alpha_t: Balancing factor to address class imbalance.
        - gamma: Focusing parameter to down-weight easy examples.
        - lambda_fn: Penalty weight for false negatives.
        - lambda_fp: Penalty weight for false positives.
        - reduction: Specifies the reduction to apply to the output ('mean', 'sum', or 'none').
        """
        super(AMPLoss, self).__init__()
        self.alpha_t = alpha_t
        self.gamma = gamma
        self.lambda_fn = lambda_fn
        self.lambda_fp = lambda_fp
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the AMPLoss function.
        Parameters:
        - inputs: Predicted logits from the model (before applying sigmoid).
        - targets: Ground truth binary labels (0 or 1).
        
        Returns:
        - Weighted loss based on the false negative and false positive penalties.
        """
        # Compute binary cross-entropy loss with logits, i.e., without applying sigmoid
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute the probability for the focal adjustment, preventing NaN errors with exp(-BCE_loss)
        pt = torch.exp(-BCE_loss)
        
        # Apply focal adjustment to focus more on difficult examples (misclassified ones)
        F_loss = self.alpha_t * (1 - pt) ** self.gamma * BCE_loss
        
        # Compute the loss contributions for false negatives and false positives
        false_negatives = targets * F_loss  # Loss where the target is 1 (FN)
        false_positives = (1 - targets) * F_loss  # Loss where the target is 0 (FP)
        
        # Apply the custom penalties for false negatives and false positives
        weighted_loss = self.lambda_fn * false_negatives + self.lambda_fp * false_positives

        # Reduce the loss according to the specified reduction mode (mean, sum, or none)
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss
