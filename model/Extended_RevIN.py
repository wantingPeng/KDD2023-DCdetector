import torch
import torch.nn as nn
import torch.nn.functional as F


class Extended_RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, n_moments=4):
        """
        Extended_RevIN: Extended RevIN normalization
        An extension of RevIN that handles higher-order moments of the distribution
        
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, Extended_RevIN has learnable affine parameters
        :param n_moments: number of moments to normalize (2: mean+var, 3: +skewness, 4: +kurtosis)
        """
        super(Extended_RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.n_moments = min(n_moments, 4)  # Support up to 4 moments
        
        if self.affine:
            self._init_params()
    
    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x
    
    def _init_params(self):
        # Initialize Extended_RevIN affine parameters for each feature
        self.affine_weight = torch.ones(self.num_features)
        self.affine_bias = torch.zeros(self.num_features)
        
        # Additional parameters for higher moments if needed
        if self.n_moments >= 3:
            self.affine_skew = torch.zeros(self.num_features)
        if self.n_moments >= 4:
            self.affine_kurt = torch.zeros(self.num_features)
            
        # Move parameters to appropriate device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.affine_weight = self.affine_weight.to(device=device)
        self.affine_bias = self.affine_bias.to(device=device)
        
        if self.n_moments >= 3:
            self.affine_skew = self.affine_skew.to(device=device)
        if self.n_moments >= 4:
            self.affine_kurt = self.affine_kurt.to(device=device)
    
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        
        # First and second moments (mean and variance)
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.var = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False).detach()
        self.stdev = torch.sqrt(self.var + self.eps).detach()
        
        # Higher order moments if enabled
        if self.n_moments >= 3 or self.n_moments >= 4:
            # Compute centered data for higher moments
            centered = x - self.mean
            normalized = centered / self.stdev
            
            if self.n_moments >= 3:
                # Third moment (skewness)
                self.skew = torch.mean(normalized**3, dim=dim2reduce, keepdim=True).detach()
                
            if self.n_moments >= 4:
                # Fourth moment (kurtosis)
                self.kurt = torch.mean(normalized**4, dim=dim2reduce, keepdim=True).detach() - 3.0  # Excess kurtosis
    
    def _normalize(self, x):
        # First normalize mean and variance (like RevIN)
        centered = x - self.mean
        standardized = centered / self.stdev
        
        # Apply higher-order moment normalization if enabled
        if self.n_moments >= 3:
            # Skewness correction via Box-Cox-like transformation
            sign = torch.sign(standardized)
            abs_val = torch.abs(standardized)
            power = 1/3 if self.n_moments == 3 else 1/2
            standardized = sign * torch.pow(abs_val + self.eps, power)
            
        if self.n_moments >= 4:
            # Kurtosis correction via adaptive scaling
            # Scale the extreme values to reduce kurtosis
            extreme_vals = (torch.abs(standardized) > 1.5)
            if extreme_vals.any():
                scale_factor = torch.ones_like(standardized)
                scale_factor[extreme_vals] = torch.sqrt(torch.abs(standardized[extreme_vals]))
                standardized = standardized / scale_factor
        
        # Apply affine transformation if enabled
        if self.affine:
            standardized = standardized * self.affine_weight + self.affine_bias
            
            # Apply additional transformations for higher moments
            if self.n_moments >= 3:
                standardized = standardized + self.affine_skew * (standardized**2) * torch.sign(standardized)
            if self.n_moments >= 4:
                standardized = standardized + self.affine_kurt * (standardized**3)
                
        return standardized
    
    def _denormalize(self, x):
        # Reverse the affine transformations if enabled
        if self.affine:
            # First reverse the higher-order moment transformations
            if self.n_moments >= 4:
                # Approximate inverse of the kurtosis transformation
                x = x - self.affine_kurt * (x**3)
                
            if self.n_moments >= 3:
                # Approximate inverse of the skewness transformation
                x = x - self.affine_skew * (x**2) * torch.sign(x)
                
            # Reverse the basic affine transformation
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        
        # Reverse the higher-order normalizations
        if self.n_moments >= 4:
            # Reverse kurtosis correction (approximate)
            extreme_vals = (torch.abs(x) > 1.0)
            if extreme_vals.any():
                scale_factor = torch.ones_like(x)
                scale_factor[extreme_vals] = torch.sqrt(torch.abs(x[extreme_vals]))
                x = x * scale_factor
                
        if self.n_moments >= 3:
            # Reverse skewness correction
            sign = torch.sign(x)
            abs_val = torch.abs(x)
            power = 3 if self.n_moments == 3 else 2
            x = sign * torch.pow(abs_val, power)
            
        # Reverse the standardization
        x = x * self.stdev + self.mean
        
        return x
