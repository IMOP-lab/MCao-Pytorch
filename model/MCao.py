import torch
import torch.nn as nn
import torch.nn.functional as F

from RieFNO import RieFNO
from kan_convs.wav_kan import WavKANConv1DLayer

class MCaoNet(nn.Module):
    def __init__(self, configs):
        super(MCaoNet, self).__init__()

        # Define the individual branches for different lead groups in ECG analysis
        self.lead_branches = nn.ModuleList([
            self._make_cnn_o_branch(20),  # LMCA branch: V1, V2, V3, V4, V5, V6, I, aVL
            self._make_cnn_o_branch(16),  # LAD branch: V1, V2, V3, V4
            self._make_cnn_o_branch(16),  # LCX branch: I, aVL, V5, V6
            self._make_cnn_o_branch(17),  # RCA branch: II, III, aVF, V1, V2
            self._make_wavelet_branch(12)  # Wavelet branch for all 12 leads
        ])
        
        # Global Average Pooling, Dropout, Flatten, and Fully Connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(192, configs.num_classes)  # 192 features from the branches combined (4 branches * 32 + 1 wavelet branch * 32)

        # RieFNO module for frequency domain processing
        self.riefno = RieFNO(hidden_size=320)
        # wKAN module for attention mechanism
        self.wKAN = wKAN(32)
        
    # Create CNN-based branch with the wKAN attention mechanism
    def _make_cnn_o_branch(self, in_channels):
        layers = []

        # Apply wKAN attention
        layers.append(wKAN(in_channels))

        # First convolution block with batch normalization and max pooling
        layers.append(nn.Conv1d(in_channels, out_channels=32, kernel_size=50, stride=2))
        layers.append(nn.BatchNorm1d(32))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Apply wKAN attention again
        layers.append(wKAN(32))

        # Second convolution block
        layers.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1))
        layers.append(nn.BatchNorm1d(32))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        
        # Dense Block 1 for feature extraction
        layers.append(DenseBlock(32, num_layers=2, growth_rate=16))
        layers.append(nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1, stride=1))  # 1x1 convolution for feature compression
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Dense Block 2 for deeper feature extraction
        layers.append(DenseBlock(32, num_layers=7, growth_rate=16))
        layers.append(nn.Conv1d(in_channels=144, out_channels=16, kernel_size=1, stride=1))  # 1x1 convolution for feature compression
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        
        # Dense Block 3 for final feature extraction
        layers.append(DenseBlock(16, num_layers=2, growth_rate=8))
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)

    # Create the wavelet-based branch using wavelet residual blocks
    def _make_wavelet_branch(self, in_channels):
        layers = []
        layers.append(RieFNO(hidden_size=12))  # RieFNO module for frequency-domain processing
        layers.append(self._make_wavelet_block(in_channels, 16))  # First wavelet block
        layers.append(self._make_wavelet_block(16, 32))  # Second wavelet block
        layers.append(self._make_wavelet_block(32, 64))  # Third wavelet block
        return nn.Sequential(*layers)
    
    # Helper function to create wavelet blocks with residual connections
    def _make_wavelet_block(self, in_channels, out_channels):
        return WaveletResidualBlock1D(in_channels, out_channels)

    # Define the forward pass of the model
    def forward(self, x):
        lead_indices_list = [
            [0, 4, 6, 7, 8, 9, 10, 11],  # LMCA branch: V1, V2, V3, V4, V5, V6, I, aVL
            [6, 7, 8, 9],  # LAD branch: V1, V2, V3, V4
            [0, 4, 10, 11],  # LCX branch: I, aVL, V5, V6
            [1, 2, 5, 6, 7],  # RCA branch: II, III, aVF, V1, V2
            list(range(12))  # Wavelet branch for all 12 leads
        ]

        # Specify which branches will have concatenated inputs with wavelet_riefno_output
        concat_branches = [0, 1, 2, 3]  # Only concatenate for specific branches

        # Process the wavelet branch first to extract RieFNO output
        wavelet_riefno_output = self.lead_branches[4][0](x[:, lead_indices_list[4], :])

        # Process each lead branch, concatenating the RieFNO output where applicable
        branch_outputs = []
        for i, (indices, branch) in enumerate(zip(lead_indices_list, self.lead_branches)):
            if i in concat_branches:  # Concatenate wavelet output for specified branches
                branch_input = torch.cat([x[:, indices, :], wavelet_riefno_output], dim=1)
                branch_output = self.global_avg_pool(branch(branch_input))
            else:
                branch_output = self.global_avg_pool(branch(x[:, indices, :]))
            branch_outputs.append(branch_output)

        # Concatenate outputs from all branches and process them through the final layers
        x = torch.cat(branch_outputs, dim=1)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

# Dense block definition, used for hierarchical feature extraction
class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = self._make_layer(in_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)
    
    # Define a single convolutional layer in the dense block
    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.Conv1d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(growth_rate),
            nn.ReLU()
        )
        return layer

    def forward(self, x):
        # Pass the input through each layer, concatenating outputs along the channel dimension
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x

# Residual block with 1D convolutions and skip connections
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=17, padding=8)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=11, padding=5)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # Ensure the skip connection matches the input/output dimensions
        self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.skip_bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.AvgPool1d(kernel_size=5)

    def forward(self, x):
        # Apply the skip connection
        residual = self.skip_bn(self.skip_conv(x))
        
        # Pass the input through the convolutions and add the residual
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += residual
        out = self.relu(out)
        out = self.pool(out)
        return out

# Channel attention module to highlight important channels
class ChannelAttention1D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        # Global average pooling and max pooling followed by two fully connected layers
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sigmoid(out)

# Spatial attention module to focus on relevant parts of the input signal
class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        self.conv1 = WavKANConv1DLayer(2, 1, kernel_size, padding=(kernel_size - 1) // 2, wavelet_type='bior13')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average and max pooling along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate the pooled outputs and apply a wavelet-based convolution
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# wKAN module, integrating both channel and spatial attention mechanisms
class wKAN(nn.Module):
    def __init__(self, in_planes, ratio=1, kernel_size=5):
        super(wKAN, self).__init__()
        self.ca = ChannelAttention1D(in_planes, ratio)
        self.sa = SpatialAttention1D(kernel_size)

    def forward(self, x):
        # Apply channel attention and then spatial attention
        x_out = x * self.ca(x)
        x_out = x_out * self.sa(x_out)
        return x_out

# Wavelet residual block for extracting multi-scale wavelet features
class WaveletResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WaveletResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=17, padding=8)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=11, padding=5)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.skip_bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.AvgPool1d(kernel_size=5)

        # wKAN attention mechanism
        self.wKAN = wKAN(out_channels)

    def forward(self, x):
        # Apply the skip connection
        residual = self.skip_bn(self.skip_conv(x))
        
        # Apply the convolutions and add the residual
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += residual
        out = self.relu(out)
        
        # Apply wKAN attention
        out = self.wKAN(out)
        out = self.pool(out)
        return out
