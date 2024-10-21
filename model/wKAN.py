import torch
import torch.nn as nn
import math

class WaveletLayerND(nn.Module):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, dimensions=2, wavelet_choice='mexican_hat'):
        super(WaveletLayerND, self).__init__()

        self.scale_param = nn.Parameter(torch.ones(1, out_channels, in_channels, *([1] * dimensions)))
        self.translation_param = nn.Parameter(torch.zeros(1, out_channels, in_channels, *([1] * dimensions)))

        self.dimensions = dimensions
        self.wavelet_choice = wavelet_choice

        self.wavelet_convolutions = nn.ModuleList([conv_module(in_channels, 1, kernel_size, stride, padding, dilation, bias=False) for _ in range(out_channels)])
        self.final_conv = conv_module(out_channels, out_channels, 1, 1, 0, dilation, bias=False)

        for conv in self.wavelet_convolutions:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.final_conv.weight, nonlinearity='linear')

    @staticmethod
    def mexican_hat_wavelet(x):
        return (2 / (math.sqrt(3) * math.pi**0.25)) * ((x**2 - 1) * torch.exp(-0.5 * x**2))

    @staticmethod
    def morlet_wavelet(x):
        return torch.exp(-0.5 * x**2) * torch.cos(5 * x)

    @staticmethod
    def dog_wavelet(x):
        return -x * torch.exp(-0.5 * x**2)

    @staticmethod
    def meyer_wavelet(x):
        abs_x = torch.abs(x)
        pi = math.pi
        def nu(t):
            return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

        def meyer_aux(v):
            return torch.where(v <= 0.5, torch.ones_like(v), torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

        return torch.sin(pi * abs_x) * meyer_aux(abs_x)

    @staticmethod
    def haar_wavelet(x):
        wavelet = torch.zeros_like(x)
        wavelet[(x >= 0) & (x < 0.5)] = 1
        wavelet[(x >= 0.5) & (x < 1)] = -1
        return wavelet

    @staticmethod
    def gaussian_wavelet(x):
        return (2 / (math.sqrt(3) * math.pi**0.25)) * ((1 - x**2) * torch.exp(-0.5 * x**2))

    @staticmethod
    def bior13_wavelet(x):
        h0 = [0.3535533906, 0.7071067812, 0.3535533906]
        wavelet = torch.zeros_like(x)
        for i, hi in enumerate(h0):
            wavelet += hi * torch.roll(x, shifts=i)
        return wavelet

    @staticmethod
    def sym2_wavelet(x):
        h0 = [0.4829629131, 0.8365163037, 0.2241438680, -0.1294095226]
        wavelet = torch.zeros_like(x)
        for i, hi in enumerate(h0):
            wavelet += hi * torch.roll(x, shifts=i)
        return wavelet

    @staticmethod
    def db4_wavelet(x):
        h0 = [0.4829629131, 0.8365163037, 0.2241438680, -0.1294095226]
        wavelet = torch.zeros_like(x)
        for i, hi in enumerate(h0):
            wavelet += hi * torch.roll(x, shifts=i)
        return wavelet

    def shannon_wavelet(self, x):
        pi = math.pi
        sinc_func = torch.sinc(x / pi)
        window = torch.hamming_window(x.size(2), periodic=False, dtype=x.dtype, device=x.device).view(1, 1, -1, *([1] * (self.dimensions - 1)))
        return sinc_func * window

    def forward(self, x):
        x_expanded = x.unsqueeze(1)
        scaled_x = (x_expanded - self.translation_param) / self.scale_param

        if self.wavelet_choice == 'mexican_hat':
            wavelet_transformed = self.mexican_hat_wavelet(scaled_x)
        elif self.wavelet_choice == 'morlet':
            wavelet_transformed = self.morlet_wavelet(scaled_x)
        elif self.wavelet_choice == 'dog':
            wavelet_transformed = self.dog_wavelet(scaled_x)
        elif self.wavelet_choice == 'meyer':
            wavelet_transformed = self.meyer_wavelet(scaled_x)
        elif self.wavelet_choice == 'haar':
            wavelet_transformed = self.haar_wavelet(scaled_x)
        elif self.wavelet_choice == 'gaussian':
            wavelet_transformed = self.gaussian_wavelet(scaled_x)
        elif self.wavelet_choice == 'bior13':
            wavelet_transformed = self.bior13_wavelet(scaled_x)
        elif self.wavelet_choice == 'sym2':
            wavelet_transformed = self.sym2_wavelet(scaled_x)
        elif self.wavelet_choice == 'db4':
            wavelet_transformed = self.db4_wavelet(scaled_x)
        elif self.wavelet_choice == 'shannon':
            wavelet_transformed = self.shannon_wavelet(scaled_x)
        else:
            raise ValueError("Invalid wavelet type specified.")

        wavelet_channels = torch.split(wavelet_transformed, 1, dim=1)
        conv_outputs = [self.wavelet_convolutions[i](ch.squeeze(1)) for i, ch in enumerate(wavelet_channels)]
        combined = torch.cat(conv_outputs, dim=1)
        output = self.final_conv(combined)
        return output


class WavKANLayerND(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1, dimensions=2, dropout=0.0, wavelet_type='mexican_hat'):
        super(WavKANLayerND, self).__init__()

        self.input_channels = input_dim
        self.output_channels = output_dim
        self.groups = groups
        self.dimensions = dimensions
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

        assert wavelet_type in ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon', 'haar', 'gaussian', 'bior13', 'sym2', 'db4'], "Unsupported wavelet type"
        self.wavelet_type = wavelet_type

        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)
        else:
            self.dropout_layer = None

        self.base_convolutions = nn.ModuleList([conv_class(input_dim // groups, output_dim // groups, kernel_size, stride, padding, dilation, groups=1, bias=False) for _ in range(groups)])
        self.wavelet_convolutions = nn.ModuleList([WaveletLayerND(conv_class, input_dim // groups, output_dim // groups, kernel_size, stride=stride, padding=padding, dilation=dilation, dimensions=dimensions, wavelet_choice=wavelet_type) for _ in range(groups)])
        self.norm_layers = nn.ModuleList([norm_class(output_dim // groups) for _ in range(groups)])
        self.activation = nn.SiLU()

    def forward(self, x):
        split_inputs = torch.split(x, self.input_channels // self.groups, dim=1)
        group_outputs = []

        for i, group_input in enumerate(split_inputs):
            base_conv_output = self.base_convolutions[i](self.activation(group_input))
            if self.dropout_layer is not None:
                group_input = self.dropout_layer(group_input)
            wavelet_conv_output = self.wavelet_convolutions[i](group_input)
            combined_output = base_conv_output + wavelet_conv_output
            group_outputs.append(self.norm_layers[i](combined_output))

        return torch.cat(group_outputs, dim=1)


class WavKANLayer3D(WavKANLayerND):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, padding=0, stride=1, dilation=1, dropout=0.0, wavelet_type='bior13'):
        super(WavKANLayer3D, self).__init__(nn.Conv3d, nn.BatchNorm3d, in_channels, out_channels, kernel_size, groups=groups, padding=padding, stride=stride, dilation=dilation, dimensions=3, dropout=dropout, wavelet_type=wavelet_type)


class WavKANLayer2D(WavKANLayerND):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, padding=0, stride=1, dilation=1, dropout=0.0, wavelet_type='bior13'):
        super(WavKANLayer2D, self).__init__(nn.Conv2d, nn.BatchNorm2d, in_channels, out_channels, kernel_size, groups=groups, padding=padding, stride=stride, dilation=dilation, dimensions=2, dropout=dropout, wavelet_type=wavelet_type)


class WavKANLayer1D(WavKANLayerND):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, padding=0, stride=1, dilation=1, dropout=0.0, wavelet_type='bior13'):
        super(WavKANLayer1D, self).__init__(nn.Conv1d, nn.BatchNorm1d, in_channels, out_channels, kernel_size, groups=groups, padding=padding, stride=stride, dilation=dilation, dimensions=1, dropout=dropout, wavelet_type=wavelet_type)
