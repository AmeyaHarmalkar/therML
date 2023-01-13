import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(ConvBlock1d, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size=(kernel_size),
                               padding=(padding),
                               stride=(stride))
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.activation = F.relu
    
    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


class ConvBlock2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(ConvBlock2d, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=(kernel_size, kernel_size),
                                     dilation=(dilation, dilation),
                                     stride=(stride, stride),
                                     padding=(padding, padding))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = F.relu
    
    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


class SmallDataCNN(nn.Module):
    
    def __init__(self, ignore_seq=False, ignore_energy=False, num_1d_blocks=1, num_2d_blocks=1):
        super(SmallDataCNN, self).__init__()
        
        self.ignore_seq = ignore_seq
        self.ignore_energy = ignore_energy
        self.num_1d_blocks = num_1d_blocks
        self.num_2d_blocks = num_2d_blocks
        
        kernel_size_1d = 7
        kernel_size_2d = 5
        num_aa_channels = 21
        num_energy_channels = 1
        num_out_channels = 5
        num_classes = 4
        
        self.seq_branch = nn.Sequential(*[
            ConvBlock1d(num_aa_channels, num_out_channels, kernel_size_1d)
            for _ in range(self.num_1d_blocks)
        ])
        
        self.adapt1d = nn.AdaptiveMaxPool1d(120)
        
        self.energy_branch = nn.Sequential(*[
            ConvBlock2d(num_energy_channels, num_out_channels, kernel_size_2d)
            for _ in range(self.num_2d_blocks)
        ])
        
        self.adapt2d = nn.AdaptiveMaxPool2d((120,120))
        
        main_branch_channels = 3 * num_out_channels
        
        self.main_branch = nn.Sequential(*[
            ConvBlock2d(main_branch_channels, num_classes+1 , kernel_size_2d, stride=2)
            for _ in range(self.num_2d_blocks)
        ])
        
        self.fc1 = nn.Linear(5*60*60, 4)
        
    def forward(self, seq_input, energy_input, dev=None):
        
        if not self.ignore_seq:
            out_1d = self.adapt1d( self.seq_branch(seq_input))
            expand_out_1d = out_1d.unsqueeze(-1).expand(
                (*out_1d.shape, out_1d.shape[-1]))
            seq_out_2d = torch.cat(
                [expand_out_1d, expand_out_1d.transpose(2, 3)], dim=1)
            print("Sequence out: ", seq_out_2d.size())
        else:
            seq_out_2d = torch.zeros(
                (seq_input.shape[0], 5*2,
                 120, 120)).to(dev)
        
        if not self.ignore_energy:
            energy_out_2d = self.adapt2d( self.energy_branch(energy_input) )
            print("Energy out: ",energy_out_2d.size())
        else:
            energy_out_2d = torch.zeros(
                (energy_input.shape[0], 5,
                 120,120)).to(dev)
            
        main_input = torch.cat([seq_out_2d, energy_out_2d], dim=1)
        print("Main Branch : ", main_input.size())
        
        main_out = self.main_branch(main_input)
        print("Main Out : ", main_out.size())
        
        out = F.relu(main_out)
        out = out.view( out.size(0), -1 )
        out = self.fc1(out)
        
        return out


class EquiCNN(nn.Module):
    
    def __init__(self, ignore_seq=False, ignore_energy=False, num_1d_blocks=1, num_2d_blocks=1, device=None):
        super(EquiCNN, self).__init__()
        
        self.ignore_seq = ignore_seq
        self.ignore_energy = ignore_energy
        self.num_1d_blocks = num_1d_blocks
        self.num_2d_blocks = num_2d_blocks
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        kernel_size_1d = 7
        kernel_size_2d = 5
        num_aa_channels = 21
        num_energy_channels = 21
        num_seq_out_channels = 5
        num_out_channels = 10
        num_classes = 4
        
        self.seq_branch = nn.Sequential(*[
            ConvBlock1d(num_aa_channels, num_seq_out_channels, kernel_size_1d)
            for _ in range(self.num_1d_blocks)
        ])
        
        self.adapt1d = nn.AdaptiveMaxPool1d(120)
        
        self.energy_branch = nn.Sequential(*[
            ConvBlock2d(num_energy_channels, num_out_channels, kernel_size_2d)
            for _ in range(self.num_2d_blocks)
        ])
        
        self.adapt2d = nn.AdaptiveMaxPool2d((120,120))
        
        main_branch_channels = 2 * num_out_channels
        
        self.main_branch = nn.Sequential(*[
            ConvBlock2d(main_branch_channels, num_classes+1 , kernel_size_2d, stride=2)
            for _ in range(self.num_2d_blocks)
        ])
        
        self.fc1 = nn.Linear(5*60*60, 4)
        
    def forward(self, seq_input, energy_input, dev=None):
        
        if not self.ignore_seq:
            out_1d = self.adapt1d( self.seq_branch(seq_input))
            expand_out_1d = out_1d.unsqueeze(-1).expand(
                (*out_1d.shape, out_1d.shape[-1]))
            seq_out_2d = torch.cat(
                [expand_out_1d, expand_out_1d.transpose(2, 3)], dim=1)
            #print("Sequence out: ", seq_out_2d.size())
        else:
            seq_out_2d = torch.zeros(
                (seq_input.shape[0], 5*2, 120, 120)).to('cpu')
        
        if not self.ignore_energy:
            energy_out_2d = self.adapt2d( self.energy_branch(energy_input) )
            #print("Energy out: ",energy_out_2d.size())
        else:
            energy_out_2d = torch.zeros(
                (energy_input.shape[0], 10, 120,120)).to('cpu')
            
        main_input = torch.cat([seq_out_2d, energy_out_2d], dim=1)
        #print("Main Branch : ", main_input.size())
        
        main_out = self.main_branch(main_input)
        #print("Main Out : ", main_out.size())
        
        out = F.relu(main_out)
        out = out.view( out.size(0), -1 )
        out = self.fc1(out)
        
        return out


class MiniCNN(nn.Module):
    
    def __init__(self, ignore_seq=False, ignore_energy=False, num_1d_blocks=1, num_2d_blocks=1):
        super(MiniCNN, self).__init__()
        
        self.ignore_seq = ignore_seq
        self.ignore_energy = ignore_energy
        self.num_1d_blocks = num_1d_blocks
        self.num_2d_blocks = num_2d_blocks
        
        kernel_size_1d = 7
        kernel_size_2d = 5
        num_aa_channels = 21
        num_energy_channels = 21
        num_seq_out_channels = 4
        num_out_channels = 8
        num_classes = 4
        
        self.seq_branch = nn.Sequential(*[
            ConvBlock1d(num_aa_channels, num_seq_out_channels, kernel_size_1d)
            for _ in range(self.num_1d_blocks)
        ])
        
        self.adapt1d = nn.AdaptiveMaxPool1d(120)
        
        self.energy_branch = nn.Sequential(*[
            ConvBlock2d(num_energy_channels, num_out_channels, kernel_size_2d)
            for _ in range(self.num_2d_blocks)
        ])
        
        self.adapt2d = nn.AdaptiveMaxPool2d((120,120))
        
        main_branch_channels = 2 * num_out_channels
        
        self.main_branch = nn.Sequential(*[
            ConvBlock2d(main_branch_channels, num_classes , kernel_size_2d, stride=2)
            for _ in range(self.num_2d_blocks)
        ])
        
        self.fc1 = nn.Linear(4*60*60, 4)
        
    def forward(self, seq_input, energy_input, dev=None):
        
        if not self.ignore_seq:
            out_1d = self.adapt1d( self.seq_branch(seq_input))
            expand_out_1d = out_1d.unsqueeze(-1).expand(
                (*out_1d.shape, out_1d.shape[-1]))
            seq_out_2d = torch.cat(
                [expand_out_1d, expand_out_1d.transpose(2, 3)], dim=1)
            #print("Sequence out: ", seq_out_2d.size())
        else:
            seq_out_2d = torch.zeros(
                (seq_input.shape[0], 4*2, 120, 120)).to(dev)
        
        if not self.ignore_energy:
            energy_out_2d = self.adapt2d( self.energy_branch(energy_input) )
            #print("Energy out: ",energy_out_2d.size())
        else:
            energy_out_2d = torch.zeros(
                (energy_input.shape[0], 8, 120,120)).to(dev)
            
        main_input = torch.cat([seq_out_2d, energy_out_2d], dim=1)
        #print("Main Branch : ", main_input.size())
        
        main_out = self.main_branch(main_input)
        #print("Main Out : ", main_out.size())
        
        out = F.relu(main_out)
        out = out.view( out.size(0), -1 )
        out = self.fc1(out)
        
        return out