# /kaggle/working/ARAA-Net/attention_gate.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        An Attention Gate for U-Net skip connections.

        Args:
            F_g (int): Number of channels in the gating signal (from the lower decoder level).
            F_l (int): Number of channels in the skip connection signal (from the encoder).
            F_int (int): Number of channels in the intermediate convolution.
        """
        super(AttentionGate, self).__init__()
        # Convolution for the gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Convolution for the skip connection signal
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Final convolution to produce the attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: Gating signal from the lower level
        # x: Skip connection signal from the encoder
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Add the processed signals and apply ReLU
        psi = self.relu(g1 + x1)
        
        # Generate the attention coefficients (alpha)
        psi = self.psi(psi)
        
        # Apply the attention coefficients to the original skip connection signal
        return x * psi