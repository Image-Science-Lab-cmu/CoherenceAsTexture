import torch
import math
import numpy as np

def roty(points, # tensor, (3,n)
        angle
):
    device = points.device

    C = math.cos(angle * math.pi / 180)
    S = math.sin(angle * math.pi / 180)

    roty_matrix = torch.tensor( 
        [ 
            [C, 0, S],
            [0, 1, 0],
            [-S, 0, C] 
        ]
    ).to(device)
    return roty_matrix @ points

def reflect(points, # tensor, (3,n)
        normal #tensor, (3,), assume unit norm 
):
    points = points.reshape(3,-1) # avoid error when n = 1
    normal = normal.reshape(3,1) # avoid error when n = 1

    proj = normal.t() @ points
    return points - 2* normal* proj.reshape(1,-1)


def mag_wave_horizontal(
    wav, # complex torch(1,1,1,1,Nx,Ny)
    mag_ratio = 2
):
    Ny = wav.size(dim=4)
    Nx = wav.size(dim=5)

    wav = torch.reshape(wav,(1,1,Ny,Nx)) # interpolate only accept 3~5 dimension

    if torch.is_complex(wav):
        real_part = torch.nn.functional.interpolate(wav.real, size=None, scale_factor=[1,mag_ratio], mode='bilinear')
        imag_part = torch.nn.functional.interpolate(wav.imag, size=None, scale_factor=[1,mag_ratio], mode='bilinear')
        wav = real_part + 1j*imag_part
    else:
        wav = torch.nn.functional.interpolate(wav.real, size=None, scale_factor=[1, mag_ratio], mode='bilinear')

    wav = wav[:,:,:,np.arange(Nx)-Nx/2 +wav.size(dim=3)/2]
    wav = torch.reshape(wav,(1,1,1,1,Ny,Nx))

    return wav

def mag_wave(
    wav, # complex torch(1,1,1,1,Nx,Ny)
    mag_ratio = 0.5
):
    Ny = wav.size(dim=4)
    Nx = wav.size(dim=5)

    wav = torch.reshape(wav,(1,1,Ny,Nx)) # interpolate only accept 3~5 dimension
    device = wav.device
    if torch.is_complex(wav):
        if mag_ratio >1:
            real_part = torch.nn.functional.interpolate(wav.real, size=None, scale_factor=[mag_ratio,mag_ratio], mode='bilinear')
            imag_part = torch.nn.functional.interpolate(wav.imag, size=None, scale_factor=[mag_ratio,mag_ratio], mode='bilinear')
        else:
            real_part = torch.nn.functional.interpolate(wav.real, size=None, scale_factor=[mag_ratio,mag_ratio], mode='bilinear', antialias=True)
            imag_part = torch.nn.functional.interpolate(wav.imag, size=None, scale_factor=[mag_ratio,mag_ratio], mode='bilinear', antialias=True)

        wav = real_part + 1j*imag_part
    else:
        if mag_ratio >1:
            wav = torch.nn.functional.interpolate(wav.real, size=None, scale_factor=[mag_ratio, mag_ratio], mode='bilinear')
        else:
            wav = torch.nn.functional.interpolate(wav.real, size=None, scale_factor=[mag_ratio, mag_ratio], mode='bilinear', antialias=True)

    if mag_ratio >1:
        new_wav = wav[:,:,-Ny//2 +wav.size(dim=2)//2:Ny//2 +wav.size(dim=2)//2,-Nx//2 +wav.size(dim=3)//2:Nx//2 +wav.size(dim=3)//2]
    else:
        new_wav = torch.zeros((1,1,Ny,Nx)).to(device) 
        if torch.is_complex(wav):   
            new_wav = new_wav+0j
        new_wav[:,:,Ny//2 -wav.size(dim=2)//2:Ny//2 +wav.size(dim=2)//2,Nx//2 -wav.size(dim=3)//2:Nx//2 +wav.size(dim=3)//2] = wav

    new_wav = torch.reshape(new_wav,(1,1,1,1,Ny,Nx))

    return new_wav

def torch_2d_gaussian(
    window_size = [9,9],
    std = 2.0,
    device = torch.device("cuda:0")
):
    wx = window_size[0]
    wy = window_size[0]

    X,Y = torch.meshgrid([torch.linspace(-wx//2, wx-wx//2, steps=wx),
                                    torch.linspace(-wy//2, wy-wy//2, steps=wy)],  indexing='xy')      
    X = X.to(device)
    Y = Y.to(device)

    return torch.exp(-(X**2 + Y**2)/2/std**2)

def torch_2d_sinc(
    window_size = [9,9],
    scl = 2.0,
    device = torch.device("cuda:0")
):
    a = scl/2 # half of the width of the central lobe
    wx = window_size[0]
    wy = window_size[0]

    X,Y = torch.meshgrid([torch.linspace(-wx//2, wx-wx//2, steps=wx),
                                    torch.linspace(-wy//2, wy-wy//2, steps=wy)],  indexing='xy')      
    X = X.to(device)
    Y = Y.to(device)

    return torch.sinc(X/a)*torch.sinc(Y/a)

class Points2Plane_Prop():
    def __init__(
        self,
        points_xyz, #Tensor: (n, 3)
        plane_xyz, #Tensor: (h, w, 3)
        lam,
        dx
    ):
        # Rayleigh-Sommerfeld
        self.points_xyz = points_xyz
        self.plane_xyz = plane_xyz
        self.dist = -2*torch.matmul(plane_xyz, torch.t(points_xyz)) #(h, w, n)
        self.dist += torch.sum(torch.square(points_xyz), dim =1).reshape(1,1,-1)
        self.dist += torch.sum( torch.square(plane_xyz), dim =2, keepdim = True) 
        self.dist = torch.sqrt(self.dist) 
        assert( torch.min(torch.abs(self.dist.reshape(-1))) > 10* dx, 'Need to satisfy far field assumption')
        self.phasor = torch.exp( 2j * torch.pi * self.dist / lam) /(self.dist) * (dx**2) /lam /(1j)
        self.phasor = torch.conj(self.phasor) # need to find out why the conj is needed

    def forward(
        self,
        point_values #(n,)
    ):
        return torch.sum(self.phasor * point_values, dim = 2)  #(h, w)