import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

import torch
from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
from holotorch.CGH_Datatypes.IntensityField import IntensityField
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Components.FT_Lens import FT_Lens
from holotorch.Optical_Components.SimpleMask import SimpleMask
from holotorch.Optical_Propagators.ASM_Prop import ASM_Prop
from holotorch.Spectra.SpacingContainer import SpacingContainer

import copy
import util

import glob
from PIL import Image
from natsort import natsorted


## define world space parameters
Nx = 2000
Ny = Nx

n_points_x = 1024 # number of target points
n_points_y = 1024

xc = Nx//2 
yc = Ny//2 

lam = 600 * nm
dx = 0.25* um # real world sample pitch, note: the wider the coherence length, the longer the depth of field
# for angular spectruem method in holotorch library, it originally assume dx > lambda/sqrt(2), we modify the library to alleviate the constraint
f = 100 * mm
dfx = lam*f/Nx/dx

save_fig = True
save_data = True


def draw_gif(
    frame_folder
):
    #string = "contrast"
    for string in ["obs","dcinterf","contrast"]:
        frames = [Image.open(image) for image in natsorted(glob.glob(f"{frame_folder}/{string}_*.png"))]
        frame_one = frames[0]
        frame_one.save(f"{frame_folder}/{string}.gif", format="GIF", append_images=frames,
                    save_all=True, duration=200, loop=0)


def sim_self_interference(
    output_path,
    coherence_size = 64, # coherent world pixel
    aperture_D_size = 250,
    camera_ds_ratio = 3,
    wave_sample_interval = 8,
    camera_view_angle = 15,
    mag_ratio = 1,
    depth_scan_step = 160/3*um,
    depth_scan_range = np.arange(-3,3.01),
    device = torch.device("cuda:0"),
    pathlength_mismatch = 0* um,
    retroreflector_horizontal_position_mismatch = 0* um
):

    # World coordinate
    world_grid_x, world_grid_y = torch.meshgrid([torch.linspace(-Nx//2+1, Nx//2, steps=Nx),
                                        torch.linspace(-Ny//2+1, Ny//2, steps=Ny)],  indexing='xy')      
    world_grid_x = world_grid_x.to(device)                                         
    world_grid_y = world_grid_y.to(device)                                         

    # Target: plane with random phase
    original_target = torch.exp(1j*2*math.pi*torch.rand(Ny,Nx))

    # Fourier transform lens
    lens = FT_Lens(focal_length  = f)

    # Built shifted aperture
    aperture_shift_pixel = int(np.round( math.tan(camera_view_angle * math.pi/ 180.0) * f / dfx  ))
    R = torch.sqrt((world_grid_x+aperture_shift_pixel)**2 + world_grid_y**2) # shift the aperture to the right
    aperture_binary = SimpleMask()
    aperture_binary.mask = torch.unsqueeze(torch.unsqueeze((R <aperture_D_size/2).float(),0),1).to(device)

    # 4f system with shifted aperture
    tilt_ortho_camera_path = [lens, aperture_binary, lens]

    # Phasors on LC cell
    LC_cell_phasors = torch.exp( torch.asarray([0,2/3,4/3])*1j* torch.pi).to(device) # three phasors for PSI

    # Simulate magnification if ratio != 1
    if mag_ratio != 1:
        p = f*(1+1/mag_ratio)
        q = f*(mag_ratio+1)

        lens_p = FT_Lens(focal_length  = p)
        lens_q = FT_Lens(focal_length  = q)
        asm_prop_mag = ASM_Prop(
            init_distance = -p-q,
        )       

        mag_path = [lens_p, asm_prop_mag, lens_q]

    # Simulate non-ideal factor
    sim_nonideal =  pathlength_mismatch != 0 or retroreflector_horizontal_position_mismatch != 0 # simulate displacement of retroreflector or not

    # Initialize values to record
    Cx = int(np.ceil(Nx/camera_ds_ratio))
    Cy = int(np.ceil(Ny/camera_ds_ratio))
    dc_all = np.zeros((Cy, Cx, len(depth_scan_range)))    
    interf_all = np.zeros((Cy, Cx, len(depth_scan_range)))    
    contrast_all = np.zeros((Cy, Cx, len(depth_scan_range)))    
    if save_fig:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    # Propagate waves
    with torch.no_grad():
        for depth_scan_id, depth_scan in enumerate(depth_scan_range):
            obs_points_sum = IntensityField(torch.zeros(1,3,1,1,Ny,Nx).to(device)) # three phasors
            dc_points_sum = IntensityField(torch.zeros(1,1,1,1,Ny,Nx).to(device))
            interf_points_sum = IntensityField(torch.zeros(1,1,1,1,Ny,Nx).to(device))

            print(f'scan {depth_scan_id}, depth = {depth_scan*depth_scan_step/um} um')

            asm_prop = ASM_Prop(
                init_distance = -depth_scan*depth_scan_step,
            )

            if sim_nonideal:
                asm_prop_nonideal = ASM_Prop(
                    init_distance = -depth_scan*depth_scan_step +  pathlength_mismatch,
                )
                nonideal_ramp_freq = retroreflector_horizontal_position_mismatch/dx/Nx
                nonideal_ramp = SimpleMask()
                nonideal_ramp.mask = torch.exp(1j * 2* torch.pi * world_grid_x* nonideal_ramp_freq)

            # simulate propagation of wave blocks one by one
            for pos_x in np.arange(-n_points_x/2,n_points_x/2, wave_sample_interval):
                for pos_y in np.arange(-n_points_y/2,n_points_y/2, wave_sample_interval):

                    # one wave contain one coherent region
                    original_wave = torch.zeros(Ny,Nx)+0j # note +0j is important as it set the type to complex
                    original_wave = original_wave.to(device)
    
                    original_wave[
                        int(Ny/2-coherence_size/2+pos_y):int(Ny/2+coherence_size/2+pos_y), 
                        int(Nx/2-coherence_size/2+pos_x):int(Nx/2+coherence_size/2+pos_x)
                    ] = original_target[ 
                        int(Ny/2-coherence_size/2+pos_y):int(Ny/2+coherence_size/2+pos_y), 
                        int(Nx/2-coherence_size/2+pos_x):int(Nx/2+coherence_size/2+pos_x)
                    ]
                    original_wave = torch.reshape(original_wave,(1,1,1,1,Ny,Nx))


                    # Record waves from two paths
                    modulated_waves = torch.zeros(2,1,1,1,Ny,Nx)+0j  
                    modulated_waves = modulated_waves.to(device)

                    for path_id in range(2):
                        waveprop = ElectricField(
                            data = copy.deepcopy(original_wave), 
                            wavelengths = lam,
                            spacing = dx
                        )
                        waveprop.wavelengths=waveprop.wavelengths.to(device)
                        waveprop.spacing = waveprop.spacing.to(device)

                        # propogate target to plane in focus of the 4f system
                        if sim_nonideal and path_id == 1:
                            # optionally simulate pathlength mismatch
                            waveprop = asm_prop_nonideal(waveprop)
                        else:
                            waveprop = asm_prop(waveprop)

                        # optional magnification
                        if mag_ratio !=1:
                            for component in mag_path:
                                waveprop = component(waveprop)
                            waveprop.data = util.mag_wave(waveprop.data, mag_ratio=mag_ratio)
                            waveprop.spacing = SpacingContainer(dx)

                        # horizontally flip the wave for one path
                        if path_id == 1:
                            waveprop.data = torch.flip(waveprop.data, dims=[5])
                            
                        # optionally simulate retroreflector horizontal position mismatch
                        if sim_nonideal and path_id == 1:
                            waveprop = nonideal_ramp(waveprop)
                    
                        # propagte over the 4f system with shifted aperture
                        for component in tilt_ortho_camera_path:
                            waveprop = component(waveprop)

                        # record
                        modulated_waves[[path_id]] += waveprop.data.detach()

                    # phase shifting interferometry
                    obs_interf = torch.cat( 
                        [   
                            torch.square( 
                                (   
                                    modulated_waves[[0]]+
                                    LC_cell_phasors[phasor_id]*modulated_waves[[1]]
                                ).abs()
                            ).detach() 
                            for phasor_id in range(3)
                        ],
                        dim = 1 # dim 1 means time 
                    )
                            
                    obs_points_sum.data = obs_points_sum.data + obs_interf.detach()

                    if pos_x == 0 and pos_y == 0:
                        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

                    del original_wave, waveprop, modulated_waves, obs_interf
            
            print('observation before downsampling in camera')

            if save_fig:
                plt.figure(figsize=(15,5))
                plt.subplot(131)
                obs_points_sum[:, 0,...].abs().visualize(rescale_factor=1,title = "|u + v|^2", flag_axis= True)
                plt.subplot(132)
                obs_points_sum[:, 1,...].abs().visualize(rescale_factor=1,title = "|u + e^(2j pi/3)v|^2", flag_axis= True)
                plt.subplot(133)
                obs_points_sum[:, 2,...].abs().visualize(rescale_factor=1,title = "|u + e^(4j pi/3)v|^2", flag_axis= True)
                
                plt.tight_layout()
                plt.show()

            print('observation after downsampling in camera') 

            average_kernel = torch.ones(3,1,int(camera_ds_ratio),int(camera_ds_ratio)).to(device)/camera_ds_ratio**2
            # average, also, change dim from [1,3,1,1,N,N] to [3,N,N]
            obs_points_sum.data= torch.nn.functional.conv2d(obs_points_sum.data[:,:,0,0,:,:],average_kernel,groups=3,padding='same')
            obs_points_sum.data= obs_points_sum.data[:,:,None,None,:,:]

            if save_fig:
                plt.figure(figsize=(15,5))
                plt.subplot(131)
                obs_points_sum[:, 0,..., 0:Ny:camera_ds_ratio, 0:Nx:camera_ds_ratio].abs().visualize(rescale_factor=1, title = "|u + v|^2", flag_axis= True)
                plt.subplot(132)
                obs_points_sum[:, 1,..., 0:Ny:camera_ds_ratio, 0:Nx:camera_ds_ratio].abs().visualize(rescale_factor=1,title = "|u + e^(2j pi/3)v|^2", flag_axis= True)
                plt.subplot(133)
                obs_points_sum[:, 2,..., 0:Ny:camera_ds_ratio, 0:Nx:camera_ds_ratio].abs().visualize(rescale_factor=1,title = "|u + e^(4j pi/3)v|^2", flag_axis= True)
            
                plt.tight_layout()
                plt.savefig(f'{output_path}/obs_{depth_scan_id}.png')

            print('dc and interf')
            dc_points_sum.data = torch.sum(
                    obs_points_sum.data,dim=1,keepdim=True
            ).detach()/3

            interf_points_sum.data = torch.sum(
                obs_points_sum.data* LC_cell_phasors[None,:,None,None,None,None],
                dim=1,keepdim=True
            ).detach()/3

            if save_fig:
                plt.figure(figsize=(15,5))
                plt.subplot(131)
                dc_points_sum[..., 0:Ny:camera_ds_ratio, 0:Nx:camera_ds_ratio].abs().visualize(rescale_factor=1,title = "dc", flag_axis= True)
                plt.subplot(132)
                interf_points_sum[...,0:Ny:camera_ds_ratio, 0:Nx:camera_ds_ratio].abs().visualize(rescale_factor=1,title = "interf abs", flag_axis= True)
                plt.subplot(133)
                interf_points_sum[...,0:Ny:camera_ds_ratio, 0:Nx:camera_ds_ratio].angle().visualize(rescale_factor=1,title = "interf phase", flag_axis= True, cmap='binary_r')
                plt.hsv()

                plt.tight_layout()
                plt.savefig(f'{output_path}/dcinterf_{depth_scan_id}.png')

            # filtered out points has low intensity/ interfernce signal
            contrast = IntensityField(2*interf_points_sum.data/dc_points_sum.data *(dc_points_sum.data>0.1*(torch.max(dc_points_sum.data))) ) 

            if save_fig:
                plt.figure(figsize=(5,5))
                plt.tight_layout()
                contrast[..., 0:Ny:camera_ds_ratio, 0:Nx:camera_ds_ratio].abs().visualize(rescale_factor=1,title = "contrast", flag_axis= True, vmax=1, vmin=0)
                plt.savefig(f'{output_path}/contrast_{depth_scan_id}.png')

            dc_all[:,:,depth_scan_id] = dc_points_sum[0,0,0,0, 0:Ny:camera_ds_ratio, 0:Nx:camera_ds_ratio].abs().data.squeeze().detach().cpu().numpy()
            interf_all[:,:,depth_scan_id] = interf_points_sum[0,0,0,0, 0:Ny:camera_ds_ratio, 0:Nx:camera_ds_ratio].abs().data.squeeze().detach().cpu().numpy()
            contrast_all[:,:,depth_scan_id] = contrast[0,0,0,0, 0:Ny:camera_ds_ratio, 0:Nx:camera_ds_ratio].abs().data.squeeze().detach().cpu().numpy()

            del obs_points_sum, dc_points_sum, interf_points_sum

    if save_fig:
        draw_gif(frame_folder=output_path)
        with open(f'{output_path}/data.npy', 'wb') as file:
            np.save(file,dc_all)
            np.save(file,interf_all)
            np.save(file,contrast_all)

    return dc_all, interf_all

if __name__ == '__main__':
    ## Setting parameters

    device = torch.device("cuda:0")
    coherence_length = 16*um # see \detla_c in the paper
    diffraction_blur_kernel = 2*um  # see \detla_Phi in the paper
    camera_pitch = 0.75*um # see \detla_x in the paper
    camera_view_angle = 15 # see atan(\beta) in the paper, effective tilt angle of the camera
    mag_ratio = 1 # see M in the paper, fixed to be 1 in analysis
    depth_scan_step = diffraction_blur_kernel/lam*coherence_length/mag_ratio # depth range will scale with these factors, see (14)
    depth_scan_range = np.arange(-3,3.01,0.75)

    # discretized parameters
    coherence_size = int(coherence_length//dx)
    aperture_D_length = f*lam/diffraction_blur_kernel
    aperture_D_size = aperture_D_length/dfx 
    camera_ds_ratio = int(camera_pitch//dx)

    # Set wave_sample_interval
    # In real world, waves from different postions are partially coherent with others
    # It can be simulated by multiple coherent blocks of waves overlapped with each other
    # To simulate real world behavior precisely, sample interveral should be smaller than diffraction_blur_kernel//dx
    # However, this dense sampling takes a long time to run
    # For some quick evaluation, one can set wave_sample_interval = coherence_size, but it will has inprecise contrast value

    #wave_sample_interval = int(2*um//dx) #precised simulation used in the paper
    wave_sample_interval = coherence_size #quick evaluation


    ## Preparing saving data

    params_dict = {
        'object_pitch': dx,
        'wave_length': lam,
        'aperture_D_length': aperture_D_length,
        'focal_length': f,
        'numerical_aperture': f/aperture_D_length,
        'coherence_length': coherence_length, #coherence_size*dx,
        'diffraction_blur_kernel': diffraction_blur_kernel, #f/aperture_D_length*lam,
        'camera_pitch': camera_pitch, #dx* camera_ds_ratio,
        'camera_view_angle': camera_view_angle,
        'depth_scan_step': depth_scan_step,
        'depth_scan_range': depth_scan_range,
        'mag_ratio': mag_ratio,
        'wave_sample_interval': wave_sample_interval
    }

    ts = time.gmtime()
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", ts)

    print(f"coherence length ={coherence_length//um}um,  diffraction blur kernel size ={diffraction_blur_kernel//um}um, camera pixel pitch ={(camera_pitch/um):.2f}um,  camera_view_angle={camera_view_angle}, depth_step={depth_scan_step}, magnification_rate={mag_ratio}")
    output_path = f'./outputs/{timestamp}_c_{coherence_length//um}_b_{diffraction_blur_kernel//um}_p_{(camera_pitch/um):.2f}_vangle_{camera_view_angle}_magratio_{mag_ratio}_dstep_{np.round(depth_scan_step/um)}'
    if save_data:
        if not os.path.exists(output_path):
            os.makedirs(output_path)


    # Run simulation and save data

    n_runs = 1
    Cx = int(np.ceil(Ny/camera_ds_ratio))
    H = int(np.ceil(n_points_y/camera_ds_ratio ))
    dc_all_epochs = np.zeros((n_runs* H, Cx, len(depth_scan_range))) 
    interf_all_epochs = np.zeros((n_runs* H, Cx, len(depth_scan_range)))

    for run in range(n_runs):
        print(f'run = {run}')

        dc_all, interf_all = sim_self_interference(
            output_path = output_path,
            coherence_size=coherence_size, 
            aperture_D_size = aperture_D_size,
            camera_ds_ratio=camera_ds_ratio,
            wave_sample_interval=wave_sample_interval,
            camera_view_angle = camera_view_angle,
            mag_ratio=mag_ratio,
            depth_scan_step = depth_scan_step,
            depth_scan_range = depth_scan_range,
            device = device
        )

        dc_all_epochs[ H*run: H*run+ H, :, :] =  dc_all[(int(Ny/camera_ds_ratio)//2) -H//2: (int(Ny/camera_ds_ratio)//2) +(H-H//2),:,:]
        interf_all_epochs[ H*run: H*run+ H, :, :] =  interf_all[ (int(Ny/camera_ds_ratio)//2) -H//2: (int(Ny/camera_ds_ratio)//2) +(H-H//2),:,:]


    if save_data:
        with open(f'{output_path}/data.npy', 'wb') as file:
            np.save(file,dc_all_epochs)
            np.save(file,interf_all_epochs)
            np.save(file,params_dict)
