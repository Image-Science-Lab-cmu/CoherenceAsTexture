#%matplotlib widget

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

import torch
import holotorch
from holotorch.utils.units import * # E.g. to get nm, um, mm etc.
from holotorch.CGH_Datatypes.IntensityField import IntensityField
from holotorch.CGH_Datatypes.ElectricField import ElectricField
from holotorch.Optical_Components.FT_Lens import FT_Lens
from holotorch.Optical_Components.SimpleMask import SimpleMask
from holotorch.Optical_Propagators.ASM_Prop import ASM_Prop
from holotorch.Spectra.SpacingContainer import SpacingContainer

import copy
import util
import argparse

import glob
from PIL import Image
from natsort import natsorted


## define world space parameters
Nx = 2000
Ny = Nx

n_points = 1024

# define observation range
Mx = Nx #previous: n_points_x 
My = Mx

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
    aperture_D_ratio = 0.25,
    coherence_size  = 64, # coherent world pixel
    wave_sample_interval = 8,
    light_angle_spread_deg = 0, #degree
    camera_view_angle = 15,
    depth_scan_step = 40/3*um,
    shift_range = np.arange(-6,7),
    ds_ratio = 4,
    mag_ratio = 1,
    device = torch.device("cuda:0")
    #light_source_num = 16 # number of light sources
):

    #Cx = int(round(Mx/ds_ratio))
    #Cy = int(round(My/ds_ratio))
    Cx = int(np.ceil(Mx/ds_ratio))
    Cy = int(np.ceil(My/ds_ratio))

    #print(f"aperture D ratio ={aperture_D_ratio}, coherence size ={coherence_size},  light_angle_spread = {light_angle_spread_deg}")
    #print(f"aperture D ratio ={aperture_D_ratio}, coherence size ={coherence_size}, light_source_num = {light_source_num}, light_angle_spread = {light_angle_spread_deg}")

    target_type = 'plane' #['plane','ball']

    shift_direction = 'sensor_depth' #['bs_normal','sensor_depth']
    #sensor_type = 'tilt_camera' # 'tilt_camera' or 'shift_aperture'
    sensor_type = 'shift_aperture' # 'tilt_camera' or 'shift_aperture'

    coherence_length = coherence_size * dx
    aperture_D = Nx * aperture_D_ratio

    # non ideal sim
    sim_nonideal = False
    depth_mismatch = 0* um
    shift_mismatch = 0* um

    #gaussian_profile = True

    if light_angle_spread_deg == 0:
        lightsource_grid = torch.tensor([[0,0,1]], dtype = torch.float32, device = device) # (1, 3)

    else:
        light_angle_interval = 1./8. # degree, determined by exp
        light_source_width = int(np.round(light_angle_spread_deg/light_angle_interval))
        light_source_height = 4

        # light_source_length = int(np.sqrt(light_source_num))
        # light_angle_spread = light_angle_spread_deg * math.pi / 180
        # lightsource_grid_x, lightsource_grid_y = torch.meshgrid([torch.linspace(-light_source_length//2+1, light_source_length//2, steps=light_source_length),
        #                                     torch.linspace(-light_source_length//2+1, light_source_length//2, steps=light_source_length)],  indexing='xy')      
        
        lightsource_grid_x, lightsource_grid_y = torch.meshgrid([torch.linspace(-light_source_width//2+1, light_source_width//2, steps=light_source_width),
                                            torch.linspace(-light_source_height//2+1, light_source_height//2, steps=light_source_height)],  indexing='xy')      
        lightsource_grid_x = lightsource_grid_x.to(device)                                         
        lightsource_grid_y = lightsource_grid_y.to(device)            
        # lightsource_grid= torch.stack( [
        #     lightsource_grid_x, 
        #     lightsource_grid_y, 
        #     light_source_length* torch.ones(light_source_length,light_source_length).to(device) / light_angle_spread 
        # ], dim = 2) # (Nx, Ny, 3)
                                
        lightsource_grid= torch.stack( [
            lightsource_grid_x * light_angle_interval * math.pi / 180,
            lightsource_grid_y * light_angle_interval * math.pi / 180,
            torch.ones(light_source_height, light_source_width).to(device)
        ], dim = 2) # (Nx, Ny, 3)


    lightsource_grid = lightsource_grid.reshape(-1,3) # (light_source_num, 3)
    #lightsource_grid = lightsource_grid + 1j # make it complex
    lightsource_grid = lightsource_grid/ torch.norm(lightsource_grid, dim = 1, keepdim = True) # normalize to unit vector

    print(f"coherence size ={coherence_size},  aperture D ratio ={aperture_D_ratio}, camera_view_angle={camera_view_angle}, light_angle_spread = {light_angle_spread_deg}, sensor_type ={sensor_type}, depth_step={depth_scan_step}, dsratio ={ds_ratio}")

    if sensor_type == 'tilt_camera':
        camera_tilt_angle = camera_view_angle
        aperture_shift_pixel = 0
    elif sensor_type == 'shift_aperture':
        camera_tilt_angle = 0
        aperture_shift_pixel = int(np.round( math.tan(camera_view_angle * math.pi/ 180.0) * f / dfx  ))
    else:
        raise ValueError('sensor type not supported')

    if save_fig:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    #coherence_size = int(np.round(coherence_length/dx))

    world_grid_x, world_grid_y = torch.meshgrid([torch.linspace(-Nx//2+1, Nx//2, steps=Nx),
                                        torch.linspace(-Ny//2+1, Ny//2, steps=Ny)],  indexing='xy')      
    world_grid_x = world_grid_x.to(device)                                         
    world_grid_y = world_grid_y.to(device)                                         
 
    ## define target space
    n_points_x = n_points#256
    n_points_y = n_points#256

    #n_cbatch_x = int(np.round(n_points_x/coherence_size))
    #n_cbatch_y = int(np.round(n_points_y/coherence_size))


    R = torch.sqrt((world_grid_x+aperture_shift_pixel)**2 + world_grid_y**2) # shift the aperture to the right
    #R = torch.sqrt((world_grid_x+aperture_shift_pixel-aperture_D/2)**2 + world_grid_y**2) # shift the aperture to the right


    aperture_binary = SimpleMask()
    aperture_binary.mask = torch.unsqueeze(torch.unsqueeze((R <aperture_D/2).float(),0),1).to(device)
    #aperture_binary.mask = torch.unsqueeze(torch.unsqueeze((R_slit <aperture_D/2).float(),0),1).to(device)

    R0 = torch.sqrt(world_grid_x**2 + world_grid_y**2)

    aperture_circ = SimpleMask()

    FFT2 = lambda sig:  torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(sig)))

    aperture_circ.mask = torch.unsqueeze(torch.unsqueeze( FFT2( (R0< 1/aperture_D_ratio).float() ),0),1)
    #aperture_circ.mask = torch.roll(aperture_circ.mask, -aperture_shift_pixel, dims=3)

    lens1 = FT_Lens(focal_length  = f)
    # ramp_freq = 2*shift_pixel/Nx
    # slm_model1= SimpleMask()
    # slm_model1.mask = torch.exp(1j * 2* torch.pi * grid_x* ramp_freq)

    #point_list[5] = point_list[5] + np.array([200*nm,0,0])

    phasors = torch.exp( torch.asarray([0,2/3,4/3])*1j* torch.pi).to(device)

    if mag_ratio != 1:
        p = f*(1+1/mag_ratio)
        q = f*(mag_ratio+1)

        lens_p = FT_Lens(focal_length  = p)
        lens_q = FT_Lens(focal_length  = q)
        asm_prop_mag = ASM_Prop(
            init_distance = -p-q,
        )       

        mag_path = [lens_p, asm_prop_mag, lens_q]


    # asm_mid_prop = ASM_Prop(
    #                 init_distance = 2*f,
    # )

    dc_all = np.zeros((Cy, Cx, len(shift_range)))    
    interf_all = np.zeros((Cy, Cx, len(shift_range)))    
    contrast_all = np.zeros((Cy, Cx, len(shift_range)))    

    original_target = torch.exp(1j*2*math.pi*torch.rand(Ny,Nx))

    with torch.no_grad():
        for physical_shift_id, physical_shift in enumerate(shift_range):
            obs_points_sum = IntensityField(torch.zeros(1,3,1,1,Ny,Nx).to(device)) # three phasors
            dc_points_sum = IntensityField(torch.zeros(1,1,1,1,Ny,Nx).to(device))
            interf_points_sum = IntensityField(torch.zeros(1,1,1,1,Ny,Nx).to(device))

            print(f'shift = {physical_shift}')

            # resample phase to avoid memory effect-like phenomena
            #object_points_phase = torch.exp(1j*2*torch.pi*torch.rand(n_cbatch_x * n_cbatch_y, coherence_size*coherence_size, 1)).to(device) 

            asm_prop = ASM_Prop(
                init_distance = -physical_shift*depth_scan_step,
            )

            path = [lens1, aperture_binary, lens1]

            if sim_nonideal:
                asm_prop_nonideal = ASM_Prop(
                    init_distance = -physical_shift*depth_scan_step +  depth_mismatch,
                )

                nonideal_ramp_freq = shift_mismatch/dx/Nx
                nonideal_ramp = SimpleMask()
                nonideal_ramp.mask = torch.exp(1j * 2* torch.pi * world_grid_x* nonideal_ramp_freq)



            # incoherent light source
            for light_dir in lightsource_grid:
                # light source: normal direction of light 
                # incoherent spatial patch

                n_cbatch_x = int(np.round(n_points_x/wave_sample_interval))
                n_cbatch_y = int(np.round(n_points_y/wave_sample_interval))


                for pos_x in np.arange(-n_cbatch_x/2,n_cbatch_x/2)*wave_sample_interval:
                    for pos_y in np.arange(-n_cbatch_y/2,n_cbatch_y/2)*wave_sample_interval:

                        fields = torch.zeros(2,1,1,1,Ny,Nx)+0j  # two paths, note +0j is important as it set the type to complex
                        fields = fields.to(device)

                        original_wave = torch.zeros(Ny,Nx)+0j
                        original_wave = original_wave.to(device)

      
                        original_wave[
                            int(Ny/2-coherence_size/2+pos_y):int(Ny/2+coherence_size/2+pos_y), 
                            int(Nx/2-coherence_size/2+pos_x):int(Nx/2+coherence_size/2+pos_x)
                        ] = original_target[ 
                            int(Ny/2-coherence_size/2+pos_y):int(Ny/2+coherence_size/2+pos_y), 
                            int(Nx/2-coherence_size/2+pos_x):int(Nx/2+coherence_size/2+pos_x)
                        ]

                        original_wave = torch.reshape(original_wave,(1,1,1,1,Ny,Nx))

                        for path_id in range(2):
                            waveprop = ElectricField(
                                data = copy.deepcopy(original_wave), 
                                wavelengths = lam,
                                spacing = dx
                            )
                            waveprop.wavelengths=waveprop.wavelengths.to(device)
                            waveprop.spacing = waveprop.spacing.to(device)


                            # propogate points with optional mismatch
                            if sim_nonideal and path_id == 1:
                                waveprop = asm_prop_nonideal(waveprop)
                            else:
                                waveprop = asm_prop(waveprop)

                            # optionally magnification
                            if mag_ratio !=1:
                                for component in mag_path:
                                    waveprop = component(waveprop)
                                waveprop.data = util.mag_wave(waveprop.data, mag_ratio=mag_ratio)
                                waveprop.spacing = SpacingContainer(dx)

                            # flip with optional additional phase ramp
                            if path_id == 1:
                                waveprop.data = torch.flip(waveprop.data, dims=[5])
                                
                                #temporary add to simulate full retroreflector                        
                                #waveprop.data = torch.flip(waveprop.data, dims=[4])

                            if sim_nonideal and path_id == 1:
                                waveprop = nonideal_ramp(waveprop)
                     
                            # prop main path
                            for component in path:
                                waveprop = component(waveprop)


                            fields[[path_id]] += waveprop.data.detach()

                        obs_interf = torch.cat( 
                            [   
                                torch.square( 
                                    (   
                                        fields[[0]]+
                                        phasors[phasor_id]*fields[[1]]
                                    ).abs()
                                ).detach() 
                                for phasor_id in range(3)
                            ],
                            dim = 1 # dim 1 means time 
                        )
                                
                        obs_points_sum.data = obs_points_sum.data + obs_interf.detach()

                        if pos_x == 0 and pos_y == 0:
                            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

                        del original_wave, waveprop, fields, obs_interf
            
            print('observation before downsampling in camera')

            if save_fig:
                plt.figure(figsize=(15,5))
                plt.subplot(131)
                obs_points_sum[:, 0,..., yc-My//2:yc+My//2, xc-Mx//2:xc+Mx//2].abs().visualize(rescale_factor=1,title = "|u + v|^2", flag_axis= True)
                plt.subplot(132)
                obs_points_sum[:, 1,..., yc-My//2:yc+My//2, xc-Mx//2:xc+Mx//2].abs().visualize(rescale_factor=1,title = "|u + e^(2j pi/3)v|^2", flag_axis= True)
                plt.subplot(133)
                obs_points_sum[:, 2,..., yc-My//2:yc+My//2, xc-Mx//2:xc+Mx//2].abs().visualize(rescale_factor=1,title = "|u + e^(4j pi/3)v|^2", flag_axis= True)
                
                plt.tight_layout()
                plt.show()

            print('observation after downsampling in camera') 

            average_kernel = torch.ones(3,1,int(ds_ratio),int(ds_ratio)).to(device)/ds_ratio**2
            # average, also, change dim from [1,3,1,1,N,N] to [3,N,N]
            obs_points_sum.data= torch.nn.functional.conv2d(obs_points_sum.data[:,:,0,0,:,:],average_kernel,groups=3,padding='same')
            obs_points_sum.data= obs_points_sum.data[:,:,None,None,:,:]

            if save_fig:
                plt.figure(figsize=(15,5))
                plt.subplot(131)
                obs_points_sum[:, 0,..., yc-My//2:yc+My//2:ds_ratio, xc-Mx//2:xc+Mx//2:ds_ratio].abs().visualize(rescale_factor=1, title = "|u + v|^2", flag_axis= True)
                plt.subplot(132)
                obs_points_sum[:, 1,..., yc-My//2:yc+My//2:ds_ratio, xc-Mx//2:xc+Mx//2:ds_ratio].abs().visualize(rescale_factor=1,title = "|u + e^(2j pi/3)v|^2", flag_axis= True)
                plt.subplot(133)
                obs_points_sum[:, 2,..., yc-My//2:yc+My//2:ds_ratio, xc-Mx//2:xc+Mx//2:ds_ratio].abs().visualize(rescale_factor=1,title = "|u + e^(4j pi/3)v|^2", flag_axis= True)
            
                plt.tight_layout()
                plt.savefig(f'{output_path}/obs_{physical_shift_id}.png')
                plt.show()

            print('dc and interf')
            dc_points_sum.data = torch.sum(
                    obs_points_sum.data,dim=1,keepdim=True
            ).detach()/3

            interf_points_sum.data = torch.sum(
                obs_points_sum.data* phasors[None,:,None,None,None,None],
                dim=1,keepdim=True
            ).detach()/3

            if save_fig:
                plt.figure(figsize=(15,5))
                plt.subplot(131)
                dc_points_sum[..., yc-My//2:yc+My//2:ds_ratio, xc-Mx//2:xc+Mx//2:ds_ratio].abs().visualize(rescale_factor=1,title = "dc", flag_axis= True)
                plt.subplot(132)
                interf_points_sum[...,yc-My//2:yc+My//2:ds_ratio, xc-Mx//2:xc+Mx//2:ds_ratio].abs().visualize(rescale_factor=1,title = "interf abs", flag_axis= True)
                plt.subplot(133)
                interf_points_sum[..., yc-My//2:yc+My//2:ds_ratio, xc-Mx//2:xc+Mx//2:ds_ratio].angle().visualize(rescale_factor=1,title = "interf phase", flag_axis= True, cmap='binary_r')
                plt.hsv()

                plt.tight_layout()
                plt.savefig(f'{output_path}/dcinterf_{physical_shift_id}.png')
                plt.show()


            # filtered out points has low intensity/ interfernce signal
            contrast = IntensityField(2*interf_points_sum.data/dc_points_sum.data *(dc_points_sum.data>0.1*(torch.max(dc_points_sum.data))) ) 

            if save_fig:
                plt.figure(figsize=(5,5))
                plt.tight_layout()
                contrast[..., yc-My//2:yc+My//2:ds_ratio, xc-Mx//2:xc+Mx//2:ds_ratio].abs().visualize(rescale_factor=1,title = "contrast", flag_axis= True, vmax=1, vmin=0)
                plt.savefig(f'{output_path}/contrast_{physical_shift_id}.png')

                plt.show()

            dc_all[:,:,physical_shift_id] = dc_points_sum[0,0,0,0, yc-My//2:yc+(My-My//2):ds_ratio, xc-Mx//2:xc+(Mx-Mx//2):ds_ratio].abs().data.squeeze().detach().cpu().numpy()
            interf_all[:,:,physical_shift_id] = interf_points_sum[0,0,0,0, yc-My//2:yc+(My-My//2):ds_ratio, xc-Mx//2:xc+(Mx-Mx//2):ds_ratio].abs().data.squeeze().detach().cpu().numpy()
            contrast_all[:,:,physical_shift_id] = contrast[0,0,0,0, yc-My//2:yc+(My-My//2):ds_ratio, xc-Mx//2:xc+(Mx-Mx//2):ds_ratio].abs().data.squeeze().detach().cpu().numpy()

            #print(interf_points_sum[..., yc, xc].data.squeeze().abs().detach().cpu().numpy())
            #print(interf_points_sum[..., yc, xc].data.squeeze().angle().detach().cpu().numpy())


            del obs_points_sum, dc_points_sum, interf_points_sum

    if save_fig:
        draw_gif(frame_folder=output_path)
        with open(f'{output_path}/data.npy', 'wb') as file:
            np.save(file,dc_all)
            np.save(file,interf_all)
            np.save(file,contrast_all)
            np.save(file,shift_range)

    return dc_all, interf_all

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dratio', help='diameter ratio', type=float, default=0.25)
    # parser.add_argument('--csize', help='coherence size', type=int, default=4)

    # args = parser.parse_args()

    # do one more time for camera 1 um
    device = torch.device("cuda:1")
    coherence_length = 16*um # see \detla_c in the paper
    diffraction_blur_kernel = 2*um  # see \detla_Phi in the paper
    camera_pitch = 0.75*um # see \detla_x in the paper


    coherence_size = int(coherence_length//dx)

    #In real world, waves from different postions are partially coherent with others
    #It can be simulated by multiple coherent blocks of waves overlapped with each other
    #To simulate real world behavior precisely, sample interveral should be smaller than diffraction_blur_kernel//dx
    #However, this dense sampling takes a long time to run
    #For some quick evaluation, one can set wave_sample_interval = coherence_size, but it will has inprecised contrast value
    wave_sample_interval = coherence_size #int(2*um//dx) 

    aperture_size = f*lam/diffraction_blur_kernel
    aperture_D_ratio = aperture_size/Nx/dfx 
    ds_ratio = int(camera_pitch//dx)

    mag_ratio = 1 # simulate the magnification effects of scanning length, fixed to be one in analysis

    # doing exp sm = 40um

    #aperture_D_ratio = 0.125 #0.125
    light_angle_spread_deg = 0 # 8.6, 2
    #coherence_size = 16 #pixels
    camera_view_angle = 15
    depth_scan_step = 5* diffraction_blur_kernel/um*coherence_length/mag_ratio/8 #um
    #depth_scan_step =  depth_scan_step*2  #7/30 temporaly double the depth scan step

    #depth_scan_step = 10*um  #1009 temporaly fix the depth scan step


    #shift_range = np.arange(-8,9)#np.arange(-8,9)#np.arange(-8,9)
    shift_range = np.arange(0,-9,-2)#np.arange(-8,9)#np.arange(-8,9)

    #ds_ratio = 4
    n_runs = 1

    H = int(np.ceil(n_points*dx/camera_pitch))

    dc_all_epochs = np.zeros((n_runs* H, int(round(Mx/ds_ratio)), len(shift_range))) 
    interf_all_epochs = np.zeros((n_runs* H, int(round(Mx/ds_ratio)), len(shift_range)))


    params_dict = {
        'object_pitch': dx,
        'wave_length': lam,
        'aperture_size': aperture_size,
        'focal_length': f,
        'numerical_aperture': f/aperture_size,
        'coherence_length': coherence_length, #coherence_size*dx,
        'diffraction_blur_kernel': diffraction_blur_kernel, #f/aperture_size*lam,
        'camera_pitch': camera_pitch, #dx* ds_ratio,
        'camera_view_angle': camera_view_angle,
        'depth_step': depth_scan_step,
        'mag_ratio': mag_ratio,
        'wave_sample_interval': wave_sample_interval
    }

    ts = time.gmtime()
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", ts)
    output_path = f'./outputs/{timestamp}_vangle_{camera_view_angle}_magratio_{mag_ratio}_dstep_{np.round(depth_scan_step/um)}_c_{coherence_length//um}_b_{diffraction_blur_kernel//um}_p_{camera_pitch//um}'

    if save_data:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    for epoch in range(n_runs):
        print(f'epoch = {epoch}')

        dc_all, interf_all = sim_self_interference(
            output_path = output_path,
            aperture_D_ratio=aperture_D_ratio,
            coherence_size=coherence_size, 
            wave_sample_interval=wave_sample_interval,
            light_angle_spread_deg=light_angle_spread_deg,
            camera_view_angle = camera_view_angle,
            depth_scan_step = depth_scan_step,
            shift_range = shift_range,
            ds_ratio=ds_ratio,
            mag_ratio=mag_ratio,
            device = device
        )

        dc_all_epochs[ H*epoch: H*epoch+ H, :, :] =  dc_all[(int(My/ds_ratio)//2) -H//2: (int(My/ds_ratio)//2) +(H-H//2),:,:]
        interf_all_epochs[ H*epoch: H*epoch+ H, :, :] =  interf_all[ (int(My/ds_ratio)//2) -H//2: (int(My/ds_ratio)//2) +(H-H//2),:,:]


    if save_data:
        with open(f'{output_path}/data.npy', 'wb') as file:
            np.save(file,dc_all_epochs)
            np.save(file,interf_all_epochs)
            np.save(file,params_dict)