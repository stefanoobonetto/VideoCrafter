import os, sys, glob
import numpy as np
from collections import OrderedDict
from decord import VideoReader, cpu
import cv2

import torch
import torchvision
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler


def batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, **kwargs):
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            #prompts = N * T * [""]  ## if is_imgbatch=True
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict):
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    
    x_T = None
    batch_variants = []
    #batch_variants1, batch_variants2 = [], []
    for _ in range(n_samples):
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            x_T=x_T,
                                            **kwargs
                                            )
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage_2DAE(samples)
        batch_variants.append(batch_images)
    ## batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    return batch_variants


def get_filelist(data_dir, ext='*'):
    file_list = glob.glob(os.path.join(data_dir, '*.%s'%ext))
    file_list.sort()
    return file_list

def get_dirlist(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                list.append(m)
    list.sort()
    return list


def load_model_checkpoint(model, ckpt):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu")
        try:
            ## deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]]=state_dict['module'][key]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        except:
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=full_strict)
        return model
    load_checkpoint(model, ckpt, full_strict=True)
    print('>>> model checkpoint loaded.')
    return model


def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def load_video_batch(filepath_list, frame_stride, video_size=(256,256), video_frames=16):
    '''
    Notice about some special cases:
    1. video_frames=-1 means to take all the frames (with fs=1)
    2. when the total video frames is less than required, padding strategy will be used (repreated last frame)
    '''
    fps_list = []
    batch_tensor = []
    assert frame_stride > 0, "valid frame stride should be a positive interge!"
    for filepath in filepath_list:
        padding_num = 0
        vidreader = VideoReader(filepath, ctx=cpu(0), width=video_size[1], height=video_size[0])
        fps = vidreader.get_avg_fps()
        total_frames = len(vidreader)
        max_valid_frames = (total_frames-1) // frame_stride + 1
        if video_frames < 0:
            ## all frames are collected: fs=1 is a must
            required_frames = total_frames
            frame_stride = 1
        else:
            required_frames = video_frames
        query_frames = min(required_frames, max_valid_frames)
        frame_indices = [frame_stride*i for i in range(query_frames)]

        ## [t,h,w,c] -> [c,t,h,w]
        frames = vidreader.get_batch(frame_indices)
        frame_tensor = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        frame_tensor = (frame_tensor / 255. - 0.5) * 2
        if max_valid_frames < required_frames:
            padding_num = required_frames - max_valid_frames
            frame_tensor = torch.cat([frame_tensor, *([frame_tensor[:,-1:,:,:]]*padding_num)], dim=1)
            print(f'{os.path.split(filepath)[1]} is not long enough: {padding_num} frames padded.')
        batch_tensor.append(frame_tensor)
        sample_fps = int(fps/frame_stride)
        fps_list.append(sample_fps)
    
    return torch.stack(batch_tensor, dim=0)

from PIL import Image
def load_image_batch(filepath_list, image_size=(256,256)):
    batch_tensor = []
    for filepath in filepath_list:
        _, filename = os.path.split(filepath)
        _, ext = os.path.splitext(filename)
        if ext == '.mp4':
            vidreader = VideoReader(filepath, ctx=cpu(0), width=image_size[1], height=image_size[0])
            frame = vidreader.get_batch([0])
            img_tensor = torch.tensor(frame.asnumpy()).squeeze(0).permute(2, 0, 1).float()
        elif ext == '.png' or ext == '.jpg':
            img = Image.open(filepath).convert("RGB")
            rgb_img = np.array(img, np.float32)
            #bgr_img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            #bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (image_size[1],image_size[0]), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        else:
            print(f'ERROR: <{ext}> image loading only support format: [mp4], [png], [jpg]')
            raise NotImplementedError
        img_tensor = (img_tensor / 255. - 0.5) * 2
        batch_tensor.append(img_tensor)
    return torch.stack(batch_tensor, dim=0)

def save_videos(batch_tensors, savedir, filenames, fps=10):
<<<<<<< HEAD
    """
    Save batch video tensors as individual MP4 files.
    Args:
        batch_tensors: Tensor of shape [B, C, T, H, W]
        savedir: Directory to save the videos
        filenames: List of filenames to save videos as
        fps: Frames per second for the videos
    """
    os.makedirs(savedir, exist_ok=True)
    
    # Iterate over each batch
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()  # Move to CPU
        video = torch.clamp(video.float(), -1.0, 1.0)  # Clamp values to [-1, 1]
        
        # Convert video tensor to [T, H, W, C] format
        video = video.permute(2, 3, 4, 1)  # [B, C, T, H, W] -> [T, H, W, C]
        video = (video + 1.0) / 2.0  # Normalize to [0, 1]
        video = (video * 255).to(torch.uint8)  # Scale to [0, 255] for uint8
        
        # Save video to disk
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        print(f"Saving video to {savepath} with shape {video.shape} and fps {fps}")
        
        try:
            torchvision.io.write_video(savepath, video, fps=fps, video_codec='h264', options={'crf': '10'})
        except Exception as e:
            print(f"Failed to save video {savepath}: {e}")
=======
    os.makedirs(savedir, exist_ok=True)

    fps = int(fps)  # Ensure FPS is an integer

    if batch_tensors.dim() == 4:  # Handle single video without batch dimension
        batch_tensors = batch_tensors.unsqueeze(0)  # Add batch dimension

    for idx, vid_tensor in enumerate(batch_tensors):
        print(f"Processing video {idx} with shape {vid_tensor.shape}")
        
        video = vid_tensor.detach().cpu()  # Move to CPU
        print(f"Before permute: {video.shape}")

        # Validate the tensor shape before permute
        if video.dim() != 4 or video.size(0) != 3:
            raise ValueError(f"Invalid tensor shape before permute: {video.shape}. Expected [C, T, H, W].")

        video = video.permute(1, 2, 3, 0)  # [C, T, H, W] -> [T, H, W, C]
        print(f"After permute: {video.shape}")

        video = torch.clamp(video.float(), -1.0, 1.0)  # Clamp values to [-1, 1]
        video = (video + 1.0) / 2.0  # Normalize to [0, 1]
        video = (video * 255).to(torch.uint8)  # Scale to [0, 255]

        # Validate tensor shape after permute
        assert video.dim() == 4, f"Expected 4D tensor [T, H, W, C], got {video.dim()} dimensions"
        assert video.size(-1) == 3, f"Expected 3 channels (RGB), got {video.size(-1)} channels"

        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        print(f"Saving video to {savepath} with shape {video.shape} and fps {fps}")

        try:
            torchvision.io.write_video(savepath, video, fps=fps, video_codec='h264')
        except Exception as e:
            print(f"Failed to save video {savepath}: {e}")
>>>>>>> 391a91b29915ea9555f5e7fe5ebfa3e61a8e6bc6
