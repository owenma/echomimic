import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

import av
import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic_acc import Audio2VideoPipeline
from src.utils.util import save_videos_grid, crop_and_pad
from src.models.face_locator import FaceLocator
from moviepy.editor import VideoFileClip, AudioFileClip
from facenet_pytorch import MTCNN

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print(
        'please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static')
elif ffmpeg_path not in os.getenv('PATH'):
    print('add ffmpeg to path')
    os.environ['PATH'] = f'{ffmpeg_path}:{os.environ["PATH"]}'


def select_face(det_bboxes, probs):
    ## max face from faces that the prob is above 0.8
    ## box: xyxy
    if det_bboxes is None or probs is None:
        return None
    filtered_bboxes = []
    for bbox_i in range(len(det_bboxes)):
        if probs[bbox_i] > 0.8:
            filtered_bboxes.append(det_bboxes[bbox_i])
    if len(filtered_bboxes) == 0:
        return None

    sorted_bboxes = sorted(filtered_bboxes, key=lambda x:(x[3]-x[1]) * (x[2] - x[0]), reverse=True)
    return sorted_bboxes[0]



def load_config(
        config='./configs/prompts/animation_acc.yaml',
):
    config = OmegaConf.load(config)
    return config


def load_models(
        config,
        device='cuda',
):
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    if device.__contains__("cuda") and not torch.cuda.is_available():
        device = "cpu"

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)

    ############# model_init started #############

    ## vae init
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    ## reference net init
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    ## denoising net init
    if os.path.exists(config.motion_module_path):
        ### stage1 + stage2
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device=device)
    else:
        ### only stage1
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
            }
        ).to(dtype=weight_dtype, device=device)
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False
    )

    ## face locator init
    face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )
    face_locator.load_state_dict(torch.load(config.face_locator_path))

    ### load audio processor params
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

    ### load face detector params
    face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709,
                          post_process=True, device=device)

    ############# model_init finished #############

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    pipe = Audio2VideoPipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        face_locator=face_locator,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)
    return {
        'pipe': pipe,
        'face_detector': face_detector,
        'weight_dtype': weight_dtype,
    }


def infer(
        models,
        config,
        width=512,
        height=512,
        video_length=1200,
        seed=32,
        facemusk_dilation_ratio=0.1,
        facecrop_dilation_ratio=0.5,
        context_frames=12,
        context_overlap=3,
        cfg=2.5,  # guidance_scale
        steps=30,
        sample_rate=16000,
        fps=24,
        ret_path="dhuman",

):
    pipe = models['pipe']
    face_detector = models['face_detector']
    weight_dtype = models['weight_dtype']

    date_str = datetime.now().strftime('%Y%m%d')
    time_str = datetime.now().strftime('%H%M%S')
    save_dir_name = f'{time_str}--seed_{seed}-{width}x{height}'
    save_dir = Path(f'{ret_path}/{date_str}/{save_dir_name}')
    save_dir.mkdir(exist_ok=True, parents=True)

    for ref_image_path in config["test_cases"].keys():
        for audio_path in config["test_cases"][ref_image_path]:

            if seed is not None and seed > -1:
                generator = torch.manual_seed(seed)
            else:
                generator = torch.manual_seed(random.randint(100, 1000000))

            # ref_name = Path(ref_image_path).stem
            # audio_name = Path(audio_path).stem
            final_fps = fps

            #### face musk prepare
            face_img = cv2.imread(ref_image_path)
            face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')

            det_bboxes, probs = face_detector.detect(face_img)
            select_bbox = select_face(det_bboxes, probs)
            if select_bbox is None:
                face_mask[:, :] = 255
            else:
                xyxy = select_bbox[:4]
                xyxy = np.round(xyxy).astype('int')
                rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
                r_pad = int((re - rb) * facemusk_dilation_ratio)
                c_pad = int((ce - cb) * facemusk_dilation_ratio)
                face_mask[rb - r_pad: re + r_pad, cb - c_pad: ce + c_pad] = 255

                #### face crop
                r_pad_crop = int((re - rb) * facecrop_dilation_ratio)
                c_pad_crop = int((ce - cb) * facecrop_dilation_ratio)
                crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]),
                             min(re + c_pad_crop, face_img.shape[0])]
                print(crop_rect)
                face_img, _ = crop_and_pad(face_img, crop_rect)
                face_mask, _ = crop_and_pad(face_mask, crop_rect)
                face_img, _  = cv2.resize(face_img, (width, height))
                face_mask, _  = cv2.resize(face_mask, (width, height))

            ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
            face_mask_tensor = torch.Tensor(face_mask).to(dtype=weight_dtype, device="cuda").unsqueeze(0).unsqueeze(
                0).unsqueeze(0) / 255.0

            video = pipe(
                ref_image_pil,
                audio_path,
                face_mask_tensor,
                width,
                height,
                video_length,
                steps,
                cfg,
                generator=generator,
                audio_sample_rate=sample_rate,
                context_frames=context_frames,
                fps=final_fps,
                context_overlap=context_overlap
            ).videos

        final_length = min(video.shape[2], face_mask_tensor.shape[2], video_length)
        face_tensor = video[:, :, :final_length, :, :].contiguous()

        video = torch.cat([face_tensor, face_mask_tensor[:, :, :final_length, :, :].detach().cpu()], dim=-1)
        path_video_grid = f'{save_dir}/{int(cfg)}_{time_str}.mp4'
        save_videos_grid(
            video,
            path_video_grid,
            n_rows=2,
            fps=final_fps,
        )

        from moviepy.editor import VideoFileClip, AudioFileClip
        video_clip = VideoFileClip(path_video_grid)
        audio_clip = AudioFileClip(audio_path)
        video_clip = video_clip.set_audio(audio_clip)
        path_video_withaudio = f'{save_dir}/{int(cfg)}_{time_str}_withaudio.mp4'
        video_clip.write_videofile(path_video_withaudio, codec="libx264", audio_codec="aac")
        print(path_video_withaudio)


if __name__ == "__main__":
    main()

