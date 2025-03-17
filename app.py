#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project App for EchoMimic
@File    app.py
@Author  mengrang.mr
@Date    2024/7/18 15:49
'''

import os
import random
from datetime import datetime
from pathlib import Path
os.system('pip install modelscope==1.16.1')

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from src.pipelines.pipeline_echo_mimic_pose_acc import AudioPose2VideoPipeline

from src.utils.util import save_videos_grid, crop_and_pad
from src.utils.img_utils import pil_to_cv2, cv2_to_pil, center_crop_cv2, pils_from_video, save_videos_from_pils, save_video_from_cv2_list

from src.models.face_locator import FaceLocator
from moviepy.editor import VideoFileClip, AudioFileClip
from facenet_pytorch import MTCNN
import argparse
import os.path as osp
import gradio as gr
from modelscope import snapshot_download

import pickle
from src.utils.draw_utils import FaceMeshVisualizer
from src.utils.motion_utils import motion_sync
from src.utils.mp_utils  import LMKExtractor

default_values = {
    "width": 512,
    "height": 512,
    "length": 240,
    "seed": 420,
    "facemask_dilation_ratio": 0.1,
    "facecrop_dilation_ratio": 0.5,
    "context_frames": 12,
    "context_overlap": 3,
    "cfg": 1.0,
    "steps": 6,
    "sample_rate": 16000,
    "fps": 24,
    "device": "cuda"
}

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"


config_path = "./configs/prompts/animation_pose_acc.yaml"
config = OmegaConf.load(config_path)
if config.weight_dtype == "fp16":
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32

device = "cuda"
if not torch.cuda.is_available():
    device = "cpu"

inference_config_path = config.inference_config
infer_config = OmegaConf.load(inference_config_path)

############# model_init started #############
## vae init
vae_dir =snapshot_download("zhuzhukeji/sd-vae-ft-mse")
vae = AutoencoderKL.from_pretrained(vae_dir).to("cuda", dtype=weight_dtype)

## reference net init
base_model_path=snapshot_download("gqy2468/sd-image-variations-diffusers")
reference_unet = UNet2DConditionModel.from_pretrained(
    base_model_path,
    subfolder="unet",
).to(dtype=weight_dtype, device=device)

os.system('modelscope download --model=BadToBest/EchoMimic --local_dir ./pretrained_weights')


reference_unet.load_state_dict(torch.load(config.reference_unet_path, map_location="cpu"))

## denoising net init
if os.path.exists(config.motion_module_path):
    ### stage1 + stage2
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)
else:
    ### only stage1
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
            "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
        }
    ).to(dtype=weight_dtype, device=device)

denoising_unet.load_state_dict(torch.load(config.denoising_unet_path, map_location="cpu"), strict=False)

## face locator init
face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(dtype=weight_dtype, device="cuda")
face_locator.load_state_dict(torch.load(config.face_locator_path))

## load audio processor params
audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

## load face detector params
face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

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
).to("cuda", dtype=weight_dtype)


def delete_pipeline():
    print("delete_pipeline ...")
    torch.cuda.empty_cache()

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

    
lmk_extractor = LMKExtractor()
def process_video(uploaded_img, uploaded_audio, width, height, length, facemask_dilation_ratio, facecrop_dilation_ratio, context_frames, context_overlap, cfg, steps, sample_rate, fps, device):
    #### face musk prepare
    face_img = cv2.imread(uploaded_img)
    face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')
    det_bboxes, probs = face_detector.detect(face_img)
    select_bbox = select_face(det_bboxes, probs)
    if select_bbox is None:
        face_mask[:, :] = 255
    else:
        xyxy = select_bbox[:4]
        xyxy = np.round(xyxy).astype('int')
        rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
        r_pad = int((re - rb) * facemask_dilation_ratio)
        c_pad = int((ce - cb) * facemask_dilation_ratio)
        face_mask[rb - r_pad : re + r_pad, cb - c_pad : ce + c_pad] = 255
        
        #### face crop
        r_pad_crop = int((re - rb) * facecrop_dilation_ratio)
        c_pad_crop = int((ce - cb) * facecrop_dilation_ratio)
        crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]), min(re + r_pad_crop, face_img.shape[0])]
        face_img = crop_and_pad(face_img, crop_rect)
        face_mask = crop_and_pad(face_mask, crop_rect)
        face_img = cv2.resize(face_img, (width, height))
        face_mask = cv2.resize(face_mask, (width, height))


    # ==================== face_locator =====================
    '''
    driver_video = "./assets/driven_videos/c.mp4"

    input_frames_cv2 = [cv2.resize(center_crop_cv2(pil_to_cv2(i)), (512, 512)) for i in pils_from_video(driver_video)]
    ref_det = lmk_extractor(face_img)

    visualizer = FaceMeshVisualizer(draw_iris=False, draw_mouse=False)
    
    pose_list = []
    sequence_driver_det = []
    try: 
        for frame in input_frames_cv2:
            result = lmk_extractor(frame)
            assert result is not None, "{}, bad video, face not detected".format(driver_video)
            sequence_driver_det.append(result)
    except:
        print("face detection failed")
        exit()
    
    sequence_det_ms = motion_sync(sequence_driver_det, ref_det)
    for p in sequence_det_ms:
        tgt_musk = visualizer.draw_landmarks((width, height), p)
        tgt_musk_pil = Image.fromarray(np.array(tgt_musk).astype(np.uint8)).convert('RGB')
        pose_list.append(torch.Tensor(np.array(tgt_musk_pil)).to(dtype=weight_dtype, device="cuda").permute(2,0,1) / 255.0)
    '''
    # face_mask_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
    face_mask_tensor = torch.Tensor(face_mask).to(dtype=weight_dtype, device="cuda").unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0
    
    ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
    
    #del pose_list, sequence_det_ms, sequence_driver_det, input_frames_cv2

    video = pipe(
        ref_image_pil,
        uploaded_audio,
        face_mask_tensor,
        width,
        height,
        length,
        steps,
        cfg,
        #generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=context_frames,
        fps=fps,
        context_overlap=context_overlap
    ).videos

    save_dir = Path("output/tmp")
    save_dir.mkdir(exist_ok=True, parents=True)
    output_video_path = save_dir / "output_video.mp4"
    save_videos_grid(video, str(output_video_path), n_rows=1, fps=fps)

    video_clip = VideoFileClip(str(output_video_path))
    audio_clip = AudioFileClip(uploaded_audio)
    final_output_path = save_dir / "output_video_with_audio.mp4"
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(str(final_output_path), codec="libx264", audio_codec="aac")

    return final_output_path

"""
example dirs
"""
example_portrait_dir = "assets/test_imgs"
example_video_dir = "assets/driven_videos"
example_audio_dir = "assets/test_audios"

data_examples = [
    [osp.join(example_portrait_dir, "a.png"), osp.join(example_audio_dir, "echomimic.wav")],
    [osp.join(example_portrait_dir, "b.png"), osp.join(example_audio_dir, "echomimic_girl.wav")],
    [osp.join(example_portrait_dir, "a.png"), osp.join(example_audio_dir, "echomimic_en.wav")],
    [osp.join(example_portrait_dir, "b.png"), osp.join(example_audio_dir, "echomimic_en_girl.wav")],
]
output_image = gr.Image(type="numpy")
output_image_paste_back = gr.Image(type="numpy")
with gr.Blocks() as demo:
    gr.Markdown('# Demo for EchoMimic')
    gr.HTML("""
    <div style="display:flex;column-gap:4px;">
        <a href='https://badtobest.github.io/echomimic.html'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
        <a href='https://huggingface.co/BadToBest/EchoMimic'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
        <a href='https://www.modelscope.cn/models/BadToBest/EchoMimic'><img src='https://img.shields.io/badge/ModelScope-Model-purple'></a>
        <a href='https://www.modelscope.cn/studios/BadToBest/BadToBest'><img src='https://img.shields.io/badge/ModelScope-Demo-purple'></a>
        <a href='https://arxiv.org/abs/2407.08136'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
        <a href='https://github.com/BadToBest/EchoMimic/blob/main/assets/echomimic.png'><img src='https://badges.aleen42.com/src/wechat.svg'></a>
    </div>
    """)
    with gr.Row():
        gr.Markdown("## 1. Load Reference Image")
        gr.Markdown("## 2. Load Audio")
        gr.Markdown("## 3. Result")

    with gr.Row():
        with gr.Column(min_width=250):
            with gr.Accordion(open=True, label="Reference Image"):
                uploaded_img = gr.Image(type="filepath")
        with gr.Column(min_width=250):
            with gr.Accordion(open=True, label="Input Audio"):
                uploaded_audio = gr.Audio(type="filepath")
        with gr.Column(min_width=250):
            with gr.Row():
                    generate_button = gr.Button("🚀 Generate Video", variant="primary")

            with gr.Row():
                with gr.Accordion(open=True, label="Result"):
                    output_video = gr.Video()
            
    with gr.Row():
        # Examples
        gr.Markdown("## You could choose the examples below ⬇️")
    with gr.Row():
        gr.Examples(
            examples=data_examples,
            inputs=[
                uploaded_img,
                uploaded_audio,
            ],
            examples_per_page=2,
            cache_examples=False,
        )
    
    def generate_video(uploaded_img, uploaded_audio,
                       facemask_dilation_ratio=default_values["facemask_dilation_ratio"],
                       facecrop_dilation_ratio=default_values["facecrop_dilation_ratio"],
                       context_frames=default_values["context_frames"],
                       context_overlap=default_values["context_overlap"],
                       cfg=default_values["cfg"],
                       steps=default_values["steps"],
                       sample_rate=default_values["sample_rate"],
                       fps=default_values["fps"],
                       device=default_values["device"],
                       width=default_values["width"],
                       height=default_values["height"],
                       length=default_values["length"] ):

        final_output_path = process_video(
            uploaded_img, 
            uploaded_audio, width, height, 
            length, facemask_dilation_ratio, 
            facecrop_dilation_ratio, context_frames, 
            context_overlap, cfg, steps, 
            sample_rate, fps, device
        )        
        output_video = final_output_path
        return final_output_path

    generate_button.click(
        generate_video,
        inputs=[
            uploaded_img,
            uploaded_audio
        ],
        outputs=output_video,
        show_progress=True
    )
parser = argparse.ArgumentParser(description='EchoMimic')
parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name')
parser.add_argument('--server_port', type=int, default=7680, help='Server port')
args = parser.parse_args()


if __name__ == '__main__':
    #demo.launch(server_name=args.server_name, server_port=args.server_port, inbrowser=True, share=True, inline=True)
    demo.queue(default_concurrency_limit=1).launch(
    server_port=args.server_port,
    share=True,
    server_name=args.server_name
)