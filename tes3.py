import os
import sys

from pathlib import Path
from infer.modules.vc.modules import VC
from infer.modules.uvr5.modules import uvr
from configs.config import Config
from sklearn.cluster import MiniBatchKMeans
#from dotenv import load_dotenv
import torch
import numpy as np
import gradio as gr
import faiss
import fairseq
import pathlib
import json
#from time import sleep
from subprocess import Popen
#from random import shuffle
#import warnings
#import traceback
#import threading
#import shutil
#import logging

load_dotenv()
config = Config()
vc = VC(config)

if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

BASE_DIR = Path.cwd()
now_dir = os.getcwd()
#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
weight_uvr5 = ('mdxnet_models')

uvr5_names = []
for name in os.listdir(weight_uvr5):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

with gr.Blocks(title="Nih Cuy") as app:
    gr.Markdown("## GassKeun")
    gr.Markdown(
        value=(
            "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
        )
    )

    with gr.TabItem(("伴奏人声分离&去混响&去回声")):
        with gr.Group():
            gr.Markdown(
                value=(
                    "人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>  (1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。"
                )
            )
            
            with gr.Row():
                with gr.Column():
                    dir_wav_input = gr.Textbox(
                        label=("输入待处理音频文件夹路径"),
                        placeholder="C:\\Users\\Desktop\\todo-songs",
                    )
                    wav_inputs = gr.File(
                        file_count="multiple", label=("也可批量输入音频文件, 二选一, 优先读文件夹")
                    )
                with gr.Column():
                    model_choose = gr.Dropdown(label=("模型"), choices=uvr5_names)
                    agg = gr.Slider(
                        minimum=0,
                        maximum=20,
                        step=1,
                        label="人声提取激进程度",
                        value=10,
                        interactive=True,
                        visible=False,  # 先不开放调整
                    )
                    opt_vocal_root = gr.Textbox(
                        label=("指定输出主人声文件夹"), value="opt"
                    )
                    opt_ins_root = gr.Textbox(
                        label=("指定输出非主人声文件夹"), value="opt"
                    )
                    format0 = gr.Radio(
                        label=("导出文件格式"),
                        choices=["wav", "flac", "mp3", "m4a"],
                        value="flac",
                        interactive=True,
                    )

            but2 = gr.Button(("转换"), variant="primary")
            vc_output4 = gr.Textbox(label=("输出信息"))
            but2.click(
                uvr,
                [
                    model_choose,
                    dir_wav_input,
                    opt_vocal_root,
                    wav_inputs,
                    opt_ins_root,
                    agg,
                    format0,
                ],
                [vc_output4],
                api_name="uvr_convert",
            )
# Lanjutkan dengan menjalankan antarmuka pengguna
if config.iscolab:
    app.queue(concurrency_count=511, max_size=1022).launch(share=True)
else:
    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        inbrowser=not config.noautoopen,
        server_port=config.listen_port,
        quiet=True,
    )
