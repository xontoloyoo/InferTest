import os
import sys
import gradio as gr
import json
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
from infer.modules.uvr5.modules import uvr
import shutil

shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)

weight_uvr5_root = os.getenv("weight_uvr5_root")

uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))
        
with gr.Blocks(title="R") as app:
    with gr.TabItem(i18n("伴奏人声分离&去混响&去回声")):
        with gr.Group():
            gr.Markdown("## RVC WebUI")  # Perbaiki spasi ekstra di sini
            gr.Markdown(
                value=i18n(
                    "人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>  (1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。"
                )
            )
            with gr.Row():
                with gr.Column():
                    dir_wav_input = gr.Textbox(
                        label=i18n("输入待处理音频文件夹路径"),
                        placeholder="C:\\Users\\Desktop\\todo-songs",
                    )
                    wav_inputs = gr.File(
                        file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹")
                    )
                with gr.Column():
                    model_choose = gr.Dropdown(label=i18n("模型"), choices=uvr5_names)
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
                        label=i18n("指定输出主人声文件夹"), value="opt"
                    )
                    opt_ins_root = gr.Textbox(
                        label=i18n("指定输出非主人声文件夹"), value="opt"
                    )
                    format0 = gr.Radio(
                        label=i18n("导出文件格式"),
                        choices=["wav", "flac", "mp3", "m4a"],
                        value="flac",
                        interactive=True,
                    )
            but2 = gr.Button(i18n("转换"), variant="primary")
            vc_output4 = gr.Textbox(label=i18n("输出信息"))
            but2.click(
                uvr.uvr_convert,  # Perbaiki pemanggilan fungsi uvr_convert
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
    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
