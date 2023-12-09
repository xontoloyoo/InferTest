import os
import sys
from infer.modules.uvr5.modules import uvr
import shutil

shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)

weight_uvr5_root = os.getenv("weight_uvr5_root")

uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))
