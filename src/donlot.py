from pathlib import Path
import requests

#donlot_model = 'https://huggingface.co/datasets/xontoloyoo/mymodel/resolve/main/Model-Training/'
donlot_rh = 'https://huggingface.co/datasets/xontoloyoo/mymodel/resolve/main/'
pisah = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
#te = 'https://github.com/CNChTu/FCPE/tree/main/torchfcpe/assets/'
BASE_DIR = Path.cwd()

# Daftar tupel dengan nama model dan direktori tujuan
download_targets = [
    ('hubert_base.pt', 'rvc_models', 'donlot_rh'),
    ('fcpe.pt', 'rvc_models', 'donlot_rh'),
    ('rmvpe.pt', 'rvc_models', 'donlot_rh'),
    ('UVR-MDX-NET-Voc_FT.onnx', 'mdxnet_models', 'pisah'),
    ('UVR_MDXNET_KARA_2.onnx', 'mdxnet_models', 'pisah'),
    ('Reverb_HQ_By_FoxJoy.onnx', 'mdxnet_models', 'pisah'),
    ('Kim_Vocal_2.onnx', 'mdxnet_models', 'pisah'),
]

# Fungsi untuk mengunduh model ke direktori yang sesuai
def dl_model(link, model_name, dir_name):
    if not dir_name.exists():
        dir_name.mkdir(parents=True)

    with requests.get(f'{link}{model_name}') as r:
        r.raise_for_status()
        with open(dir_name / model_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

if __name__ == '__main__':
    for model, dir_path, source_url in download_targets:
        print(f'Downloading {model}...')
        dl_model(globals()[source_url], model, BASE_DIR / dir_path)

    print('All models downloaded!')
