import os
import json
import hashlib
import librosa
import soundfile as sf
import subprocess
import shlex
import argparse
from src.mdx import run_mdx  # pastikan ini sudah diimpor dengan benar

BASE_DIR = os.getcwd()
mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

def get_hash(filepath):
    """Generate a unique hash for the song file."""
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()[:11]

def convert_to_stereo(audio_path):
    """Ensure the input file is in stereo format."""
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)
    if len(wave.shape) == 1:  # Check if mono
        stereo_path = f'{os.path.splitext(audio_path)[0]}_stereo.wav'
        command = shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 "{stereo_path}"')
        subprocess.run(command)
        return stereo_path
    return audio_path

def load_model_params():
    """Load model parameters from the 'model_data.json' file."""
    params_path = os.path.join(mdxnet_models_dir, 'model_data.json')  # Load model parameters from this file
    with open(params_path, 'r') as infile:
        mdx_model_params = json.load(infile)
    return mdx_model_params

def preprocess_song(song_input, output_format='mp3'):
    """Main processing pipeline: Separate vocals, instrumentals, and remove reverb from vocals."""
    # Generate a unique ID for the song based on the file hash
    song_id = get_hash(song_input)
    song_output_dir = os.path.join(output_dir, song_id)

    if not os.path.exists(song_output_dir):
        os.makedirs(song_output_dir)

    # Convert to stereo if needed
    print("[INFO] Checking and converting input to stereo if needed...")
    song_input = convert_to_stereo(song_input)

    # Load model parameters
    print("[INFO] Loading model parameters...")
    model_params = load_model_params()

    # Step 1: Separate Vocals and Instrumentals
    print("[INFO] Separating vocals and instrumentals...")
    vocals_path, instrumentals_path = run_mdx(
        model_params=model_params,  # Mengirimkan model_params yang benar
        output_dir=song_output_dir,
        model_path=os.path.join(mdxnet_models_dir, 'UVR-MDX-NET-Voc_FT.onnx'),
        filename=song_input,  # Mengirimkan path file audio input
        suffix='Vocals',  # Label untuk vokal
        invert_suffix='Instrumental',  # Label untuk instrumental
        denoise=True  # Optional, applies denoising if available
    )

    # Step 2: Remove Reverb from Main Vocals
    print("[INFO] Removing reverb from vocals...")
    _, main_vocals_dereverb_path = run_mdx(
        model_params=model_params,  # Mengirimkan model_params yang benar
        output_dir=song_output_dir,
        model_path=os.path.join(mdxnet_models_dir, 'Reverb467.onnx'),
        filename=vocals_path,  # Mengirimkan path file vokal untuk penghilangan reverb
        invert_suffix='DeReverb',
        #exclude_main=True,
        denoise=True  # Optional, applies denoising if available
    )

    # Step 3: Save Output in Chosen Format
    if output_format == 'mp3':
        print(f"[INFO] Converting outputs to {output_format}...")
        vocals_mp3 = f"{os.path.splitext(vocals_path)[0]}.{output_format}"
        instrumentals_mp3 = f"{os.path.splitext(instrumentals_path)[0]}.{output_format}"
        dereverb_mp3 = f"{os.path.splitext(main_vocals_dereverb_path)[0]}.{output_format}"

        for input_file, output_file in [(vocals_path, vocals_mp3),
                                        (instrumentals_path, instrumentals_mp3),
                                        (main_vocals_dereverb_path, dereverb_mp3)]:
            command = shlex.split(f'ffmpeg -y -loglevel error -i "{input_file}" "{output_file}"')
            subprocess.run(command)

        print(f"[INFO] Outputs saved as MP3 files in: {song_output_dir}")
    else:
        print(f"[INFO] Outputs saved as WAV files in: {song_output_dir}")

    return vocals_path, instrumentals_path, main_vocals_dereverb_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Separate vocals, instrumentals, and remove reverb from vocals from a local audio file.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the local audio file (MP3/WAV).')
    parser.add_argument('-oformat', '--output-format', type=str, choices=['mp3', 'wav'], default='mp3', help='Output format for separated files.')
    args = parser.parse_args()

    vocals, instrumentals, main_vocals_dereverb = preprocess_song(args.input, output_format=args.output_format)
    print(f"[SUCCESS] Processing completed! Separated files are saved in the output directory.")
