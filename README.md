# Audio-Denoising-AutoEncoder
This repository implements noise reduction models using deep learning techniques, including DCUnet and DAAE.
## Models
### 1. DCUnet
- Status: Implemented
### 2. DAAE(Denoising Adversarial AutoEncoder)
- Status: Work in Progress
## Requirements
- Python >= 3.0, <= 3.10
- Audio dataset of 'pure noise'
- Clean audio dataset
- Audio files to denoise
## Installation
1. Clone this repository
```sh
git clone https://github.com/ludwig7959/Audio-Denoising-AutoEncoder.git
```
2. Install required packages
```sh
pip install -r requirements.txt
```
3. Set up environment variables
Rename the ```.env.template``` file to ```.env``` for the initial setup. For detailed explanations of each environment variable, please refer to each section.
## Common options
```dotenv
DEVICE=cuda
N_FFT=2046
HOP_LENGTH=512
SAMPLING_RATE=16000
MODEL_TYPE=DCUNET
```
- DEVICE
  - Description: Specifies the device to be used for training or inference. cuda is used to leverage an NVIDIA GPU, which can significantly speed up training.
  - Possible options:
    - cpu: Use CPU for computations (if no GPU is available).
    - cuda: Use NVIDIA GPU for computations.
    - mps: Use Apple’s Metal Performance Shaders (MPS) backend for computations on Apple Silicon devices (such as the M1 or M2 chips).
- N_FFT
  - Description: Defines the size of the FFT (Fast Fourier Transform) window used for converting audio signals to the frequency domain. This parameter is crucial for analyzing the frequency components of the audio. Keeping this value at 2046 is recommended for optimal training performance.
- HOP_LENGTH
  - Description: Determines the number of audio samples between successive FFT windows. It controls the overlap between FFT frames and is usually set to 1/4 of N_FFT. This value is important for balancing the time and frequency resolution in audio analysis.
- SAMPLING_RATE
  - Description: Sets the sampling rate for audio processing, meaning the number of audio samples per second. A rate of 16,000 Hz is commonly used for speech processing and ensures that the audio data is sampled at a rate appropriate for human voice frequencies.
- MODEL_TYPE
  - Description: Specifies the type of model to use.
  - Possible options
    - DCUNET
    - DAAE (Not available yet)
## Preprocess data
1. Prepare clean and noise audio
2. Perform common preprocessing
```dotenv
PREPROCESSING_INPUT_PATH=noise_raw
PREPROCESSING_OUTPUT_PATH=noise_processed
TARGET_RMS=0.1
```
- TARGET_RMS
  - Description: Defines the target Root Mean Square (RMS) level for the audio files. This ensures uniformity in volume across all processed files.
```shell
python preprocess.py
```
3. Concatenate clean audio
```dotenv
CONCATENATING_INPUT_PATH=segments
CONCATENATING_OUTPUT_PATH=concatenated
CONCATENATING_COUNT=100
```
- CONCATENATING_COUNT
  - Description: Defines the number of concatenated audio files to be generated. Setting this value to 100 means that the script will create 100 output files, each composed of randomly selected and concatenated audio segments from the input directory.
```shell
python concatenate.py
```
4. Pad noise audio
```dotenv
PADDING_INPUT_PATH=noise_unprocessed
PADDING_OUTPUT_PATH=noise_processed
```
```shell
python pad_noise.py
```
5. Synthesize noise audio and clean audio to create noisy audio
```dotenv
SYNTHESIZING_CLEAN_PATH=clean
SYNTHESIZING_NOISE_PATH=noise
SYNTHESIZING_OUTPUT_INPUT=input
SYNTHESIZING_OUTPUT_TARGET=target
NOISES_PER_CLEAN=20
```
- SYNTHESIZING_OUTPUT_INPUT
  - Description: Path where synthesized noisy audio files will be saved. They will be used as inputs for deep learning models.
- SYNTHESIZING_OUTPUT_TARGET
  - Description: Path where target audio files(clean) will be saved.
```shell
python synthesize.py
```
## Train the model
```
python train.py
```
The models will be saved to ```models/model_(epoch).pth``` on each epoch.
## Denoise audio files
```
python denoise.py
```
The denoised audio files will be saved as same names.
## References
- [Speech Denoising Without Clean Training Data: A Noise2Noise Approach](https://arxiv.org/abs/2104.03838)
- [Noisier2Noise: Learning to Denoise from Unpaired Noisy Data](https://arxiv.org/abs/1910.11908)