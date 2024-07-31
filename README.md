# Audio-Denoising-AutoEncoder
This repository contains an implementation of an Audio-Denoising AutoEncoder based on the Noisier2Noise approach described in the paper [Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/abs/1910.11908).
## Requirements
- Python >= 3.0, <= 3.10
- Audio dataset of 'pure noise'
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
Make a file name ```.env``` in the project directory and set it up as follows:
```dosini
# Keep this 2046 for the best training option  
N_FFT=2046  
HOP_LENGTH=512  
# The number of segements the model sees in one iteration  
# If you encounter out-of-memory error of your processor, decrease this number  
# If training is way too slow, increase this number  
BATCH_SIZE=4
```
## Training models
```
python train.py
Enter the path of the directory that contains audio files of noise: Noise
Epochs: 30
```
The models will be saved to ```models/model_(epoch).pth``` on each epoch.
## Denoise audio files
```
python denoise.py
Enter the path of the model: models/model_30.pth
Enter the path of the directory that contains audio files to denoise: Noisy_Audio
Enter the output path: Denoised_Audio
```
The denoised audio files will be saved as same names.
## References
- [Speech Denoising Without Clean Training Data: A Noise2Noise Approach](https://arxiv.org/abs/2104.03838)
- [Noisier2Noise: Learning to Denoise from Unpaired Noisy Data](https://arxiv.org/abs/1910.11908)