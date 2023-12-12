from mmldm.models.DAC.dac_model import __model_version__, load_model

import librosa
import torch
import soundfile


# Init an empty model
#model = DAC()

# Load compatible pre-trained model
model = load_model(__model_version__)
model.eval()

'''
default sample rate: 44100
compression rate: 512
latent dim: 1024
param: 76651890
latent shape: test 1: (1024, 690)
              test 2: (1024, 776)
              test 3: (1024, 690)
'''


sample_rate = 44100

audio, sr = librosa.load('../../000369.wav', sr=sample_rate)

audio = torch.FloatTensor(audio)
audio = audio[None, None, ...]

with torch.no_grad():
    out_dict = model.encode(audio, sample_rate=sample_rate)

    recon_dict = model.decode(out_dict['z'], length=out_dict['length'])
soundfile.write('../../recon.wav', recon_dict['audio'].squeeze().detach().numpy(), samplerate=sample_rate)
'''
# Load audio signal file
signal = AudioSignal('000190.wav')

# Encode audio signal
encoded_out = encode(signal, 'cuda', model)

# Decode audio signal
recon = decode(encoded_out, 'cuda', model, preserve_sample_rate=True)

# Write to file
recon.write('recon.wav')
'''