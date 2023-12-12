import torch
import codecs as cs
from os.path import join as pjoin
import librosa
from torch.nn.functional import mse_loss

from mmldm.models.DAC.dac_model import __model_version__, load_model
from mmldm.modules.meta_vqvae_module import MetaVQVAE


#meta_dir = 'D:/local_documentation/deep_learning/ChuangGan/fma_medium'
#music_dir = 'D:/local_documentation/deep_learning/ChuangGan/fma_medium'
music_dir = "/nobackup/users/yiningh/yh/fma/data/fma_medium"
meta_dir = "/nobackup/users/yiningh/yh/fma/data/fma_medium"
device = torch.device('cuda')
#meta_vqvae_ckpt = 'C:\\Users\\Mahlering\\.cache\\huggingface\\hub\\models--facebook--musicgen-small\\snapshots\\bf842007f70cf1b174daa4f42b5e45a21d59d3ec\\compression_state_dict.bin'
#meta_vqvae_ckpt = '/nobackup/users/yiningh/yh/music_dance/weight/musicgen_vqvae.bin'
meta_vqvae_ckpt = 'tempt.bin'
#meta_vqvae_fine_tuned = '/nobackup/users/yiningh/yh/music_dance/submission/music_piano_vqvae_logs/2023-06-22T10-51-55_meta_vqvae/checkpoints/epoch=000001.ckpt'
# e1: 0.01371  e2: 0.01395  e3:

# prepare data
music_ignore = []
music_data = []

with cs.open(pjoin(meta_dir, 'fma_medium_ignore.txt'), "r") as f:
    for line in f.readlines():
        music_ignore.append(line.strip())

with cs.open(pjoin(meta_dir, 'fma_medium_test.txt'), "r") as f:
    for line in f.readlines():
        if line.strip() in music_ignore:
            continue
        music_data.append(line.strip())

print('total data number: ', len(music_data))


# load model
dac = load_model(__model_version__)
dac.eval().to(device)


meta = MetaVQVAE(
    pretrained_ckpt=meta_vqvae_ckpt
).eval().to(device)

'''
if meta_vqvae_fine_tuned is not None:
    ckpt = torch.load(meta_vqvae_fine_tuned)
    model_dict = meta.state_dict()
    #ckpt['state_dict'] = dict(ckpt['state_dict'])
    pretrained_dict = {k: v for k, v in (ckpt['state_dict']).items() if k in model_dict}

    meta.load_state_dict(pretrained_dict)

    ckpt_to_save = meta.vqvae.state_dict()
    pkg = torch.load(meta_vqvae_ckpt, map_location='cpu')
    old_dict = pkg['best_state']
    print(type(ckpt_to_save), type(old_dict))
    for k, v in old_dict.items():
        if k in ckpt_to_save:
            print('in new weight: ', k)
            old_dict[k] = ckpt_to_save[k]
        else:
            print('not in new weight: ', k)
    torch.save(pkg, './tempt.bin')

    print('load model from tempt.bin')
    meta = MetaVQVAE(
        pretrained_ckpt='./tempt.bin'
    ).eval().to(device)
  '''


# traverse the data
meta_loss = []
dac_loss = []


for i, music_id in enumerate(music_data):
    music_path = pjoin(music_dir, music_id + ".wav")
    with torch.no_grad():
        # first MusicGen
        waveform, sr = librosa.load(music_path, sr=32000)
        waveform = torch.FloatTensor(waveform)[None, None, ...].to(device)

        recon_meta = meta(waveform)

        loss_meta = mse_loss(recon_meta, waveform)
        meta_loss.append(loss_meta.item())

        '''
        waveform, sr = librosa.load(music_path, sr=44100)
        waveform = torch.FloatTensor(waveform)[None, None, ...].to(device)

        recon_dac = dac(waveform)['audio']

        loss_dac = mse_loss(recon_dac, waveform)'''
        loss_dac = torch.tensor([0])
        dac_loss.append(loss_dac.item())

    print('%d/%d, meta loss: %.6f, dac loss: %.6f' % (i+1, len(music_data), loss_meta.item(), loss_dac.item()))

print('mean of meta: ', sum(meta_loss) / len(meta_loss))
print('mean of dac: ', sum(dac_loss) / len(dac_loss))




