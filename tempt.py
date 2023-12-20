from unimumo.models import UniMuMo
from unimumo.motion.utils import visualize_music_motion

model = UniMuMo.from_checkpoint('final_model/unimumo_model.ckpt')
model = model.cuda()
model.music_motion_lm = model.music_motion_lm.cuda()

text_description = 'The audio is a jazz fusion piece with elements of funk. The genre of the dance is LA style hip-hop.'
waveform_gen, motion_gen = model.generate_music_motion(
    text_description=[text_description], batch_size=5
)

visualize_music_motion(waveform_gen, motion_gen['joint'], 'gen_results')
