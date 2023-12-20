import importlib
import os
import torch
import json


def instantiate_from_config(config):
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model_from_config(config, ckpt, verbose=False):
    model = instantiate_from_config(config.model)

    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

    model.eval()
    return model


def get_music_motion_prompt_list(meta_dir):
    with open(os.path.join(meta_dir, 'music4all_captions_val_test.json'), 'r') as caption_fd:
        music_caption = json.load(caption_fd)
    music_prompt_list = [v for k, v in music_caption.items()]

    aist_genres = ['break', 'pop', 'lock', 'middle hip-hop', 'LA style hip-hop', 'house', 'waack', 'krump',
                   'street jazz', 'ballet jazz']
    motion_prompt_list = []
    for genre in aist_genres:
        motion_prompt_list.append(f'The genre of the dance is {genre}.')
        motion_prompt_list.append(f'The style of the dance is {genre}.')
        motion_prompt_list.append(f'This is a {genre} style dance.')

    return music_prompt_list, motion_prompt_list
