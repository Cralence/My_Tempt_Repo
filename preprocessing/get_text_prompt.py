import os.path

from openai import OpenAI
import json
import pandas as pd
import random

OPENAI_API_KEY = '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'
dropout_prob = 0.3
metadata_path = 'text_prompt.csv'
save_path = 'result.json'
n_prompt_per_audio = 5

client = OpenAI(
    api_key=OPENAI_API_KEY
)
model = "gpt-3.5-turbo"

text_df = pd.read_csv(metadata_path, index_col=0)

music_id_list = list(text_df.index)

if os.path.exists(save_path):
    with open(save_path, 'r') as file:
        generated_caption = json.load(file)
    generated_id_list = generated_caption.keys()
    music_id_list = [k for k in music_id_list if k not in generated_id_list]

for id_idx, music_id in enumerate(music_id_list):
    generated_prompt_list = []

    tag_list = text_df.loc[music_id, 'tags']
    if pd.isna(tag_list):
        tag_list = ['nan value']
        tag_is_empty = True
    else:
        tag_list = tag_list.split('\t')
        tag_is_empty = False

    tag_list = [t for t in tag_list if 'vocalist' not in t]

    for i in range(n_prompt_per_audio):
        # choose for tempo descriptor
        tempo = text_df.loc[music_id, 'tempo']
        if tempo < 60:
            s1 = ['extremely', 'very']
            s2 = ['slow', 'languid', 'lethargic', 'relaxed', 'leisure', 'chilled']
            tempo_description = f'{random.choice(s1)} {random.choice(s2)}'
        elif 60 <= tempo < 75:
            tempo_description = random.choice(['slow', 'languid', 'lethargic', 'relaxed', 'leisure', 'chilled'])
        elif 75 <= tempo < 110:
            tempo_description = random.choice(['moderate', 'easy-going', 'laid-back', 'medium', 'balanced', 'neutral'])
        elif 110 <= tempo < 150:
            tempo_description = random.choice(['fast', 'upbeat', 'high', 'brisk', 'quick', 'rapid', 'swift'])
        else:
            s1 = ['extremely', 'very', 'highly']
            s2 = ['fast', 'upbeat', 'high', 'brisk', 'quick', 'rapid', 'swift']
            tempo_description = f'{random.choice(s1)} {random.choice(s2)}'

        # choose for energy descriptor
        energy = text_df.loc[music_id, 'energy']
        if energy < 0.1:
            s1 = ['extremely', 'very']
            s2 = ['soft', 'calm', 'peaceful', 'serene', 'gentle', 'light', 'tranquil', 'mild', 'mellow']
            energy_description = f'{random.choice(s1)} {random.choice(s2)}'
        elif 0.1 <= energy < 0.4:
            energy_description = random.choice(
                ['soft', 'calm', 'peaceful', 'serene', 'gentle', 'light', 'tranquil', 'mild', 'mellow'])
        elif 0.4 <= energy < 0.7:
            energy_description = random.choice(['moderate', 'comfortable', 'balanced', 'relaxing'])
        elif 0.7 <= energy < 0.95:
            energy_description = random.choice(
                ['intense', 'powerful', 'strong', 'vigorous', 'fierce', 'potent', 'energetic'])
        else:
            s1 = ['extremely', 'very', 'highly']
            s2 = ['intense', 'powerful', 'strong', 'vigorous', 'fierce', 'potent', 'energetic']
            energy_description = f'{random.choice(s1)} {random.choice(s2)}'

        # drop out some tags
        p = random.uniform(0, 1)
        if p < dropout_prob:
            tempo_description = None
        p = random.uniform(0, 1)
        if p < dropout_prob:
            energy_description = None
        filtered_tag_list = []
        for tag in tag_list:
            p = random.uniform(0, 1)
            if p > dropout_prob:
                filtered_tag_list.append(tag)

        # construct phrases
        noun_choices = ['tempo', 'speed', 'pace', 'BPM', 'rhythm', 'beat']
        tempo_choices = [f'with a {tempo_description} {random.choice(noun_choices)}',
                         f'whose {random.choice(noun_choices)} is {tempo_description}',
                         f'a {tempo_description} music',
                         f'set in a {tempo_description} {random.choice(noun_choices)}']

        noun_choices = ['intensity', 'energy']
        energy_choices = [f'which is {energy_description}',
                          f'with {energy_description} {random.choice(noun_choices)}',
                          f'a {energy_description} music',
                          f'whose {random.choice(noun_choices)} is {energy_description}']

        noun_choices = ['genre', 'style', 'type', 'category']
        if len(filtered_tag_list) == 0:
            filtered_tag_list = [random.choice(tag_list)]  # ensure at least have 1 tag
        if len(filtered_tag_list) == 1:
            tag_string = filtered_tag_list[0]
        else:
            tag_string = ', '.join(filtered_tag_list[:-1]) + f' and {filtered_tag_list[-1]}'
        tag_choices = [f'this is a track which is {tag_string}',
                       f'this song has the {random.choice(noun_choices)} of {tag_string}',
                       f'the music is {tag_string}',
                       f'the {random.choice(noun_choices)} of the music is {tag_string}']

        phrase_tempo = random.choice(tempo_choices)
        phrase_energy = random.choice(energy_choices)
        phrase_tag = random.choice(tag_choices)

        text_prompt = []
        if tempo_description is not None:
            text_prompt.append(phrase_tempo)
        if energy_description is not None:
            text_prompt.append(phrase_energy)
        if not tag_is_empty:
            text_prompt.append(phrase_tag)

        random.shuffle(text_prompt)
        music_text_prompt = ', '.join(text_prompt) + '.'

        try:
            completion = client.chat.completions.create(
              model=model,
              messages=[
                {"role": "system", "content": "You are an expert in music, skilled in writing music comments and descriptions."},
                {"role": "user", "content": f"Help me polish the following music description in concise language: {music_text_prompt}"}
              ]
            )

            print(f'{id_idx + 1}/{len(music_id_list)}, {i} prompt: [{music_text_prompt}]\ngenerated: [{completion.choices[0].message.content}]')
            generated_prompt_list.append(completion.choices[0].message.content)
        except:
            with open(save_path, 'w') as file:
                json.dump(generated_caption, file, indent=4)

    generated_caption[music_id] = generated_prompt_list

with open(save_path, 'w') as file:
    json.dump(generated_caption, file, indent=4)
