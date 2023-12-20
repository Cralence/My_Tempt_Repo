import json


with open('data/music/music4all_captions.json', 'r') as caption_fd:
    captions = json.load(caption_fd)

names = []
with open('music4all_val.txt', 'r') as f:
    for line in f.readlines():
        names.append(line.strip())
with open('music4all_test.txt', 'r') as f:
    for line in f.readlines():
        names.append(line.strip())
print('here')
captions = {k: v for k, v in captions.items() if k in names}
