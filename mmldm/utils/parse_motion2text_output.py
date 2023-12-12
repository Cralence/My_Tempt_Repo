# open gen_captions.txt as read
with open('gen_captions.txt', 'r') as f:
    with open('gen_caption_clean.txt', 'w') as fw:
        for line in f.readlines():
            line = line.strip()

            music_id = line.split('\t')[0]
            caption = line.split('\t')[1]

            sentences = caption.split('.')
            sentences = [s for s in sentences if ' music ' not in s and ' audio ' not in s
                         and 'vocal' not in s and ' sound ' not in s]

            if len(sentences) == 0:
                print('here')
            if len(sentences) == 1:
                print('here')


            sentences = '.'.join(sentences)

            if 'The motion is that ' in sentences:
                sentences = sentences.replace('The motion is that ', '')
            if 'The dance is that ' in sentences:
                sentences = sentences.replace('The dance is that ', '')
            sentences = sentences.strip().capitalize()

            print(sentences)

            fw.write(music_id + '\t' + sentences + '\n')


