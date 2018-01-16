import json
import textwrap
import argparse
import soundfile as sf

from speech_to_text import speech_to_text as stt
from natural_language_processing import natural_language_processing as nlp

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=textwrap.dedent('''\
        Script to generate the "manhattan plot" for a conversation.
    '''))
    parser.add_argument(
        'transcript', type=str,
        help='Path to json file containing the text to speech transcript'
    )
    parser.add_argument(
        'sentiment', type=str,
        help='Path to json file containing the sentiment analysis'
    )
    parser.add_argument(
        '--outdir', type=str, default='plots/',
        help='If provided, path to directory to save the plot'
    )
    parser.add_argument(
        '--credentials', type=str, default='resources/credentials.json',
        help='Credentials for IBM Watson services'
    )

    args = parser.parse_args()
    conv_file = args.transcript
    nlp_file = args.sentiment
    out_dir = args.outdir
    cred_file = args.credentials

    creds = json.load(open(cred_file))
    tts_c = creds['tts']

    speech_engine = stt(
        tts_c['username'],
        tts_c['password']
    )

    dialogue = speech_engine.load_json(conv_file)

    avg, targets = nlp.load_json(nlp_file)

    hist = []
    labels = []
    speakers = []
    dots = []

    for sentence in dialogue:
        found = False
        ts = sentence['timestamp']
        speaker = sentence['speaker']
        conf = sentence['confidence']
        sentence = sentence['sentence']
        # labels[-1] = ts
        for i in range(len(targets) - 1, -1, -1):
            index = len(targets) - (i + 1)
            unit = targets[index]
            target = unit['text']
            if target in sentence:
                hist.append(unit['score'])
                speakers.append(speaker)
                label = str(ts) + '\n' + target + '\nspeaker:' + str(speaker)
                labels.append(label)
                hist.extend([0] * 10)
                speakers.extend([speaker] * 10)
                labels.extend([''] * 10)
                del targets[index]
                sentence = sentence.replace(target, '', 1)
                found = True
        if not found:
            for i in range(len(sentence.split(' '))//2):
                hist.append(0)
                labels.append('')
                speakers.append(speaker)

    labels[-1] = ts

    for i in range(len(hist)):
        if hist[i] == 0 and labels[i] != '':
            dots.append(i)

    x = np.arange(len(labels))
    y = hist
    plt.figure(figsize=(60,20))
    ax = sns.barplot(x, y)
    ax.set_autoscale_on(False)
    plt.ylim(-1, 1)
    ax.axhline(color='black')
    ax.axhline(y=avg, color='red', linestyle='dashed')
    ax.set(xlabel='Keywords and key attributes', ylabel='Sentiment')
    ax.set_xticklabels(labels[:-1], fontsize=15)
    for dot in dots:
        ax.plot(dot, 0, 'o', c='black')
    filename = out_dir + conv_file.split('/')[-1].split('.')[0] + '_' + nlp_file.split('/')[-1].split('.')[0] + '.png'
    plt.tight_layout()
    plt.savefig(filename)
