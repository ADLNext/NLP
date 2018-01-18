import json
import textwrap
import argparse

from speech_to_text import speech_to_text as stt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=textwrap.dedent('''\
        Script to compute SNR from a transcript.
    '''))
    parser.add_argument(
        'transcript', type=str,
        help='Path to json file containing the speech to text transcript'
    )

    args = parser.parse_args()
    conv_file = args.transcript

    response = json.load(open(conv_file))
    text = response['results']

    silence = 0

    for j in range(len(text[:-1])):
        elem = text[j]
        for i in range(len(elem['alternatives'][0]['timestamps'][:-1])):
            unit = elem['alternatives'][0]['timestamps'][i]
            next_unit = elem['alternatives'][0]['timestamps'][i+1]
            end_time = unit[2]
            next_start_time = next_unit[1]
            silence += (next_start_time - end_time)
        last_ending_time = elem['alternatives'][0]['timestamps'][-1][2]
        next_elem = text[j+1]
        next_start_time = next_elem['alternatives'][0]['timestamps'][0][1]
        silence += (next_start_time - last_ending_time)

    conv_len = elem['alternatives'][0]['timestamps'][-1][2]

    if silence == 0:
        print('SNR: +inf')
    else:
        SNR = silence/conv_len
        print('SNR: %f' % SNR)
