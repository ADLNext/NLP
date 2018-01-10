import textwrap
import argparse

from speech_to_text import speech_to_text as stt
from natural_language_processing import natural_language_processing as nlp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=textwrap.dedent('''\
        Script to print a summary of the transcribed conversation
        and the identified keywords with their sentiment score.
    '''))
    parser.add_argument(
        'speech_to_text', type=str,
        help='Path to json file with speech_to_text transcription'
    )
    parser.add_argument(
        'natural_language', type=str, default=None,
        help='Path to json file with sentiment analysis'
    )
    parser.add_argument(
        '--outfile', type=str, default=None,
        help='If provided, the output is also saved on this file'
    )

    args = parser.parse_args()
    stt_file = args.speech_to_text
    nlp_file = args.natural_language
    outfile = args.outfile

    speech_engine = stt(
        'b957a8f7-cc9f-408d-a8a5-46a486be371e',
        'CzrwFjQD6ofa',
    )

    dialogue = speech_engine.load_json(stt_file)

    score, targets = nlp.load_json(nlp_file)

    for unit in dialogue:
        print(str(unit['speaker']) + ' [' + str(unit['timestamp']) + ']: ' + unit['sentence'])

    print('Conversation sentiment score: %f' % score)

    for target in targets:
        print('%s: %f' % (target['text'], target['score']))

    if outfile != None:
        print('INFO: logging to %s' % outfile)
        with open(outfile, 'w') as f:
            for unit in dialogue:
                f.write(str(unit['speaker']) + ' [' + str(unit['timestamp']) + ']: ' + unit['sentence'] + '\n')
            f.write('Conversation sentiment score: %f\n' % score)
            for target in targets:
                f.write('%s: %f\n' % (target['text'], target['score']))
