'''
Example usage:

python main.py data/001.flac --log json_dumps/001_tts.json --skip-nlp True
python main.py json_dumps/001_tts.json --log json_dumps/001_nlp.json
'''

import json
import textwrap
import argparse

from speech_to_text import speech_to_text as stt
from natural_language_processing import natural_language_processing as nlp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=textwrap.dedent('''\
        Script to perform speech to text and natural language processing.
        If a .flac file is provided, the script will log the output of the speech to text process.
        If the input is a json file, the script will load the conversation from the file and log the NLP.
        It is possible to skip the NLP step with the --skip-nlp flag.
    '''))
    parser.add_argument(
        'path', type=str,
        help='Path to an audio(flac)/json file'
    )
    parser.add_argument(
        '--log', type=str, default=None,
        help='If provided, path to json file to dump transcript'
    )
    parser.add_argument(
        '--flags', type=str, default='resources/flags.json',
        help='If provided, path to json file containing the red flags'
    )
    parser.add_argument(
        '--skip-nlp', type=bool, default=False,
        help='Skip NLP and only do speech to text'
    )

    args = parser.parse_args()
    input_file = args.path
    flags_file = args.flags
    log_file = args.log
    skip_nlp = args.skip_nlp

    flags = json.load(open(flags_file))
    if log_file == None:
        print('WARNING: log file not specified')
    else:
        print('INFO: logging to %s\n' % log_file)

    speech_engine = stt(
        'b957a8f7-cc9f-408d-a8a5-46a486be371e',
        'CzrwFjQD6ofa',
        log_file=log_file
    )
    if input_file.endswith('.flac'):
        text, speakers = speech_engine.transcribe(input_file)
        dialogue = speech_engine.parse_conv(text, speakers)
    elif input_file.endswith('.json'):
        dialogue = speech_engine.load_json(input_file)
    else:
        print('ERROR: %s in not a valid file! (flac/json)' % input_file)

    if skip_nlp:
        print('WARNING: not performing NLP')
        exit(0)

    if input_file.endswith('.json') and log_file != None:
        print('INFO: logging NLP to %s\n' % log_file)
        log_nlp = log_file
    else:
        print('WARNING: not logging NLP')
        log_nlp = None

    nlp_engine = nlp(
        '30c38aa5-66aa-4f3a-93b1-11f2d32c39b9',
        'c0ddm1bOKzZr',
        flags,
        log_file=log_nlp
    )

    nlp_engine.process_batch(dialogue)
