import json
from watson_developer_cloud import SpeechToTextV1

class speech_to_text:

    def __init__(self, uname, passwd, log_file = None):
        self.speech_api = SpeechToTextV1(
            username=uname,
            password=passwd,
            x_watson_learning_opt_out=False
        )
        self.log_file = log_file


    def transcribe(self, audio_file):
        with open(audio_file, 'rb') as audio:
            response = self.speech_api.recognize(
                audio, content_type='audio/flac', timestamps=True,
                word_confidence=True,
                speaker_labels=True,
                model='es-ES_NarrowbandModel'
            )
        if self.log_file != None:
            print('INFO: logging speech to text to %s\n' % self.log_file)
            with open(self.log_file, 'w') as outfile:
                json.dump(response, outfile)
        text = response['results']
        speakers = response['speaker_labels']

        return text, speakers


    def parse_conv(self, text, speakers):
        dialogue = []

        for elem in text:
            for unit, conf in zip(elem['alternatives'][0]['timestamps'], elem['alternatives'][0]['word_confidence']):
                elem['alternatives'][0]['word_confidence']
                word = unit[0]
                start_time = unit[1]
                end_time = unit[2]
                confidence = conf[1]
                dialogue_step = {
                    'text': word,
                    'start': start_time,
                    'end': end_time,
                    'confidence': confidence
                }
                dialogue.append(dialogue_step)

        labeled_dialogue = []

        for unit in dialogue:
            for token in speakers:
                start = token['from']
                end = token['to']
                speaker = token['speaker']
                if unit['start'] >= start and unit['end'] <= end:
                    labeled_unit = {
                        'text': unit['text'],
                        'timestamp': unit['start'],
                        'confidence': unit['confidence'],
                        'speaker': speaker
                    }
                    labeled_dialogue.append(labeled_unit)

        final_dialogue = []

        n_words = len(labeled_dialogue)
        current_speaker = labeled_dialogue[0]['speaker']
        timestamp = labeled_dialogue[0]['timestamp']
        total_conf = 0
        current_sentence = ''
        for i in range(n_words):
            current_word = labeled_dialogue[i]['text']
            speaker = labeled_dialogue[i]['speaker']
            word_conf = labeled_dialogue[i]['confidence']
            if speaker == current_speaker:
                current_sentence += current_word + ' '
                total_conf += word_conf
            else:
                total_conf /= len(current_sentence)
                final_dialogue.append(
                    {
                        'sentence': current_sentence,
                        'speaker': current_speaker,
                        'timestamp': timestamp,
                        'confidence': total_conf
                    })
                current_sentence = current_word + ' '
                current_speaker = speaker
                timestamp = labeled_dialogue[i]['timestamp']
                total_conf = labeled_dialogue[i]['confidence']

        return final_dialogue


    def load_json(self, json_file):
        response = json.load(open(json_file))
        text = response['results']
        speakers = response['speaker_labels']

        return self.parse_conv(text, speakers)
