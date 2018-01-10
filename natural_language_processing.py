import json
import unicodedata

from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, KeywordsOptions, SentimentOptions, CategoriesOptions

class natural_language_processing:

    def __init__(self, uname, passwd, flags, log_file=None):
        self.nlp_api = NaturalLanguageUnderstandingV1(
            version='2017-02-27',
            username=uname,
            password=passwd
        )
        self.flags = flags
        self.log_file = log_file


    def process_batch(self, collection):
        target_score = []
        doc_score = 0
        found = 0
        for conv in collection:
            conv = conv['sentence']
            unicodedata.normalize('NFKD', conv).encode('ascii', 'ignore')
            nlp = self.nlp_api.analyze(
                text=conv,
                language='es',
                features=Features(
                    keywords=KeywordsOptions(
                        emotion=True,
                        sentiment=True
                    ),
                    categories=CategoriesOptions(),
                    sentiment=SentimentOptions(
                        targets=self.flags
                    )
                )
            )

            try:
                doc_score += nlp['sentiment']['document']['score']
                for target in nlp['sentiment']['targets']:
                    target_score.append({
                        'text': target['text'],
                        'score': target['score']
                    })
                found += 1
            except KeyError:
                print('INFO: no target found in sentence')
        doc_score /= found

        json_out = {
            'sentiment':{
                'document': {
                    'score': doc_score
                },
                'targets': target_score
            }
        }

        if self.log_file != None:
            print('INFO: logging NLP to %s\n' % self.log_file)
            with open(self.log_file, 'w') as outfile:
                json.dump(json_out, outfile)

        return doc_score, target_score


    def process_text(self, conv):
        unicodedata.normalize('NFKD', conv).encode('ascii', 'ignore')
        nlp = self.nlp_api.analyze(
            text=conv,
            language='es',
            features=Features(
                keywords=KeywordsOptions(
                    emotion=True,
                    sentiment=True
                ),
                categories=CategoriesOptions(),
                sentiment=SentimentOptions(
                    targets=self.flags
                )
            )
        )

        if self.log_file != None:
            print('INFO: logging NLP to %s\n' % self.log_file)
            with open(self.log_file, 'w') as outfile:
                json.dump(nlp, outfile)

        doc_score = 0
        target_score = []
        try:
            doc_score = nlp['sentiment']['document']['score']
            for target in nlp['sentiment']['targets']:
                target_score.append({
                    'text': target['text'],
                    'score': target['score']
                })
        except KeyError:
            print('INFO: no target found')

        return doc_score, target_score


    @staticmethod
    def load_json(json_file):
        nlp = json.load(open(json_file))

        doc_score = 0
        target_score = []
        try:
            doc_score = nlp['sentiment']['document']['score']
            for target in nlp['sentiment']['targets']:
                target_score.append({
                    'text': target['text'],
                    'score': target['score']
                })
        except KeyError:
            print('INFO: no target found')

        return doc_score, target_score
