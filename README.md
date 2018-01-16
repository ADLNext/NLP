# Speech to text and Natural Language Processing

## Usage

This are advices, you can run `main.py`, `plot.py` and `summary.py` with the `-h` flag to see all parameters.

Step 1 is performing speech-to-text transcription from an audio file:
```
python main.py data/audio.flac --log logs/tts.json --skip-nlp True
```

Then it is possible to use the json file previously generated to perform sentiment analysis:
```
python main.py logs/tts.json --log logs/nlp.json
```

It is then possible to either create the "manhattan plot" for the conversation or to generate a summary of the operations:

```
python plot.py logs/tts.json logs/nlp.json
```

```
python summary.py logs/tts.json logs/nlp.json --outfile conv
```
