# Speech Processing :microphone:

[Speech and Language Processing (3rd. Ed.)](https://web.stanford.edu/~jurafsky/slp3/)

[Introduction to Speech Processing (2nd. Ed.)](https://speechprocessingbook.aalto.fi/)

[[2023] Transformers in Speech Processing: A Survey (arXiv)](https://arxiv.org/abs/2303.11607)

## Speech Recognition

## Speech Synthesis | Voice Cloning | Speaker Adaptation | Speaker Encoding

## Speech Emotion Recognition

## Speaker Recognition

## Speech Segmentation

## Speech Enhancement

## Speech Translation

### Whisper - A model that can convert audio into text | [OpenAI](https://openai.com/blog/whisper/) | [GitHub](https://github.com/openai/whisper) | [Paper](https://cdn.openai.com/papers/whisper.pdf)

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language indentification.

---

### Speech Processing (SP) Project Pipeline

* Preprocessing
  * Speech Segmentation: pyAudioAnalysis, Voice Activity Detection (VAD)
  * Speaker Diarization: pyannote.audio, kaldi, End-to-End Neural Diarization (EEND)
* Speech Feature Extraction
  * Speaker Identification: x-vector, d-vector (Deep Speaker Embeddings), Resemblyzer
  * Speech Feature Extraction: Wav2Vec2, HuBERT, DeepSpeech
* Speech Synthesis Preparation
  * Text-to-Speech (TTS)
  * Voice Cloning: Tacotron 2 + WaveNet, FastSpeech + HiFi-GAN, Variational Inference TTS (VITS)
* Speech Synthesis
  * Text Generation (NLP): GPT-3, GPT-4, T5
  * Speech Synthesis: Tacotron 2, FastSpeech 2, VITS
* Postprocessing
  * Speech Enhancement: DeepSpeech, Speech Enhancement Generative Adverssarial Network (SEGAN)
 
i.e., Preprocessing (Segmentation) > Feature Extraction (Speaker Identification, Feature Extraction) > Voice Cloning > Speech Enhancement

---

### Reference
- Whisper, https://openai.com/blog/whisper/, 2022-12-10-Sat.
- Whisper GitHub, https://github.com/openai/whisper, 2022-12-10-Sat.
- Whisper Paper, https://cdn.openai.com/papers/whisper.pdf, 2022-12-10-Sat.
- Speech and Language Processing 3rd. Ed., https://web.stanford.edu/~jurafsky/slp3/, 2025-01-20-Mon.
- Introduction to Speech Processing 2nd. Ed., https://speechprocessingbook.aalto.fi/, 2025-01-20-Mon.
- [2023] Transformers in Speech Processing: A Survey (arXiv), https://arxiv.org/abs/2303.11607, 2025-01-20-Mon.
