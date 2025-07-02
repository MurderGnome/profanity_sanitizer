# Profanity Sanitizer 🔇
Batch upload MP4 videos to Google Colab, transcribe them with OpenAI's Whisper, and automatically mute profanity using ffmpeg.

Click below to open a google colab notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MurderGnome/profanity_sanitizer/blob/main/censor_pipeline.ipynb)

## or

Paste this into a Colab cell to clone & run:

```python
!git clone https://github.com/MurderGnome/profanity_sanitizer.git
%cd profanity_sanitizer
!pip install -q -r requirements.txt
!apt -q update && apt install -y ffmpeg
!python censor_pipeline.py
