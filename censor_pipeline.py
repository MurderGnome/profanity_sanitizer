# censor_pipeline.py

import whisper
from pydub import AudioSegment
from better_profanity import profanity
import os
import ffmpeg

from google.colab import files

# Upload MP4s via browser interface
print("📂 Upload MP4 videos (1–∞)")
uploaded = files.upload()

# Load Whisper and profanity filter
model = whisper.load_model("base")
profanity.load_censor_words()

# Process each file
for filename in uploaded:
    base_name = os.path.splitext(filename)[0]
    audio_output = f"{base_name}_audio.wav"
    transcript_file = f"{base_name}_censored_transcript.txt"
    output_video = f"{base_name}_censored.mp4"

    print(f"\n🎬 Processing: {filename}")

    # Extract audio from video
    ffmpeg.input(filename).output(audio_output, ac=1, ar='16000').run(overwrite_output=True)

    # Transcribe audio with word-level timestamps
    result = model.transcribe(audio_output, word_timestamps=True, verbose=False)
    transcript = result["text"]
    censored_text = profanity.censor(transcript)

    # Save transcript
    with open(transcript_file, "w") as f:
        f.write(censored_text)

    # Identify profane word time ranges
    mute_ranges = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            if profanity.contains_profanity(word["word"].strip().lower()):
                mute_ranges.append((word["start"], word["end"]))

    # Build FFmpeg volume filter
    def generate_combined_volume_filter(ranges):
        if not ranges:
            return None
        conds = [f"between(t,{s},{e})" for s, e in ranges]
        return f"volume=enable='{'+'.join(conds)}':volume=0"

    volume_filter = generate_combined_volume_filter(mute_ranges)

    # Apply filter and render final video
    if volume_filter:
        ffmpeg.input(filename).output(output_video, af=volume_filter, vcodec='copy', acodec='aac').run(overwrite_output=True)
    else:
        ffmpeg.input(filename).output(output_video, vcodec='copy', acodec='copy').run(overwrite_output=True)

    # Auto-download result files
    files.download(transcript_file)
    files.download(output_video)

print("\n✅ All files processed.")
