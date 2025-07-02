# censor_pipeline.py

import os
import re
import whisper
import ffmpeg
from better_profanity import profanity

try:
    import google.colab
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def normalize_word(word):
    return re.sub(r"[^\w\s]", "", word.lower())

def generate_combined_volume_filter(ranges):
    if not ranges:
        return None
    conds = [f"between(t,{s},{e})" for s, e in ranges]
    return f"volume=enable='{'+'.join(conds)}':volume=0"

def process_video(filename, model, output_dir="."):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    audio_output = os.path.join(output_dir, f"{base_name}_audio.wav")
    transcript_file = os.path.join(output_dir, f"{base_name}_censored_transcript.txt")
    output_video = os.path.join(output_dir, f"{base_name}_censored.mp4")

    print(f"\n🎬 Processing: {filename}")
    ffmpeg.input(filename).output(audio_output, ac=1, ar='16000').run(overwrite_output=True)

    result = model.transcribe(audio_output, word_timestamps=True, verbose=False)
    transcript = result["text"]
    censored_text = profanity.censor(transcript)

    with open(transcript_file, "w") as f:
        f.write(censored_text)

    mute_ranges = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            if profanity.contains_profanity(normalize_word(word["word"])):
                mute_ranges.append((word["start"], word["end"]))

    volume_filter = generate_combined_volume_filter(mute_ranges)
    if volume_filter:
        ffmpeg.input(filename).output(output_video, af=volume_filter, vcodec='copy', acodec='aac').run(overwrite_output=True)
    else:
        ffmpeg.input(filename).output(output_video, vcodec='copy', acodec='copy').run(overwrite_output=True)

    if IN_COLAB:
        files.download(transcript_file)
        files.download(output_video)
    else:
        print(f"✅ Saved to:\n- {transcript_file}\n- {output_video}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input folder with MP4s")
    parser.add_argument("--output", type=str, default=".", help="Output folder")
    args = parser.parse_args()

    profanity.load_censor_words()
    profanity.add_censor_words([
        "hell", "hells", "hell's", "damn", "wtf", "crap", "shit", "fuck", "holy shit", "ass", "bastard"
    ])
    model = whisper.load_model("base")

    if IN_COLAB:
        print("📂 Upload MP4 videos (1–∞)")
        uploaded = files.upload()
        for f in uploaded:
            process_video(f, model)
    else:
        input_dir = args.input
        if not input_dir or not os.path.isdir(input_dir):
            raise ValueError("❌ Input folder is required outside Colab. Use --input /your/folder")
        os.makedirs(args.output, exist_ok=True)
        for f in os.listdir(input_dir):
            if f.lower().endswith(".mp4"):
                process_video(os.path.join(input_dir, f), model, output_dir=args.output)

if __name__ == "__main__":
    main()
