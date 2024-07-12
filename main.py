from datetime import datetime

import srt
import torchaudio
from faster_whisper import WhisperModel
from pydub import AudioSegment
from speechbrain.inference import WaveformEnhancement, SpectralMaskEnhancement
from yt_dlp import YoutubeDL
import yt_dlp
import demucs.separate
import os

VIDEO_URL = ['https://www.youtube.com/watch?v=JIHAc55uzj4']


def extract_vocal(filename):
    demucs.separate.main(["--two-stems", "vocals", "-o", "./temp/separated", "-n", "htdemucs_ft", filename])
    return "./temp/separated/htdemucs_ft/out0/vocals.wav"


def normalize_sound(filename):
    vocal = AudioSegment.from_file(filename, format="flac")
    normalized_vocal = vocal.normalize()
    boosted_vocal = normalized_vocal + 20
    output = "./temp/out1.wav"
    boosted_vocal.export(output, format="wav")
    return output


def enhance_audio(filename):
    enhance_model1 = WaveformEnhancement.from_hparams(
        source="speechbrain/mtl-mimic-voicebank",
        savedir="pretrained_models/mtl-mimic-voicebank",
        run_opts={"device": "cuda"}
    )
    enhanced = enhance_model1.enhance_file(filename)
    torchaudio.save('./temp/out2_0.wav', enhanced.unsqueeze(0).cpu(), 16000)

    enhance_model2 = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
        run_opts={"device": "cuda"}
    )
    enhanced = enhance_model2.enhance_file(filename)
    torchaudio.save('./temp/out2_1.wav', enhanced.unsqueeze(0).cpu(), 16000)

    vocal1 = AudioSegment.from_file('./temp/out2_0.wav', format="wav")
    vocal2 = AudioSegment.from_file('./temp/out2_1.wav', format="wav")

    combined_vocal = vocal1.overlay(vocal2.apply_gain(-2), gain_during_overlay=-2)

    output = "./temp/out3.wav"
    combined_vocal.export(output, format="wav")
    return output


def create_srt(filename):
    audio = AudioSegment.from_file('./temp/out1.wav', format="wav")
    resampled_audio = audio.set_frame_rate(48000)
    stereo_audio = resampled_audio.set_channels(2)
    stereo_audio.export('./temp/out4.wav', format="wav", bitrate="512k")

    model = WhisperModel("large-v3", device="cuda", compute_type="float32")

    segments, info = model.transcribe(
        './temp/out4.wav',
        beam_size=50,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    subtitles = []

    print("Start parse =", datetime.now())
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        subtitles.append(srt.Subtitle(index=1, start=srt.timedelta(seconds=segment.start),
                                      end=srt.timedelta(seconds=segment.end), content=f"{segment.text}"))
    print("End parse =", datetime.now())

    print("Start create srt file =", datetime.now())
    srt_text = srt.compose(subtitles)

    output = f"{filename}.srt"

    with open(output, "w", encoding="utf-8") as f:
        f.write(srt_text)

    print("End create srt file =", datetime.now())
    return output


class V2SrtPP(yt_dlp.postprocessor.PostProcessor):
    def run(self, info):
        self.to_screen('Done download video')
        video_file = info["requested_downloads"][0]["filepath"]
        base_name, extension = os.path.splitext(video_file)
        new_video_file_name = f"./temp/out0{extension}"
        os.replace(video_file, new_video_file_name)

        self.to_screen('Start extract vocal')
        vocal_file_name = extract_vocal(new_video_file_name)
        self.to_screen('End extract vocal')

        self.to_screen('Start normalize sound')
        normalized_file_name = normalize_sound(vocal_file_name)
        self.to_screen('End normalize sound')

        self.to_screen('Start enhance sound')
        enhanced_file_name = enhance_audio(normalized_file_name)
        self.to_screen('End enhance sound')

        self.to_screen('Start create srt')
        srt_file = create_srt(enhanced_file_name)
        self.to_screen(f'End create srt result in: {srt_file}')

        return [], info


ydl_opts = {
    "paths": {"home": "./temp"},
    "quiet": False,
    "verbose": False
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.add_post_processor(V2SrtPP(), when='after_video')
    ydl.download(VIDEO_URL)
