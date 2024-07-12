from datetime import datetime

import srt
from faster_whisper import WhisperModel
from pydub import AudioSegment

#demucs.separate.main(["--two-stems", "vocals", "-n", "htdemucs_ft", filename])

filename = "1_講師から直伝！ロシア語勉強配信！изучаю русский [SdJbWIt2zZ0]_(Vocals).flac"

vocal = AudioSegment.from_file(filename, format="flac")
normalized_vocal = vocal.normalize()
boosted_vocal = normalized_vocal + 20
boosted_vocal.export("./temp/out1.wav", format="wav")

# enhance_model1 = WaveformEnhancement.from_hparams(
#     source="speechbrain/mtl-mimic-voicebank",
#     savedir="pretrained_models/mtl-mimic-voicebank",
#     run_opts={"device":"cuda"}
# )
# enhanced = enhance_model1.enhance_file("./temp/out1.wav")
# torchaudio.save('./temp/out2_0.wav', enhanced.unsqueeze(0).cpu(), 16000)
#
# enhance_model2 = SpectralMaskEnhancement.from_hparams(
#     source="speechbrain/metricgan-plus-voicebank",
#     savedir="pretrained_models/metricgan-plus-voicebank",
#     run_opts={"device":"cuda"}
# )
# enhanced = enhance_model2.enhance_file("./temp/out1.wav")
# torchaudio.save('./temp/out2_1.wav', enhanced.unsqueeze(0).cpu(), 16000)
#
# vocal1 = AudioSegment.from_file('./temp/out2_0.wav', format="wav")
# vocal2 = AudioSegment.from_file('./temp/out2_1.wav', format="wav")
#
# combined_vocal = vocal1.overlay(vocal2.apply_gain(-2), gain_during_overlay=-2)
#
# combined_vocal.export("./temp/out3.wav", format="wav")

audio = AudioSegment.from_file('./temp/out1.wav', format="wav")
resampled_audio = audio.set_frame_rate(48000)
sterio_audio = resampled_audio.set_channels(2)
sterio_audio.export('./temp/out4.wav', format="wav", bitrate="512k")


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

with open(f"{filename}.srt", "w", encoding="utf-8") as f:
    f.write(srt_text)

print("End create srt file =", datetime.now())
