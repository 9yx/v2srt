from datetime import datetime

import srt
import torch
from transformers import pipeline

fileName = "1_【歌】文野環お姉ちゃん♪【にじさんじ】 [Lb5TnW7NZX0]_(Vocals).flac"

torch.cuda.empty_cache()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=device
)

print("Start parse =", datetime.now())

subtitles = []
index = 0

pp = pipe(f"F:\\PyProjects\\JapToRus\\{fileName}",
                generate_kwargs={"task": "transcribe", "language": "<|ja|>"},
                batch_size=16,
                return_timestamps=True,
                max_new_tokens=128,
                chunk_length_s=15
                #chunk_length_s=10,
                #stride_length_s=(4, 2)
                )

for out in pp["chunks"]:
    if out['timestamp'][0] is not None and out['timestamp'][1] is not None:
        subtitles.append(srt.Subtitle(index=1, start=srt.timedelta(seconds=out['timestamp'][0]),
                                  end=srt.timedelta(seconds=out['timestamp'][1]), content=f"{out['text']}"))
        index += 1
    else:
        print(out)
    print(out['text'])
print("End parse =", datetime.now())

print("Start create srt file =", datetime.now())
srt_text = srt.compose(subtitles)

with open(f"{fileName}.srt", "w", encoding="utf-8") as f:
    f.write(srt_text)
print("End create srt file =", datetime.now())
