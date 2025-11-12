from pathlib import Path

dataset_name = "/training-1/asr_dataset_files/asr_auf_zwei_planeten"

split = "train_nodev" # train_dev test

id_wav, id_text = {}, {}

with open(f"{dataset_name}/{split}/wav.scp", encoding="UTF-8") as rf:
    for line in rf:
        id, wav_path = line.strip().split("\t")
        id_wav[id] = wav_path

with open(f"{dataset_name}/{split}/text", encoding="UTF-8") as rf:
    for line in rf:
        id, text = line.strip().split("\t")
        id_text[id] = text

for id in id_wav:
    wav_path = id_wav[wav_path]
    text = id_text