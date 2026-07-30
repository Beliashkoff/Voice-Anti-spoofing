import random
from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class ASVspoofDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        protocol_path,
        name,
        train=True,
        sample_rate=16000,
        max_duration_sec=4.0,
        *args,
        **kwargs,
    ):
        self.data_dir = Path(data_dir)
        self.train = train
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * max_duration_sec)

        cache_dir = ROOT_PATH / "data" / "asvspoof" / name
        cache_dir.mkdir(exist_ok=True, parents=True)
        index_path = cache_dir / "index.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(protocol_path, index_path)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, protocol_path, index_path):
        index = []
        flac_dir = self.data_dir / "flac"

        with open(protocol_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            utt_id = parts[1]
            label = parts[-1]

            audio_path = flac_dir / f"{utt_id}.flac"

            index.append(
                {
                    "path": str(audio_path),
                    "label": 1 if label == "bonafide" else 0,
                    "utt_id": utt_id,
                }
            )

        write_json(index, str(index_path))
        return index

    def load_object(self, path):
        wav, sr = torchaudio.load(path)
        assert sr == self.sample_rate

        wav = self._pad_or_crop(wav)
        return wav

    def _pad_or_crop(self, wav):
        length = wav.shape[1]

        if length < self.max_len:
            n_repeats = self.max_len // length + 1
            wav = wav.repeat(1, n_repeats)
            length = wav.shape[1]

        if length > self.max_len:
            if self.train:
                start = random.randint(0, length - self.max_len)
            else:
                start = 0
            wav = wav[:, start : start + self.max_len]

        return wav

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        data_object = self.load_object(data_dict["path"])

        instance_data = {
            "data_object": data_object,
            "labels": data_dict["label"],
            "utt_id": data_dict["utt_id"],
        }
        instance_data = self.preprocess_data(instance_data)

        return instance_data
