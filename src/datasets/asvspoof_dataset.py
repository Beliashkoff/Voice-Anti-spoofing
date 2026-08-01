import random
from pathlib import Path

import torch
import torchaudio

from src.datasets.base_dataset import BaseDataset


class ASVspoofDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        protocol_path,
        name,
        train=True,
        sample_rate=16000,
        max_duration_sec=4.10625,
        eval_segments=3,
        noise_std=0.005,
        balanced_limit=None,
        *args,
        **kwargs,
    ):
        self.data_dir = Path(data_dir)
        self.name = name
        self.train = train
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * max_duration_sec)
        self.eval_segments = int(eval_segments)
        self.noise_std = float(noise_std)

        if self.max_len <= 0:
            raise ValueError("неверная длина аудио")
        if self.eval_segments <= 0:
            raise ValueError("неверное число сегментов")

        index = self._create_index(protocol_path)
        if balanced_limit is not None:
            per_class = int(balanced_limit) // 2
            if per_class <= 0:
                raise ValueError("слишком маленький лимит")
            bona = [item for item in index if item["label"] == 1][:per_class]
            spoof = [item for item in index if item["label"] == 0][:per_class]
            if len(bona) != per_class or len(spoof) != per_class:
                raise ValueError("не хватает данных")
            index = bona + spoof
        super().__init__(index, *args, **kwargs)

    def _create_index(self, protocol_path):
        index = []
        seen_ids = set()
        flac_dir = self.data_dir / "flac"

        with open(protocol_path, "r") as protocol:
            for line_number, line in enumerate(protocol, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    raise ValueError(f"ошибка протокола в строке {line_number}")

                utt_id = parts[1]
                label = parts[-1]
                if label not in {"bonafide", "spoof"}:
                    raise ValueError(f"неверная метка {label}")
                if utt_id in seen_ids:
                    raise ValueError(f"повтор id {utt_id}")
                seen_ids.add(utt_id)

                index.append(
                    {
                        "path": str(flac_dir / f"{utt_id}.flac"),
                        "label": 1 if label == "bonafide" else 0,
                        "utt_id": utt_id,
                    }
                )

        if not index:
            raise ValueError(f"пустой протокол {protocol_path}")
        return index

    def load_object(self, path):
        wav, sample_rate = torchaudio.load(path)
        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.numel() == 0:
            raise ValueError(f"пустое аудио {path}")

        if self.train:
            wav = self._repeat_to_length(wav)
            start = random.randint(0, wav.shape[1] - self.max_len)
            wav = wav[:, start : start + self.max_len]
            if self.noise_std > 0:
                wav = wav + torch.randn_like(wav) * self.noise_std
            return wav

        return self._evaluation_segments(wav)

    def _repeat_to_length(self, wav):
        if wav.shape[1] < self.max_len:
            repeats = (self.max_len + wav.shape[1] - 1) // wav.shape[1]
            wav = wav.repeat(1, repeats)
        return wav

    def _evaluation_segments(self, wav):
        wav = self._repeat_to_length(wav)
        max_start = wav.shape[1] - self.max_len
        if self.eval_segments == 1 or max_start == 0:
            starts = [0] * self.eval_segments
        else:
            starts = torch.linspace(0, max_start, self.eval_segments).round().long()
            starts = starts.tolist()
        return torch.stack(
            [wav[:, start : start + self.max_len] for start in starts], dim=0
        )

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
