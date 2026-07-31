import csv

import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        )

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        self.save_path = save_path

        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part, rows):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        probs = torch.softmax(batch["logits"], dim=-1)
        scores = probs[:, 1]

        for i in range(scores.shape[0]):
            rows.append((batch["utt_id"][i], scores[i].item()))

        return batch

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            for met in self.metrics["inference"]:
                if hasattr(met, "reset"):
                    met.reset()
            self.evaluation_metrics.reset()

        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        rows = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                    rows=rows,
                )

        if self.save_path is not None:
            csv_path = self.save_path / part / "scores.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                for utt_id, score in rows:
                    writer.writerow([utt_id, score])

        if self.evaluation_metrics is not None:
            return self.evaluation_metrics.result()
        return {}
