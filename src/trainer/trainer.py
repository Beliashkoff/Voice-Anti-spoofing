import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            self.grad_scaler.scale(batch["loss"]).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self._clip_grad_norm()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            value = met(**batch)
            if value is not None:
                metrics.update(met.name, value)
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        for met in self.metrics["inference"]:
            if hasattr(met, "reset"):
                met.reset()
        return super()._evaluation_epoch(epoch, part, dataloader)

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode == "train":
            pass
        else:
            pass
