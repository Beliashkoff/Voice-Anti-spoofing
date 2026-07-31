import numpy as np
import torch

from src.metrics.base_metric import BaseMetric


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(bonafide_scores, other_scores):
    frr, far, thresholds = compute_det_curve(bonafide_scores, other_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self.scores = []
        self.labels = []

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        probs = torch.softmax(logits, dim=-1)
        score = probs[:, 1]

        self.scores.extend(score.detach().cpu().tolist())
        self.labels.extend(labels.detach().cpu().tolist())

        scores = np.array(self.scores)
        labels_arr = np.array(self.labels)

        bona = scores[labels_arr == 1]
        spoof = scores[labels_arr == 0]

        if len(bona) == 0 or len(spoof) == 0:
            return 0.0

        eer, _ = compute_eer(bona, spoof)
        return eer * 100
