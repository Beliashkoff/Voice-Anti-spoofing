import argparse
import csv
import math
from pathlib import Path

import numpy as np


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]
    target_sums = np.cumsum(labels)
    nontarget_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - target_sums
    )
    frr = np.concatenate((np.atleast_1d(0), target_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_sums / nontarget_scores.size)
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )
    return frr, far, thresholds


def compute_eer(bonafide_scores, spoof_scores):
    frr, far, thresholds = compute_det_curve(bonafide_scores, spoof_scores)
    index = np.argmin(np.abs(frr - far))
    return float(np.mean((frr[index], far[index]))), float(thresholds[index])


def read_protocol(path):
    entries = []
    seen = set()
    with path.open() as protocol:
        for line_number, line in enumerate(protocol, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"ошибка протокола в строке {line_number}")
            utterance_id = parts[1]
            label = parts[-1]
            if label not in {"bonafide", "spoof"}:
                raise ValueError(f"неверная метка {label}")
            if utterance_id in seen:
                raise ValueError(f"повтор id {utterance_id}")
            seen.add(utterance_id)
            entries.append((utterance_id, label))
    return entries


def read_scores(path):
    scores = {}
    with path.open(newline="") as source:
        for line_number, row in enumerate(csv.reader(source), start=1):
            if len(row) != 2:
                raise ValueError(f"ошибка csv в строке {line_number}")
            utterance_id, raw_score = row
            if utterance_id in scores:
                raise ValueError(f"повтор id {utterance_id}")
            try:
                score = float(raw_score)
            except ValueError as error:
                raise ValueError(f"неверный скор в строке {line_number}") from error
            if not math.isfinite(score):
                raise ValueError(f"неверный скор в строке {line_number}")
            scores[utterance_id] = score
    return scores


def validate(score_path, protocol_path, expected_rows=None):
    protocol = read_protocol(protocol_path)
    scores = read_scores(score_path)
    protocol_ids = {utterance_id for utterance_id, _ in protocol}
    score_ids = set(scores)

    missing = protocol_ids - score_ids
    unknown = score_ids - protocol_ids
    if missing:
        raise ValueError(f"не хватает {len(missing)} id")
    if unknown:
        raise ValueError(f"лишние id {len(unknown)}")
    if len(scores) != len(protocol):
        raise ValueError("неверное число строк")
    if expected_rows is not None and len(scores) != expected_rows:
        raise ValueError("неверное число строк")

    ordered_scores = np.asarray([scores[key] for key, _ in protocol], dtype=np.float64)
    labels = np.asarray([label == "bonafide" for _, label in protocol])
    eer, threshold = compute_eer(ordered_scores[labels], ordered_scores[~labels])
    inverted_eer, _ = compute_eer(-ordered_scores[labels], -ordered_scores[~labels])
    if inverted_eer + 1e-12 < eer:
        raise ValueError("скоры перевернуты")
    return eer * 100.0, threshold, len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--expected-rows", type=int, default=None)
    args = parser.parse_args()

    eer, threshold, row_count = validate(
        args.scores, args.protocol, expected_rows=args.expected_rows
    )
    print(f"Validated {row_count} scores")
    print(f"Official-compatible EER: {eer:.4f}%")
    print(f"EER threshold: {threshold:.8g}")


if __name__ == "__main__":
    main()
