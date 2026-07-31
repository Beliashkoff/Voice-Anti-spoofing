import torch


def collate_fn(dataset_items: list[dict]):
    result_batch = {}

    result_batch["data_object"] = torch.stack(
        [elem["data_object"] for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

    if "utt_id" in dataset_items[0]:
        result_batch["utt_id"] = [elem["utt_id"] for elem in dataset_items]

    return result_batch
