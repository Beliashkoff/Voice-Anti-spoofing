from itertools import repeat

import torch
from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):

    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device):
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    datasets = instantiate(config.datasets)  
    _validate_asvspoof_partitions(datasets)

    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        assert config.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        sampler = None
        if dataset_partition == "train" and config.get("balance_train", False):
            labels = torch.tensor([item["label"] for item in dataset._index])
            class_counts = torch.bincount(labels, minlength=2).float()
            if (class_counts == 0).any():
                raise ValueError("нет одного из классов")
            sample_weights = (1.0 / class_counts)[labels]
            sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train" and sampler is None),
            sampler=sampler,
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms


def _validate_asvspoof_partitions(datasets):
    partition_ids = {}
    for name, dataset in datasets.items():
        index = getattr(dataset, "_index", None)
        if index is None or not index or "utt_id" not in index[0]:
            continue
        ids = [item["utt_id"] for item in index]
        labels = [item["label"] for item in index]
        if len(ids) != len(set(ids)):
            raise ValueError(f"повтор id в {name}")
        if not set(labels).issubset({0, 1}):
            raise ValueError(f"неверные метки в {name}")
        if name != "eval" and set(labels) != {0, 1}:
            raise ValueError(f"нет одного из классов в {name}")
        partition_ids[name] = set(ids)

    names = list(partition_ids)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = partition_ids[left_name] & partition_ids[right_name]
            if overlap and not (
                len(partition_ids[left_name]) <= 16
                and partition_ids[left_name] == partition_ids[right_name]
            ):
                raise ValueError(f"пересечение {left_name} и {right_name}")
