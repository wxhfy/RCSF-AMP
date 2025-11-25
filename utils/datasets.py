#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
import logging

logger = logging.getLogger(__name__)


def custom_collate_fn(data_list: List[Data]) -> Batch:
    """
    Custom collate function that filters out None values before batching.
    This prevents errors if the Dataset's get() method returns None for a corrupted file.
    """
    # Filter out any samples that failed to load (returned as None)
    valid_data_list = [data for data in data_list if data is not None]
    if not valid_data_list:
        return Batch()
    
    # Use the default PyG batching mechanism on the filtered list
    return Batch.from_data_list(valid_data_list)


class AMPGraphDataset(Dataset):
    """
    A PyTorch Geometric Dataset for loading pre-computed AMP graph data.
    """
    def __init__(
            self,
            root_dir: str,
            split_file: Optional[str] = None,
            transform=None,
            extra_embeddings: Optional[Dict] = None
    ):
        self.root_dir = os.path.expanduser(root_dir)
        self.data_files_dir = os.path.join(self.root_dir, 'graphs')
        self.file_list: List[str] = []
        self.extra_embeddings = extra_embeddings or {}

        if not os.path.isdir(self.data_files_dir):
            logger.warning(f"Graph data directory not found: {self.data_files_dir}. Dataset will be empty.")
            super().__init__(root=self.root_dir, transform=transform)
            return

        if split_file and os.path.exists(split_file):
            with open(split_file, "r") as f:
                self.file_list = [line.strip() for line in f if line.strip().endswith(".pt")]
            logger.info(f"Loaded {len(self.file_list)} graph references from split file: {split_file}")
        else:
            logger.info(f"No split file provided. Loading all .pt files from {self.data_files_dir}.")
            self.file_list = [f for f in os.listdir(self.data_files_dir) if f.endswith(".pt")]
            logger.info(f"Found {len(self.file_list)} graph files in directory.")

        if not self.file_list:
            logger.warning(f"No .pt files found in {self.data_files_dir} with split file '{split_file}'.")
            
        super().__init__(root=self.root_dir, transform=transform)

    @property
    def processed_file_names(self) -> List[str]:
        # This list tells PyG what files to expect in the processed_dir.
        return self.file_list

    def len(self) -> int:
        return len(self.file_list)

    def get(self, idx: int) -> Optional[Data]:
        if idx >= len(self.file_list):
            logger.error(f"Index {idx} is out of bounds for dataset with size {len(self.file_list)}.")
            return None

        file_path = os.path.join(self.data_files_dir, self.file_list[idx])
        try:
            data = torch.load(file_path, map_location=torch.device('cpu'))
            if not isinstance(data, Data):
                logger.error(f"File {file_path} does not contain a valid PyG Data object.")
                return None

            # Attach additional embeddings if requested
            if self.extra_embeddings:
                seq_id = getattr(data, 'seq_id', Path(self.file_list[idx]).stem)
                num_nodes = getattr(data, 'num_nodes', data.x.size(0) if hasattr(data, 'x') else None)

                for name, cfg in self.extra_embeddings.items():
                    attr_name = cfg.get('attr', name)
                    source_type = cfg.get('type', 'file')

                    if source_type == 'file':
                        base_dir = cfg.get('path')
                        if not base_dir:
                            continue
                        embedding_path = os.path.join(base_dir, f"{seq_id}.npy")
                        if not os.path.exists(embedding_path):
                            logger.warning(f"Missing embedding file for '{name}': {embedding_path}")
                            continue
                        try:
                            embedding = np.load(embedding_path)
                        except Exception as exc:
                            logger.error(f"Failed to load embedding '{name}' from {embedding_path}: {exc}")
                            continue

                        if embedding.ndim == 3 and embedding.shape[0] == 1:
                            embedding = embedding[0]

                        if num_nodes is not None and embedding.shape[0] == num_nodes + 2:
                            # Assume CLS/SEP tokens and trim them
                            embedding = embedding[1:-1]

                        if num_nodes is not None and embedding.shape[0] != num_nodes:
                            logger.warning(
                                f"Embedding length mismatch for '{name}' (seq_id={seq_id}): "
                                f"expected {num_nodes}, got {embedding.shape[0]}. Skipping."
                            )
                            continue

                        setattr(data, attr_name, torch.from_numpy(embedding).float())

                    elif source_type == 'one_hot_from_x':
                        if not hasattr(data, 'x'):
                            logger.warning(f"Data sample {seq_id} missing 'x', cannot derive one-hot embedding.")
                            continue
                        one_hot_dim = cfg.get('dim', 21)
                        one_hot = data.x[:, :one_hot_dim].float()
                        setattr(data, attr_name, one_hot)

            return data
        except Exception as e:
            logger.error(f"Failed to load or process data from {file_path}: {e}")
            return None


def create_dataloader(
        dataset: Dataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        generator: torch.Generator = None,
        worker_init_fn=None
) -> PyGDataLoader:
    """Creates a PyG DataLoader with a custom collate function."""
    if len(dataset) == 0:
        logger.warning("Dataset is empty. Returning an empty DataLoader.")
        return PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Pass generator and worker_init_fn through to ensure reproducible shuffling
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )

    if generator is not None:
        loader_kwargs['generator'] = generator
    if worker_init_fn is not None:
        loader_kwargs['worker_init_fn'] = worker_init_fn

    return PyGDataLoader(**loader_kwargs)


def create_data_loaders(config: Dict) -> tuple:
    """
    Creates the training, validation, and optional test data loaders.
    """
    data_root = config.get('data_root', './data')
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    pin_memory = config.get('pin_memory', True)

    train_txt = os.path.join(data_root, 'train.txt')
    val_txt = os.path.join(data_root, 'val.txt')
    test_txt = os.path.join(data_root, 'test.txt')
    
    if not os.path.exists(train_txt) or not os.path.exists(val_txt):
        raise FileNotFoundError(f"train.txt or val.txt not found in data root: {data_root}")

    data_cfg = config.get('data', {})
    extra_embeddings_cfg = data_cfg.get('extra_embeddings', {})

    train_dataset = AMPGraphDataset(data_root, split_file=train_txt, extra_embeddings=extra_embeddings_cfg)
    val_dataset = AMPGraphDataset(data_root, split_file=val_txt, extra_embeddings=extra_embeddings_cfg)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"Train or validation dataset is empty. Check your data files and split files.")

    # Prepare a torch.Generator seeded for reproducible shuffling
    seed = config.get('random_seed', None)
    generator = None
    worker_init = None
    if seed is not None:
        try:
            generator = torch.Generator()
            generator.manual_seed(int(seed))

            # worker_init_fn to ensure numpy/python random are seeded in each worker
            def make_worker_init_fn(base_seed: int):
                def _worker_init(worker_id: int):
                    worker_seed = base_seed + worker_id
                    try:
                        import numpy as _np
                        _np.random.seed(worker_seed)
                    except Exception:
                        pass
                    try:
                        import random as _random
                        _random.seed(worker_seed)
                    except Exception:
                        pass
                    try:
                        torch.manual_seed(worker_seed)
                    except Exception:
                        pass
                return _worker_init

            worker_init = make_worker_init_fn(int(seed))
        except Exception:
            logger.warning("Failed to create torch.Generator for reproducible DataLoader shuffling.")

    train_loader = create_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        generator=generator, worker_init_fn=worker_init
    )
    
    val_loader = create_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        generator=generator, worker_init_fn=worker_init
    )
    
    logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    
    test_loader = None
    if os.path.exists(test_txt):
        test_dataset = AMPGraphDataset(data_root, split_file=test_txt, extra_embeddings=extra_embeddings_cfg)
        if len(test_dataset) > 0:
            test_loader = create_dataloader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                generator=generator, worker_init_fn=worker_init
            )
            logger.info(f"Loaded {len(test_dataset)} test samples.")
        else:
            logger.warning(f"Test split file {test_txt} was found but resulted in an empty dataset.")

    return (train_loader, val_loader, test_loader) if test_loader is not None else (train_loader, val_loader)