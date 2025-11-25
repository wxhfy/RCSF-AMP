#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


class ESMEmbedder:
    """
    Computes protein sequence embeddings using ESM models from HuggingFace Transformers.
    """
    def __init__(self,
                 model_name: str = "facebook/esm2_t36_3B_UR50D",
                 local_model_path_root: Optional[str] = None,
                 device: Union[str, torch.device] = None,
                 repr_layer: Optional[int] = 36,
                 include_bos_eos: bool = False,
                 max_sequence_length: int = 1022):
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.repr_layer_index = repr_layer
        self.include_bos_eos = include_bos_eos
        self.max_sequence_length = max_sequence_length

        # Build a prioritized list of candidate locations for the ESM weights.
        model_load_paths = []

        def _collect_valid_paths(raw_entry: Optional[str]):
            if not raw_entry:
                return []
            # Allow os.pathsep separated entries so users can pass multiple roots.
            entries = [seg for seg in raw_entry.split(os.pathsep) if seg]
            valid = []
            for entry in entries:
                path_obj = Path(entry).expanduser().resolve()
                if path_obj.is_dir():
                    valid.append(str(path_obj))
            return valid

        # 1) Explicitly provided path(s)
        model_load_paths.extend(_collect_valid_paths(local_model_path_root))

        # 2) Environment override (ESM_MODEL_DIRS allows quick reconfiguration without editing code)
        model_load_paths.extend(_collect_valid_paths(os.environ.get("ESM_MODEL_DIRS")))

        # 3) Built-in fallbacks for this project
        default_local_roots = [
            "/home/4T-1/fyh/work/utils/esm_lora/esm2/",
            "/home/fyh0106/SUCF/baseline/esm-AxP-GDL/models/esm2/checkpoints/",
        ]
        for fallback in default_local_roots:
            if str(Path(fallback).expanduser().resolve()) not in model_load_paths:
                model_load_paths.extend(_collect_valid_paths(fallback))

        # Finally, fall back to pulling from HuggingFace Hub
        model_load_paths.append(model_name)

        last_error = None
        for load_path in model_load_paths:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
                model_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
                load_kwargs = {"torch_dtype": model_dtype, "trust_remote_code": True}
                self.model = AutoModel.from_pretrained(load_path, **load_kwargs)
                self.model.to(self.device)
                self.model.eval()

                # Determine the total number of layers to set the default representation layer
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_hidden_layers'):
                    num_model_layers = self.model.config.num_hidden_layers
                else:
                    # Fallback for models without a standard config attribute
                    known_layers = {"esm2_t36_3B_UR50D": 36, "esm2_t33_650M_UR50D": 33}
                    model_key = model_name.split('/')[-1]
                    num_model_layers = known_layers.get(model_key, 36) # Default to 36 if unknown

                if self.repr_layer_index is None or not (0 <= self.repr_layer_index <= num_model_layers):
                    self.repr_layer_index = num_model_layers # Default to the last layer

                break  # Success, exit loop
            except Exception as e:
                logger.warning(f"Failed to load ESM model from {load_path}: {e}")
                last_error = e
        else:
            logger.error(f"Failed to initialize ESMEmbedder model from all sources. Last error: {last_error}", exc_info=True)
            raise last_error

        logger.info(f"ESMEmbedder initialized on device '{self.device}'. Embeddings will be from layer {self.repr_layer_index}.")

    def embed_batch(self, batch_ids_seqs: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        """
        Embeds a batch of sequences.
        """
        if not batch_ids_seqs:
            return {}

        batch_strs = [seq for _, seq in batch_ids_seqs]
        tokenizer_max_len = self.max_sequence_length + self.tokenizer.num_special_tokens_to_add(pair=False)

        inputs = self.tokenizer(
            batch_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer_max_len,
        ).to(self.device)

        batch_embeddings = {}
        with torch.no_grad(), autocast(enabled=(self.device.type == 'cuda')):
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            
            # Extract representations from the specified layer
            hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
            if hidden_states is None:
                logger.warning("Model output does not contain hidden states.")
                return {seq_id: np.array([], dtype=np.float32) for seq_id, _ in batch_ids_seqs}
            
            token_representations = hidden_states[self.repr_layer_index]

            for i, (seq_id, _) in enumerate(batch_ids_seqs):
                seq_len = inputs['attention_mask'][i].sum().item()
                
                # Slice to get embeddings for actual tokens, excluding padding
                start_idx = 1 if not self.include_bos_eos else 0
                end_idx = seq_len - 1 if not self.include_bos_eos else seq_len
                
                if start_idx >= end_idx:
                    embedding = np.array([], dtype=np.float32)
                else:
                    embedding = token_representations[i, start_idx:end_idx].cpu().numpy()
                
                batch_embeddings[seq_id] = embedding
                
        return batch_embeddings



def embed_sequences_and_save(
    sequence_data: List[Dict[str, str]],
    output_dir: str,
    model_name: str,
    local_model_path_root: Optional[str] = None,
    repr_layer: Optional[int] = None,
    include_bos_eos: bool = False,
):
    """
    Computes and saves protein sequence embeddings using a single process.
    This function is a simplified wrapper around the ESMEmbedder class.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not sequence_data:
        logger.error("No sequences provided to embed.")
        return {}

    # Initialize the embedder on the primary CUDA device if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedder = ESMEmbedder(
        model_name=model_name,
        local_model_path_root=local_model_path_root,
        device=device,
        repr_layer=repr_layer,
        include_bos_eos=include_bos_eos
    )

    saved_paths = {}
    for item in tqdm(sequence_data, desc="Generating ESM Embeddings", unit="seq"):
        seq_id, seq_str = item.get("id"), item.get("sequence")
        if not seq_id or not seq_str:
            continue

        embedding_dict = embedder.embed_batch([(seq_id, seq_str)])
        embedding = embedding_dict.get(seq_id)

        if embedding is not None:
            out_file = output_path / f"{seq_id}.npy"
            try:
                np.save(out_file, embedding)
                saved_paths[seq_id] = str(out_file)
            except Exception as e:
                logger.error(f"Failed to save embedding for {seq_id}: {e}")
                saved_paths[seq_id] = None
    
    return saved_paths