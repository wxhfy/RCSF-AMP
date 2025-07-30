#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
from multiprocessing import cpu_count  #

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import logging

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel  # Added for LoRA #
from torch.cuda.amp import autocast
import sys
logger = logging.getLogger(__name__)


# 为了更好地调试多进程，可以在日志格式中加入进程ID
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s:%(levelname)s:%(process)d] %(message)s")


class _SequenceDataset(torch.utils.data.Dataset):
    """用于ESM序列嵌入的内部PyTorch数据集类。"""

    def __init__(self, sequence_ids: List[str], sequences: List[str]):
        if len(sequence_ids) != len(sequences):
            raise ValueError("序列ID列表和序列列表的长度必须一致。")
        self.sequence_ids = sequence_ids
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.sequence_ids[idx], self.sequences[idx]


class ESMEmbedder:
    """
    使用Transformers库加载的ESM模型计算蛋白质序列嵌入的类。
    支持从本地路径加载基础模型并应用LoRA adapter。
    """

    def __init__(self,
                 model_name: str = "facebook/esm2_t36_3B_UR50D",
                 local_model_path_root: Optional[str] = None,
                 device: Union[str, torch.device] = None,
                 lora_weights_path: Optional[str] = None,
                 repr_layer: Optional[int] = 36,
                 include_bos_eos: bool = False,
                 max_sequence_length: int = 1022
                 ):
        self.model_hub_name_original = model_name
        # 强制优先使用GPU
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lora_weights_path = lora_weights_path
        self.repr_layer_index = repr_layer
        self.include_bos_eos = include_bos_eos
        self.max_sequence_length = max_sequence_length

        # 仅保留必要的错误日志
        base_model_load_path = model_name
        if local_model_path_root:
            resolved_local_path = Path(local_model_path_root).resolve()
            if resolved_local_path.is_dir() and (resolved_local_path / "config.json").exists():
                base_model_load_path = str(resolved_local_path)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_load_path, trust_remote_code=True)
            model_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
            load_kwargs = {"torch_dtype": model_dtype, "trust_remote_code": True}
            if Path(base_model_load_path).is_dir() and (Path(base_model_load_path) / "pytorch_model.bin").exists() or \
                    Path(base_model_load_path).is_dir() and list(Path(base_model_load_path).glob("*.safetensors")):
                pass
            base_model_for_peft = AutoModel.from_pretrained(base_model_load_path, **load_kwargs)
            use_lora = False
            if self.lora_weights_path and self.lora_weights_path.lower() not in ['null', 'none', '']:
                resolved_lora_path = Path(self.lora_weights_path).resolve()
                if resolved_lora_path.is_dir():
                    adapter_config_file = resolved_lora_path / "adapter_config.json"
                    if adapter_config_file.exists() and adapter_config_file.is_file():
                        try:
                            self.model = PeftModel.from_pretrained(base_model_for_peft, str(resolved_lora_path))
                            use_lora = True
                        except Exception as e_peft:
                            logger.error(f"ESMEmbedder: 应用LoRA adapter '{resolved_lora_path}' 失败: {e_peft}")
                            self.model = base_model_for_peft
                    else:
                        self.model = base_model_for_peft
                else:
                    self.model = base_model_for_peft
            else:
                self.model = base_model_for_peft
            self.model.to(self.device)
            self.model.eval()
            effective_config = None
            try:
                if hasattr(self.model, 'get_base_model') and callable(self.model.get_base_model):
                    effective_config = self.model.get_base_model().config
                elif hasattr(self.model, 'config'):
                    effective_config = self.model.config
            except Exception as e_config:
                pass
            num_model_layers = -1
            if effective_config and hasattr(effective_config, 'num_hidden_layers'):
                num_model_layers = effective_config.num_hidden_layers
            else:
                known_esm_layers = {
                    "esm2_t48_15B_UR50D": 48, "esm2_t36_3B_UR50D": 36,
                    "esm2_t33_650M_UR50D": 33, "esm2_t30_150M_UR50D": 30,
                    "esm2_t12_35M_UR50D": 12, "esm2_t6_8M_UR50D": 6
                }
                model_key_part = self.model_hub_name_original.split('/')[-1]
                guessed_layers = known_esm_layers.get(model_key_part)
                if guessed_layers:
                    num_model_layers = guessed_layers
                else:
                    num_model_layers = 36
            if self.repr_layer_index is None:
                self.repr_layer_index = num_model_layers
            elif not (0 <= self.repr_layer_index <= num_model_layers):
                self.repr_layer_index = num_model_layers
        except Exception as e_init:
            logger.error(f"ESMEmbedder: 初始化模型或Tokenizer时发生严重错误: {e_init}", exc_info=True)
            raise

        try:
            logger.info(f"ESMEmbedder: 从 {base_model_load_path} 加载Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_load_path, trust_remote_code=True)

            logger.info(f"ESMEmbedder: 从 {base_model_load_path} 加载基础ESM模型...")
            model_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
            if self.device.type == 'cuda':
                logger.info(f"ESMEmbedder: 将在CUDA上以 {model_dtype} 加载模型。")

            # 尝试加载基础模型，优先使用 local_files_only=True 如果路径看起来像完整下载
            load_kwargs = {"torch_dtype": model_dtype, "trust_remote_code": True}
            if Path(base_model_load_path).is_dir() and (Path(base_model_load_path) / "pytorch_model.bin").exists() or \
                    Path(base_model_load_path).is_dir() and list(Path(base_model_load_path).glob("*.safetensors")):
                # 如果是目录且包含典型权重文件，尝试local_files_only
                # load_kwargs["local_files_only"] = True # 可选，如果网络慢或模型已完整下载
                pass

            base_model_for_peft = AutoModel.from_pretrained(base_model_load_path, **load_kwargs)
            logger.info(f"ESMEmbedder: 基础模型 {base_model_load_path} 加载成功。类型: {type(base_model_for_peft)}")

            # 应用LoRA adapter（支持条件性使用）
            use_lora = False
            if self.lora_weights_path and self.lora_weights_path.lower() not in ['null', 'none', '']:
                resolved_lora_path = Path(self.lora_weights_path).resolve()
                
                if resolved_lora_path.is_dir():
                    adapter_config_file = resolved_lora_path / "adapter_config.json"
                    if adapter_config_file.exists() and adapter_config_file.is_file():
                        try:
                            # 加载LoRA时，基础模型应已在期望的dtype
                            logger.info(f"ESMEmbedder: 正在应用LoRA adapter: {resolved_lora_path}")
                            self.model = PeftModel.from_pretrained(base_model_for_peft, str(resolved_lora_path))
                            use_lora = True
                            logger.info("ESMEmbedder: LoRA adapter应用成功！")
                        except Exception as e_peft:
                            logger.error(f"ESMEmbedder: 应用LoRA adapter '{resolved_lora_path}' 失败: {e_peft}")
                            logger.warning("ESMEmbedder: 将回退到使用没有LoRA的基础模型。")
                            self.model = base_model_for_peft
                    else:
                        logger.warning(
                            f"ESMEmbedder: LoRA adapter路径 '{resolved_lora_path}' 中未找到 'adapter_config.json' 文件。"
                            "请确保路径正确且adapter已正确保存。将使用没有LoRA的基础模型。"
                        )
                        self.model = base_model_for_peft
                else:
                    logger.warning(
                        f"ESMEmbedder: 提供的LoRA adapter路径 '{self.lora_weights_path}' (resolved: {resolved_lora_path}) "
                        "不是一个有效的目录。将使用没有LoRA的基础模型。"
                    )
                    self.model = base_model_for_peft
            else:
                logger.info("ESMEmbedder: 未提供有效的LoRA权重路径，将使用原始基础模型。")
                self.model = base_model_for_peft

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"ESMEmbedder: 最终模型已移至 {self.device} 并设置为评估模式。")
            
            # 打印详细的模型信息
            logger.info("=" * 60)
            logger.info("ESM2 模型配置信息:")
            logger.info(f"  基础模型路径: {base_model_load_path}")
            logger.info(f"  是否为本地模型: {Path(base_model_load_path).is_dir()}")
            logger.info(f"  使用LoRA适配器: {'是' if use_lora else '否'}")
            if use_lora:
                logger.info(f"  LoRA适配器路径: {self.lora_weights_path}")
            logger.info(f"  模型类型: {type(self.model).__name__}")
            logger.info(f"  设备: {self.device}")
            logger.info(f"  数据类型: {model_dtype}")
            
            # 获取模型参数信息
            try:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                logger.info(f"  总参数数量: {total_params:,}")
                logger.info(f"  可训练参数数量: {trainable_params:,}")
                logger.info(f"  参数冻结比例: {(total_params - trainable_params) / total_params * 100:.1f}%")
            except Exception as e:
                logger.warning(f"  无法获取参数统计信息: {e}")
            logger.info("=" * 60)

            # 确定隐藏层数量 (num_model_layers)
            # PeftModel 会将其基础模型的配置存储在 self.model.config (如果基础模型有config)
            # 或者更可靠地通过 self.model.get_base_model().config
            effective_config = None
            try:
                if hasattr(self.model, 'get_base_model') and callable(self.model.get_base_model):
                    effective_config = self.model.get_base_model().config
                elif hasattr(self.model, 'config'):  # 对于非PEFT或PEFT直接暴露config
                    effective_config = self.model.config
            except Exception as e_config:
                logger.warning(f"ESMEmbedder: 获取模型配置时出错: {e_config}")

            num_model_layers = -1
            if effective_config and hasattr(effective_config, 'num_hidden_layers'):
                num_model_layers = effective_config.num_hidden_layers
                logger.info(f"ESMEmbedder: 从模型配置中获取 num_hidden_layers: {num_model_layers}")
            else:
                # Fallback logic based on model name if config is not accessible
                known_esm_layers = {
                    "esm2_t48_15B_UR50D": 48, "esm2_t36_3B_UR50D": 36,
                    "esm2_t33_650M_UR50D": 33, "esm2_t30_150M_UR50D": 30,
                    "esm2_t12_35M_UR50D": 12, "esm2_t6_8M_UR50D": 6
                }
                model_key_part = self.model_hub_name_original.split('/')[-1]
                guessed_layers = known_esm_layers.get(model_key_part)

                if guessed_layers:
                    num_model_layers = guessed_layers
                    logger.warning(
                        f"ESMEmbedder: 无法从模型配置 ({type(self.model)}) 中可靠地获取num_hidden_layers。 "
                        f"基于模型名称 '{self.model_hub_name_original}' 猜测为: {num_model_layers} 层。"
                    )
                else:  # Final fallback
                    num_model_layers = 36  # Default for common large ESM
                    logger.error(
                        f"ESMEmbedder: 无法确定模型的层数。将默认使用 {num_model_layers} 层。这可能不正确！"
                    )

            # hidden_states 返回的元组长度是 num_model_layers + 1 (0是输入嵌入, 1 to num_model_layers 是transformer层输出)
            # 所以有效的索引是 0 到 num_model_layers
            if self.repr_layer_index is None:  # 用户未指定，默认为最后一层transformer的输出
                self.repr_layer_index = num_model_layers  # Index for outputs.hidden_states
                logger.info(
                    f"ESMEmbedder: repr_layer 未指定，将从最后一层 (hidden_states 索引 {self.repr_layer_index}) 提取表示。")
            elif not (0 <= self.repr_layer_index <= num_model_layers):
                logger.warning(
                    f"ESMEmbedder: 指定的 repr_layer_index {self.repr_layer_index} 超出有效范围 "
                    f"[0 (输入嵌入层), {num_model_layers} (最后一Transformer层)]. "
                    f"将使用最后一层 (索引 {num_model_layers}) 作为替代。"
                )
                self.repr_layer_index = num_model_layers
            else:
                logger.info(f"ESMEmbedder: 将从指定的 hidden_states 索引 {self.repr_layer_index} 提取表示。")


        except Exception as e_init:
            logger.error(f"ESMEmbedder: 初始化模型或Tokenizer时发生严重错误: {e_init}", exc_info=True)
            raise

        logger.info(
            f"ESMEmbedder: 模型和Tokenizer准备完毕。Include BOS/EOS: {self.include_bos_eos}, Representation Layer Index: {self.repr_layer_index}")

    def embed_batch(self, batch_ids_seqs: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        if not batch_ids_seqs:
            return {}

        processed_batch_data = []
        for item in batch_ids_seqs:
            if isinstance(item, tuple) and len(item) == 2:
                processed_batch_data.append(item)
            elif isinstance(item, dict) and "id" in item and "sequence" in item:
                processed_batch_data.append((item["id"], item["sequence"]))
            else:
                logger.warning(f"ESMEmbedder: 跳过格式不正确的序列数据项: {item}")
                continue

        if not processed_batch_data:
            return {}

        batch_strs = [seq_str for _, seq_str in processed_batch_data]
        # tokenizer.num_special_tokens_to_add(pair=False) 通常是2 (BOS, EOS)
        tokenizer_max_len = self.max_sequence_length + self.tokenizer.num_special_tokens_to_add(pair=False)

        inputs = self.tokenizer(
            batch_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer_max_len,
        ).to(self.device)

        batch_embeddings = {}
        with torch.no_grad(), autocast(enabled=(
                self.device.type == 'cuda' and self.model.dtype == torch.float16)):  # Autocast if model is fp16 on CUDA
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

            if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None or not outputs.hidden_states:
                logger.error("ESMEmbedder: 模型输出中未找到 'hidden_states'。")
                if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    logger.warning("ESMEmbedder: 使用 'last_hidden_state' 作为备选。")
                    token_representations = outputs.last_hidden_state
                else:
                    logger.error("ESMEmbedder: 模型输出中也未找到 'last_hidden_state'。无法提取嵌入。")
                    for seq_id, _ in processed_batch_data:
                        batch_embeddings[seq_id] = np.array([], dtype=np.float32)  # 返回空数组表示失败
                    return batch_embeddings  # 提前返回，因为无法提取表示
            else:
                # self.repr_layer_index 已在 __init__ 中验证并设置为有效值
                token_representations = outputs.hidden_states[self.repr_layer_index]

            for i, (seq_id, original_seq_str) in enumerate(processed_batch_data):
                true_len_with_special_tokens = inputs['attention_mask'][i].sum().item()

                if self.include_bos_eos:
                    # 包含BOS和EOS (以及之间的所有tokens)
                    embedding = token_representations[i, :true_len_with_special_tokens].cpu().numpy()
                else:
                    # 不包含BOS和EOS，只取氨基酸残基的表示
                    # 假设BOS是第0个token, EOS是最后一个有效token (true_len_with_special_tokens - 1)
                    # 氨基酸序列在 token_representations[i, 1 : true_len_with_special_tokens-1]
                    start_idx = 1
                    end_idx = true_len_with_special_tokens - 1
                    if start_idx >= end_idx:  # 例如，序列在截断后只剩下BOS和EOS或更少
                        logger.warning(
                            f"ESMEmbedder: 序列 {seq_id} (原始长度 {len(original_seq_str)}) 在移除BOS/EOS后没有剩余的残基表示 "
                            f"(tokenized length with special tokens: {true_len_with_special_tokens})。返回空嵌入。"
                        )
                        embedding = np.array([], dtype=np.float32)
                    else:
                        embedding = token_representations[i, start_idx:end_idx].cpu().numpy()
                batch_embeddings[seq_id] = embedding
        return batch_embeddings


def _worker_process_embedding(args_tuple: Tuple) -> Optional[Dict[str, str]]:
    worker_id, device_id_str, model_hub_name_param, local_model_dir_param, lora_path_param, \
        layer_to_extract_param, include_special_tokens_param, seq_chunk_tuples_param, out_dir_param, BATCH_SIZE_GPU_param = args_tuple

    # 为每个工作进程配置独立的日志记录器或使用主进程的（如果配置允许）
    # logger = logging.getLogger(f"_worker_process_embedding_{worker_id}") # 可选：独立logger
    # handler = logging.StreamHandler()
    # formatter = logging.Formatter("%(asctime)s [%(name)s:%(levelname)s:%(process)d] %(message)s")
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    # logger.setLevel(logging.INFO) # 或者从主进程继承

    logger.info(
        f"Worker {worker_id} (PID {os.getpid()}) on device {device_id_str}: Starting, processing {len(seq_chunk_tuples_param)} sequences.")
    saved_file_paths = {}  # type: Dict[str, Optional[str]]

    try:
        embedder_instance = ESMEmbedder(
            model_name=model_hub_name_param,
            local_model_path_root=local_model_dir_param,
            device=device_id_str,
            lora_weights_path=lora_path_param,
            repr_layer=layer_to_extract_param,
            include_bos_eos=include_special_tokens_param
        )

        num_batches_in_chunk = (len(seq_chunk_tuples_param) + BATCH_SIZE_GPU_param - 1) // BATCH_SIZE_GPU_param
        # 为工作进程创建进度条
        worker_pbar = tqdm(
            total=len(seq_chunk_tuples_param),
            desc=f"Worker {worker_id} (GPU {device_id_str})"
        )

        for batch_idx in range(num_batches_in_chunk):
            batch_start = batch_idx * BATCH_SIZE_GPU_param
            batch_end = min((batch_idx + 1) * BATCH_SIZE_GPU_param, len(seq_chunk_tuples_param))
            batch_tuples = seq_chunk_tuples_param[batch_start:batch_end]
            batch_embeddings = embedder_instance.embed_batch(batch_tuples)
            for seq_id, embedding in batch_embeddings.items():
                out_file = Path(out_dir_param) / f"{seq_id}.npy"
                # 确保目录存在
                out_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    np.save(out_file, embedding)
                    saved_file_paths[seq_id] = str(out_file)
                except Exception as e_save:
                    logger.error(f"Worker {worker_id}: 保存序列 {seq_id} 的嵌入到 {out_file} 时出错: {e_save}")
                    saved_file_paths[seq_id] = None
            worker_pbar.update(len(batch_tuples))

        worker_pbar.close()

        logger.info(
            f"Worker {worker_id} (PID {os.getpid()}) on device {device_id_str}: Finished. "
            f"Successfully processed {len([p for p in saved_file_paths.values() if p is not None])} sequences in this chunk."
        )
        return saved_file_paths
    except Exception as e_proc_init:
        logger.error(
            f"Worker {worker_id} (PID {os.getpid()}) on device {device_id_str}: CRITICAL ERROR during initialization or execution: {e_proc_init}",
            exc_info=True)
        failed_results = {seq_identifier: None for seq_identifier, _ in seq_chunk_tuples_param}
        return failed_results


def embed_sequences_multi_gpu(
    sequence_data,
    output_dir,
    model_name,
    local_model_path_root=None,
    lora_weights_path=None,
    repr_layer=None,
    include_bos_eos=False,
    batch_size_per_gpu=8,
    gpu_ids=None,
    num_workers=None
):
    """
    多GPU/多进程并行计算蛋白质序列嵌入，并保存为npy文件。
    """
    from pathlib import Path
    import math

    # 目录创建由主训练脚本负责，这里不再创建多余目录
    # 预处理输入数据
    sequences_to_process_tuples = []
    for item in sequence_data:
        if isinstance(item, dict) and "id" in item and "sequence" in item:
            sequences_to_process_tuples.append((item["id"], item["sequence"]))
        elif isinstance(item, tuple) and len(item) == 2:
            sequences_to_process_tuples.append(item)
        else:
            logger.warning(f"embed_sequences_multi_gpu: 跳过格式不正确的序列数据项: {item}")

    if not sequences_to_process_tuples:
        logger.error("embed_sequences_multi_gpu: 没有可处理的序列。")
        return {}

    # 只用单进程，强制使用cuda:0（如可用）
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedder = ESMEmbedder(
        model_name=model_name,
        local_model_path_root=local_model_path_root,
        device=device,
        lora_weights_path=lora_weights_path,
        repr_layer=repr_layer,
        include_bos_eos=include_bos_eos
    )
    final_saved_paths = {}
    from tqdm import tqdm
    for seq_id, seq_str in tqdm(sequences_to_process_tuples, desc="ESM嵌入", unit="seq"):
        embedding_dict = embedder.embed_batch([(seq_id, seq_str)])
        embedding = embedding_dict.get(seq_id, None)
        out_file = Path(output_dir) / f"{seq_id}.npy"
        # 确保目录存在
        out_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.save(out_file, embedding)
            final_saved_paths[seq_id] = str(out_file)
        except Exception as e_save:
            logger.error(f"保存序列 {seq_id} 的嵌入到 {out_file} 时出错: {e_save}")
            final_saved_paths[seq_id] = None
    return final_saved_paths


if __name__ == "__main__":
    # --- Basic Logging Setup for Standalone Script Execution ---
    # Note: If this module is imported, the calling script should configure logging.
    if not logging.getLogger().hasHandlers():  # Configure only if no handlers are set
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s:%(levelname)s:%(process)d] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]  # Changed to sys.stdout
        )

    # --- Multiprocessing Start Method (Crucial for CUDA) ---
    # It's best to set this once if the script can be the main entry point.
    # 'spawn' is generally safer for CUDA.
    try:
        current_start_method = torch.multiprocessing.get_start_method(allow_none=True)
        desired_start_method = "spawn"
        if current_start_method is None:
            torch.multiprocessing.set_start_method(desired_start_method)
            logger.info(f"Multiprocessing start method set to '{desired_start_method}'.")
        elif current_start_method != desired_start_method:
            # Forcing can be risky if other parts of an application expect a different method.
            # mp.set_start_method(desired_start_method, force=True)
            logger.warning(
                f"Multiprocessing start method is already '{current_start_method}'. "
                f"For CUDA, '{desired_start_method}' is often recommended."
            )
        else:  # Already set to desired_start_method
            logger.info(f"Multiprocessing start method is already '{desired_start_method}'.")
    except RuntimeError as e_mp_set:
        logger.warning(f"Could not set multiprocessing start method to 'spawn' (Reason: {e_mp_set}). "
                       "This might lead to issues with CUDA in subprocesses.")

    parser = argparse.ArgumentParser(description="使用ESM模型并行计算蛋白质序列的嵌入。")
    parser.add_argument("--sequence_file", type=str, required=True,
                        help="包含序列数据的JSON文件路径。JSON文件应为一个列表，每个元素是一个包含 'id' 和 'sequence' 键的字典。")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="保存嵌入的输出目录。每个序列的嵌入将保存为 <id>.npy。")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t36_3B_UR50D",
                        help="要使用的ESM模型名称 (例如 'esm2_t30_150M_UR50D')。")
    parser.add_argument("--local_model_path_root", type=str, default=None,
                        help="ESM基础模型文件所在的本地根目录 (例如 /path/to/esm_models/esm2_t36_3B_UR50D/)。")
    parser.add_argument("--lora_weights_path", type=str, default=None,
                        help="可选的LoRA adapter目录路径 (包含adapter_config.json和权重)。")
    parser.add_argument("--repr_layer", type=int, default=None,
                        help="从中提取表示的Transformer层索引 (0-based for hidden_states)。默认为最后一层。")
    parser.add_argument("--include_bos_eos", action="store_true",
                        help="在每个残基的嵌入中包含BOS和EOS token的表示 (默认为不包含)。")
    parser.add_argument("--batch_size_per_gpu", type=int, default=8,  # 之前是1，可以适当调大
                        help="在每个GPU上进行嵌入计算时的内部批处理大小。")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="要使用的GPU ID列表，用逗号分隔 (例如 '0,1,2')。如果未提供或为空，则使用CPU模式。")
    parser.add_argument("--num_cpu_workers", type=int, default=None,
                        help="在CPU模式下使用的进程数。默认为 os.cpu_count()。")

    args = parser.parse_args()
    import sys  # Add sys import for StreamHandler(sys.stdout)

    try:
        with open(args.sequence_file, 'r') as f:
            sequence_data_list = json.load(f)
        if not isinstance(sequence_data_list, list) or \
                not all(isinstance(item, dict) and "id" in item and "sequence" in item for item in sequence_data_list):
            raise ValueError("输入JSON文件格式不正确或内容不符合预期（缺少'id'或'sequence'）。")
    except FileNotFoundError:
        logger.error(f"序列文件未找到: {args.sequence_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"无法解析JSON序列文件: {args.sequence_file}")
        sys.exit(1)
    except ValueError as e_val:
        logger.error(f"序列文件内容错误: {e_val}")
        sys.exit(1)
    except Exception as e_file:
        logger.error(f"加载序列文件 {args.sequence_file} 时发生未知错误: {e_file}", exc_info=True)
        sys.exit(1)

    parsed_gpu_ids = None
    if args.gpu_ids:
        try:
            parsed_gpu_ids = [int(gid.strip()) for gid in args.gpu_ids.split(',') if gid.strip()]
            if not torch.cuda.is_available():
                logger.warning("CUDA 不可用，请求的 GPU 将被忽略。将使用 CPU。")
                parsed_gpu_ids = None
            elif parsed_gpu_ids and not all(0 <= gid < torch.cuda.device_count() for gid in parsed_gpu_ids):
                # 过滤掉无效的GPU ID，只保留有效的
                valid_gids = [gid for gid in parsed_gpu_ids if 0 <= gid < torch.cuda.device_count()]
                if len(valid_gids) < len(parsed_gpu_ids):
                    logger.warning(
                        f"部分请求的 GPU ID {parsed_gpu_ids} 无效或超出范围 (可用GPU数量: {torch.cuda.device_count()})。 "
                        f"将仅使用有效的GPU ID: {valid_gids}。"
                    )
                parsed_gpu_ids = valid_gids
                if not parsed_gpu_ids:  # 如果所有提供的ID都无效
                    logger.warning("所有提供的GPU ID均无效。将使用 CPU。")
                    parsed_gpu_ids = None
        except ValueError:
            logger.error(f"无效的 GPU ID 格式: '{args.gpu_ids}'。将使用 CPU。")
            parsed_gpu_ids = None

    embed_sequences_multi_gpu(
        sequence_data=sequence_data_list,
        output_dir=args.output_dir,
        model_name=args.model_name,
        local_model_path_root=args.local_model_path_root,
        lora_weights_path=args.lora_weights_path,
        repr_layer=args.repr_layer,
        include_bos_eos=args.include_bos_eos,
        batch_size_per_gpu=args.batch_size_per_gpu,
        gpu_ids=parsed_gpu_ids,
        num_workers=args.num_cpu_workers
    )