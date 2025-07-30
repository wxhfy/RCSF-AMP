#!/usr/bin/env python3
"""
AMP多模态模型数据预处理器 - 简化版本
去掉一阶段内容，直接处理抗菌肽数据进行训练
"""
import os
import argparse
import json
import logging
import warnings
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple
import glob
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# 在 main_preprocess_simplified.py 中
from .pdb_parser import PDBProcessor
from .graph_constructor import GraphConstructor
from .esm_embedder import ESMEmbedder, embed_sequences_multi_gpu
from .feature_calculator import FeatureCalculator

# ---- 日志设置 ----
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s:%(levelname)s:%(process)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
logger = logging.getLogger(__name__)



def _mp_graph_constructor_task(args_bundle: Tuple) -> dict | None:
    """多进程图构建任务"""
    pdb_file_path, protein_id, base_embedding_dir, embedding_stage, activity_label, cutoff_distance, max_seq_sep = args_bundle

    try:
        # 在工作进程中创建局部的图构建器实例
        from .graph_constructor import GraphConstructor
        graph_constructor = GraphConstructor(
            cutoff_distance=cutoff_distance,
            max_seq_sep=max_seq_sep
        )
        
        graph_data = graph_constructor.create_graph_from_pdb(
            pdb_gz_path=pdb_file_path,
            sequence_id=protein_id,
            embedding_dir=base_embedding_dir,
            embedding_stage=embedding_stage,
            activity_label=activity_label,
        )
        
        return graph_data
        
    except Exception as e:
        # 简单返回None，不打印详细错误
        return None
    finally:
        # 显式清理内存
        import gc
        gc.collect()
        if 'graph_constructor' in locals():
            del graph_constructor
        if 'graph_data' in locals():
            del graph_data


class AMPPreprocessor:
    """AMP数据预处理器 - 简化版本，只处理训练数据"""
    
    def __init__(self, output_dir: str, cutoff_distance: float = 10.0, max_seq_sep: int = 32,
                esm_model_name: str = "esm2_t36_3B_UR50D",
                 esm_model_base_path: Optional[str] = None, max_seq_len: int = 200):
        """
        初始化预处理器
        
        Args:
            output_dir: 输出目录
            cutoff_distance: 距离截断值
            max_seq_sep: 最大序列分离度
            esm_model_name: ESM模型名称
            esm_model_base_path: ESM模型本地路径
            max_seq_len: 提取序列的最大长度（包含/不包含特殊符号，默认50）
        """
        self.output_dir_base = output_dir
        self.graph_constructor = GraphConstructor(
            cutoff_distance=cutoff_distance, max_seq_sep=max_seq_sep
        )
        self.feature_calc = FeatureCalculator()
        self.pdb_processor = PDBProcessor()
        self.esm_model_name = esm_model_name
        self.esm_model_base_path = esm_model_base_path
        self.max_seq_len = max_seq_len

    def _extract_sequences_from_pdbs(self, pdb_file_paths: List[str], desc_prefix: str = "") -> List[Dict]:
        """从PDB文件中提取序列（长度<=max_seq_len）"""
        sequence_data = []
        logger.info(f"{desc_prefix}从 {len(pdb_file_paths)} 个 PDB 文件中提取序列 (最大长度: {self.max_seq_len})")
        for pdb_file in tqdm(pdb_file_paths, desc=f"{desc_prefix}提取序列"):
            try:
                protein_id = Path(pdb_file).stem.split('.')[0]
                pdb_data = self.pdb_processor.parse_pdb_gz(pdb_file)
                if pdb_data and 'sequence' in pdb_data:
                    sequence = pdb_data['sequence']
                    if 0 < len(sequence) <= self.max_seq_len:
                        sequence_data.append({"id": protein_id, "sequence": sequence, "original_path": pdb_file})
                    elif len(sequence) == 0:
                        logger.warning(f"跳过 {pdb_file} 中的 {protein_id}：序列为空。")
                    else:
                        logger.debug(f"跳过 {pdb_file} 中的 {protein_id}：序列过长 ({len(sequence)})，不符合L<={self.max_seq_len}。")
                else:
                    logger.warning(f"无法从 {pdb_file} 解析 {protein_id} 的序列。")
            except Exception as e:
                logger.error(f"从 {pdb_file} 提取序列时出错: {str(e)}")
        return sequence_data


    def preprocess_dataset(self,
                           data_root: str = None,
                           benchmark_mode: str = "benchmark1",
                           pdb_file_type: str = "pdb.gz",
                           lora_weights_path: Optional[str] = None,
                           num_workers: int = 16,
                           gpu_ids_for_embedding: Optional[List[int]] = None,
                           process_embeddings: bool = True,
                           force_regenerate_embeddings: bool = False,
                           force_regenerate_graphs: bool = False,
                           batch_size: int = 500):
        """

        预处理AMP数据集，支持benchmark1/2两种模式和pdb/pdb.gz两种文件类型。
        Args:
            data_root: 数据根目录（包含各子集文件夹）
            benchmark_mode: "benchmark1" 或 "benchmark2"
            pdb_file_type: "pdb" 或 "pdb.gz"
            其余参数同原逻辑
        """
        output_dir = Path(self.output_dir_base)
        # 确保最外层输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        graphs_output_dir = output_dir / "graphs"
        embeddings_output_dir = output_dir / "embeddings"

        logger.info("--- 开始AMP数据预处理 ---")
        logger.info(f"输出将保存到: {output_dir}")

        # 1. 收集PDB文件（支持两种benchmark模式和两种文件类型）
        if data_root is None:
            logger.error("必须指定数据根目录 data_root")
            self._create_splits(output_dir, [])
            return

        if pdb_file_type not in ["pdb", "pdb.gz"]:
            logger.error(f"不支持的pdb文件类型: {pdb_file_type}")
            self._create_splits(output_dir, [])
            return

        if benchmark_mode == "benchmark1":
            pos_folders = ["AMP_eval", "AMP_test", "AMP_train"]
            neg_folders = ["DECOY_eval", "DECOY_test", "DECOY_train"]
        elif benchmark_mode == "benchmark2":
            pos_folders = ["amp_eval", "amp_test", "amp_train"]
            neg_folders = ["non_amp_eval", "non_amp_test", "non_amp_train"]
        else:
            logger.error(f"不支持的benchmark模式: {benchmark_mode}")
            self._create_splits(output_dir, [])
            return

        file_ext = f"*.{pdb_file_type}"
        pos_pdb_files = []
        neg_pdb_files = []
        for folder in pos_folders:
            folder_path = os.path.join(data_root, folder)
            if os.path.isdir(folder_path):
                pos_pdb_files.extend(glob.glob(os.path.join(folder_path, file_ext)))
        for folder in neg_folders:
            folder_path = os.path.join(data_root, folder)
            if os.path.isdir(folder_path):
                neg_pdb_files.extend(glob.glob(os.path.join(folder_path, file_ext)))

        all_pdb_files_with_labels: List[Tuple[str, str, int]] = []
        for pdb_path in pos_pdb_files:
            protein_id = Path(pdb_path).stem.split('.')[0]
            all_pdb_files_with_labels.append((pdb_path, protein_id, 1))
        for pdb_path in neg_pdb_files:
            protein_id = Path(pdb_path).stem.split('.')[0]
            all_pdb_files_with_labels.append((pdb_path, protein_id, 0))

        if not all_pdb_files_with_labels:
            logger.warning("在指定目录中未找到PDB文件。")
            self._create_splits(output_dir, [])
            return

        logger.info(f"找到 {len(pos_pdb_files)} 个阳性样本和 {len(neg_pdb_files)} 个阴性样本")


        # 1. 直接处理所有PDB文件提取序列
        logger.info("开始从所有PDB文件中提取序列...")
        sequence_data_list: List[Dict] = []
        pos_count = 0
        neg_count = 0
        length_filtered_count = 0
        
        # 直接处理所有PDB文件
        pdb_paths_for_seq_extraction = [item[0] for item in all_pdb_files_with_labels]
        
        logger.info(f"正在处理 {len(pdb_paths_for_seq_extraction)} 个PDB文件...")
        for pdb_path, protein_id, activity_label in tqdm(all_pdb_files_with_labels, desc="提取序列并统计"):
            try:
                pdb_data = self.pdb_processor.parse_pdb_gz(pdb_path)
                if pdb_data and 'sequence' in pdb_data:
                    sequence = pdb_data['sequence']
                    if 0 < len(sequence) <= self.max_seq_len:
                        sequence_data_list.append({
                            "id": protein_id,
                            "sequence": sequence,
                            "original_path": pdb_path,
                            "activity_label": activity_label
                        })
                        if activity_label == 1:
                            pos_count += 1
                        else:
                            neg_count += 1
                    elif len(sequence) > self.max_seq_len:
                        length_filtered_count += 1
                        logger.debug(f"跳过 {protein_id}：序列过长 ({len(sequence)})，不符合L<={self.max_seq_len}。")
                    else:
                        logger.warning(f"跳过 {protein_id}：序列为空。")
                else:
                    logger.warning(f"无法从 {pdb_path} 解析 {protein_id} 的序列。")
            except Exception as e:
                logger.error(f"从 {pdb_path} 提取序列时出错: {str(e)}")
        
        # 统计信息
        total_valid = len(sequence_data_list)
        logger.info(f"序列提取完成！统计信息：")
        logger.info(f"  - 总计有效序列 (L<={self.max_seq_len}): {total_valid}")
        logger.info(f"  - 阳性样本: {pos_count}")
        logger.info(f"  - 阴性样本: {neg_count}")
        logger.info(f"  - 因长度过滤的序列: {length_filtered_count}")
        
        # 保存序列信息到JSON文件（包含标签信息和原始路径）
        sequence_json_path = output_dir / f"sequences_L_lt_{self.max_seq_len}.json"
        sequences_to_save_json = [
            {
                "id": s_data["id"],
                "sequence": s_data["sequence"],
                "activity_label": s_data["activity_label"],
                "original_path": s_data["original_path"]
            }
            for s_data in sequence_data_list
        ]
        try:
            with open(sequence_json_path, "w") as f:
                json.dump(sequences_to_save_json, f, indent=2)
            logger.info(f"序列信息已保存到 {sequence_json_path}")
        except Exception as e_save_seq:
            logger.error(f"保存序列文件 {sequence_json_path} 失败: {e_save_seq}", exc_info=True)

        if not sequence_data_list:
            logger.warning(f"未提取/加载到有效序列 (L<={self.max_seq_len})。")
            self._create_splits(output_dir, [])
            return

        sequences_for_esm_json = [{"id": s_data["id"], "sequence": s_data["sequence"]} for s_data in sequence_data_list]

        # 2. ESM嵌入计算
        esm_embedding_subdir = "amp_embedding"
        stage_specific_esm_output_dir = embeddings_output_dir / esm_embedding_subdir
        
        if process_embeddings:
            if force_regenerate_embeddings and stage_specific_esm_output_dir.exists():
                logger.info(f"强制重新生成ESM嵌入，删除旧目录: {stage_specific_esm_output_dir}")
                import shutil
                try:
                    shutil.rmtree(stage_specific_esm_output_dir)
                except Exception as e_rm_embed:
                    logger.error(f"删除旧嵌入目录失败: {e_rm_embed}")

            pass  # 目录创建由主训练脚本负责
            logger.info(f"计算ESM嵌入 (模型: '{self.esm_model_name}', LoRA: {'是' if lora_weights_path else '否'})")
            embed_sequences_multi_gpu(
                sequence_data=sequences_for_esm_json,
                output_dir=str(stage_specific_esm_output_dir),
                model_name=self.esm_model_name,
                local_model_path_root=self.esm_model_base_path,
                lora_weights_path=lora_weights_path,
                gpu_ids=gpu_ids_for_embedding,
            )
            logger.info(f"ESM嵌入已计算/检查完毕，保存在 {stage_specific_esm_output_dir}")
        else:
            logger.info("跳过ESM嵌入计算步骤。")

        # 4. 并行构建图 - 使用批次处理
        pdb_info_map = {item[1]: (item[0], item[2]) for item in all_pdb_files_with_labels}

        processed_ids_for_graphing = set()
        if not force_regenerate_graphs:
            for existing_graph_file in graphs_output_dir.glob("*.pt"):
                processed_ids_for_graphing.add(existing_graph_file.stem)
            logger.info(f"找到 {len(processed_ids_for_graphing)} 个已存在的图文件，将跳过它们。")

        # 收集要处理的所有样本
        all_tasks_for_mp = []
        
        for seq_entry in sequence_data_list:
            protein_id = seq_entry["id"]
            if not force_regenerate_graphs and protein_id in processed_ids_for_graphing:
                continue

            original_pdb_path, activity_label = pdb_info_map.get(protein_id, (seq_entry["original_path"], None))
            if not Path(original_pdb_path).exists():
                logger.warning(f"PDB {original_pdb_path} (ID: {protein_id}) 未找到。")
                continue


            all_tasks_for_mp.append(
                (original_pdb_path, protein_id, str(embeddings_output_dir), esm_embedding_subdir,
                 activity_label, self.graph_constructor.feature_calculator.cutoff_distance,
                 self.graph_constructor.feature_calculator.max_seq_sep)
            )
        
        # 显示总任务数
        total_tasks = len(all_tasks_for_mp)
        if total_tasks == 0:
            logger.info("没有新的任务用于图构建。")
        else:
            logger.info(f"总计需要为 {total_tasks} 个PDB文件构建新的图")
            
            # 计算批次数
            num_batches = (total_tasks + batch_size - 1) // batch_size  # 向上取整
            logger.info(f"将分 {num_batches} 个批次处理，每批次最多 {batch_size} 个任务")
            
            # 设置批次处理进度条
            batch_pbar = tqdm(total=num_batches, desc="批次进度")
            
            # 批次处理
            newly_saved_graph_paths_count = 0
            for batch_idx in range(num_batches):
                # 计算当前批次的起始和结束索引
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_tasks)
                current_batch_tasks = all_tasks_for_mp[start_idx:end_idx]
                
                logger.info(f"处理批次 {batch_idx + 1}/{num_batches}，包含 {len(current_batch_tasks)} 个任务")
                
                # 处理当前批次
                batch_results_raw = []
                actual_num_workers = min(num_workers, mp.cpu_count() if mp.cpu_count() else 1, len(current_batch_tasks))
                if actual_num_workers <= 0 < len(current_batch_tasks):
                    actual_num_workers = 1
                
                if actual_num_workers > 1:
                    logger.info(f"使用 {actual_num_workers} 个并行进程处理当前批次")
                    try:
                        # 使用spawn启动方法避免问题
                        ctx = mp.get_context("spawn")
                        
                        with ctx.Pool(processes=actual_num_workers) as pool:
                            batch_results_raw = list(tqdm(
                                pool.imap(_mp_graph_constructor_task, current_batch_tasks),
                                total=len(current_batch_tasks), 
                                desc=f"构建图 批次 {batch_idx + 1}/{num_batches}"
                            ))
                            
                            # 确保pool被关闭和终止
                            pool.close()
                            pool.join()
                    
                    except Exception as e_pool:
                        logger.error(f"批次 {batch_idx + 1} 多进程池失败: {e_pool}")
                        logger.warning("将回退到单进程模式")
                        actual_num_workers = 0
                
                if actual_num_workers <= 1 and len(current_batch_tasks) > 0:
                    if len(current_batch_tasks) > 1:
                        logger.info("使用单进程进行图构建。")
                    for task_args in tqdm(current_batch_tasks, desc=f"构建图 批次 {batch_idx + 1}/{num_batches} (单进程)"):
                        batch_results_raw.append(_mp_graph_constructor_task(task_args))
                
                # 处理和保存当前批次的图
                batch_results_graphs_data_objects = [g for g in batch_results_raw if g is not None and isinstance(g, Data)]
                logger.info(f"批次 {batch_idx + 1}: 从工作进程返回的结果中提取到 {len(batch_results_graphs_data_objects)} 个有效Data对象")
                
                if batch_results_graphs_data_objects:
                    batch_saved_count = 0
                    for graph_data_obj in tqdm(batch_results_graphs_data_objects, desc=f"保存批次 {batch_idx + 1} 图文件"):
                        if not hasattr(graph_data_obj, 'seq_id') or not graph_data_obj.seq_id:
                            logger.error(f"图对象缺少有效 'seq_id' 属性")
                            continue
                        output_graph_file_path = graphs_output_dir / f"{graph_data_obj.seq_id}.pt"
                        # 确保目录存在
                        output_graph_file_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            torch.save(graph_data_obj, output_graph_file_path)
                            batch_saved_count += 1
                        except Exception as e_save:
                            logger.error(f"保存图文件 {output_graph_file_path} 失败: {e_save}")
                    
                    newly_saved_graph_paths_count += batch_saved_count
                    logger.info(f"批次 {batch_idx + 1}: 成功保存了 {batch_saved_count} 个图文件")
                
                # 清理内存
                del batch_results_raw
                del batch_results_graphs_data_objects
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 更新批次进度条
                batch_pbar.update(1)
                
                logger.info(f"批次 {batch_idx + 1}/{num_batches} 处理完成，累计保存 {newly_saved_graph_paths_count} 个图文件")
            
            # 关闭批次进度条
            batch_pbar.close()
            
            logger.info(f"所有批次处理完成，总共构建并保存了 {newly_saved_graph_paths_count} 个图文件")

        # 创建数据集切分
        all_available_graph_paths = [str(p) for p in graphs_output_dir.glob("*.pt")]
        logger.info(f"总共找到 {len(all_available_graph_paths)} 个图文件用于创建切分。")
        self._create_splits(output_dir, all_available_graph_paths)

    def _create_splits(self, current_output_dir: Path, successfully_saved_graph_paths: List[str]):
        """严格根据PDB文件来源文件夹名进行训练/验证/测试集切分"""
        if not successfully_saved_graph_paths:
            logger.warning("未找到图文件以创建切分。将创建空切分文件。")
            self._save_split_file([], current_output_dir / "train.txt")
            self._save_split_file([], current_output_dir / "val.txt")
            self._save_split_file([], current_output_dir / "test.txt")
            return

        # 读取序列信息以获取标签和原始路径
        sequence_json_path = current_output_dir / f"sequences_L_lt_{self.max_seq_len}.json"
        id_to_path = {}
        id_to_label = {}
        if sequence_json_path.exists():
            try:
                with open(sequence_json_path, "r") as f:
                    sequences_data = json.load(f)
                for item in sequences_data:
                    if "id" in item and "original_path" in item:
                        id_to_path[item["id"]] = item["original_path"]
                    if "id" in item and "activity_label" in item:
                        id_to_label[item["id"]] = item["activity_label"]
            except Exception as e:
                logger.warning(f"无法读取序列标签信息: {e}，将使用随机分割")

        # 按照原始路径的文件夹名进行分组
        train_files, val_files, test_files = [], [], []
        for graph_path in successfully_saved_graph_paths:
            filename = os.path.basename(graph_path)
            protein_id = filename.replace('.pt', '')
            orig_path = id_to_path.get(protein_id, None)
            if orig_path is None:
                # 无法找到原始路径，归为训练集
                train_files.append(graph_path)
                continue
            # 获取上一级文件夹名
            parent_folder = Path(orig_path).parent.name.lower()
            if any(x in parent_folder for x in ["train"]):
                train_files.append(graph_path)
            elif any(x in parent_folder for x in ["val", "eval"]):
                val_files.append(graph_path)
            elif any(x in parent_folder for x in ["test"]):
                test_files.append(graph_path)
            else:
                # 未知文件夹，归为训练集
                train_files.append(graph_path)

        logger.info(f"数据集切分（基于文件夹名）: 训练集={len(train_files)}, 验证集={len(val_files)}, 测试集={len(test_files)}")

        # 保存分割文件
        self._save_split_file(train_files, current_output_dir / "train.txt")
        self._save_split_file(val_files, current_output_dir / "val.txt")
        self._save_split_file(test_files, current_output_dir / "test.txt")

    def _save_split_file(self, file_path_list: List[str], output_split_filepath: Path):
        """保存切分文件"""
        pass  # 目录创建由主训练脚本负责
        with open(output_split_filepath, 'w') as f:
            for p_path_full in file_path_list:
                filename = os.path.basename(p_path_full)
                f.write(f"{filename}\n")


if __name__ == "__main__":
    # 设置多进程启动方法 - 优先使用fork以减少内存开销
    try:
        current_start_method = mp.get_start_method(allow_none=True)
        # 在Linux系统上优先使用fork，减少内存开销
        desired_start_method = "fork" if hasattr(os, 'fork') else "spawn"
        if current_start_method is None or current_start_method != desired_start_method:
            mp.set_start_method(desired_start_method, force=True)
            print(f"[INFO] 已设置多进程启动方法为 '{desired_start_method}'。")
        else:
            print(f"[INFO] 当前多进程启动方法: '{current_start_method}'")
    except RuntimeError as e_mp_start:
        print(f"[WARNING] 设置多进程启动方法失败: {e_mp_start}。"
              f"当前启动方法: {mp.get_start_method(allow_none=True)}")
    except Exception as e_mp_general:
        print(f"[ERROR] 设置多进程启动方法时发生未知错误: {e_mp_general}")


    parser = argparse.ArgumentParser(description="AMP数据预处理器")
    parser.add_argument("--output_dir", type=str, required=True, help="保存所有处理后数据的根目录")
    parser.add_argument("--data_root", type=str, required=True, help="包含所有子集文件夹的根目录（如 AMP_train/DECOY_train/...）")
    parser.add_argument("--benchmark_mode", type=str, required=True, choices=["benchmark1", "benchmark2"], help="数据集模式: benchmark1 或 benchmark2")
    parser.add_argument("--cutoff", type=float, default=10.0, help="距离截断值 (Å)")
    parser.add_argument("--esm_model_name", type=str, default="facebook/esm2_t36_3B_UR50D", help="ESM模型名")
    parser.add_argument("--esm_model_base_path", type=str, help="ESM本地模型根目录")
    parser.add_argument("--max_seq_len", type=int, default=500, help="提取序列的最大长度 (默认200)")
    # 强制重新生成选项
    parser.add_argument("--force_regenerate_sequences", action="store_true", help="强制重新提取序列，即使JSON文件已存在。")
    parser.add_argument("--force_regenerate_embeddings", action="store_true", help="强制重新计算ESM嵌入，即使.npy文件已存在。")
    parser.add_argument("--force_regenerate_graphs", action="store_true", help="强制重新构建图，即使.pt文件已存在。")
    parser.add_argument("--skip_embeddings", action="store_true", help="完全跳过ESM嵌入计算步骤。")
    parser.add_argument("--num_workers", type=int, default=1, help="CPU工作进程数")
    parser.add_argument("--batch_size", type=int, default=500, help="图构建批次大小，较小的值有助于减少内存使用")
    parser.add_argument("--gpus_embed", type=str, default=None, help="ESM嵌入用GPU ID (逗号分隔, 如 0 或 0,1,2)")

    args = parser.parse_args()

    # 解析GPU ID
    parsed_gpu_ids = None
    if args.gpus_embed:
        try:
            parsed_gpu_ids = [int(gid.strip()) for gid in args.gpus_embed.split(',') if gid.strip()]
            if not torch.cuda.is_available():
                logger.warning("CUDA不可用，GPU将被忽略。")
                parsed_gpu_ids = None
            elif parsed_gpu_ids:
                valid_gids = [gid for gid in parsed_gpu_ids if 0 <= gid < torch.cuda.device_count()]
                if len(valid_gids) < len(parsed_gpu_ids):
                    logger.warning(f"部分GPU ID无效。将使用: {valid_gids}。")
                parsed_gpu_ids = valid_gids
                if not parsed_gpu_ids:
                    logger.warning("所有GPU ID均无效。")
        except ValueError:
            logger.error(f"无效GPU ID格式: '{args.gpus_embed}'。")
            parsed_gpu_ids = None


    # 创建预处理器并运行
    preprocessor = AMPPreprocessor(
        output_dir=args.output_dir,
        cutoff_distance=args.cutoff,
        esm_model_name=args.esm_model_name,
        esm_model_base_path=args.esm_model_base_path,
        max_seq_len=args.max_seq_len
    )

    preprocessor.preprocess_dataset(
        data_root=args.data_root,
        benchmark_mode=args.benchmark_mode,
        pdb_file_type="pdb.gz",  # 可根据需要添加参数
        lora_weights_path=None,   # 明确不使用LoRA
        num_workers=args.num_workers,
        gpu_ids_for_embedding=parsed_gpu_ids,
        process_embeddings=(not args.skip_embeddings),
        force_regenerate_embeddings=args.force_regenerate_embeddings,
        force_regenerate_graphs=args.force_regenerate_graphs,
        batch_size=args.batch_size
    )

    logger.info(f"AMP数据预处理完成。结果保存在 {args.output_dir}")


if __name__ == "__main__":
    # 直接运行脚本时，测试两种benchmark模式的数据收集
    # 无命令行参数时，自动跑测试样例
    pass
