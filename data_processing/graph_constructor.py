#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import traceback  # 确保导入 traceback

# 假设 pdb_parser 和 feature_calculator 在同一个包或路径下
from .pdb_parser import PDBProcessor
from .feature_calculator import FeatureCalculator

logger = logging.getLogger(__name__)


class GraphConstructor:
    """从蛋白质结构和其他信息构建PyTorch Geometric图对象的类。"""

    def __init__(self,
                 cutoff_distance: float = 10.0,
                 max_seq_sep: int = 32):
        """
        初始化图构建器。

        参数:
            cutoff_distance (float): 构建空间边时使用的距离阈值 (单位Å)。
            max_seq_sep (int): 序列分离度编码的最大值。
        """
        self.pdb_processor = PDBProcessor()
        self.feature_calculator = FeatureCalculator(
            cutoff_distance=cutoff_distance,
            max_seq_sep=max_seq_sep
        )
        # logger.info(f"GraphConstructor 初始化完毕，空间边截断距离: {cutoff_distance} Å")

    def _load_esm_embedding(self, embedding_path: str) -> Optional[np.ndarray]:
        """加载预计算的ESM嵌入。"""
        if embedding_path and os.path.exists(embedding_path):
            try:
                embedding = np.load(embedding_path)
                return embedding
            except Exception as e:
                logger.error(f"加载ESM嵌入文件 {embedding_path} 失败: {e}")
                return None
        else:
            logger.debug(f"ESM嵌入文件路径未提供或文件不存在: {embedding_path}")
            return None

    def create_graph_from_pdb(self,
                              pdb_gz_path: str,
                              sequence_id: str,
                              embedding_dir: Optional[str],
                              embedding_stage: str,  # 这个参数来自 main_preprocess.py 中的 esm_embedding_subdir_name
                              activity_label: Optional[int] = None,
                              ) -> Dict:  # MODIFIED FOR DEBUGGING: 返回字典而不是 Optional[Data]
        """
        从PDB文件和预计算的ESM嵌入构建一个PyG图对象。
        在调试模式下，返回一个包含状态的字典。
        """
        try:
            # 1. 解析PDB文件
            parsed_pdb = self.pdb_processor.parse_pdb_gz(pdb_gz_path)
            if parsed_pdb is None:
                logger.warning(f"PDB文件解析失败: {pdb_gz_path}")
                return {"status": "pdb_parse_failed", "seq_id": sequence_id, "path": pdb_gz_path}

            original_sequence = parsed_pdb['sequence']
            ca_coords = parsed_pdb['coords']
            plddt_scores = parsed_pdb['plddt']

            if not original_sequence or ca_coords.shape[0] == 0:
                logger.warning(f"PDB文件 {pdb_gz_path} (ID: {sequence_id}) 未包含有效的Cα原子或序列为空。")
                return {"status": "no_alpha_carbon_or_empty_seq", "seq_id": sequence_id, "path": pdb_gz_path}
            if ca_coords.shape[0] != len(original_sequence):
                logger.warning(
                    f"PDB文件 {pdb_gz_path} (ID: {sequence_id}) Cα原子数 ({ca_coords.shape[0]}) 与序列长度 ({len(original_sequence)}) 不匹配。")
                return {"status": "coord_seq_mismatch", "seq_id": sequence_id, "path": pdb_gz_path}
            # 3. 计算节点标量特征
            node_scalar_feat = self.feature_calculator.calculate_node_scalar_features(original_sequence, plddt_scores)

            # 4. 计算节点矢量特征 (归一化坐标)
            node_vector_feat = self.feature_calculator.normalize_coordinates(ca_coords)
            # 保证 node_vector 形状为 [N, 1, 3]
            if node_vector_feat.ndim == 2:
                node_vector_feat = node_vector_feat[:, None, :]
            elif node_vector_feat.ndim == 3 and node_vector_feat.shape[1] != 1:
                node_vector_feat = node_vector_feat.reshape(node_vector_feat.shape[0], 1, 3)

            # 5. 构建边和计算边特征
            edge_index, edge_scalar_attr, edge_vector_attr = self.feature_calculator.build_graph_edges(ca_coords)
            # 保证 edge_vector 形状为 [E, 1, 3]
            if edge_vector_attr.ndim == 2:
                edge_vector_attr = edge_vector_attr[:, None, :]
            elif edge_vector_attr.ndim == 3 and edge_vector_attr.shape[1] != 1:
                edge_vector_attr = edge_vector_attr.reshape(edge_vector_attr.shape[0], 1, 3)

            # 6. 加载预计算的ESM嵌入
            loaded_embedding = None
            if embedding_dir:
                specific_embedding_dir = os.path.join(embedding_dir, embedding_stage)
                embedding_file_path = os.path.join(specific_embedding_dir, f"{sequence_id}.npy")
                loaded_embedding = self._load_esm_embedding(embedding_file_path)
                if loaded_embedding is None:
                    logger.debug(f"未能为 {sequence_id} (嵌入类型: {embedding_stage}) 加载ESM嵌入。")
                elif loaded_embedding.shape[0] != len(original_sequence):
                    logger.error(
                        f"序列 {sequence_id} 的长度 ({len(original_sequence)}) 与其ESM嵌入的长度 ({loaded_embedding.shape[0]}) 不匹配 (嵌入类型: {embedding_stage})。将不使用此嵌入。")
                    loaded_embedding = None

            # 7. 创建PyG Data对象 (仍然创建它以检查是否有错误，但不直接返回它用于ancdata调试)
            data = Data(
                x=torch.from_numpy(node_scalar_feat).float(),
                y=torch.tensor([activity_label], dtype=torch.long) if activity_label is not None else torch.tensor([0], dtype=torch.long),
                node_vector=torch.from_numpy(node_vector_feat).float(),
                edge_index=torch.from_numpy(edge_index).long(),
                edge_attr=torch.from_numpy(edge_scalar_attr).float(),
                edge_vector=torch.from_numpy(edge_vector_attr).float(),
                coords=torch.from_numpy(ca_coords).float(),
                original_seq=original_sequence,
                plddt=torch.from_numpy(plddt_scores).float(),
                seq_id=sequence_id
            )
            data.num_nodes = len(original_sequence)

            # --- 正确的 ESM 嵌入存储逻辑 ---
            if loaded_embedding is not None:
                setattr(data, embedding_stage, torch.from_numpy(loaded_embedding).float())
                logger.debug(f"在Data对象上为序列 {sequence_id} 成功设置属性 '{embedding_stage}'。")
            # else:
            #    logger.debug(f"序列 {sequence_id} (嵌入类型: {embedding_stage}) 没有加载的ESM嵌入，Data对象将不包含属性 '{embedding_stage}'。")

            # 不再添加 activity_label 字段，只保留 y 字段


            return data

        except Exception as e:
            logger.error(f"从PDB {pdb_gz_path} (ID: {sequence_id}) 构建图时发生未捕获的异常: {e}", exc_info=True)
            # --- DEBUGGING RETURN ---
            return {"status": "exception_in_create_graph", "seq_id": sequence_id, "path": pdb_gz_path, "error": str(e),
                    "traceback": traceback.format_exc()}

    def create_graph_from_data(self,
                               sequence_id: str,
                               original_sequence: str,
                               ca_coords: np.ndarray,
                               plddt_scores: np.ndarray,
                               esm_embedding: Optional[np.ndarray] = None,
                               embedding_stage: Optional[str] = "amp_embedding",  # 默认为简化后的名称
                               activity_label: Optional[int] = None,
                               ) -> Optional[Data]:  # 此方法暂不修改为返回字典，除非也需要调试它
        try:
            if ca_coords.shape[0] == 0 or len(original_sequence) == 0:
                logger.warning(f"序列 {sequence_id} 未包含有效的Cα原子或序列。")
                return None
            if ca_coords.shape[0] != len(original_sequence):
                logger.warning(
                    f"序列 {sequence_id} Cα原子数 ({ca_coords.shape[0]})与序列长度 ({len(original_sequence)})不匹配。")
                return None

            node_scalar_feat = self.feature_calculator.calculate_node_scalar_features(original_sequence)
            node_vector_feat = self.feature_calculator.normalize_coordinates(ca_coords)
            edge_index, edge_scalar_attr, edge_vector_attr = self.feature_calculator.build_graph_edges(ca_coords)

            data = Data(
                x=torch.from_numpy(node_scalar_feat).float(),
                y=torch.tensor([activity_label], dtype=torch.float32),  # 默认标签为0
                node_vector=torch.from_numpy(node_vector_feat).float(),
                edge_index=torch.from_numpy(edge_index).long(),
                edge_attr=torch.from_numpy(edge_scalar_attr).float(),
                edge_vector=torch.from_numpy(edge_vector_attr).float(),
                coords=torch.from_numpy(ca_coords).float(),
                original_seq=original_sequence,
                plddt=torch.from_numpy(plddt_scores).float(),
                seq_id=sequence_id
            )
            data.num_nodes = len(original_sequence)

            if esm_embedding is not None:
                if esm_embedding.shape[0] != len(original_sequence):
                    logger.error(
                        f"序列 {sequence_id} 的长度 ({len(original_sequence)}) 与其ESM嵌入的长度 ({esm_embedding.shape[0]}) 不匹配。跳过此样本。")
                    return None

                    # 使用 setattr 统一处理
                if embedding_stage:  # 确保 embedding_stage 不是 None 或空字符串
                    setattr(data, embedding_stage, torch.from_numpy(esm_embedding).float())
                    logger.debug(
                        f"在Data对象上为序列 {sequence_id} (create_graph_from_data) 成功设置属性 '{embedding_stage}'。")
                else:
                    logger.warning(f"在 create_graph_from_data 中未提供 embedding_stage，ESM嵌入未存储。")

            # 添加活性标签
            if activity_label is not None:
                print(f"添加活性标签: {activity_label} 到图数据")
                data.y = torch.tensor([activity_label], dtype=torch.float32)

            return data
        except Exception as e:
            logger.error(f"从数据为 {sequence_id} 构建图时出错: {e}", exc_info=True)
            return None