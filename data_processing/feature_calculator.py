#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, List, Tuple, Optional
import torch  # 虽然此处计算主要是numpy，但后续会转为torch.tensor
from scipy.spatial import distance_matrix  # 用于计算距离矩阵
import logging
from Bio.SeqUtils.ProtParam import ProteinAnalysis # 用于理化性质计算

logger = logging.getLogger(__name__)


class FeatureCalculator:
    """为蛋白质图计算节点和边特征的类。"""

    # 氨基酸到索引的映射 (20种标准氨基酸 + 'X'代表未知或非标准)
    AA_TO_IDX = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
        'X': 20, 'U': 20, 'O': 20 # 将U和O也映射到X (索引20)
    }
    IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}


    def __init__(self, cutoff_distance: float = 10.0, max_seq_sep: int = 32, num_gaussian_seq_sep: int = 5):
        """
        初始化特征计算器。

        参数:
            cutoff_distance (float): 构建空间边时使用的距离阈值 (单位Å)。
            max_seq_sep (int): 边特征中序列分离度高斯编码能覆盖的最大分离值。
                               高斯函数的中心会基于此和num_gaussian_seq_sep分布。
            num_gaussian_seq_sep (int): 用于序列分离度编码的高斯基函数的数量。
        """
        self.cutoff_distance = cutoff_distance
        self.max_seq_sep = max_seq_sep
        self.num_gaussian_seq_sep = num_gaussian_seq_sep

    def calculate_node_scalar_features(self, sequence: str, plddt: np.ndarray) -> np.ndarray:
        """
        生成节点标量特征: AA One-hot(21) + 归一化plddt(1) => [L, 22]
        参数:
            sequence (str): 氨基酸序列。
            plddt (np.ndarray): pLDDT分数数组 [L]。
        返回:
            np.ndarray: 节点特征 [L, 22]
        """
        L = len(sequence)
        node_s_features = np.zeros((L, 22), dtype=np.float32)
        valid_sequence = "".join([aa if aa in self.AA_TO_IDX else 'X' for aa in sequence.upper()])
        for i, aa_char in enumerate(valid_sequence):
            aa_idx = self.AA_TO_IDX[aa_char]
            node_s_features[i, aa_idx] = 1.0
            # 归一化plddt加到最后一位
            if plddt is not None and i < len(plddt):
                node_s_features[i, 21] = np.clip(plddt[i] / 100.0, 0.0, 1.0)
            else:
                node_s_features[i, 21] = 0.0
        return node_s_features

    def normalize_coordinates(self, ca_coordinates: np.ndarray) -> np.ndarray:
        """
        对Cα坐标进行质心归一化，并返回[N, 1, 3]格式。

        参数:
            ca_coordinates (np.ndarray): 原始Cα坐标矩阵 [L, 3]。

        返回:
            np.ndarray: 质心归一化后的坐标矩阵 [L, 1, 3]。
        """
        if ca_coordinates.shape[0] == 0: # 处理空坐标数组
            return np.empty((0, 1, 3), dtype=np.float32)
        centroid = np.mean(ca_coordinates, axis=0, keepdims=True)
        normalized_coords = ca_coordinates - centroid
        # 保证输出为[N, 1, 3]
        normalized_coords = normalized_coords[:, None, :]
        return normalized_coords


    def build_graph_edges(self, ca_coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        构建图的边，并计算边特征。

        参数:
            ca_coordinates (np.ndarray): Cα坐标矩阵 [L, 3]。

        返回:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - edge_index: 边索引 [2, num_edges], int64
                - edge_scalar_attributes: 边标量特征 [num_edges, 10], float32
                - edge_vector_attributes: 边矢量特征 [num_edges, 1, 3], float32
        """
        L = ca_coordinates.shape[0]
        if L == 0:
            return (np.empty((2, 0), dtype=np.int64),
                    np.empty((0, 10), dtype=np.float32),
                    np.empty((0, 3), dtype=np.float32))

        # 1. 序列边 (Type 1)
        adj_seq_src = np.arange(L - 1)
        adj_seq_dst = np.arange(1, L)
        # 双向
        seq_edge_index_forward = np.stack([adj_seq_src, adj_seq_dst], axis=0)
        seq_edge_index_backward = np.stack([adj_seq_dst, adj_seq_src], axis=0)
        seq_edge_index = np.concatenate([seq_edge_index_forward, seq_edge_index_backward], axis=1) if L > 1 else np.empty((2,0), dtype=np.int64)
        seq_edge_types = np.ones(seq_edge_index.shape[1], dtype=int)  # Type 1

        # 2. 空间边 (Type 0) - Radius Graph
        dist_mat = distance_matrix(ca_coordinates, ca_coordinates)  # [L, L]

        spatial_src_nodes, spatial_dst_nodes = np.where(
            (dist_mat < self.cutoff_distance) &
            (np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) > 1)
        )

        if spatial_src_nodes.size > 0:
            spatial_edge_index = np.stack([spatial_src_nodes, spatial_dst_nodes], axis=0)
            spatial_edge_types = np.zeros(spatial_edge_index.shape[1], dtype=int)  # Type 0
        else:
            spatial_edge_index = np.empty((2, 0), dtype=np.int64)
            spatial_edge_types = np.empty(0, dtype=int)

        # 合并序列边和空间边
        edge_index = np.concatenate([seq_edge_index, spatial_edge_index], axis=1)
        edge_types_all = np.concatenate([seq_edge_types, spatial_edge_types])

        num_total_edges = edge_index.shape[1]
        if num_total_edges == 0:
            return (np.empty((2, 0), dtype=np.int64),
                    np.empty((0, 10), dtype=np.float32),
                    np.empty((0, 3), dtype=np.float32))

        # 3. 计算边特征 (前8维: 距离的RBF展开, 后2维: 边类型one-hot)
        edge_scalar_attributes = np.zeros((num_total_edges, 10), dtype=np.float32)
        edge_vector_attributes = np.zeros((num_total_edges, 3), dtype=np.float32)

        # RBF参数
        rbf_dim = 8
        rbf_centers = np.linspace(0, self.cutoff_distance, rbf_dim)
        rbf_width = (rbf_centers[1] - rbf_centers[0]) if rbf_dim > 1 else 1.0

        for k in range(num_total_edges):
            src_idx, dst_idx = edge_index[0, k], edge_index[1, k]
            distance = dist_mat[src_idx, dst_idx]
            # RBF展开
            rbf_feat = np.exp(-((distance - rbf_centers) ** 2) / (2 * (rbf_width ** 2)))
            edge_scalar_attributes[k, :rbf_dim] = rbf_feat
            # 边类型one-hot
            edge_type_one_hot = np.zeros(2, dtype=np.float32)
            edge_type_one_hot[edge_types_all[k]] = 1.0
            edge_scalar_attributes[k, 8:10] = edge_type_one_hot
            # 边矢量
            edge_vector_attributes[k, :] = ca_coordinates[dst_idx] - ca_coordinates[src_idx]

        # 保证 edge_vector_attributes 形状为 [E, 1, 3]
        if edge_vector_attributes.ndim == 2:
            edge_vector_attributes = edge_vector_attributes[:, None, :]
        elif edge_vector_attributes.ndim == 3 and edge_vector_attributes.shape[1] != 1:
            edge_vector_attributes = edge_vector_attributes.reshape(edge_vector_attributes.shape[0], 1, 3)
        return edge_index.astype(np.int64), edge_scalar_attributes, edge_vector_attributes
