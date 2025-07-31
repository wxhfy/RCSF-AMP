#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, List, Tuple
import torch  # Although calculations are numpy-based, results are often converted to torch.Tensors.
from scipy.spatial import distance_matrix
import logging

logger = logging.getLogger(__name__)


class FeatureCalculator:
    """A class to compute node and edge features for protein graphs."""

    # Mapping from amino acid to index (20 standard AAs + 'X' for unknown/non-standard)
    AA_TO_IDX = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
        'X': 20, 'U': 20, 'O': 20  # Map unusual amino acids U and O to the 'X' index
    }

    def __init__(self, cutoff_distance: float = 10.0):
        """
        Initializes the feature calculator.

        Parameters:
            cutoff_distance (float): The distance threshold in Angstroms (Å) for creating spatial edges.
        """
        self.cutoff_distance = cutoff_distance
    
    def calculate_node_scalar_features(self, sequence: str, plddt: np.ndarray) -> np.ndarray:
        """
        Generates scalar features for nodes: AA One-hot(21) + normalized pLDDT(1) => [L, 22].

        Parameters:
            sequence (str): The amino acid sequence.
            plddt (np.ndarray): An array of pLDDT scores of shape [L].

        Returns:
            np.ndarray: Node features of shape [L, 22].
        """
        seq_len = len(sequence)
        node_s_features = np.zeros((seq_len, 22), dtype=np.float32)

        # Create one-hot encoding for the amino acid sequence
        valid_sequence = "".join([aa if aa in self.AA_TO_IDX else 'X' for aa in sequence.upper()])
        for i, aa_char in enumerate(valid_sequence):
            aa_idx = self.AA_TO_IDX[aa_char]
            node_s_features[i, aa_idx] = 1.0

        # Add normalized pLDDT score as the last feature
        if plddt is not None:
            normalized_plddt = np.clip(plddt / 100.0, 0.0, 1.0)
            node_s_features[:, 21] = normalized_plddt[:seq_len]
            
        return node_s_features

    def normalize_coordinates(self, ca_coordinates: np.ndarray) -> np.ndarray:
        """
        Performs centroid normalization on C-alpha coordinates and returns them in [L, 1, 3] format.

        Parameters:
            ca_coordinates (np.ndarray): The original C-alpha coordinate matrix of shape [L, 3].

        Returns:
            np.ndarray: Centroid-normalized coordinate matrix of shape [L, 1, 3].
        """
        if ca_coordinates.shape[0] == 0:
            return np.empty((0, 1, 3), dtype=np.float32)
        
        centroid = np.mean(ca_coordinates, axis=0, keepdims=True)
        normalized_coords = ca_coordinates - centroid
        
        # Ensure output shape is [L, 1, 3] for compatibility with downstream models
        return normalized_coords[:, np.newaxis, :]

    def build_graph_edges(self, ca_coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds graph edges and computes their features based on C-alpha coordinates.

        Parameters:
            ca_coordinates (np.ndarray): C-alpha coordinate matrix of shape [L, 3].

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - edge_index: Edge indices of shape [2, num_edges] (int64).
                - edge_scalar_attributes: Scalar edge features of shape [num_edges, 10] (float32).
                - edge_vector_attributes: Vector edge features of shape [num_edges, 1, 3] (float32).
        """
        seq_len = ca_coordinates.shape[0]
        if seq_len <= 1:
            return (np.empty((2, 0), dtype=np.int64),
                    np.empty((0, 10), dtype=np.float32),
                    np.empty((0, 1, 3), dtype=np.float32))

        # 1. Sequential Edges (Type 1)
        adj_seq_src = np.arange(seq_len - 1)
        adj_seq_dst = np.arange(1, seq_len)
        seq_edge_index = np.array([
            np.concatenate([adj_seq_src, adj_seq_dst]),
            np.concatenate([adj_seq_dst, adj_seq_src])
        ])
        seq_edge_types = np.ones(seq_edge_index.shape[1], dtype=int)

        # 2. Spatial Edges (Type 0) - Radius Graph
        dist_mat = distance_matrix(ca_coordinates, ca_coordinates)
        
        # Find pairs within cutoff distance, excluding self-loops and sequential neighbors
        mask = (dist_mat < self.cutoff_distance) & (np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :]) > 1)
        spatial_src_nodes, spatial_dst_nodes = np.where(mask)

        if spatial_src_nodes.size > 0:
            spatial_edge_index = np.stack([spatial_src_nodes, spatial_dst_nodes], axis=0)
            spatial_edge_types = np.zeros(spatial_edge_index.shape[1], dtype=int)
        else:
            spatial_edge_index = np.empty((2, 0), dtype=np.int64)
            spatial_edge_types = np.empty(0, dtype=int)

        # Combine sequential and spatial edges
        edge_index = np.concatenate([seq_edge_index, spatial_edge_index], axis=1)
        edge_types = np.concatenate([seq_edge_types, spatial_edge_types])
        num_total_edges = edge_index.shape[1]

        # 3. Compute Edge Features
        edge_scalar_attributes = np.zeros((num_total_edges, 10), dtype=np.float32)
        edge_vector_attributes = np.zeros((num_total_edges, 3), dtype=np.float32)

        # RBF parameters for distance encoding
        rbf_dim = 8
        rbf_centers = np.linspace(0, self.cutoff_distance, rbf_dim)
        rbf_width = rbf_centers[1] - rbf_centers[0] if rbf_dim > 1 else 1.0

        src_indices, dst_indices = edge_index[0], edge_index[1]
        distances = dist_mat[src_indices, dst_indices]

        # Batch compute RBF features
        rbf_feat = np.exp(-((distances[:, None] - rbf_centers) ** 2) / (2 * rbf_width ** 2))
        edge_scalar_attributes[:, :rbf_dim] = rbf_feat

        # Batch compute one-hot for edge types
        edge_scalar_attributes[np.arange(num_total_edges), 8 + edge_types] = 1.0

        # Batch compute vector features
        edge_vector_attributes = ca_coordinates[dst_indices] - ca_coordinates[src_indices]

        return (edge_index.astype(np.int64),
                edge_scalar_attributes,
                edge_vector_attributes[:, np.newaxis, :])