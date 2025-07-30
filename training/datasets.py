#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
import logging

logger = logging.getLogger(__name__)


def custom_collate_fn(data_list: List[Data]) -> Batch:
    """
    自定义的collate函数，适配新的数据格式
    支持动态节点特征维度（22维 baseline 或 21维 no_plddt_gate）
    - x: [L, 22/21] - 节点标量特征（根据消融类型动态调整）
    - node_vector: [L, 1, 3] - 节点向量特征
    - edge_attr: [E, 10] - 边标量特征
    - edge_vector: [E, 1, 3] - 边向量特征
    - y: [1] - 活性标签
    
    Args:
        data_list: List of PyG Data objects
        
    Returns:
        Batch: PyG Batch object
    """
    if not data_list:
        return Batch()
    
    # 使用PyG默认的批处理逻辑，新数据格式已经正确
    batch = Batch.from_data_list(data_list)
    
    # 验证关键字段是否存在
    expected_fields = ['x', 'node_vector', 'amp_embedding', 'edge_index', 'edge_attr', 'edge_vector', 'y']
    missing_fields = [field for field in expected_fields 
                     if not hasattr(batch, field) or getattr(batch, field) is None]
    
    if missing_fields:
        logger.warning(f"批次数据缺少字段: {missing_fields}")
    
    # 验证维度是否符合格式（支持22维baseline或21维no_plddt_gate）
    if hasattr(batch, 'x') and batch.x is not None:
        actual_dim = batch.x.size(-1)
        if actual_dim not in [21, 22]:
            logger.warning(f"节点标量特征维度异常: 期望21或22维，实际{actual_dim}维")
        else:
            logger.debug(f"节点特征维度: {actual_dim}维 ({'baseline' if actual_dim == 22 else 'no_plddt_gate'})")
    
    if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
        if batch.edge_attr.size(-1) != 10:
            logger.warning(f"边标量特征维度不匹配: 期望10维，实际{batch.edge_attr.size(-1)}维")
    
    if hasattr(batch, 'node_vector') and batch.node_vector is not None:
        expected_shape = (batch.node_vector.size(0), 1, 3)
        if batch.node_vector.shape[1:] != (1, 3):
            logger.warning(f"节点向量特征格式不匹配: 期望[N, 1, 3]，实际{batch.node_vector.shape}")
    
    if hasattr(batch, 'edge_vector') and batch.edge_vector is not None:
        if batch.edge_vector.shape[1:] != (1, 3):
            logger.warning(f"边向量特征格式不匹配: 期望[E, 1, 3]，实际{batch.edge_vector.shape}")
    
    logger.debug(f"批次数据验证完成: {len(data_list)}个样本")
    return batch


class AMPGraphDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            split_file: Optional[str] = None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            config: Optional[Dict] = None  # 新增：接收配置参数
    ):
        self.root_dir = os.path.expanduser(root_dir)
        # root_dir 就是 graphs 目录，直接使用
        self.data_files_dir = self.root_dir
        
        # 保存配置以便在数据加载时使用
        self.config = config or {}
        
        # 检查是否为 no_plddt_gate 消融实验
        self.is_no_plddt_ablation = False
        if self.config:
            model_config = self.config.get('model', {})
            architecture_config = model_config.get('architecture', {})
            ablation_type = architecture_config.get('ablation_type', '')
            use_plddt_gate = architecture_config.get('use_plddt_gate', True)
            
            # 检查是否为 no_plddt_gate 消融实验
            if ablation_type == 'no_plddt_gate' or use_plddt_gate is False:
                self.is_no_plddt_ablation = True
                logger.info("🔍 检测到 no_plddt_gate 消融实验，将移除节点特征中的 pLDDT 维度")

        self.file_list: List[str] = []  # 存储 .pt 文件的基本名称

        if not os.path.isdir(self.data_files_dir):
            logger.warning(f"指定的图文件目录 {self.data_files_dir} 不存在。")
            # 即使目录不存在，也调用super，但数据集将为空
            super(AMPGraphDataset, self).__init__(root=self.root_dir, transform=transform, pre_transform=pre_transform,
                                                  pre_filter=pre_filter)
            logger.warning(f"数据集为空，因为图文件目录 {self.data_files_dir} 未找到。")
            return

        if split_file and os.path.exists(split_file):
            with open(split_file, "r") as f:
                for line in f:
                    filename = line.strip()
                    if filename and filename.endswith(".pt"):
                        # 确保文件存在于 data_files_dir 中
                        if os.path.exists(os.path.join(self.data_files_dir, filename)):
                            self.file_list.append(filename)
                        else:
                            logger.warning(
                                f"划分文件 {split_file} 中的文件 {filename} 在目录 {self.data_files_dir} 中未找到，已跳过。")
            logger.info(f"从划分文件 {split_file} 加载了 {len(self.file_list)} 个图的引用。")
        elif split_file is None:  # 如果不提供划分文件，则加载目录中所有.pt文件
            logger.info(f"未提供划分文件，将从 {self.data_files_dir} 加载所有 .pt 文件。")
            try:
                for filename in os.listdir(self.data_files_dir):
                    if filename.endswith(".pt"):
                        self.file_list.append(filename)
                logger.info(f"从目录 {self.data_files_dir} 加载了 {len(self.file_list)} 个图的引用。")
            except FileNotFoundError:
                logger.warning(f"图文件目录 {self.data_files_dir} 未找到，即使在尝试加载所有文件时。")
        else:  # split_file 提供但不存在
            logger.warning(f"提供的划分文件 {split_file} 未找到。数据集将为空。")

        effective_num_samples = len(self.file_list)
        if effective_num_samples == 0:
            logger.warning(
                f"在 {self.data_files_dir} (使用划分文件: {split_file if split_file else '无'}) 未找到有效的 .pt 文件或引用。数据集为空。")

        # super().__init__ 应该在确定 self.file_list 之后调用，
        # 因为 processed_file_names 依赖它。
        # PyG Dataset 的 'root' 参数通常用于存储 'processed' 和 'raw' 子目录。
        # 如果您的图文件直接在 root_dir/graphs，那么 processed_dir 应该是 root_dir/graphs。
        # 为了让PyG的内部机制（如检查processed文件是否存在）正常工作，
        # 我们可能需要调整这里的root和processed_dir的理解。
        # 一个简单的做法是让 root_dir 指向包含 'graphs' 的父目录，
        # 并在 processed_paths 中返回相对于 self.processed_dir 的路径。
        # 或者，如果图已完全预处理，我们可以简化一些PyG的Dataset机制。

        # 简化：假设图文件已存在于 self.data_files_dir，我们只负责加载它们。
        # PyG 的 root 参数是期望的，但如果 processed_file_names 返回绝对路径或我们自己处理加载，
        # 它的确切用途可能不那么关键。
        super(AMPGraphDataset, self).__init__(root=self.root_dir, transform=transform, pre_transform=pre_transform,
                                              pre_filter=pre_filter)
        # logger.info(f"AMPGraphDataset 初始化完毕: {effective_num_samples} 个样本，阶段 {stage}。")

    @property
    def raw_file_names(self) -> List[str]:
        # 如果有原始数据处理步骤，在这里返回原始文件名
        return []

    @property
    def processed_file_names(self) -> List[str]:
        # 返回期望在 self.processed_dir 中找到的文件名列表
        # 在我们的情况下，这些是 self.file_list 中的基本文件名
        # 如果 self.processed_dir 就是 self.data_files_dir，那么这直接对应
        if not self.file_list:
            # PyG 要求 processed_file_names 至少返回一个占位符，即使文件不存在，以避免某些内部错误
            return ["dummy_placeholder_to_avoid_pyg_error.pt"]
        return self.file_list

    def len(self) -> int:
        return len(self.file_list)

    def get(self, idx: int) -> Data:
        if not self.file_list or idx >= len(self.file_list):
            logger.error(f"尝试从索引 {idx} 获取样本，但文件列表为空或索引越界。")
            # 返回一个空的Data对象，或者根据策略抛出错误
            return self._create_empty_data_for_error_case()

        filename = self.file_list[idx]
        # 文件路径是相对于 self.data_files_dir (即 processed_dir)
        file_path = os.path.join(self.data_files_dir, filename)

        try:
            data = torch.load(file_path, map_location=torch.device('cpu'))  # 加载到CPU以避免多进程问题
            if not isinstance(data, Data):
                logger.error(f"文件 {file_path} 未包含有效的 PyG Data 对象，得到类型 {type(data)}。")
                return self._create_empty_data_for_error_case(seq_id=filename.replace(".pt", ""))
            
            # 核心修复：先移除pLDDT特征，再进行验证
            if self.is_no_plddt_ablation:
                data = self._remove_plddt_features(data, filename)

            # 验证并可能填充缺失属性
            self._verify_data_attributes(data, filename)  
            
            return data
        except FileNotFoundError:
            logger.error(f"数据文件未找到: {file_path}")
            return self._create_empty_data_for_error_case(seq_id=filename.replace(".pt", ""))
        except Exception as e:
            logger.error(f"加载数据 {file_path} 失败: {str(e)}")
            return self._create_empty_data_for_error_case(seq_id=filename.replace(".pt", ""))

    def _remove_plddt_features(self, data: Data, filename: str) -> Data:
        """
        在 no_plddt_gate 消融实验中移除节点特征中的 pLDDT 相关维度
        
        根据特征格式说明，pLDDT 分数在第21列（索引20）
        移除该列后，node_scalar_dim 从 22 降到 21
        """
        seq_id = getattr(data, 'seq_id', filename.replace(".pt", ""))
        
        if not hasattr(data, 'x') or data.x is None:
            logger.warning(f"{seq_id}: 节点特征 x 不存在，无法移除 pLDDT 维度")
            return data

        original_shape = data.x.shape
        
        # 核心逻辑：如果处于pLDDT消融模式，则必须确保最终维度是21
        if self.is_no_plddt_ablation:
            if original_shape[1] == 22:
                # 标准情况：从22维移除pLDDT（索引20）
                data.x = torch.cat([data.x[:, :20], data.x[:, 21:]], dim=1)
                new_shape = data.x.shape
                logger.debug(f"{seq_id}: 成功移除 pLDDT 维度。原始: {original_shape} -> 新: {new_shape}")
                
                if new_shape[1] != 21:
                     logger.error(f"{seq_id}: pLDDT移除后维度异常: 期望21，实际{new_shape[1]}。这是一个Bug。")

            elif original_shape[1] == 21:
                # 已经是21维，无需操作，但记录一下
                logger.debug(f"{seq_id}: 节点特征已经是21维，无需移除pLDDT。")
            
            else:
                # 维度异常，这是一个需要关注的问题
                logger.warning(f"{seq_id}: 节点特征维度异常({original_shape[1]})，无法执行pLDDT移除。")
        
        # 如果存在单独的 plddt 字段，也清除它
        if hasattr(data, 'plddt'):
            logger.debug(f"{seq_id}: 同时清除单独的 plddt 字段")
            del data.plddt # 直接删除属性更干净
        
        return data

    def _verify_data_attributes(self, data: Data, filename: str):
        """验证Data对象是否包含新数据格式的必需属性"""
        # 新数据格式的必需属性
        required_attrs = ['x', 'node_vector', 'edge_index', 'edge_attr', 'edge_vector', 'amp_embedding', 'y']
        
        # 确保 num_nodes 存在且正确
        if not hasattr(data, 'num_nodes') or data.num_nodes != data.x.shape[0]:
            data.num_nodes = data.x.shape[0] if hasattr(data, 'x') and data.x is not None else 0

        missing_attrs = [attr for attr in required_attrs if not hasattr(data, attr) or getattr(data, attr) is None]
        seq_id_for_log = getattr(data, 'seq_id', filename.replace(".pt", ""))

        if missing_attrs:
            logger.warning(f"数据 {filename} (ID: {seq_id_for_log}) 缺少属性: {', '.join(missing_attrs)}")
            
            # 为缺失的关键属性创建占位符
            for attr in missing_attrs:
                if attr == "x":
                    # 根据是否为 no_plddt_gate 消融实验确定维度
                    node_dim = 21 if self.is_no_plddt_ablation else 22
                    setattr(data, attr, torch.zeros(data.num_nodes, node_dim, dtype=torch.float))
                elif attr == "node_vector":
                    setattr(data, attr, torch.zeros(data.num_nodes, 1, 3, dtype=torch.float))
                elif attr == "edge_attr":
                    num_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0
                    setattr(data, attr, torch.zeros(num_edges, 10, dtype=torch.float))
                elif attr == "edge_vector":
                    num_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0
                    setattr(data, attr, torch.zeros(num_edges, 1, 3, dtype=torch.float))
                elif attr == "amp_embedding":
                    setattr(data, attr, torch.zeros(data.num_nodes, 2560, dtype=torch.float))
                elif attr == "y":
                    setattr(data, attr, torch.tensor([0], dtype=torch.long))
                    logger.debug(f"为 {seq_id_for_log} 的缺失标签创建默认值 [0]")
        
        # 验证数据维度
        self._verify_data_dimensions(data, seq_id_for_log)
    
    def _verify_data_dimensions(self, data: Data, seq_id: str):
        """验证数据维度是否符合格式要求（考虑 pLDDT 消融）"""
        # 根据是否为 no_plddt_gate 消融实验确定期望的节点特征维度
        expected_node_dim = 21 if self.is_no_plddt_ablation else 22
        
        if hasattr(data, 'x') and data.x is not None:
            if data.x.shape[1] != expected_node_dim:
                logger.warning(f"{seq_id}: 节点标量特征维度 {data.x.shape[1]} != {expected_node_dim} (no_plddt: {self.is_no_plddt_ablation})")
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if data.edge_attr.shape[1] != 10:
                logger.warning(f"{seq_id}: 边标量特征维度 {data.edge_attr.shape[1]} != 10")
        
        if hasattr(data, 'node_vector') and data.node_vector is not None:
            if data.node_vector.shape[1:] != (1, 3):
                logger.warning(f"{seq_id}: 节点向量格式 {data.node_vector.shape} != [N, 1, 3]")
        
        if hasattr(data, 'edge_vector') and data.edge_vector is not None:
            if data.edge_vector.shape[1:] != (1, 3):
                logger.warning(f"{seq_id}: 边向量格式 {data.edge_vector.shape} != [E, 1, 3]")

    def _create_empty_data_for_error_case(self, seq_id: str = "error_empty_data") -> Data:
        """为错误情况创建一个空的Data对象，使用正确的维度（考虑 pLDDT 消融）"""
        # 根据是否为 no_plddt_gate 消融实验确定节点特征维度
        node_scalar_dim = 21 if self.is_no_plddt_ablation else 22
        node_vector_channels = 1  # 节点向量通道数
        edge_scalar_dim = 10      # 边标量特征维度
        esm_dim = 2560           # ESM嵌入维度

        data = Data(
            x=torch.empty((0, node_scalar_dim), dtype=torch.float),
            node_vector=torch.empty((0, node_vector_channels, 3), dtype=torch.float),  # [N, 1, 3]
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, edge_scalar_dim), dtype=torch.float),
            edge_vector=torch.empty((0, node_vector_channels, 3), dtype=torch.float),  # [E, 1, 3]
            original_seq="",
            coords=torch.empty((0, 3), dtype=torch.float),
            plddt=torch.empty(0, dtype=torch.float),
            amp_embedding=torch.empty((0, esm_dim), dtype=torch.float),
            y=torch.tensor([0], dtype=torch.long),  # 使用y字段而不是activity_label
            num_nodes=0,
            seq_id=seq_id
        )
        return data


def create_dataloader(
        dataset: Dataset,  # 修改：直接接收Dataset对象
        batch_size: int = 16,
        shuffle: bool = True,  # 当使用sampler时，这里应为False
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        sampler: Optional[torch.utils.data.Sampler] = None,  # 新增sampler参数
        use_custom_collate: bool = True  # 新增：是否使用自定义collate函数
) -> PyGDataLoader:
    actual_batch_size = max(1, batch_size)

    if len(dataset) == 0:
        logger.warning(f"数据集为空。将创建一个空的 DataLoader。")
        # 对于空数据集，sampler应为None，shuffle为False
        return PyGDataLoader(dataset, batch_size=actual_batch_size, shuffle=False, num_workers=0, sampler=None)

    # persistent_workers 只在 num_workers > 0 时有意义
    actual_persistent_workers = persistent_workers if num_workers > 0 else False

    # 如果提供了sampler，则DataLoader的shuffle参数必须为False，因为sampler负责打乱
    effective_shuffle = shuffle if sampler is None else False

    # 选择collate函数
    collate_fn = custom_collate_fn if use_custom_collate else None

    dataloader = PyGDataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=effective_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,  # pin_memory只在数据加载到CUDA时有用
        persistent_workers=actual_persistent_workers,
        sampler=sampler,  # <--- 使用sampler
        collate_fn=collate_fn  # <--- 使用自定义collate函数
    )
    return dataloader


def create_data_loaders(config):
    """
    使用AMPGraphDataset和create_dataloader读取真实数据。
    支持分布式训练和测试集
    """
    logger = logging.getLogger(__name__)
    data_root = config.get('data_root', './data')
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    pin_memory = config.get('pin_memory', True)
    is_distributed = False

    # 直接使用graphs目录
    graphs_dir = os.path.join(data_root, 'graphs')
    if not os.path.exists(graphs_dir):
        raise FileNotFoundError(f"找不到graphs目录: {graphs_dir}")

    # 读取train/val/test分割文件
    train_txt = os.path.join(data_root, 'train.txt')
    val_txt = os.path.join(data_root, 'val.txt')
    test_txt = os.path.join(data_root, 'test.txt')
    
    assert os.path.exists(train_txt), f"找不到train.txt: {train_txt}"
    assert os.path.exists(val_txt), f"找不到val.txt: {val_txt}"

    train_dataset = AMPGraphDataset(graphs_dir, split_file=train_txt, config=config)
    val_dataset = AMPGraphDataset(graphs_dir, split_file=val_txt, config=config)
    
    # 测试集是可选的
    test_dataset = None
    if os.path.exists(test_txt):
        test_dataset = AMPGraphDataset(graphs_dir, split_file=test_txt, config=config)
        logger.info(f"找到测试集文件: {test_txt}")

    # 检查数据集是否为空
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"数据集为空 - 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    train_sampler = None
    val_sampler = None
    test_sampler = None

    # 创建数据加载器
    train_loader = create_dataloader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None),  # 如果有sampler则不shuffle
        num_workers=num_workers, 
        pin_memory=pin_memory,
        sampler=train_sampler
    )
    
    val_loader = create_dataloader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        sampler=val_sampler
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = create_dataloader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=pin_memory,
            sampler=test_sampler
        )

    logger.info(f"使用真实数据加载器，数据根目录: {data_root}")
    logger.info(f"  graphs目录: {graphs_dir}")
    logger.info(f"  训练集: {train_txt}, 验证集: {val_txt}")
    if test_dataset is not None:
        logger.info(f"  测试集: {test_txt}")
    logger.info(f"  训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")
    if test_dataset is not None:
        logger.info(f"  测试样本数: {len(test_dataset)}")
    # 分布式训练已移除
    
    if test_loader is not None:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader