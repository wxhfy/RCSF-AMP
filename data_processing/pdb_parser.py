#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import os
from pathlib import Path

import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import logging

logger = logging.getLogger(__name__)


class PDBProcessor:
    """处理PDB文件以提取坐标和结构信息的类。"""

    def __init__(self):
        """
        初始化PDB处理器。

        参数:
            dssp_executable (str): DSSP可执行文件的路径 (默认: 假定 'mkdssp' 在系统PATH中)。
        """
        self.parser = PDBParser(QUIET=True)  # 初始化PDB解析器，QUIET=True避免过多警告

    def _three_to_one(self, three_letter_code: str) -> str:
        """
        将氨基酸三字母缩写转换为单字母缩写。

        参数:
            three_letter_code (str): 氨基酸三字母缩写。

        返回:
            str: 氨基酸单字母缩写，未知残基返回 'X'。
        """
        mapping = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            'SEC': 'U', 'PYL': 'O',  # 特殊氨基酸
            # 根据biotite常见的非标准残基映射，这里可以补充更多
            'MSE': 'M',  # 甲硒氨酸通常视为M
        }
        return mapping.get(three_letter_code.upper(), 'X')  # 转换为大写以匹配，未知返回X

    def parse_pdb_gz(self, pdb_gz_path: str) -> Optional[Dict]:
        """
        解析gzipped PDB文件并提取相关信息。

        参数:
            pdb_gz_path (str): gzipped PDB文件的路径。

        返回:
            Optional[Dict]: 包含解析信息的字典，如果出错则返回None。
                           字典包含: 'sequence', 'coords', 'plddt', 'chain_id', 'residue_ids'
        """
        if not os.path.exists(pdb_gz_path):
            logger.error(f"PDB文件未找到: {pdb_gz_path}")
            return None

        pdb_id = Path(pdb_gz_path).stem.split('.')[0]  # 获取PDB ID (例如文件名 '1abc.pdb.gz' -> '1abc')

        try:
            with warnings.catch_warnings():  # 忽略Bio.PDB可能产生的警告
                warnings.simplefilter('ignore', PDBConstructionWarning)
                with gzip.open(pdb_gz_path, 'rt') as f_in:  # 'rt'模式读取文本
                    structure = self.parser.get_structure(pdb_id, f_in)
        except Exception as e:
            logger.error(f"解析PDB文件 {pdb_gz_path} 失败: {e}")
            return None

        model = structure[0]  # 通常取第一个模型

        # 尝试获取第一个链，如果链ID命名不规范，可能需要更鲁棒的逻辑
        if not list(model.get_chains()):
            logger.error(f"PDB文件 {pdb_gz_path} 中未找到链。")
            return None
        chain = list(model.get_chains())[0]  # 取第一个链

        chain_id = chain.id

        sequence_list = []
        ca_coords_list = []
        plddt_values_list = []
        residue_ids_list = []  # 存储残基编号

        for residue in chain.get_residues():
            # 跳过水分子、配体等非标准残基 (HETATM记录)
            # residue.id[0] 是 hetero-flag, ' ' 表示标准氨基酸, 'W'表示水, 'H_'开头的通常是HETATM
            if residue.id[0] != ' ':
                continue

            res_name = residue.get_resname()
            one_letter_code = self._three_to_one(res_name)

            if "CA" not in residue:  # 确保Cα原子存在
                logger.warning(f"残基 {res_name}{residue.id[1]} 在 {pdb_id} 中缺少Cα原子, 跳过此残基。")
                continue

            sequence_list.append(one_letter_code)
            ca_coords_list.append(residue["CA"].get_coord())  # 获取Cα坐标
            plddt_values_list.append(residue["CA"].get_bfactor())  # ESMFold将pLDDT存储在B-factor字段
            residue_ids_list.append(residue.id[1])  # 残基编号 (通常是整数)

        if not sequence_list:  # 如果链中没有有效的氨基酸残基
            logger.warning(f"PDB文件 {pdb_id} 的链 {chain_id} 中未找到有效氨基酸残基。")
            return None

        return {
            'sequence': "".join(sequence_list),
            'coords': np.array(ca_coords_list, dtype=np.float32),
            'plddt': np.array(plddt_values_list, dtype=np.float32),
            'chain_id': chain_id,
            'residue_ids': residue_ids_list
        }
