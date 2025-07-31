#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import os
from pathlib import Path
import numpy as np
from typing import Dict, Optional
import warnings
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import logging

logger = logging.getLogger(__name__)


class PDBProcessor:
    """A class to process PDB files for extracting coordinates and structural information."""

    def __init__(self):
        """Initializes the PDB processor."""
        # Initialize the PDB parser, QUIET=True suppresses verbose warnings.
        self.parser = PDBParser(QUIET=True)

    def _three_to_one(self, three_letter_code: str) -> str:
        """
        Converts a three-letter amino acid code to a one-letter code.

        Parameters:
            three_letter_code (str): The three-letter amino acid code.

        Returns:
            str: The one-letter amino acid code, or 'X' for unknown residues.
        """
        mapping = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            'SEC': 'U', 'PYL': 'O',  # Special amino acids
            'MSE': 'M',  # Selenomethionine is often treated as Methionine
        }
        # Return 'X' for unknown codes; convert input to uppercase for matching.
        return mapping.get(three_letter_code.upper(), 'X')

    def parse_pdb_gz(self, pdb_gz_path: str) -> Optional[Dict]:
        """
        Parses a gzipped PDB file and extracts relevant information.

        Parameters:
            pdb_gz_path (str): Path to the gzipped PDB file.

        Returns:
            Optional[Dict]: A dictionary with parsed info, or None on error.
                            The dictionary contains: 'sequence', 'coords', 'plddt', 'chain_id', 'residue_ids'.
        """
        if not os.path.exists(pdb_gz_path):
            logger.error(f"PDB file not found: {pdb_gz_path}")
            return None

        pdb_id = Path(pdb_gz_path).stem.split('.')[0]

        try:
            # Suppress common warnings from Bio.PDB during file parsing.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', PDBConstructionWarning)
                with gzip.open(pdb_gz_path, 'rt') as f_in:
                    structure = self.parser.get_structure(pdb_id, f_in)
        except Exception as e:
            logger.error(f"Failed to parse PDB file {pdb_gz_path}: {e}")
            return None

        # Typically, we only consider the first model in the PDB file.
        model = structure[0]

        chains = list(model.get_chains())
        if not chains:
            logger.error(f"No chains found in PDB file {pdb_gz_path}.")
            return None
        # Process only the first chain found.
        chain = chains[0]

        sequence_list = []
        ca_coords_list = []
        plddt_values_list = []
        residue_ids_list = []

        for residue in chain.get_residues():
            # Skip non-standard residues (e.g., water, ligands) which are marked as HETATM.
            # residue.id[0] is the hetero-flag: ' ' for standard residues, 'W' for water.
            if residue.id[0] != ' ':
                continue

            # Ensure the C-alpha atom exists to form the backbone trace.
            if "CA" not in residue:
                res_name = residue.get_resname()
                logger.warning(f"Skipping residue {res_name}{residue.id[1]} in {pdb_id}: missing C-alpha atom.")
                continue

            one_letter_code = self._three_to_one(residue.get_resname())

            sequence_list.append(one_letter_code)
            ca_coords_list.append(residue["CA"].get_coord())
            # ESMFold and AlphaFold store the pLDDT confidence score in the B-factor field.
            plddt_values_list.append(residue["CA"].get_bfactor())
            residue_ids_list.append(residue.id[1])

        if not sequence_list:
            logger.warning(f"No valid amino acid residues found in chain {chain.id} of PDB {pdb_id}.")
            return None

        return {
            'sequence': "".join(sequence_list),
            'coords': np.array(ca_coords_list, dtype=np.float32),
            'plddt': np.array(plddt_values_list, dtype=np.float32),
            'chain_id': chain.id,
            'residue_ids': residue_ids_list
        }