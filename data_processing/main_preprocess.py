#!/usr/bin/env python3
"""
AMP Multimodal Model Data Preprocessor
"""
import os
import argparse
import json
import logging
from pathlib import Path
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple
import glob
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from .pdb_parser import PDBProcessor
from .graph_constructor import GraphConstructor
from .esm_embedder import embed_sequences_and_save
from .feature_calculator import FeatureCalculator

# ---- Logging Setup ----
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
logger = logging.getLogger(__name__)


def _mp_graph_constructor_task(args_bundle: Tuple) -> Optional[Data]:
    """
    A wrapper function for parallel graph construction in a worker process.
    """
    pdb_file_path, protein_id, base_embedding_dir, embedding_stage, activity_label, cutoff_distance = args_bundle

    try:
        # Create a local GraphConstructor instance to ensure process safety
        from .graph_constructor import GraphConstructor
        graph_constructor = GraphConstructor(
            cutoff_distance=cutoff_distance,
        )
        graph_data = graph_constructor.create_graph_from_pdb(
            pdb_gz_path=pdb_file_path,
            sequence_id=protein_id,
            embedding_dir=base_embedding_dir,
            embedding_stage=embedding_stage,
            activity_label=activity_label,
        )
        return graph_data
    except Exception:
        # Return None on failure without logging from the child process
        return None
    finally:
        # Explicitly clean up memory in the worker process
        import gc
        gc.collect()


class AMPPreprocessor:
    """
    Handles sequence extraction, ESM embedding, and graph construction for AMP datasets.
    """
    def __init__(self, output_dir: str, cutoff_distance: float = 10.0, max_seq_sep: int = 32,
                 esm_model_name: str = "esm2_t36_3B_UR50D",
                 esm_model_base_path: Optional[str] = None, max_seq_len: int = 200):
        """
        Initializes the preprocessor.

        Parameters:
            output_dir (str): Root directory for all output files.
            cutoff_distance (float): Distance cutoff for graph construction (in Å).
            max_seq_sep (int): Maximum sequence separation for graph edges.
            esm_model_name (str): Name of the ESM model for embeddings.
            esm_model_base_path (Optional[str]): Local path to ESM model files.
            max_seq_len (int): Maximum allowed sequence length.
        """
        self.output_dir_base = output_dir
        self.graph_constructor = GraphConstructor(
            cutoff_distance=cutoff_distance, max_seq_sep=max_seq_sep
        )
        self.pdb_processor = PDBProcessor()
        self.esm_model_name = esm_model_name
        self.esm_model_base_path = esm_model_base_path
        self.max_seq_len = max_seq_len

    def preprocess_dataset(self,
                           data_root: str,
                           benchmark_mode: str = "benchmark1",
                           pdb_file_type: str = "pdb.gz",
                           num_workers: int = 16,
                           process_embeddings: bool = True,
                           force_regenerate_embeddings: bool = False,
                           force_regenerate_graphs: bool = False,
                           batch_size: int = 500):
        """
        Main entry point for preprocessing an AMP dataset.
        """
        output_dir = Path(self.output_dir_base)
        output_dir.mkdir(parents=True, exist_ok=True)
        graphs_output_dir = output_dir / "graphs"
        embeddings_output_dir = output_dir / "embeddings"

        logger.info("--- Starting AMP data preprocessing ---")
        logger.info(f"Output will be saved to: {output_dir}")

        if benchmark_mode == "benchmark1":
            pos_folders = ["AMP_eval", "AMP_test", "AMP_train"]
            neg_folders = ["DECOY_eval", "DECOY_test", "DECOY_train"]
        elif benchmark_mode == "benchmark2":
            pos_folders = ["amp_eval", "amp_test", "amp_train"]
            neg_folders = ["non_amp_eval", "non_amp_test", "non_amp_train"]
        else:
            logger.error(f"Unsupported benchmark mode: {benchmark_mode}")
            self._create_splits(output_dir, [])
            return

        file_ext = f"*.{pdb_file_type}"
        pos_pdb_files = [f for folder in pos_folders for f in glob.glob(os.path.join(data_root, folder, file_ext))]
        neg_pdb_files = [f for folder in neg_folders for f in glob.glob(os.path.join(data_root, folder, file_ext))]
        
        all_pdb_files_with_labels = [
            (path, Path(path).stem.split('.')[0], 1) for path in pos_pdb_files
        ] + [
            (path, Path(path).stem.split('.')[0], 0) for path in neg_pdb_files
        ]

        if not all_pdb_files_with_labels:
            logger.warning("No PDB files found in the specified directories.")
            self._create_splits(output_dir, [])
            return

        logger.info(f"Found {len(pos_pdb_files)} positive and {len(neg_pdb_files)} negative PDB files.")

        # --- 1. Sequence Extraction ---
        sequence_data_list = []
        for pdb_path, protein_id, label in tqdm(all_pdb_files_with_labels, desc="Extracting sequences"):
            try:
                pdb_data = self.pdb_processor.parse_pdb_gz(pdb_path)
                if pdb_data and 'sequence' in pdb_data and 0 < len(pdb_data['sequence']) <= self.max_seq_len:
                    sequence_data_list.append({
                        "id": protein_id,
                        "sequence": pdb_data['sequence'],
                        "original_path": pdb_path,
                        "activity_label": label
                    })
            except Exception as e:
                logger.error(f"Error processing {pdb_path}: {e}")

        logger.info(f"Extracted {len(sequence_data_list)} valid sequences (length <= {self.max_seq_len}).")

        sequence_json_path = output_dir / f"sequences_L_lt_{self.max_seq_len}.json"
        with open(sequence_json_path, "w") as f:
            json.dump(sequence_data_list, f, indent=2)
        logger.info(f"Sequence information saved to {sequence_json_path}")

        if not sequence_data_list:
            self._create_splits(output_dir, [])
            return

        # --- 2. ESM Embedding ---
        if process_embeddings:
            esm_embedding_subdir = "amp_embedding"
            stage_specific_esm_output_dir = embeddings_output_dir / esm_embedding_subdir
            if force_regenerate_embeddings and stage_specific_esm_output_dir.exists():
                logger.info(f"Forcing ESM embedding regeneration, deleting: {stage_specific_esm_output_dir}")
                import shutil
                shutil.rmtree(stage_specific_esm_output_dir)

            sequences_for_esm = [{"id": s["id"], "sequence": s["sequence"]} for s in sequence_data_list]
            embed_sequences_and_save(
                sequence_data=sequences_for_esm,
                output_dir=str(stage_specific_esm_output_dir),
                model_name=self.esm_model_name,
                local_model_path_root=self.esm_model_base_path,
            )
            logger.info(f"ESM embeddings saved in {stage_specific_esm_output_dir}")
        else:
            logger.info("Skipping ESM embedding computation.")

        # --- 3. Graph Construction ---
        processed_ids = set()
        if not force_regenerate_graphs:
            processed_ids = {p.stem for p in graphs_output_dir.glob("*.pt")}
            logger.info(f"Found {len(processed_ids)} existing graph files, skipping them.")
        
        tasks_for_mp = [
            (s["original_path"], s["id"], str(embeddings_output_dir), "amp_embedding", s["activity_label"],
             self.graph_constructor.feature_calculator.cutoff_distance)
            for s in sequence_data_list if s["id"] not in processed_ids
        ]

        if not tasks_for_mp:
            logger.info("No new graphs to construct.")
        else:
            logger.info(f"Constructing {len(tasks_for_mp)} new graphs using {num_workers} workers...")
            num_batches = (len(tasks_for_mp) + batch_size - 1) // batch_size
            
            with tqdm(total=len(tasks_for_mp), desc="Constructing Graphs") as pbar:
                for i in range(num_batches):
                    batch_tasks = tasks_for_mp[i * batch_size:(i + 1) * batch_size]
                    if not batch_tasks:
                        continue
                    
                    if num_workers > 1:
                        try:
                            ctx = mp.get_context("spawn")
                            with ctx.Pool(processes=num_workers) as pool:
                                batch_results = pool.map(_mp_graph_constructor_task, batch_tasks)
                        except Exception as e:
                            logger.error(f"Multiprocessing failed: {e}. Falling back to single process for this batch.")
                            batch_results = [_mp_graph_constructor_task(task) for task in batch_tasks]
                    else:
                        batch_results = [_mp_graph_constructor_task(task) for task in batch_tasks]

                    # Save valid results from the batch
                    for graph_data in batch_results:
                        if graph_data and isinstance(graph_data, Data):
                            out_path = graphs_output_dir / f"{graph_data.seq_id}.pt"
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            torch.save(graph_data, out_path)
                        pbar.update(1)

        # --- 4. Create Splits ---
        all_graph_paths = [str(p) for p in graphs_output_dir.glob("*.pt")]
        logger.info(f"Found {len(all_graph_paths)} total graph files for dataset splitting.")
        self._create_splits(output_dir, all_graph_paths)

    def _create_splits(self, output_dir: Path, all_graph_paths: List[str]):
        """
        Splits graph files into train/val/test sets based on their original folder names.
        """
        if not all_graph_paths:
            logger.warning("No graph files found. Creating empty split files.")
            for split in ["train", "val", "test"]:
                self._save_split_file([], output_dir / f"{split}.txt")
            return

        sequence_json_path = output_dir / f"sequences_L_lt_{self.max_seq_len}.json"
        id_to_orig_path = {}
        if sequence_json_path.exists():
            with open(sequence_json_path, "r") as f:
                sequences_data = json.load(f)
            id_to_orig_path = {item["id"]: item["original_path"] for item in sequences_data}
        
        splits = {"train": [], "val": [], "test": []}
        for graph_path in all_graph_paths:
            protein_id = Path(graph_path).stem
            orig_path = id_to_orig_path.get(protein_id)
            if orig_path:
                parent_folder = Path(orig_path).parent.name.lower()
                if "train" in parent_folder:
                    splits["train"].append(graph_path)
                elif "val" in parent_folder or "eval" in parent_folder:
                    splits["val"].append(graph_path)
                elif "test" in parent_folder:
                    splits["test"].append(graph_path)
                else:
                    splits["train"].append(graph_path) # Default to train
            else:
                splits["train"].append(graph_path) # Default to train if original path not found

        logger.info(f"Dataset split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

        for split_name, file_list in splits.items():
            self._save_split_file(file_list, output_dir / f"{split_name}.txt")

    def _save_split_file(self, file_list: List[str], output_path: Path):
        """
        Writes a list of graph filenames to a split file.
        """
        with open(output_path, 'w') as f:
            for full_path in file_list:
                f.write(f"{Path(full_path).name}\n")
        logger.info(f"Saved {len(file_list)} entries to {output_path}")


if __name__ == "__main__":
    try:
        if hasattr(os, 'fork'):
            mp.set_start_method("fork", force=True)
    except (ValueError, RuntimeError):
        logger.info("Fork start method not available, using default.")

    parser = argparse.ArgumentParser(description="AMP data preprocessing utility")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory to save all processed data.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the input PDB files.")
    parser.add_argument("--benchmark_mode", type=str, required=True, choices=["benchmark1", "benchmark2"], help="Dataset version.")
    parser.add_argument("--cutoff", type=float, default=10.0, help="Distance cutoff for graph construction (Å).")
    parser.add_argument("--esm_model_name", type=str, default="facebook/esm2_t36_3B_UR50D", help="ESM model name.")
    parser.add_argument("--esm_model_base_path", type=str, help="Local root directory for the ESM model.")
    parser.add_argument("--max_seq_len", type=int, default=500, help="Maximum sequence length to process.")
    parser.add_argument("--force_regenerate_embeddings", action="store_true", help="Force re-computation of all ESM embeddings.")
    parser.add_argument("--force_regenerate_graphs", action="store_true", help="Force re-construction of all graph files.")
    parser.add_argument("--skip_embeddings", action="store_true", help="Skip the ESM embedding computation step.")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count(), help="Number of CPU worker processes for graph construction.")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for processing graphs to manage memory.")
    parser.add_argument("--gpus_embed", type=str, default=None, help="GPU IDs for ESM embedding (e.g., '0,1').")

    args = parser.parse_args()

    gpu_ids = None
    if args.gpus_embed:
        try:
            gpu_ids = [int(gid.strip()) for gid in args.gpus_embed.split(',') if gid.strip()]
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, ignoring specified GPU IDs.")
                gpu_ids = None
        except ValueError:
            logger.error(f"Invalid GPU ID format: '{args.gpus_embed}'.")
            sys.exit(1)

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
        num_workers=args.num_workers,
        process_embeddings=(not args.skip_embeddings),
        force_regenerate_embeddings=args.force_regenerate_embeddings,
        force_regenerate_graphs=args.force_regenerate_graphs,
        batch_size=args.batch_size
    )

    logger.info(f"Preprocessing complete. Results are in {args.output_dir}")