from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_layout():
    required_files = {
        "README.md",
        "requirements.txt",
        "train_sucf.py",
        "configs/training_config.yaml",
        "docs/reproducibility.md",
        "results/README.md",
    }
    required_directories = {
        "configs",
        "data_processing",
        "experiments",
        "models",
        "scripts",
        "tests",
        "utils",
    }

    assert all((ROOT / path).is_file() for path in required_files)
    assert all((ROOT / path).is_dir() for path in required_directories)


def test_no_machine_specific_paths_in_release_sources():
    source_files = list(ROOT.glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    assert "/home/20T-1/" not in source_text
    assert "/home/fyh0106/" not in source_text
