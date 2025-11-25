import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a configuration file (YAML or JSON).
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        A dictionary containing the configuration.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Successfully loaded configuration from: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        raise

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validates the integrity and correctness of the configuration dictionary.
    
    Args:
        config: The configuration dictionary.
        
    Raises:
        ValueError: If the configuration validation fails.
    """
    # 1. Check for required top-level sections
    required_sections = ['model', 'training', 'paths']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config is missing required section: '{section}'")
    
    # 2. Validate 'model' section
    if 'architecture' not in config['model'] or 'hidden_dim' not in config['model']['architecture']:
        raise ValueError("Config is missing 'model.architecture.hidden_dim'")

    # 3. Validate 'training' section for two-stage setup
    if 'sub_stages' not in config['training']:
        raise ValueError("Config is missing 'training.sub_stages' for two-stage training.")
    
    for stage_name, stage_config in config['training']['sub_stages'].items():
        if 'optimizer' not in stage_config or 'lr' not in stage_config['optimizer']:
            raise ValueError(f"Training stage '{stage_name}' is missing optimizer learning rate ('optimizer.lr')")

    # 4. Validate 'paths' section
    if 'data_root' not in config['paths']:
        raise ValueError("Config is missing required path: 'paths.data_root'")
    
    logger.info("Configuration validation successful.")