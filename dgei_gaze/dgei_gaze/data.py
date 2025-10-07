import yaml
import os
from typing import Dict, Any, Optional


def read_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Read a YAML file and return its contents as a dictionary.
    
    Args:
        file_path (str): Path to the YAML file
        
    Returns:
        Dict[str, Any]: Dictionary containing the YAML file contents
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If there's an error parsing the YAML file
        
    Example:
        >>> config = read_yaml_file('config.yaml')
        >>> print(config['BaselinePitch'])
        0
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            
        # Handle case where file is empty or contains only comments
        if data is None:
            return {}
            
        return data
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error reading YAML file {file_path}: {e}")


def read_gaze_config(file_path: str) -> Dict[str, float]:
    """
    Read a gaze configuration YAML file with expected baseline and threshold values.
    
    Args:
        file_path (str): Path to the gaze configuration YAML file
        
    Returns:
        Dict[str, float]: Dictionary containing gaze configuration values
        
    Expected YAML structure:
        BaselinePitch: 0
        BaselineYaw: 0
        PitchThreshold: 0
        YawThreshold: 0
    """
    config = read_yaml_file(file_path)
    
    # Define expected keys with default values
    expected_keys = {
        'BaselinePitch': 0.0,
        'BaselineYaw': 0.0,
        'PitchThreshold': 0.0,
        'YawThreshold': 0.0
    }
    
    # Ensure all expected keys are present, use defaults if missing
    gaze_config = {}
    for key, default_value in expected_keys.items():
        gaze_config[key] = float(config.get(key, default_value))
    
    return gaze_config


def write_yaml_file(data: Dict[str, Any], file_path: str) -> None:
    """
    Write a dictionary to a YAML file.
    
    Args:
        data (Dict[str, Any]): Dictionary to write to YAML
        file_path (str): Path where the YAML file should be written
        
    Raises:
        yaml.YAMLError: If there's an error writing the YAML file
    """
    def round_floats(obj):
        """Recursively round float values to 2 decimal places and ensure proper type"""
        if isinstance(obj, (float, int)):
            # Convert to standard Python float and round to 2 decimal places
            return float(round(float(obj), 2))
        elif isinstance(obj, dict):
            return {key: round_floats(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [round_floats(item) for item in obj]
        else:
            return obj
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Round all float values to 2 decimal places and ensure proper types
        rounded_data = round_floats(data)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(
                rounded_data, 
                file, 
                default_flow_style=False, 
                sort_keys=False,
                allow_unicode=True
            )
            
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error writing YAML file {file_path}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error writing YAML file {file_path}: {e}")