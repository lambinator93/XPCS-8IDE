from pathlib import Path
from dataclasses import dataclass
import json
import re


@dataclass
class XPCSConfig:
    """Configuration container for XPCS analysis."""
    work_path: Path
    base_dir: Path
    beamline: str
    geometry: str
    compound: str
    particle: str
    temperature: int | None
    scan: int


def load_xpcs_config(config_name: str, scan: int, config_dir: Path = Path('../configs')) -> XPCSConfig:
    """
    Load XPCS configuration and extract parameters for a given scan.
    
    Args:
        config_name: Name of the JSON config file (e.g., "config_IPA_NBH_A4.json")
        scan: Scan number to load
        config_dir: Path to configs directory
    
    Returns:
        XPCSConfig with all extracted parameters
    """
    config_file = config_dir / config_name
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    base_dir = Path(config["Base"])
    dataset_path = Path(config[str(scan)])
    work_path = base_dir / dataset_path
    
    # Extract temperature from filename (e.g., "080K" → 80)
    match = re.search(r'(\d+)K', str(dataset_path))
    temperature = int(match.group(1)) if match else None
    
    return XPCSConfig(
        work_path=work_path,
        base_dir=base_dir,
        beamline=config.get("Beamline"),
        geometry=config.get("Geometry"),
        compound=config.get("Compound"),
        particle=config.get("Particle"),
        temperature=temperature,
        scan=scan,
    )

