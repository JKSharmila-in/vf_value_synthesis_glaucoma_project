"""
VF Value Synthesis - Utility Functions

Helper functions for image preprocessing, clinical data handling, and CSV logging.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import io
from PIL import Image


# ============================================================================
# CSV Logging
# ============================================================================

def ensure_records_csv(records_dir="records", filename="predictions_log.csv"):
    """
    Ensure predictions CSV exists with headers.
    Creates directory and file if they don't exist.
    
    Parameters
    ----------
    records_dir : str
        Directory to store records (will be created if missing)
    filename : str
        CSV filename
    
    Returns
    -------
    csv_path : Path
        Full path to CSV file
    """
    records_path = Path(records_dir)
    records_path.mkdir(exist_ok=True)
    
    csv_path = records_path / filename
    
    # If file doesn't exist, create with headers
    if not csv_path.exists():
        headers = [
            "timestamp",
            "age",
            "gender",
            "iop_mmhg",
            "cct_um",
            "vf_mean_db",
            "vf_min_db",
            "vf_max_db",
            "vf_std_db",
            "model_output_id"
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    
    return csv_path


def log_prediction(
    age=None,
    gender=None,
    iop=None,
    cct=None,
    vf_array=None,
    records_dir="records",
    filename="predictions_log.csv"
):
    """
    Log prediction to CSV file (append mode).
    
    Non-identifiable data only:
    - Timestamp (ISO format)
    - Optional clinical inputs (or empty if not provided)
    - VF summary statistics
    - Unique model output ID
    
    Parameters
    ----------
    age : float, optional
    gender : str, optional
        "M" or "F"
    iop : float, optional
        Intraocular pressure (mmHg)
    cct : float, optional
        Central corneal thickness (µm)
    vf_array : np.array, optional
        Synthesised VF values (61 points)
    records_dir : str
        Directory containing CSV
    filename : str
        CSV filename
    """
    csv_path = ensure_records_csv(records_dir, filename)
    
    # Generate timestamp and unique ID
    timestamp = datetime.utcnow().isoformat() + "Z"
    model_output_id = f"VF_{timestamp.replace('-', '').replace(':', '').replace('.', '_')}"
    
    # Compute VF statistics (if provided)
    if vf_array is not None:
        vf_array = np.asarray(vf_array)
        vf_mean = float(np.mean(vf_array))
        vf_min = float(np.min(vf_array))
        vf_max = float(np.max(vf_array))
        vf_std = float(np.std(vf_array))
    else:
        vf_mean = vf_min = vf_max = vf_std = None
    
    # Build record (use empty string for None values)
    record = {
        "timestamp": timestamp,
        "age": age if age is not None else "",
        "gender": gender if gender is not None else "",
        "iop_mmhg": iop if iop is not None else "",
        "cct_um": cct if cct is not None else "",
        "vf_mean_db": vf_mean if vf_mean is not None else "",
        "vf_min_db": vf_min if vf_min is not None else "",
        "vf_max_db": vf_max if vf_max is not None else "",
        "vf_std_db": vf_std if vf_std is not None else "",
        "model_output_id": model_output_id,
    }
    
    # Append to CSV
    try:
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp", "age", "gender", "iop_mmhg", "cct_um",
                    "vf_mean_db", "vf_min_db", "vf_max_db", "vf_std_db",
                    "model_output_id"
                ]
            )
            writer.writerow(record)
        return model_output_id
    except Exception as e:
        print(f"[WARNING] Failed to log prediction: {e}")
        return None


# ============================================================================
# VF Visualization Utilities
# ============================================================================

def vf_to_grid(vf_array):
    """
    Convert linear VF array (61 points) to spatial grid (10x10).
    
    Uses standard Humphrey 24-2 visual field layout.
    
    Parameters
    ----------
    vf_array : np.array [61]
        Linear VF values
    
    Returns
    -------
    grid : np.array [10, 10]
        Spatial grid with VF values at mapped positions.
        Unmapped positions are filled with NaN.
    """
    grid = np.full((10, 10), np.nan)
    
    # Standard Humphrey 24-2 field positions
    positions = [
        (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
        (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
        (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
        (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8),
        (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
        (8, 3), (8, 4), (8, 5), (8, 6),
    ]
    
    vf_array = np.asarray(vf_array)
    for i, (r, c) in enumerate(positions):
        if i < len(vf_array):
            grid[r, c] = vf_array[i]
    
    return grid


def load_image_as_pil(image_source):
    """
    Load image from file path or BytesIO object.
    
    Parameters
    ----------
    image_source : str or BytesIO or PIL.Image
        Image source (file path, bytes buffer, or PIL Image)
    
    Returns
    -------
    img : PIL.Image
        RGB image
    """
    if isinstance(image_source, str):
        return Image.open(image_source).convert("RGB")
    elif isinstance(image_source, io.BytesIO):
        return Image.open(image_source).convert("RGB")
    elif isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    else:
        raise TypeError(f"Unsupported image source type: {type(image_source)}")


# ============================================================================
# Clinical Data Validation
# ============================================================================

def validate_age(age_str):
    """
    Validate and parse age input.
    
    Parameters
    ----------
    age_str : str
    
    Returns
    -------
    age : float or None
        Parsed age (1-120), or None if invalid
    """
    if not age_str or str(age_str).strip() == "":
        return None
    try:
        age = float(age_str)
        if 1.0 <= age <= 120.0:
            return age
        else:
            return None
    except (ValueError, TypeError):
        return None


def validate_iop(iop_str):
    """
    Validate and parse intraocular pressure (IOP).
    
    Parameters
    ----------
    iop_str : str
    
    Returns
    -------
    iop : float or None
        Parsed IOP (0-60 mmHg), or None if invalid
    """
    if not iop_str or str(iop_str).strip() == "":
        return None
    try:
        iop = float(iop_str)
        if 0.0 <= iop <= 60.0:
            return iop
        else:
            return None
    except (ValueError, TypeError):
        return None


def validate_cct(cct_str):
    """
    Validate and parse central corneal thickness (CCT).
    
    Parameters
    ----------
    cct_str : str
    
    Returns
    -------
    cct : float or None
        Parsed CCT (200-900 µm), or None if invalid
    """
    if not cct_str or str(cct_str).strip() == "":
        return None
    try:
        cct = float(cct_str)
        if 200.0 <= cct <= 900.0:
            return cct
        else:
            return None
    except (ValueError, TypeError):
        return None


def validate_gender(gender_str):
    """
    Validate and normalize gender input.
    
    Parameters
    ----------
    gender_str : str
    
    Returns
    -------
    gender : str or None
        "M", "F", or None if invalid/empty
    """
    if not gender_str or str(gender_str).strip() == "":
        return None
    
    gender_upper = str(gender_str).strip().upper()
    
    if gender_upper.startswith("M") or gender_upper == "MALE":
        return "M"
    elif gender_upper.startswith("F") or gender_upper == "FEMALE":
        return "F"
    else:
        return None


if __name__ == "__main__":
    print("VF Value Synthesis Utilities Module")
    print("Loaded successfully. Use for preprocessing and logging.")
    
    # Example: ensure CSV exists
    csv_path = ensure_records_csv()
    print(f"CSV file: {csv_path}")
    
    # Example: log a sample prediction
    sample_vf = np.random.uniform(0, 35, 61)
    output_id = log_prediction(
        age=55.0,
        gender="F",
        iop=14.5,
        cct=540.0,
        vf_array=sample_vf
    )
    print(f"Logged prediction: {output_id}")
