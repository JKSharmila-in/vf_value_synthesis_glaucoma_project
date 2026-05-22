import os
from PIL import Image


# -----------------------------------
# ALLOWED IMAGE TYPES
# -----------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    """
    Check if uploaded file is a valid image type
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# -----------------------------------
# IMAGE VALIDATION
# -----------------------------------
def validate_image(image_path):
    """
    Ensure uploaded file is a valid image
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


# -----------------------------------
# OPTIONAL: VF POST-PROCESSING
# -----------------------------------
def format_vf_values(vf_values, decimals=2):
    """
    Format VF values for display or export
    """
    return [round(float(v), decimals) for v in vf_values]
