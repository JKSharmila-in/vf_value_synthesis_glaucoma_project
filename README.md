# VF Value Synthesis – Visual Field Prediction from Fundus Images

A medical AI research prototype for synthesizing Visual Field (VF) values from fundus (retinal) images using Conditional Diffusion Probabilistic Models (CDPM).

---

## ⚠️ Research Prototype Notice

**This is a research prototype, not a diagnostic or production system.** 

- **Use by qualified ophthalmologists / medical professionals only**
- **Not intended for clinical diagnosis**
- **All predictions must be validated by qualified ophthalmologists in appropriate clinical context**
- This prototype is for research and evaluation purposes under institutional oversight

---

## Features

- **Image-Based Prediction**: Leverages fundus image analysis with Vision Transformer embeddings
- **Optional Clinical Data**: Accepts supplementary parameters (Age, Gender, IOP, CCT) for improved context
- **Diffusion-Based Synthesis**: Advanced generative modeling using DDPM for VF field synthesis
- **Learned Null Tokens**: Missing optional fields use learned representations, not population means
- **Local Processing**: All inference runs locally on your machine; no data transmission
- **CSV Logging**: Non-identifiable prediction records stored locally
- **Static Frontend**: Deployable to GitHub Pages (frontend is HTML/CSS/JS only)

---

## Project Structure

```
VF_Value_Synthesis/
├── index.html                    # Landing page
├── patient-details.html          # Patient input form
├── prediction-result.html        # Results display
├── feedback.html                 # Feedback form
├── css/
│   └── styles.css                # Comprehensive styling
├── js/
│   └── app.js                    # Frontend logic
├── inference.py                  # Model loading & inference
├── utils.py                      # Utilities (preprocessing, logging)
├── best_model.pt                 # Trained PyTorch model (NOT committed)
├── records/
│   └── predictions_log.csv       # Local prediction log
├── .gitignore                    # Excludes model and data files
└── README.md                     # This file
```

---

## Requirements

### Frontend (HTML/CSS/JavaScript)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- No server required for UI display
- Can be deployed directly to GitHub Pages

### Backend (Python Inference)
- **Python 3.8+**
- **PyTorch** (with CUDA support recommended for performance)
- **torchvision**
- **timm** (PyTorch Image Models)
- **scikit-learn**
- **Pillow**
- **numpy**
- **pandas** (optional, for data inspection)

### Input Requirements
- **Fundus Image** (REQUIRED): JPG or PNG, minimum 256×256 pixels, recommended 512×512+
- **Age** (optional): 1–120 years
- **Gender** (optional): Male / Female
- **IOP** (optional): 0–60 mmHg
- **CCT** (optional): 200–900 µm

---

## Installation & Setup

### 1. Clone or Download the Repository

```bash
git clone https://github.com/your-org/VF_Value_Synthesis.git
cd VF_Value_Synthesis
```

### 2. Set Up Python Environment

#### Using `venv`:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Using `conda`:
```bash
conda create -n vf_synthesis python=3.9
conda activate vf_synthesis
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install timm scikit-learn pillow numpy pandas
```

### 3. Place the Model

- **Download** `best_model.pt` from your trained model checkpoint
- **Place** it in the project root directory: `./best_model.pt`
- The `.gitignore` already excludes it from version control

### 4. Verify Installation

```bash
python -c "import torch; import timm; import inference; print('Setup successful!')"
```

---

## Usage

### Frontend Only (Demo Mode)

1. Open `index.html` in your web browser
2. Click "Start Prediction"
3. Upload a fundus image and (optionally) enter clinical parameters
4. View simulated results

**Note**: In demo mode, VF predictions are generated randomly. For real predictions, integrate the Python backend (see below).

### Backend Integration (Local Inference)

#### Option 1: Flask Server (Recommended)

Create `app.py`:

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from inference import infer_single_vf
from utils import log_prediction
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

CHECKPOINT_PATH = "./best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for VF prediction"""
    try:
        # Get uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        # Get optional clinical data
        age = request.form.get('age', type=float)
        gender = request.form.get('gender', type=str)
        iop = request.form.get('iop', type=float)
        cct = request.form.get('cct', type=float)
        n_steps = request.form.get('steps', default=300, type=int)
        
        # Save uploaded image temporarily
        temp_image_path = "/tmp/fundus_temp.jpg"
        image_file.save(temp_image_path)
        
        # Run inference
        vf_denorm, vf_zscore = infer_single_vf(
            temp_image_path,
            CHECKPOINT_PATH,
            age=age,
            gender=gender,
            iop=iop,
            cct=cct,
            n_steps=n_steps,
            device=DEVICE
        )
        
        # Log to CSV
        output_id = log_prediction(
            age=age,
            gender=gender,
            iop=iop,
            cct=cct,
            vf_array=vf_denorm
        )
        
        # Clean up temp file
        os.remove(temp_image_path)
        
        # Return results
        return jsonify({
            'vf': vf_denorm.tolist(),
            'vf_mean': float(np.mean(vf_denorm)),
            'vf_min': float(np.min(vf_denorm)),
            'vf_max': float(np.max(vf_denorm)),
            'vf_std': float(np.std(vf_denorm)),
            'model_output_id': output_id,
            'age': age,
            'gender': gender,
            'iop': iop,
            'cct': cct,
            'steps': n_steps
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

Install Flask:
```bash
pip install flask flask-cors
```

Run the server:
```bash
python app.py
```

Then update `js/app.js` to call the backend:

```javascript
// In handlePredictionSubmit()
const response = await fetch('http://localhost:5000/api/predict', {
    method: 'POST',
    body: formData
});
const predictionResult = await response.json();
```

#### Option 2: Direct Python Integration

For Jupyter notebooks or scripts:

```python
from inference import infer_single_vf
from utils import log_prediction, vf_to_grid
import matplotlib.pyplot as plt

# Run inference
vf_denorm, vf_zscore = infer_single_vf(
    image_path="fundus.jpg",
    checkpoint_path="best_model.pt",
    age=55.0,
    gender="F",
    iop=14.5,
    cct=540.0,
    n_steps=300
)

# Log prediction
output_id = log_prediction(
    age=55.0,
    gender="F",
    iop=14.5,
    cct=540.0,
    vf_array=vf_denorm
)

# Visualize
grid = vf_to_grid(vf_denorm)
plt.imshow(grid, cmap='YlGnBu', vmin=0, vmax=35)
plt.title(f"Visual Field - {output_id}")
plt.colorbar(label="Sensitivity (dB)")
plt.show()
```

---

## CSV Logging

Prediction metadata is logged locally to `records/predictions_log.csv`:

```csv
timestamp,age,gender,iop_mmhg,cct_um,vf_mean_db,vf_min_db,vf_max_db,vf_std_db,model_output_id
2024-05-20T14:32:10.123456Z,55,F,14.5,540,18.5,2.1,34.8,7.2,VF_20240520T143210123456
```

**Data logged:**
- Timestamp (ISO format)
- Optional clinical inputs (empty if not provided)
- VF summary statistics (mean, min, max, std)
- Unique model output ID

**What is NOT logged:**
- Raw fundus images
- Patient names or identifiers
- Personally identifiable information

---

## Model Architecture

### Components

1. **FundusViTEncoder**: Vision Transformer (ViT-B/16) for fundus image encoding
2. **ClinicalMLP**: Processes clinical features (Age, Gender, IOP, CCT) with learnable null tokens
3. **ConditioningFusion**: Fuses image and clinical embeddings (256-d)
4. **VFDenoiser**: Diffusion network (denoiser) for VF synthesis
5. **DiffusionSchedule**: DDPM forward/reverse sampling schedule

### Key Hyperparameters

- Image size: 224×224
- ViT embedding: 768-d
- Clinical embedding: 128-d
- Conditioning: 256-d
- VF output: 61 points
- Diffusion steps: 1000 (forward), configurable (reverse)

### Optional Field Handling

Missing clinical fields use **learned null tokens** (per-field scalar parameters) optimized during training, not population-level means.

---

## Deployment

### GitHub Pages (Frontend Only)

1. Push to GitHub with `index.html` at repository root
2. Enable GitHub Pages in repository settings
3. Frontend is immediately accessible at `https://your-org.github.io/VF_Value_Synthesis/`
4. **Note**: Real inference requires local Python backend; GitHub Pages cannot execute Python

### Local Server

1. Run Flask app on `http://localhost:5000`
2. Open `index.html` from the repository
3. Predictions will use the local backend

### Institutional Deployment

- Deploy Flask app on secure server with authentication
- Store model securely (not in version control)
- Log all predictions to database
- Implement audit trails for clinical use
- Document model validation and performance metrics

---

## File Reference

### inference.py

Core inference module with model classes and prediction functions:

```python
# Load model and run inference
vf_denorm, vf_zscore = infer_single_vf(
    image_path_or_pil="fundus.jpg",
    checkpoint_path="best_model.pt",
    age=55.0,
    gender="F",
    iop=14.5,
    cct=540.0,
    n_steps=300,
    device=None
)
```

### utils.py

Utility functions for preprocessing and logging:

```python
# Ensure CSV exists and log prediction
output_id = log_prediction(
    age=55.0,
    gender="F",
    iop=14.5,
    cct=540.0,
    vf_array=vf_array
)

# Validate inputs
age = validate_age("55")
gender = validate_gender("F")
iop = validate_iop("14.5")
cct = validate_cct("540")

# Visualize VF
grid = vf_to_grid(vf_array)
```

---

## Clinical Guidelines

1. **Validation**: Always compare synthesized VF with recent actual VF tests
2. **Context**: Consider clinical history, visual symptoms, and imaging findings
3. **Interpretation**: VF range typically 0–35 dB; lower values = localized defects
4. **Documentation**: Record model predictions and clinical assessment separately
5. **Oversight**: Implement institutional review for research use

---

## Troubleshooting

### Model File Not Found
```
FileNotFoundError: Checkpoint not found: ./best_model.pt
```
**Solution**: Download and place `best_model.pt` in the project root.

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: 
- Reduce `n_steps` (e.g., 100 instead of 300)
- Use CPU: `device = torch.device("cpu")`
- Reduce batch size if processing multiple images

### Import Errors
```
ModuleNotFoundError: No module named 'timm'
```
**Solution**: Install missing packages
```bash
pip install timm torch torchvision scikit-learn pillow numpy
```

### CSV Logging Fails
```
[WARNING] Failed to log prediction: Permission denied
```
**Solution**: Ensure `records/` directory is writable or create it manually:
```bash
mkdir records
```

---

## Contributing

For research improvements or bug reports:

1. Document observations in `feedback.html` form
2. Provide fundus image (de-identified) if possible
3. Note model hyperparameters used
4. Include clinical context if relevant

---

## License

[Specify your license, e.g., CC-BY-NC for research use]

---

## Citation

If you use this prototype in research, please cite:

```
[Citation information to be added]
```

---

## Support

For questions or issues:
- Open an issue on GitHub
- Contact the development team
- For clinical matters, consult a qualified ophthalmologist

---

## Disclaimer

This research prototype is provided "as-is" without warranty. The developers and institutions are not liable for:

- Incorrect predictions or clinical outcomes
- Data loss or privacy breaches
- Any use outside research contexts
- Decisions based on model outputs without professional review

**All predictions must be validated by qualified ophthalmologists.**

---

**Last Updated**: May 2026  
**Version**: 1.0 (Research Prototype)
