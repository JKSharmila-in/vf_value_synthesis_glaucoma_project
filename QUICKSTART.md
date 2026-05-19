# Quick Start Guide - VF Value Synthesis

Get up and running with the VF Value Synthesis prototype in minutes.

---

## 1. Frontend Only (Demo Mode) – 2 Minutes

**No Python setup required!**

### Option A: Local File
1. Download the repository
2. Open `index.html` in your web browser
3. Navigate to "Patient Details"
4. Upload any image file (JPG/PNG)
5. Click "Run Prediction" → See simulated VF results

**Note**: In demo mode, predictions are generated randomly. For real VF synthesis, proceed to Option B.

### Option B: GitHub Pages
1. Repository must be on GitHub
2. Enable GitHub Pages in repo settings
3. Navigate to `https://your-org.github.io/VF_Value_Synthesis/`
4. Use the frontend normally (demo mode)

---

## 2. Backend Integration – 10 Minutes

For real VF predictions using the trained model.

### Step 1: Install Python Dependencies

```bash
# Navigate to project directory
cd VF_Value_Synthesis

# Option A: Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Option B: Conda
conda create -n vf_synthesis python=3.9
conda activate vf_synthesis
pip install torch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Step 2: Add the Model File

```bash
# Place your trained model in the project root
cp /path/to/best_model.pt ./best_model.pt
```

**Important**: Do NOT commit `best_model.pt` to GitHub. It's already in `.gitignore`.

### Step 3: Start the Flask Server

```bash
# Copy the template (optional - use your own Flask app if you have one)
cp app_template.py app.py

# Run the server
python app.py

# Server will start on http://localhost:5000
```

You should see:
```
VF Value Synthesis - Flask Backend
Starting server on http://localhost:5000
Device: cuda (or cpu)
Model: ./best_model.pt
```

### Step 4: Update Frontend to Use Backend

Edit `js/app.js` and uncomment the backend call in `simulatePrediction()`:

```javascript
// Around line 265, replace the setTimeout block with:
const response = await fetch('http://localhost:5000/api/predict', {
    method: 'POST',
    body: formData
});

if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
}

const predictionResult = await response.json();
resolve(predictionResult);
```

### Step 5: Use the Application

1. Open `index.html` in your browser
2. Go to "Patient Details"
3. Upload a fundus image
4. (Optional) Enter clinical parameters
5. Click "Run Prediction"
6. View results on "Prediction Results" page

**Note**: First inference may take 30–60 seconds (DDPM sampling). Subsequent runs will be faster due to caching.

---

## 3. Prediction Pipeline

```
┌─────────────────────────────────────────────────────┐
│  1. Upload Fundus Image + Clinical Data (Optional) │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  2. Frontend Sends to Backend (Flask)               │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  3. Backend (inference.py) Loads Model & Encodes   │
│     - FundusViTEncoder: fundus image → 768-d        │
│     - ClinicalMLP: features → 128-d                 │
│     - ConditioningFusion: concatenate → 256-d       │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  4. DDPM Reverse Sampling (300 steps)               │
│     Generates VF field from noise + conditioning    │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  5. Denormalize to Real VF Space (0–35 dB)          │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  6. Log to CSV (records/predictions_log.csv)        │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  7. Return Results to Frontend & Display            │
│     - VF grid visualization                         │
│     - Statistics (mean, min, max, std)              │
│     - Input parameters echoed back                  │
└─────────────────────────────────────────────────────┘
```

---

## 4. Key Files

| File | Purpose |
|------|---------|
| `index.html` | Landing page & project info |
| `patient-details.html` | Patient data input form |
| `prediction-result.html` | Results display & visualization |
| `feedback.html` | Feedback form |
| `css/styles.css` | Responsive styling |
| `js/app.js` | Frontend logic & form handling |
| `inference.py` | Model classes & prediction function |
| `utils.py` | Preprocessing & CSV logging |
| `app_template.py` | Flask backend server |
| `records/predictions_log.csv` | Local prediction log |

---

## 5. Verification Checklist

- [ ] `best_model.pt` is in the project root
- [ ] `python -c "import torch; import timm; print('OK')"` works
- [ ] Flask server starts without errors
- [ ] `/api/health` returns `{"status": "ok"}`
- [ ] Frontend form validates correctly
- [ ] Upload image appears in preview
- [ ] Prediction completes in 30–60 seconds
- [ ] Results page displays VF grid and stats
- [ ] CSV file `records/predictions_log.csv` has new entry

---

## 6. Troubleshooting

### "No module named 'timm'"
```bash
pip install timm
```

### "best_model.pt not found"
```bash
# Verify file exists
ls -la best_model.pt

# If missing, download and place in project root
```

### "CUDA out of memory"
```python
# In Flask app, use CPU or reduce steps:
DEVICE = torch.device("cpu")
# Or in form: use 100–150 steps instead of 300
```

### Port 5000 already in use
```bash
# Use a different port
python app.py --port 5001
# Update frontend to use http://localhost:5001
```

### Predictions look wrong
- Check image quality (should be clear fundus photo)
- Verify model was trained on similar dataset
- Try with multiple images to confirm consistency
- Check that clinical fields are within valid ranges

---

## 7. Next Steps

### For Research Use
1. Validate model predictions against actual VF tests
2. Document performance metrics (MSE, MAE, correlation)
3. Publish findings with appropriate disclaimers
4. Implement audit trail for institutional use

### For Production Deployment
1. Set up HTTPS/SSL
2. Implement user authentication
3. Use secure model storage (not in Git)
4. Log all predictions to database
5. Implement privacy safeguards
6. Add error handling and monitoring
7. Set rate limiting
8. Document API contracts

### For Additional Features
- Add batch prediction mode
- Implement confidence scores
- Add comparison with previous VF tests
- Integrate with EHR systems
- Generate PDF reports

---

## 8. Performance Notes

### Timing (on NVIDIA GPU)
- Model loading: ~2–3 seconds
- Preprocessing: <0.1 seconds
- DDPM sampling (300 steps): ~30–50 seconds
- Denormalization & logging: <1 second
- **Total: 30–60 seconds per prediction**

### On CPU
- Add 2–3x multiplier to DDPM timing
- Recommended: Use GPU for acceptable performance

### Optimization Tips
- Reduce `n_steps` to 100–150 for faster inference (slightly lower quality)
- Cache model in memory (Flask app already does this)
- Use GPU with sufficient VRAM (8GB+ recommended)

---

## 9. Sample Workflow

```bash
# Terminal 1: Start Flask server
cd VF_Value_Synthesis
source venv/bin/activate  # or conda activate vf_synthesis
python app.py

# Terminal 2 (or browser directly)
# Open: file:///path/to/VF_Value_Synthesis/index.html
# Or:   http://localhost:8000 (if using local server)
```

Then:
1. Click "Start Prediction"
2. Upload a fundus image
3. Enter optional clinical data
4. Click "Run Prediction"
5. Wait 30–60 seconds
6. View results

---

## 10. Important Reminders

⚠️ **This is a research prototype, not a diagnostic tool**

- All predictions must be reviewed by qualified ophthalmologists
- Do not make clinical decisions based on predictions alone
- Compare with recent actual VF tests when available
- Document model use in clinical records
- Maintain audit trails for institutional oversight
- Publish validation studies with appropriate disclaimers

---

## Questions?

Refer to:
- `README.md` for detailed documentation
- `inference.py` docstrings for API details
- `utils.py` for utility functions
- `feedback.html` to submit issues or suggestions

**Good luck with your research!** 🔍👀
