/**
 * VF Value Synthesis - Application JavaScript
 * 
 * Handles:
 * - Patient details form validation and submission
 * - Image preview
 * - Clinical parameter validation
 * - Feedback form handling
 * - Results display and visualization
 * - VF grid rendering
 * - Local storage of session data
 */

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Show/hide elements by ID
 */
function showElement(elementId) {
    const elem = document.getElementById(elementId);
    if (elem) elem.style.display = '';
}

function hideElement(elementId) {
    const elem = document.getElementById(elementId);
    if (elem) elem.style.display = 'none';
}

/**
 * Display error message on form field
 */
function showError(fieldId, message) {
    const errorElem = document.getElementById(fieldId + 'Error');
    if (errorElem) {
        errorElem.textContent = message;
        errorElem.classList.add('show');
    }
}

function clearError(fieldId) {
    const errorElem = document.getElementById(fieldId + 'Error');
    if (errorElem) {
        errorElem.textContent = '';
        errorElem.classList.remove('show');
    }
}

/**
 * Validate file is an image
 */
function isValidImageFile(file) {
    if (!file) return false;
    const validTypes = ['image/jpeg', 'image/png'];
    return validTypes.includes(file.type);
}

/**
 * Convert image file to base64
 */
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

/**
 * Store data in session storage
 */
function storeSessionData(key, value) {
    try {
        sessionStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
        console.warn('Session storage not available:', e);
    }
}

/**
 * Retrieve data from session storage
 */
function getSessionData(key) {
    try {
        const data = sessionStorage.getItem(key);
        return data ? JSON.parse(data) : null;
    } catch (e) {
        console.warn('Session storage not available:', e);
        return null;
    }
}

// ============================================================================
// Patient Details Form
// ============================================================================

function initPatientDetailsForm() {
    const form = document.getElementById('predictionForm');
    if (!form) return;

    const fundusImageInput = document.getElementById('fundusImage');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const predictBtn = document.getElementById('predictBtn');
    
    // ---- Image Upload Handler ----
    if (fundusImageInput) {
        fundusImageInput.addEventListener('change', async (event) => {
            const file = event.target.files[0];
            clearError('image');

            if (!file) {
                hideElement('imagePreview');
                return;
            }

            if (!isValidImageFile(file)) {
                showError('image', 'Please upload a valid image file (JPG or PNG).');
                fundusImageInput.value = '';
                hideElement('imagePreview');
                return;
            }

            // Show preview
            try {
                const base64 = await fileToBase64(file);
                previewImg.src = base64;
                showElement('imagePreview');
            } catch (e) {
                showError('image', 'Error reading image file.');
                console.error(e);
            }

            updatePredictButtonState();
        });
    }

    // ---- Clinical Field Toggles ----
    const fieldToggles = {
        useAge: 'age',
        useGender: 'gender',
        useIOP: 'iop',
        useCCT: 'cct'
    };

    Object.entries(fieldToggles).forEach(([checkboxId, inputId]) => {
        const checkbox = document.getElementById(checkboxId);
        const input = document.getElementById(inputId);
        if (checkbox && input) {
            checkbox.addEventListener('change', (event) => {
                input.disabled = !event.target.checked;
                if (!event.target.checked) {
                    input.value = '';
                    clearError(inputId);
                }
                updatePredictButtonState();
            });
        }
    });

    // ---- Diffusion Steps Slider ----
    const stepsSlider = document.getElementById('diffusionSteps');
    const stepsValue = document.getElementById('stepsValue');
    if (stepsSlider && stepsValue) {
        stepsSlider.addEventListener('input', (event) => {
            stepsValue.textContent = event.target.value;
        });
    }

    // ---- Form Submission ----
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        await handlePredictionSubmit();
    });

    // Initialize button state
    updatePredictButtonState();
}

/**
 * Update predict button state based on form validity
 */
function updatePredictButtonState() {
    const fundusImageInput = document.getElementById('fundusImage');
    const predictBtn = document.getElementById('predictBtn');
    
    const hasImage = fundusImageInput && fundusImageInput.files && fundusImageInput.files.length > 0;
    if (predictBtn) {
        predictBtn.disabled = !hasImage;
    }
}

/**
 * Validate form inputs
 */
function validatePredictionForm() {
    let isValid = true;

    // Validate image
    const fundusImageInput = document.getElementById('fundusImage');
    if (!fundusImageInput || !fundusImageInput.files || fundusImageInput.files.length === 0) {
        showError('image', 'Fundus image is required.');
        isValid = false;
    } else {
        clearError('image');
    }

    // Validate optional clinical fields if enabled
    const validations = {
        useAge: { input: 'age', min: 1, max: 120 },
        useGender: { input: 'gender' },
        useIOP: { input: 'iop', min: 0, max: 60 },
        useCCT: { input: 'cct', min: 200, max: 900 }
    };

    Object.entries(validations).forEach(([checkboxId, config]) => {
        const checkbox = document.getElementById(checkboxId);
        const input = document.getElementById(config.input);

        if (checkbox && checkbox.checked && input) {
            const value = input.value.trim();

            if (config.input === 'gender') {
                if (!value || (value !== 'M' && value !== 'F')) {
                    showError(config.input, 'Please select a gender.');
                    isValid = false;
                } else {
                    clearError(config.input);
                }
            } else {
                const numValue = parseFloat(value);
                if (!value || isNaN(numValue) || numValue < config.min || numValue > config.max) {
                    showError(config.input, `Value must be between ${config.min} and ${config.max}.`);
                    isValid = false;
                } else {
                    clearError(config.input);
                }
            }
        } else {
            clearError(config.input);
        }
    });

    return isValid;
}

/**
 * Handle prediction form submission
 */
async function handlePredictionSubmit() {
    if (!validatePredictionForm()) {
        return;
    }

    // Show loading spinner
    const loadingSpinner = document.getElementById('loadingSpinner');
    const formError = document.getElementById('formError');
    const predictBtn = document.getElementById('predictBtn');
    
    if (loadingSpinner) showElement('loadingSpinner');
    if (formError) hideElement('formError');
    if (predictBtn) predictBtn.disabled = true;

    try {
        // Collect form data
        const fundusImageInput = document.getElementById('fundusImage');
        const fundusFile = fundusImageInput.files[0];

        const formData = new FormData();
        formData.append('image', fundusFile);
        
        // Optional fields
        const fields = ['age', 'gender', 'iop', 'cct'];
        fields.forEach(field => {
            const checkboxId = 'use' + field.charAt(0).toUpperCase() + field.slice(1);
            const checkbox = document.getElementById(checkboxId);
            const input = document.getElementById(field);

            if (checkbox && checkbox.checked && input && input.value) {
                formData.append(field, input.value);
            }
        });

        const stepsSlider = document.getElementById('diffusionSteps');
        if (stepsSlider) {
            formData.append('steps', stepsSlider.value);
        }

        // IMPORTANT: This is where you would call the Python backend
        // Example (requires Flask server or similar):
        // const response = await fetch('http://localhost:5000/predict', {
        //     method: 'POST',
        //     body: formData
        // });
        
        // For now, simulate the prediction (demo mode)
        const predictionResult = await simulatePrediction(formData);

        // Store result in session
        storeSessionData('predictionResult', predictionResult);

        // Redirect to results page
        if (loadingSpinner) hideElement('loadingSpinner');
        window.location.href = 'prediction-result.html';

    } catch (error) {
        console.error('Prediction error:', error);
        if (formError) {
            formError.textContent = `Error: ${error.message}`;
            showElement('formError');
        }
        if (loadingSpinner) hideElement('loadingSpinner');
        if (predictBtn) predictBtn.disabled = false;
    }
}

/**
 * Simulate prediction (demo mode)
 * In production, this would call the Python backend
 */
async function simulatePrediction(formData) {
    return new Promise((resolve) => {
        // Simulate 3-second processing
        setTimeout(() => {
            // Generate dummy VF data (61 points)
            const vf = Array.from({ length: 61 }, () => 
                Math.random() * 35
            );

            const result = {
                timestamp: new Date().toISOString(),
                age: formData.get('age') || null,
                gender: formData.get('gender') || null,
                iop: formData.get('iop') ? parseFloat(formData.get('iop')) : null,
                cct: formData.get('cct') ? parseFloat(formData.get('cct')) : null,
                steps: formData.get('steps') || 100,
                vf: vf,
                vf_mean: vf.reduce((a, b) => a + b) / vf.length,
                vf_min: Math.min(...vf),
                vf_max: Math.max(...vf),
                vf_std: Math.sqrt(vf.reduce((sum, val) => sum + Math.pow(val - (vf.reduce((a, b) => a + b) / vf.length), 2), 0) / vf.length),
                model_output_id: 'VF_' + new Date().toISOString().replace(/[-:.]/g, '').slice(0, 15),
                imageBase64: null // Will be set below
            };

            // Read image as base64
            const imageInput = document.getElementById('fundusImage');
            if (imageInput && imageInput.files[0]) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    result.imageBase64 = e.target.result;
                    resolve(result);
                };
                reader.readAsDataURL(imageInput.files[0]);
            } else {
                resolve(result);
            }
        }, 3000);
    });
}

// ============================================================================
// Prediction Results Display
// ============================================================================

function initPredictionResults() {
    const predictionResult = getSessionData('predictionResult');
    if (!predictionResult) {
        document.body.innerHTML = '<div style="padding:40px;text-align:center;"><h2>No prediction result found.</h2><p><a href="patient-details.html">Start a new prediction</a></p></div>';
        return;
    }

    // Display input parameters
    document.getElementById('resultAge').textContent = predictionResult.age || '—';
    document.getElementById('resultGender').textContent = predictionResult.gender || '—';
    document.getElementById('resultIOP').textContent = predictionResult.iop !== null ? predictionResult.iop + ' mmHg' : '—';
    document.getElementById('resultCCT').textContent = predictionResult.cct !== null ? predictionResult.cct + ' µm' : '—';
    document.getElementById('resultSteps').textContent = predictionResult.steps || '—';
    document.getElementById('resultOutputID').textContent = predictionResult.model_output_id || '—';

    // Display input image
    if (predictionResult.imageBase64) {
        document.getElementById('inputImage').src = predictionResult.imageBase64;
    }

    // Display VF statistics
    document.getElementById('vfMean').textContent = predictionResult.vf_mean.toFixed(2);
    document.getElementById('vfMin').textContent = predictionResult.vf_min.toFixed(2);
    document.getElementById('vfMax').textContent = predictionResult.vf_max.toFixed(2);
    document.getElementById('vfStd').textContent = predictionResult.vf_std.toFixed(2);

    // Render VF grid
    renderVFGrid(predictionResult.vf);

    // Populate VF values table
    populateVFTable(predictionResult.vf);

    // Download canvas button
    const downloadBtn = document.getElementById('downloadCanvasBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadVFGridImage);
    }
}

/**
 * Render VF grid on canvas
 */
function renderVFGrid(vfValues) {
    const canvas = document.getElementById('vfCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const cellSize = 60;
    const padding = 30;
    const cols = 10;
    const rows = 10;
    
    canvas.width = cols * cellSize + padding * 2;
    canvas.height = rows * cellSize + padding * 2;

    // Map VF 61 points to grid
    const grid = new Array(rows * cols).fill(null);
    const positions = [
        [0, 3], [0, 4], [0, 5], [0, 6],
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7],
        [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8],
        [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8],
        [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
        [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8],
        [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8],
        [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7],
        [8, 3], [8, 4], [8, 5], [8, 6],
    ];

    positions.forEach((pos, idx) => {
        if (idx < vfValues.length) {
            const cellIdx = pos[0] * cols + pos[1];
            grid[cellIdx] = vfValues[idx];
        }
    });

    // Draw grid
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const x = padding + c * cellSize;
            const y = padding + r * cellSize;
            const value = grid[r * cols + c];

            if (value !== null) {
                // Color gradient: yellow (low) to blue (high)
                const normalized = Math.min(value / 35, 1.0);
                const hue = 240 * (1 - normalized); // Blue to Yellow
                ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
                ctx.fillRect(x, y, cellSize - 2, cellSize - 2);

                // Draw border
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 1;
                ctx.strokeRect(x, y, cellSize - 2, cellSize - 2);

                // Draw text
                ctx.fillStyle = value > 17.5 ? '#000' : '#fff';
                ctx.font = 'bold 12px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(value.toFixed(1), x + cellSize / 2, y + cellSize / 2);
            } else {
                ctx.fillStyle = '#1a1a2e';
                ctx.fillRect(x, y, cellSize - 2, cellSize - 2);
            }
        }
    }

    // Draw title
    ctx.fillStyle = '#333';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Visual Field Grid (61 Points, Humphrey 24-2)', canvas.width / 2, 15);
}

/**
 * Populate VF values table
 */
function populateVFTable(vfValues) {
    const tbody = document.getElementById('vfTableBody');
    if (!tbody) return;

    tbody.innerHTML = '';
    vfValues.forEach((value, index) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${index + 1}</td>
            <td>${value.toFixed(2)}</td>
        `;
        tbody.appendChild(tr);
    });
}

/**
 * Download VF grid image
 */
function downloadVFGridImage() {
    const canvas = document.getElementById('vfCanvas');
    if (!canvas) return;

    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = `vf_grid_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.png`;
    link.click();
}

// ============================================================================
// Feedback Form
// ============================================================================

function initFeedbackForm() {
    const form = document.getElementById('feedbackForm');
    if (!form) return;

    const messageInput = document.getElementById('message');
    const charCount = document.getElementById('charCount');

    // Character counter
    if (messageInput && charCount) {
        messageInput.addEventListener('input', (e) => {
            charCount.textContent = e.target.value.length;
            if (e.target.value.length > 2000) {
                messageInput.value = messageInput.value.slice(0, 2000);
                charCount.textContent = '2000';
            }
        });
    }

    // Form submission
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        await handleFeedbackSubmit();
    });
}

/**
 * Handle feedback form submission
 */
async function handleFeedbackSubmit() {
    const feedbackType = document.getElementById('feedbackType').value;
    const name = document.getElementById('name').value.trim();
    const email = document.getElementById('email').value.trim();
    const message = document.getElementById('message').value.trim();
    const rating = document.querySelector('input[name="rating"]:checked');
    const consent = document.getElementById('consent').checked;

    // Validate
    let hasError = false;
    if (!feedbackType) {
        showError('type', 'Please select a feedback type.');
        hasError = true;
    } else {
        clearError('type');
    }

    if (!message) {
        showError('message', 'Please enter your feedback.');
        hasError = true;
    } else {
        clearError('message');
    }

    if (hasError) return;

    // Prepare feedback record
    const feedback = {
        timestamp: new Date().toISOString(),
        type: feedbackType,
        name: name || null,
        email: email || null,
        message: message,
        rating: rating ? rating.value : null,
        consent: consent
    };

    try {
        // Log feedback locally (would normally send to server)
        console.log('Feedback submitted:', feedback);
        
        // In production, you would send this to a backend:
        // await fetch('/api/feedback', {
        //     method: 'POST',
        //     headers: { 'Content-Type': 'application/json' },
        //     body: JSON.stringify(feedback)
        // });

        // Show success message
        document.getElementById('feedbackForm').style.display = 'none';
        showElement('feedbackSuccess');

    } catch (error) {
        console.error('Feedback submission error:', error);
        const feedbackError = document.getElementById('feedbackError');
        if (feedbackError) {
            feedbackError.textContent = 'Error submitting feedback. Please try again.';
            showElement('feedbackError');
        }
    }
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Check which page we're on
    const pathname = window.location.pathname;

    if (pathname.includes('patient-details')) {
        initPatientDetailsForm();
    } else if (pathname.includes('prediction-result')) {
        initPredictionResults();
    } else if (pathname.includes('feedback')) {
        initFeedbackForm();
    }
});
