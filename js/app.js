/* ==========================================
   VF VALUE SYNTHESIS - APPLICATION JS
   ========================================== */

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

/**
 * Show/hide elements
 */
function show(el) {
  if (el) el.style.display = '';
}

function hide(el) {
  if (el) el.style.display = 'none';
}

/**
 * Clear error message
 */
function clearError(errorEl) {
  if (errorEl) errorEl.textContent = '';
}

/**
 * Show error message
 */
function showError(errorEl, message) {
  if (errorEl) errorEl.textContent = message;
}

/**
 * Validate email
 */
function isValidEmail(email) {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
}

/**
 * Format date
 */
function formatDate(date) {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

// ==========================================
// PATIENT DETAILS PAGE (patient-details.html)
// ==========================================

document.addEventListener('DOMContentLoaded', function () {
  // Initialize patient details form if on patient-details page
  if (document.getElementById('predictionForm')) {
    initPatientDetailsPage();
  }

  // Initialize feedback form if on feedback page
  if (document.getElementById('feedbackForm')) {
    initFeedbackPage();
  }

  // Initialize results page if on prediction-result page
  if (document.getElementById('vfTable')) {
    initResultsPage();
  }
});

/**
 * Initialize Patient Details Page
 */
function initPatientDetailsPage() {
  const form = document.getElementById('predictionForm');
  const fundusImageInput = document.getElementById('fundusImage');
  const imagePreview = document.getElementById('imagePreview');
  const previewImg = document.getElementById('previewImg');
  const predictBtn = document.getElementById('predictBtn');
  const loadingSpinner = document.getElementById('loadingSpinner');
  const loadingText = document.getElementById('loadingText');
  const formError = document.getElementById('formError');

  // Clinical parameter checkboxes
  const useAgeCheckbox = document.getElementById('useAge');
  const useGenderCheckbox = document.getElementById('useGender');
  const useIOPCheckbox = document.getElementById('useIOP');
  const useCCTCheckbox = document.getElementById('useCCT');

  const ageInput = document.getElementById('age');
  const genderSelect = document.getElementById('gender');
  const iopInput = document.getElementById('iop');
  const cctInput = document.getElementById('cct');

  const diffusionStepsInput = document.getElementById('diffusionSteps');
  const stepsValueDisplay = document.getElementById('stepsValue');

  // Handle image file selection
  fundusImageInput.addEventListener('change', function (e) {
    const file = e.target.files[0];
    clearError(document.getElementById('imageError'));

    if (!file) {
      hide(imagePreview);
      predictBtn.disabled = true;
      return;
    }

    // Validate file type
    if (!file.type.match('image/(jpeg|png)')) {
      showError(document.getElementById('imageError'), 'Please select a valid JPG or PNG image.');
      predictBtn.disabled = true;
      hide(imagePreview);
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      showError(document.getElementById('imageError'), 'File size must be less than 10MB.');
      predictBtn.disabled = true;
      hide(imagePreview);
      return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = function (event) {
      previewImg.src = event.target.result;
      show(imagePreview);
      predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  });

  // Toggle clinical parameter inputs
  useAgeCheckbox.addEventListener('change', function () {
    ageInput.disabled = !this.checked;
    if (!this.checked) {
      ageInput.value = '';
      clearError(document.getElementById('ageError'));
    }
  });

  useGenderCheckbox.addEventListener('change', function () {
    genderSelect.disabled = !this.checked;
    if (!this.checked) {
      genderSelect.value = '';
      clearError(document.getElementById('genderError'));
    }
  });

  useIOPCheckbox.addEventListener('change', function () {
    iopInput.disabled = !this.checked;
    if (!this.checked) {
      iopInput.value = '';
      clearError(document.getElementById('iopError'));
    }
  });

  useCCTCheckbox.addEventListener('change', function () {
    cctInput.disabled = !this.checked;
    if (!this.checked) {
      cctInput.value = '';
      clearError(document.getElementById('cctError'));
    }
  });

  // Update slider display value
  diffusionStepsInput.addEventListener('input', function () {
    stepsValueDisplay.textContent = this.value;
  });

  // Handle form submission
  form.addEventListener('submit', async function (e) {
    e.preventDefault();
    clearError(formError);

    // Validate required fields
    const imageFile = fundusImageInput.files[0];
    if (!imageFile) {
      showError(document.getElementById('imageError'), 'Please select a fundus image.');
      return;
    }

    // Validate optional clinical parameters if provided
    let validationErrors = false;

    if (useAgeCheckbox.checked) {
      const age = parseFloat(ageInput.value);
      if (!ageInput.value || age < 1 || age > 120) {
        showError(document.getElementById('ageError'), 'Age must be between 1 and 120.');
        validationErrors = true;
      }
    }

    if (useIOPCheckbox.checked) {
      const iop = parseFloat(iopInput.value);
      if (!iopInput.value || iop < 0 || iop > 60) {
        showError(document.getElementById('iopError'), 'IOP must be between 0 and 60 mmHg.');
        validationErrors = true;
      }
    }

    if (useCCTCheckbox.checked) {
      const cct = parseFloat(cctInput.value);
      if (!cctInput.value || cct < 200 || cct > 900) {
        showError(document.getElementById('cctError'), 'CCT must be between 200 and 900 µm.');
        validationErrors = true;
      }
    }

    if (validationErrors) return;

    // Prepare form data
    const formData = new FormData();
    formData.append('fundusImage', imageFile);
    formData.append('useAge', useAgeCheckbox.checked);
    if (useAgeCheckbox.checked) formData.append('age', ageInput.value);

    formData.append('useGender', useGenderCheckbox.checked);
    if (useGenderCheckbox.checked) formData.append('gender', genderSelect.value);

    formData.append('useIOP', useIOPCheckbox.checked);
    if (useIOPCheckbox.checked) formData.append('iop', iopInput.value);

    formData.append('useCCT', useCCTCheckbox.checked);
    if (useCCTCheckbox.checked) formData.append('cct', cctInput.value);

    formData.append('diffusionSteps', diffusionStepsInput.value);

    // Show loading spinner
    show(loadingSpinner);
    form.style.opacity = '0.5';
    form.style.pointerEvents = 'none';

    try {
      // Simulate API call to backend prediction service
      // In a real implementation, this would call your Flask backend
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // For demonstration: store form data in sessionStorage
      sessionStorage.setItem(
        'predictionData',
        JSON.stringify({
          age: useAgeCheckbox.checked ? ageInput.value : null,
          gender: useGenderCheckbox.checked ? genderSelect.value : null,
          iop: useIOPCheckbox.checked ? iopInput.value : null,
          cct: useCCTCheckbox.checked ? cctInput.value : null,
          diffusionSteps: diffusionStepsInput.value,
          imageBase64: previewImg.src,
          timestamp: new Date().toISOString(),
        })
      );

      // Generate mock VF results
      const mockVFValues = generateMockVFValues();
      sessionStorage.setItem('vfResults', JSON.stringify(mockVFValues));

      // Redirect to results page
      setTimeout(() => {
        window.location.href = 'prediction-result.html';
      }, 500);
    } catch (error) {
      showError(formError, 'Error processing prediction. Please try again.');
      console.error('Prediction error:', error);
      hide(loadingSpinner);
      form.style.opacity = '1';
      form.style.pointerEvents = 'auto';
    }
  });
}

/**
 * Generate mock VF values (simulating API response)
 */
function generateMockVFValues() {
  // 61-point Humphrey 24-2 pattern
  const values = [];
  for (let i = 0; i < 61; i++) {
    // Generate realistic VF values (0-35 dB range, with some variation)
    const baseValue = 28 + (Math.random() - 0.5) * 8;
    const value = Math.max(0, Math.min(35, baseValue));
    values.push(parseFloat(value.toFixed(1)));
  }
  return values;
}

/**
 * Calculate VF statistics
 */
function calculateVFStats(values) {
  const mean = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2);
  const min = Math.min(...values).toFixed(2);
  const max = Math.max(...values).toFixed(2);

  const variance =
    values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  const std = Math.sqrt(variance).toFixed(2);

  return { mean, min, max, std };
}

// ==========================================
// FEEDBACK PAGE (feedback.html)
// ==========================================

/**
 * Initialize Feedback Page
 */
function initFeedbackPage() {
  const form = document.getElementById('feedbackForm');
  const messageTextarea = document.getElementById('message');
  const charCount = document.getElementById('charCount');
  const submitBtn = document.getElementById('submitFeedback');
  const successMessage = document.getElementById('feedbackSuccess');
  const errorMessage = document.getElementById('feedbackError');

  // Update character count
  messageTextarea.addEventListener('input', function () {
    charCount.textContent = this.value.length;
    if (this.value.length > 2000) {
      this.value = this.value.substring(0, 2000);
      charCount.textContent = '2000';
    }
  });

  // Handle form submission
  form.addEventListener('submit', async function (e) {
    e.preventDefault();
    hide(errorMessage);

    // Validate required fields
    const feedbackType = document.getElementById('feedbackType').value;
    const message = messageTextarea.value.trim();

    let hasErrors = false;

    clearError(document.getElementById('typeError'));
    clearError(document.getElementById('messageError'));

    if (!feedbackType) {
      showError(document.getElementById('typeError'), 'Please select a feedback type.');
      hasErrors = true;
    }

    if (!message) {
      showError(document.getElementById('messageError'), 'Please provide your feedback.');
      hasErrors = true;
    }

    if (hasErrors) return;

    // Validate email if provided
    const email = document.getElementById('email').value.trim();
    if (email && !isValidEmail(email)) {
      showError(document.getElementById('emailError'), 'Please enter a valid email address.');
      return;
    }

    // Prepare feedback data
    const feedbackData = {
      type: feedbackType,
      name: document.getElementById('name').value.trim() || null,
      email: email || null,
      message: message,
      rating: document.querySelector('input[name="rating"]:checked')?.value || null,
      consent: document.getElementById('consent').checked,
      timestamp: new Date().toISOString(),
    };

    // Save feedback to local storage (simulating backend storage)
    try {
      submitBtn.disabled = true;
      submitBtn.textContent = 'Submitting...';

      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Save to localStorage
      const feedbackHistory = JSON.parse(localStorage.getItem('feedbackHistory') || '[]');
      feedbackHistory.push(feedbackData);
      localStorage.setItem('feedbackHistory', JSON.stringify(feedbackHistory));

      // Show success message
      hide(form);
      show(successMessage);

      // Log for demonstration
      console.log('Feedback submitted:', feedbackData);
    } catch (error) {
      showError(errorMessage, 'Error submitting feedback. Please try again.');
      console.error('Feedback submission error:', error);
      submitBtn.disabled = false;
      submitBtn.textContent = 'Submit Feedback';
    }
  });
}

// ==========================================
// RESULTS PAGE (prediction-result.html)
// ==========================================

/**
 * Initialize Results Page
 */
function initResultsPage() {
  // Retrieve data from sessionStorage
  const predictionData = JSON.parse(sessionStorage.getItem('predictionData') || '{}');
  const vfValues = JSON.parse(sessionStorage.getItem('vfResults') || '[]');

  if (!vfValues || vfValues.length === 0) {
    console.error('No VF results found');
    return;
  }

  // Display input parameters
  displayInputParameters(predictionData);

  // Calculate and display statistics
  displayVFStatistics(vfValues);

  // Render VF grid visualization
  renderVFGrid(vfValues);

  // Populate VF table
  populateVFTable(vfValues);

  // Handle download button
  const downloadBtn = document.getElementById('downloadCanvasBtn');
  if (downloadBtn) {
    downloadBtn.addEventListener('click', downloadCanvasAsImage);
  }
}

/**
 * Display input parameters
 */
function displayInputParameters(data) {
  const resultAge = document.getElementById('resultAge');
  const resultGender = document.getElementById('resultGender');
  const resultIOP = document.getElementById('resultIOP');
  const resultCCT = document.getElementById('resultCCT');
  const resultSteps = document.getElementById('resultSteps');
  const resultOutputID = document.getElementById('resultOutputID');
  const inputImage = document.getElementById('inputImage');

  if (resultAge) resultAge.textContent = data.age ? `${data.age} years` : '—';
  if (resultGender) resultGender.textContent = data.gender
    ? data.gender === 'M'
      ? 'Male'
      : 'Female'
    : '—';
  if (resultIOP) resultIOP.textContent = data.iop ? `${data.iop} mmHg` : '—';
  if (resultCCT) resultCCT.textContent = data.cct ? `${data.cct} µm` : '—';
  if (resultSteps) resultSteps.textContent = data.diffusionSteps
    ? `${data.diffusionSteps} steps`
    : '—';

  // Generate output ID
  const outputID = `OUT-${Date.now()}-${Math.random().toString(36).substr(2, 9).toUpperCase()}`;
  if (resultOutputID) resultOutputID.textContent = outputID;

  // Display input image
  if (inputImage && data.imageBase64) {
    inputImage.src = data.imageBase64;
  }
}

/**
 * Display VF statistics
 */
function displayVFStatistics(values) {
  const stats = calculateVFStats(values);

  const vfMean = document.getElementById('vfMean');
  const vfMin = document.getElementById('vfMin');
  const vfMax = document.getElementById('vfMax');
  const vfStd = document.getElementById('vfStd');

  if (vfMean) vfMean.textContent = stats.mean;
  if (vfMin) vfMin.textContent = stats.min;
  if (vfMax) vfMax.textContent = stats.max;
  if (vfStd) vfStd.textContent = stats.std;
}

/**
 * Render VF grid on canvas
 */
function renderVFGrid(values) {
  const canvas = document.getElementById('vfCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const cellSize = 60;
  const padding = 40;
  const cols = 9;
  const rows = 9;

  // Set canvas size
  canvas.width = cols * cellSize + 2 * padding;
  canvas.height = rows * cellSize + 2 * padding;

  // Fill background
  ctx.fillStyle = '#071428';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw border
  ctx.strokeStyle = '#3a6fd8';
  ctx.lineWidth = 2;
  ctx.strokeRect(padding, padding, cols * cellSize, rows * cellSize);

  // 61-point Humphrey 24-2 layout (indices)
  const vfLayout = [
    [-1, -1, -1, -1, 0, 1, -1, -1, -1],
    [-1, -1, -1, 2, 3, 4, 5, -1, -1],
    [-1, -1, 6, 7, 8, 9, 10, 11, -1],
    [-1, 12, 13, 14, 15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24, 25, 26, 27, 28],
    [29, 30, 31, 32, 33, 34, 35, 36, 37],
    [-1, 38, 39, 40, 41, 42, 43, 44, 45],
    [-1, -1, 46, 47, 48, 49, 50, 51, -1],
    [-1, -1, -1, 52, 53, 54, 55, -1, -1],
  ];

  // Color scale function
  function getColor(value) {
    // Green (good) to red (poor) gradient
    const normalized = Math.max(0, Math.min(35, value)) / 35;
    if (normalized > 0.66) {
      return '#10b981'; // Green
    } else if (normalized > 0.33) {
      return '#f59e0b'; // Orange
    } else {
      return '#ef4444'; // Red
    }
  }

  // Draw grid cells
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const valueIndex = vfLayout[r][c];
      const x = padding + c * cellSize + cellSize / 2;
      const y = padding + r * cellSize + cellSize / 2;

      if (valueIndex >= 0) {
        const value = values[valueIndex] || 0;
        const color = getColor(value);

        // Draw circle
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, cellSize / 2.5, 0, 2 * Math.PI);
        ctx.fill();

        // Draw value
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 14px Barlow';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(value.toFixed(1), x, y);
      }
    }
  }

  // Draw legend
  const legendY = canvas.height - 30;
  const legendItems = [
    { label: '>23 dB', color: '#10b981' },
    { label: '12-23 dB', color: '#f59e0b' },
    { label: '<12 dB', color: '#ef4444' },
  ];

  let legendX = padding;
  ctx.font = '12px Barlow';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#a0aec0';

  legendItems.forEach((item, index) => {
    ctx.fillStyle = item.color;
    ctx.fillRect(legendX, legendY, 12, 12);

    ctx.fillStyle = '#a0aec0';
    ctx.fillText(item.label, legendX + 16, legendY + 6);

    legendX += 120;
  });
}

/**
 * Populate VF values table
 */
function populateVFTable(values) {
  const tableBody = document.getElementById('vfTableBody');
  if (!tableBody) return;

  tableBody.innerHTML = '';

  values.forEach((value, index) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${index + 1}</td>
      <td>${value.toFixed(2)}</td>
    `;
    tableBody.appendChild(row);
  });
}

/**
 * Download canvas as image
 */
function downloadCanvasAsImage() {
  const canvas = document.getElementById('vfCanvas');
  if (!canvas) return;

  const link = document.createElement('a');
  link.href = canvas.toDataURL('image/png');
  link.download = `vf-grid-${new Date().toISOString().split('T')[0]}.png`;
  link.click();
}

// ==========================================
// EXPORT FOR TESTING
// ==========================================
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    generateMockVFValues,
    calculateVFStats,
    isValidEmail,
    formatDate,
  };
}
