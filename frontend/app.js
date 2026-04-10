const ENDPOINTS = {
    local: 'http://localhost:5000/detect',
    sagemaker: 'YOUR_API_GATEWAY_URL',
    bedrock: 'http://localhost:5000/detect-bedrock',
    localLlm: 'http://localhost:5000/detect-local-llm'
};

const imageInput = document.getElementById('imageInput');
const uploadBtn = document.getElementById('uploadBtn');
const selectedImagePreview = document.getElementById('selectedImagePreview');
const selectedImageEmptyState = document.getElementById('selectedImageEmptyState');
const selectedImageStatus = document.getElementById('selectedImageStatus');
const processedCanvas = document.getElementById('processedCanvas');
const processedPlaceholder = document.getElementById('processedImagePlaceholder');
const processedSpinner = document.getElementById('processedSpinner');
const processedImageStatus = document.getElementById('processedImageStatus');
const detectionInfo = document.getElementById('detectionInfo');
const canvas = processedCanvas;
const ctx = canvas.getContext('2d');

let selectedImage = null;
let selectedImageDataUrl = null;
let previewObjectUrl = null;

const bedrockBtn = document.getElementById('bedrockBtn');
const localLlmBtn = document.getElementById('localLlmBtn');
const bedrockStatus = document.getElementById('bedrockStatus');
const bedrockSpinner = document.getElementById('bedrockSpinner');
const bedrockEmpty = document.getElementById('bedrockEmpty');
const bedrockResults = document.getElementById('bedrockResults');
const bedrockSummary = document.getElementById('bedrockSummary');
const bedrockDetections = document.getElementById('bedrockDetections');

uploadBtn.disabled = true;

imageInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];

    if (!file) {
        selectedImage = null;
        selectedImageDataUrl = null;
        uploadBtn.disabled = true;
        bedrockBtn.disabled = true;
        localLlmBtn.disabled = true;
        resetSelectedPreview();
        resetProcessedState();
        return;
    }

    selectedImage = file;
    uploadBtn.disabled = false;
    bedrockBtn.disabled = false;
    localLlmBtn.disabled = false;

    if (previewObjectUrl) {
        URL.revokeObjectURL(previewObjectUrl);
    }

    previewObjectUrl = URL.createObjectURL(file);
    selectedImagePreview.src = previewObjectUrl;
    selectedImagePreview.classList.remove('d-none');
    selectedImageEmptyState.classList.add('is-hidden');
    selectedImageStatus.textContent = 'Ready';

    selectedImageDataUrl = await fileToBase64(file);
    resetProcessedState();
});

uploadBtn.addEventListener('click', async () => {
    if (!selectedImage || !selectedImageDataUrl) return;

    uploadBtn.disabled = true;
    imageInput.disabled = true;
    setProcessingState(true);

    try {
        const mode = document.querySelector('input[name="mode"]:checked').value;
        const response = await fetch(ENDPOINTS[mode], {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: selectedImageDataUrl.split(',')[1]
            })
        });

        const result = await response.json();

        displayResults(selectedImageDataUrl, result.detections || []);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error processing image. Please try again.');
        setProcessingState(false);
        processedImageStatus.textContent = 'Processing failed';
    } finally {
        imageInput.disabled = false;
        uploadBtn.disabled = !selectedImage;
    }
});

async function runAnalysis(endpoint) {
    if (!selectedImageDataUrl) return;

    bedrockBtn.disabled = true;
    localLlmBtn.disabled = true;
    bedrockEmpty.classList.add('d-none');
    bedrockResults.classList.add('d-none');
    bedrockSpinner.classList.remove('d-none');
    bedrockStatus.textContent = 'Processing';

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: selectedImageDataUrl.split(',')[1] })
        });
        const result = await response.json();

        bedrockSummary.textContent = result.summary || '';

        let html = '';

        // Image quality
        const q = result.image_quality;
        if (q) {
            const qBadge = q.sufficient_for_human_detection ? 'text-bg-success' : 'text-bg-danger';
            const qLabel = q.sufficient_for_human_detection ? 'Sufficient' : 'Insufficient';
            html += `<div class="detection-item"><span class="detection-label">Human detection quality</span><span class="badge ${qBadge}">${qLabel}</span></div>`;
            if (q.issues && q.issues.length) html += `<div class="detection-item"><span class="text-body-secondary">${q.issues.join(', ')}</span></div>`;
        }

        // Soil
        const s = result.soil_assessment;
        if (s) {
            const tColors = {good:'text-bg-success',moderate:'text-bg-warning',poor:'text-bg-danger',impassable:'text-bg-danger'};
            html += `<div class="detection-item"><span class="detection-label">Soil: ${s.condition}</span><span class="badge ${tColors[s.traversability] || 'text-bg-secondary'}">${s.traversability}</span></div>`;
            if (s.concerns && s.concerns.length) html += `<div class="detection-item"><span class="text-body-secondary">${s.concerns.join(', ')}</span></div>`;
        }

        // Obstacles
        if (result.obstacles && result.obstacles.length) {
            const sevColors = {critical:'text-bg-danger',warning:'text-bg-warning',info:'text-bg-info'};
            result.obstacles.forEach(o => {
                html += `<div class="detection-item"><span class="detection-label">${o.type}</span><span class="badge ${sevColors[o.severity] || 'text-bg-secondary'}">${o.severity}</span></div>`;
                html += `<div class="detection-item"><span class="text-body-secondary">${o.description}</span></div>`;
            });
        } else {
            html += '<div class="detection-item"><span class="text-body-secondary">No obstacles detected</span></div>';
        }

        bedrockDetections.innerHTML = html;

        bedrockSpinner.classList.add('d-none');
        bedrockResults.classList.remove('d-none');
        bedrockStatus.textContent = `${(result.obstacles || []).length} obstacles`;
    } catch (error) {
        console.error('Analysis error:', error);
        bedrockSpinner.classList.add('d-none');
        bedrockEmpty.classList.remove('d-none');
        bedrockStatus.textContent = 'Error';
    } finally {
        bedrockBtn.disabled = !selectedImage;
        localLlmBtn.disabled = !selectedImage;
    }
}

bedrockBtn.addEventListener('click', () => runAnalysis(ENDPOINTS.bedrock));
localLlmBtn.addEventListener('click', () => runAnalysis(ENDPOINTS.localLlm));

function resetSelectedPreview() {
    if (previewObjectUrl) {
        URL.revokeObjectURL(previewObjectUrl);
        previewObjectUrl = null;
    }

    selectedImagePreview.removeAttribute('src');
    selectedImagePreview.classList.add('d-none');
    selectedImageEmptyState.classList.remove('is-hidden');
    selectedImageStatus.textContent = 'Waiting for upload';
}

function resetProcessedState() {
    processedCanvas.classList.add('d-none');
    processedPlaceholder.classList.remove('is-hidden');
    processedSpinner.classList.add('is-hidden');
    processedImageStatus.textContent = 'Waiting for processing';
    detectionInfo.innerHTML = '';
    ctx.clearRect(0, 0, processedCanvas.width, processedCanvas.height);
}

function setProcessingState(isProcessing) {
    if (isProcessing) {
        processedCanvas.classList.add('d-none');
        processedPlaceholder.classList.add('is-hidden');
        processedSpinner.classList.remove('is-hidden');
        processedImageStatus.textContent = 'Processing';
        detectionInfo.innerHTML = '';
    } else {
        processedSpinner.classList.add('is-hidden');
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function displayResults(imageData, detections) {
    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        
        // Draw image
        ctx.drawImage(img, 0, 0);
        
        // Draw bounding boxes
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 3;
        ctx.font = '16px Arial';
        ctx.fillStyle = '#00ff00';
        
        detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;
            const width = x2 - x1;
            const height = y2 - y1;
            
            // Draw box
            ctx.strokeRect(x1, y1, width, height);
            
            // Draw label
            const label = `Human ${(detection.confidence * 100).toFixed(1)}%`;
            ctx.fillText(label, x1, y1 - 5);
        });

        processedPlaceholder.classList.add('is-hidden');
        processedSpinner.classList.add('is-hidden');
        processedCanvas.classList.remove('d-none');
        processedImageStatus.textContent = `Complete · ${detections.length} detected`;
        detectionInfo.innerHTML = detections.length
            ? detections.map((detection, index) => {
                return `
                    <div class="detection-item">
                        <span class="detection-label">Human ${index + 1}</span>
                        <span class="detection-confidence">${(detection.confidence * 100).toFixed(1)}%</span>
                    </div>
                `;
            }).join('')
            : '<div class="text-body-secondary">No humans detected.</div>';
    };
    img.src = imageData;
}
