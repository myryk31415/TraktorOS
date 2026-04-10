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
const processedCanvas = document.getElementById('processedCanvas');
const processedPlaceholder = document.getElementById('processedImagePlaceholder');
const processedSpinner = document.getElementById('processedSpinner');
const detectionInfo = document.getElementById('detectionInfo');
const canvas = processedCanvas;
const ctx = canvas.getContext('2d');

let selectedImage = null;
let selectedImageDataUrl = null;
let previewObjectUrl = null;

const bedrockBtn = document.getElementById('bedrockBtn');
const localLlmBtn = document.getElementById('localLlmBtn');
const bedrockSkeleton = document.getElementById('bedrockSkeleton');
const bedrockEmpty = document.getElementById('bedrockEmpty');
const bedrockResults = document.getElementById('bedrockResults');
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
    bedrockSkeleton.classList.remove('d-none');

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: selectedImageDataUrl.split(',')[1] })
        });

        if (!response.ok) {
            throw new Error(`Request failed with status ${response.status}`);
        }

        const result = await response.json();
        if (result.error) {
            throw new Error(result.error);
        }

        const analysisData = {
            imageQuality: result.image_quality || {},
            soilAssessment: result.soil_assessment || {},
            obstacles: Array.isArray(result.obstacles) ? result.obstacles : [],
            summary: result.summary || ''
        };

        bedrockDetections.innerHTML = buildAnalysisCards(analysisData);

        bedrockSkeleton.classList.add('d-none');
        bedrockResults.classList.remove('d-none');
    } catch (error) {
        console.error('Analysis error:', error);
        bedrockSkeleton.classList.add('d-none');
        bedrockEmpty.querySelector('p').textContent = 'Analysis failed';
        bedrockEmpty.querySelector('.text-body-secondary').textContent = String(error.message || 'Please try again.');
        bedrockEmpty.classList.remove('d-none');
    } finally {
        bedrockBtn.disabled = !selectedImage;
        localLlmBtn.disabled = !selectedImage;
    }
}

function buildAnalysisCards(analysisData) {
    const iconPath = 'icons/image-outline.svg';
    const quality = analysisData.imageQuality || {};
    const soil = analysisData.soilAssessment || {};
    const obstacles = Array.isArray(analysisData.obstacles) ? analysisData.obstacles : [];
    const summaryText = analysisData.summary || '';

    const qualityIssues = Array.isArray(quality.issues) ? quality.issues.filter(Boolean) : [];
    const qualityDescription = qualityIssues.length
        ? qualityIssues.join(', ')
        : quality.sufficient_for_human_detection === true
            ? 'Image quality is sufficient for human detection.'
            : 'No image quality details reported.';

    const soilConcerns = Array.isArray(soil.concerns) ? soil.concerns.filter(Boolean) : [];
    const soilBase = [soil.condition, soil.traversability].filter(Boolean).join(' / ');
    const soilDescription = [soilBase, soilConcerns.join(', ')].filter(Boolean).join(' - ') || 'No soil assessment reported.';

    const obstacleDescription = obstacles.length
        ? obstacles.map((item) => {
            const type = item.type || 'obstacle';
            const description = item.description || 'No description';
            return `${type}: ${description}`;
        }).join(' | ')
        : 'No obstacles detected.';

    const cards = [
        {
            key: 'image-quality',
            icon: 'icons/image-outline.svg',
            title: 'Image quality',
            status: qualityStatus(quality),
            description: qualityDescription
        },
        {
            key: 'soil-assessment',
            icon: 'icons/golf-outline.svg',
            title: 'Soil assessment',
            status: traversabilityStatus(soil.traversability),
            description: soilDescription
        },
        {
            key: 'obstacles',
            icon: 'icons/cube-outline.svg',
            title: 'Obstacles',
            status: obstacleStatus(obstacles),
            description: obstacleDescription
        },
        {
            key: 'summary',
            icon: 'icons/book-outline.svg',
            title: 'Summary',
            status: obstacleStatus(obstacles),
            description: summaryText || 'No summary reported.'
        }
    ];

    return cards.map((card) => {
        return `
            <div class="analysis-card" data-card="${card.key}">
                <div class="analysis-card-header">
                    <div class="analysis-card-title-wrap">
                        <span class="analysis-card-left-icon" aria-hidden="true">
                            <img src="${card.icon}" alt="" aria-hidden="true">
                        </span>
                        <h3 class="analysis-card-title">${card.title}</h3>
                    </div>
                    <span class="analysis-status-icon ${statusClass(card.status)}" aria-hidden="true">
                        ${statusImage(card.status) ? `<img src="${statusImage(card.status)}" alt="" aria-hidden="true" style="transform: scale(1.3);">` : ''}
                    </span>
                </div>
                <p class="analysis-card-description">${escapeHtml(card.description)}</p>
            </div>
        `;
    }).join('');
}

function qualityStatus(imageQuality) {
    if (imageQuality.sufficient_for_human_detection === true) return 'success';
    if (imageQuality.sufficient_for_human_detection === false) return 'danger';
    return 'info';
}

function traversabilityStatus(traversability) {
    const value = String(traversability || '').toLowerCase();
    if (value === 'good') return 'success';
    if (value === 'moderate') return 'warning';
    if (value === 'poor' || value === 'impassable') return 'danger';
    return 'info';
}

function obstacleStatus(obstacles) {
    if (!obstacles.length) return 'success';

    const severities = obstacles.map((item) => String(item.severity || '').toLowerCase());
    if (severities.includes('critical')) return 'danger';
    if (severities.includes('warning')) return 'warning';
    return 'info';
}

function statusClass(status) {
    const map = {
        success: 'status-success',
        warning: 'status-warning',
        danger: 'status-danger',
        info: 'status-info'
    };
    return map[status] || 'status-info';
}

function statusImage(status) {
    const map = {
        success: 'icons/checkmark-outline.svg',
        warning: 'icons/alert-outline.svg',
        danger: 'icons/close-outline.svg',
        info: ''
    };
    return map[status] || '';
 }


function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
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
}

function resetProcessedState() {
    processedCanvas.classList.add('d-none');
    processedPlaceholder.classList.remove('is-hidden');
    processedSpinner.classList.add('is-hidden');
    detectionInfo.innerHTML = '';
    ctx.clearRect(0, 0, processedCanvas.width, processedCanvas.height);
}

function setProcessingState(isProcessing) {
    if (isProcessing) {
        processedCanvas.classList.add('d-none');
        processedPlaceholder.classList.add('is-hidden');
        processedSpinner.classList.remove('is-hidden');
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
