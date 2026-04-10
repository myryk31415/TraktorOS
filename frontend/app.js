const ENDPOINTS = {
    local: 'http://localhost:5000/detect',
    sagemaker: 'YOUR_API_GATEWAY_URL',
    bedrock: 'http://localhost:5000/detect-bedrock',
    quality: 'http://localhost:5000/quality'
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
let lastDetections = [];
let lastImageData = null;

const tractorWidthSlider = document.getElementById('tractorWidth');
const tractorWidthLabel = document.getElementById('tractorWidthLabel');
const horizonSlider = document.getElementById('horizonLine');
const horizonLabel = document.getElementById('horizonLineLabel');

tractorWidthSlider.addEventListener('input', () => {
    tractorWidthLabel.textContent = tractorWidthSlider.value + '%';
    if (lastImageData) displayResults(lastImageData, lastDetections);
});
horizonSlider.addEventListener('input', () => {
    horizonLabel.textContent = horizonSlider.value + '%';
    if (lastImageData) displayResults(lastImageData, lastDetections);
});

const bedrockBtn = document.getElementById('bedrockBtn');
const bedrockSkeleton = document.getElementById('bedrockSkeleton');
const bedrockEmpty = document.getElementById('bedrockEmpty');
const bedrockResults = document.getElementById('bedrockResults');
const bedrockDetections = document.getElementById('bedrockDetections');
const activeModeBadge = document.getElementById('activeModeBadge');
const selectedFileMeta = document.getElementById('selectedFileMeta');
const detectionCountBadge = document.getElementById('detectionCountBadge');
const analysisStateBadge = document.getElementById('analysisStateBadge');
const statusHint = document.getElementById('statusHint');
const detectionSummary = document.getElementById('detectionSummary');

uploadBtn.disabled = true;
setInitialUiState();

document.querySelectorAll('input[name="mode"]').forEach((radio) => {
    radio.addEventListener('change', () => {
        updateModeBadge();
    });
});

imageInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];

    if (!file) {
        selectedImage = null;
        selectedImageDataUrl = null;
        uploadBtn.disabled = true;
        bedrockBtn.disabled = true;
        if (statusHint) statusHint.textContent = 'Choose an image to start the pipeline.';
        if (analysisStateBadge) analysisStateBadge.textContent = 'Idle';
        if (detectionCountBadge) detectionCountBadge.textContent = '0';
        if (selectedFileMeta) selectedFileMeta.textContent = 'No image selected';
        if (detectionSummary) detectionSummary.textContent = 'No detections yet.';
        resetSelectedPreview();
        resetProcessedState();
        return;
    }

    selectedImage = file;
    uploadBtn.disabled = false;
    bedrockBtn.disabled = false;
    if (selectedFileMeta) selectedFileMeta.textContent = `${file.name} (${formatFileSize(file.size)})`;
    if (statusHint) statusHint.textContent = 'Image loaded. Run detection or request analysis.';

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
    if (statusHint) statusHint.textContent = 'Running object detection...';

    try {
        const mode = document.querySelector('input[name="mode"]:checked').value;
        const imagePayload = JSON.stringify({ image: selectedImageDataUrl.split(',')[1] });

        // Run detection and quality check in parallel
        const [detectionRes, qualityRes] = await Promise.all([
            fetch(ENDPOINTS[mode], {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: imagePayload
            }),
            fetch(ENDPOINTS.quality, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: imagePayload
            }).catch(() => null)
        ]);

        const result = await detectionRes.json();

        // Show quality banner
        const qualityBanner = document.getElementById('qualityBanner');
        if (qualityRes) {
            const q = await qualityRes.json();
            qualityBanner.classList.remove('d-none', 'quality-good', 'quality-warn');
            if (q.sufficient) {
                qualityBanner.classList.add('quality-good');
                qualityBanner.innerHTML = '✓ Image quality OK <span class="quality-details">blur=' + q.metrics.blur + ' bright=' + q.metrics.brightness + ' contrast=' + q.metrics.contrast + '</span>';
            } else {
                qualityBanner.classList.add('quality-warn');
                qualityBanner.innerHTML = '⚠ ' + q.issues.join(', ') + ' <span class="quality-details">blur=' + q.metrics.blur + ' bright=' + q.metrics.brightness + ' contrast=' + q.metrics.contrast + '</span>';
            }
        }

        displayResults(selectedImageDataUrl, result.detections || []);
        if (statusHint) statusHint.textContent = 'Detection completed. You can now inspect details or run analysis.';
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error processing image. Please try again.');
        if (statusHint) statusHint.textContent = 'Detection failed. Please retry with another image.';
        setProcessingState(false);
    } finally {
        imageInput.disabled = false;
        uploadBtn.disabled = !selectedImage;
    }
});

async function runAnalysis(endpoint) {
    if (!selectedImageDataUrl) return;

    bedrockBtn.disabled = true;
    bedrockEmpty.classList.add('d-none');
    bedrockResults.classList.add('d-none');
    bedrockSkeleton.classList.remove('d-none');
    if (analysisStateBadge) analysisStateBadge.textContent = 'Running';

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
        if (analysisStateBadge) analysisStateBadge.textContent = 'Ready';
    } catch (error) {
        console.error('Analysis error:', error);
        bedrockSkeleton.classList.add('d-none');
        bedrockEmpty.querySelector('p').textContent = 'Analysis failed';
        bedrockEmpty.querySelector('.text-body-secondary').textContent = String(error.message || 'Please try again.');
        bedrockEmpty.classList.remove('d-none');
        if (analysisStateBadge) analysisStateBadge.textContent = 'Error';
    } finally {
        bedrockBtn.disabled = !selectedImage;
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
            ? 'Image quality is sufficient for object detection.'
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
    const qb = document.getElementById('qualityBanner');
    if (qb) qb.classList.add('d-none');
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
    lastImageData = imageData;
    lastDetections = detections;
    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        
        // Draw image
        ctx.drawImage(img, 0, 0);

        // Draw tractor width corridor (perspective trapezoid)
        const widthPct = parseInt(tractorWidthSlider.value) / 100;
        const horizonPct = parseInt(horizonSlider.value) / 100;
        const horizonY = img.height * horizonPct;
        const bottomW = img.width * widthPct;
        const cx = img.width / 2;
        const bottomLeft = cx - bottomW / 2;
        const bottomRight = cx + bottomW / 2;

        // Corridor converges to vanishing point at horizon
        // Dim area outside corridor
        ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
        ctx.beginPath();
        ctx.moveTo(0, 0); ctx.lineTo(img.width, 0);
        ctx.lineTo(img.width, horizonY); ctx.lineTo(0, horizonY);
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(0, horizonY); ctx.lineTo(cx, horizonY);
        ctx.lineTo(bottomLeft, img.height); ctx.lineTo(0, img.height);
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(cx, horizonY); ctx.lineTo(img.width, horizonY);
        ctx.lineTo(img.width, img.height); ctx.lineTo(bottomRight, img.height);
        ctx.fill();

        // Draw corridor edge lines
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.setLineDash([10, 6]);
        ctx.beginPath();
        ctx.moveTo(cx, horizonY); ctx.lineTo(bottomLeft, img.height);
        ctx.moveTo(cx, horizonY); ctx.lineTo(bottomRight, img.height);
        ctx.stroke();
        // Horizon line
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.5)';
        ctx.beginPath();
        ctx.moveTo(0, horizonY); ctx.lineTo(img.width, horizonY);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Draw bounding boxes
        ctx.lineWidth = 3;
        ctx.font = '16px Arial';
        
        detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;
            const width = x2 - x1;
            const height = y2 - y1;

            // Check if detection center is inside the trapezoid
            const bCenterX = (x1 + x2) / 2;
            const bCenterY = (y1 + y2) / 2;
            let inCorridor = false;
            if (bCenterY > horizonY) {
                const t = (bCenterY - horizonY) / (img.height - horizonY);
                const edgeLeft = cx + (bottomLeft - cx) * t;
                const edgeRight = cx + (bottomRight - cx) * t;
                inCorridor = bCenterX > edgeLeft && bCenterX < edgeRight;
            }

            ctx.strokeStyle = inCorridor ? '#ef4444' : '#00ff00';
            ctx.fillStyle = ctx.strokeStyle;
            
            // Draw box
            ctx.strokeRect(x1, y1, width, height);
            
            // Draw label
            const detectionClass = formatDetectionClass(detection.class || detection.label || detection.category || 'object');
            const label = `${detectionClass} ${(detection.confidence * 100).toFixed(1)}%` + (inCorridor ? ' ⚠ IN PATH' : '');
            ctx.fillText(label, x1, y1 - 5);
        });

        processedPlaceholder.classList.add('is-hidden');
        processedSpinner.classList.add('is-hidden');
        processedCanvas.classList.remove('d-none');
        if (detectionCountBadge) detectionCountBadge.textContent = String(detections.length);
        if (detectionSummary) detectionSummary.textContent = detections.length
            ? `Detected ${formatDetectionSummary(detections)} in the current frame.`
            : 'No detections in the latest run.';
        detectionInfo.innerHTML = detections.length
            ? detections.map((detection, index) => {
                const detectionClass = formatDetectionClass(detection.class || detection.label || detection.category || 'object');
                return `
                    <div class="detection-item">
                        <span class="detection-label">${detectionClass} ${index + 1}</span>
                        <span class="detection-confidence">${(detection.confidence * 100).toFixed(1)}%</span>
                    </div>
                `;
            }).join('')
            : '<div class="text-body-secondary">No detections.</div>';
    };
    img.src = imageData;
}

function setInitialUiState() {
    updateModeBadge();
    if (analysisStateBadge) analysisStateBadge.textContent = 'Idle';
    if (detectionCountBadge) detectionCountBadge.textContent = '0';
}

function formatDetectionClass(value) {
    const normalized = String(value || 'object').trim().replace(/[_-]+/g, ' ');
    return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function formatDetectionSummary(detections) {
    const counts = new Map();

    detections.forEach((detection) => {
        const key = String(detection.class || detection.label || detection.category || 'object').trim().toLowerCase().replace(/[_-]+/g, ' ');
        counts.set(key, (counts.get(key) || 0) + 1);
    });

    return Array.from(counts.entries())
        .map(([className, count]) => `${count} ${count === 1 ? className : `${className}s`}`)
        .join(', ');
}

function updateModeBadge() {
    const mode = document.querySelector('input[name="mode"]:checked')?.value || 'local';
    if (!activeModeBadge) return;
    activeModeBadge.textContent = mode === 'sagemaker' ? 'SageMaker' : 'Local';
}

function formatFileSize(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';

    const units = ['B', 'KB', 'MB', 'GB'];
    const power = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / (1024 ** power);
    return `${value.toFixed(power === 0 ? 0 : 1)} ${units[power]}`;
}
