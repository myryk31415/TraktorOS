const API = location.protocol === 'file:' ? 'http://localhost:5000' : '';
const ENDPOINTS = {
    local: `${API}/detect`,
    bedrock: `${API}/detect-bedrock`,
    quality: `${API}/quality`
};

const imageInput = document.getElementById('imageInput');
const uploadBtn = document.getElementById('uploadBtn');
const selectedImagePreview = document.getElementById('selectedImagePreview');
const selectedImageEmptyState = document.getElementById('selectedImageEmptyState');
const processedCanvas = document.getElementById('processedCanvas');
const processedPlaceholder = document.getElementById('processedImagePlaceholder');
const processedSpinner = document.getElementById('processedSpinner');
const depthMapAccordion = document.getElementById('depthMapAccordion');
const detectionInfoHeading = document.getElementById('detectionInfoHeading');
const detectionInfo = document.getElementById('detectionInfo');
const canvas = processedCanvas;
const ctx = canvas.getContext('2d');

let selectedImage = null;
let selectedImageDataUrl = null;
let previewObjectUrl = null;
let lastDetections = [];
let lastImageData = null;
let lastQualityData = null;
let lastBedrockData = null;

const tractorWidthSlider = document.getElementById('tractorWidth');
const tractorWidthLabel = document.getElementById('tractorWidthLabel');
const horizonSlider = document.getElementById('horizonLine');
const horizonLabel = document.getElementById('horizonLineLabel');

tractorWidthSlider.addEventListener('input', () => {
    tractorWidthLabel.textContent = tractorWidthSlider.value + '%';
    if (lastImageData) { displayResults(lastImageData, lastDetections); computeActions(lastDetections, lastQualityData); }
});
horizonSlider.addEventListener('input', () => {
    horizonLabel.textContent = horizonSlider.value + '%';
    if (lastImageData) { displayResults(lastImageData, lastDetections); computeActions(lastDetections, lastQualityData); }
});

const bedrockBtn = document.getElementById('bedrockBtn');
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
        resetSelectedPreview();
        resetProcessedState();
        return;
    }

    selectedImage = file;
    uploadBtn.disabled = false;
    bedrockBtn.disabled = false;

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
    const model = document.querySelector('input[name="model"]:checked').value;
    await runOnDeviceAnalysis(ENDPOINTS.local, model);
});

async function runOnDeviceAnalysis(endpoint, model) {
    if (!selectedImage || !selectedImageDataUrl) return;

    uploadBtn.disabled = true;
    imageInput.disabled = true;
    setProcessingState(true);

    try {
        const imagePayload = JSON.stringify({
            image: selectedImageDataUrl.split(',')[1],
            model
        });

        // Run detection and quality check in parallel
        const [detectionRes, qualityRes] = await Promise.all([
            fetch(endpoint, {
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

        if (!detectionRes.ok) {
            const text = await detectionRes.text();
            throw new Error(`Server returned ${detectionRes.status}: ${text.slice(0, 200)}`);
        }

        const result = await detectionRes.json();

        // Show quality accordion
        let qualityData = null;
        if (qualityRes) {
            qualityData = await qualityRes.json();
            lastQualityData = qualityData;
            const q = qualityData;
            const qualityAccordion = document.getElementById('qualityAccordion');
            const qualityTitle = document.getElementById('qualityBannerTitle');
            const qualityDetails = document.getElementById('qualityDetails');
            const qualityItem = qualityAccordion.querySelector('.accordion-item');
            qualityItem.classList.remove('quality-good', 'quality-warn');
            qualityAccordion.classList.remove('d-none');
            if (q.sufficient) {
                qualityItem.classList.add('quality-good');
                qualityTitle.textContent = '✓ Image quality OK';
            } else {
                qualityItem.classList.add('quality-warn');
                qualityTitle.textContent = '⚠ ' + (q.issues || []).join(', ');
            }
            qualityDetails.innerHTML = formatQualityMetrics(q.metrics || {});
        }

        displayResults(selectedImageDataUrl, result.detections || []);
        computeActions(result.detections || [], qualityData);

        // Show depth map if available
        const depthContainer = document.getElementById('depthMapContainer');
        const depthImg = document.getElementById('depthMapImage');
        if (result.depth_map && depthContainer && depthImg) {
            depthImg.src = 'data:image/png;base64,' + result.depth_map;
            depthContainer.classList.remove('d-none');
            document.getElementById('depthMapEmpty')?.classList.add('d-none');
        }
        if (typeof statusHint !== 'undefined' && statusHint) statusHint.textContent = 'Detection completed. You can now inspect details or run analysis.';
        
    } catch (error) {
        console.error('Detection error:', error);
        let msg = `Detection failed (${model}):\n\n${error.message}`;
        if (error.message === 'Failed to fetch') {
            msg += `\n\nCannot reach server at ${endpoint}. Is the backend running?`;
        }
        alert(msg);
        setProcessingState(false);
    } finally {
        imageInput.disabled = false;
        uploadBtn.disabled = !selectedImage;
    }
}

async function runAnalysis(endpoint) {
    if (!selectedImageDataUrl) return;

    bedrockBtn.disabled = true;
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
            groundAssessment: result.ground_assessment || {},
            pathAnalysis: result.path_analysis || {},
            maintenance: result.maintenance || {},
            summary: result.summary || ''
        };

        bedrockDetections.innerHTML = buildAnalysisCards(analysisData);
        lastBedrockData = analysisData;

        bedrockSkeleton.classList.add('d-none');
        bedrockResults.classList.remove('d-none');

        // Recompute actions with Bedrock data
        computeActions(lastDetections, lastQualityData);
    } catch (error) {
        console.error('Analysis error:', error);
        bedrockSkeleton.classList.add('d-none');
        bedrockEmpty.querySelector('p').textContent = 'Analysis failed';
        bedrockEmpty.querySelector('.text-body-secondary').textContent = String(error.message || 'Please try again.');
        bedrockEmpty.classList.remove('d-none');
    } finally {
        bedrockBtn.disabled = !selectedImage;
    }
}

function buildAnalysisCards(analysisData) {
    const ground = analysisData.groundAssessment || {};
    const path = analysisData.pathAnalysis || {};
    const maint = analysisData.maintenance || {};
    const summaryText = analysisData.summary || '';

    const groundHazards = Array.isArray(ground.hazards) ? ground.hazards.filter(Boolean) : [];
    const groundBase = [ground.surface_type, ground.safety_to_traverse].filter(Boolean).join(' / ');
    const groundDescription = [groundBase, groundHazards.join(', ')].filter(Boolean).join(' — ') || 'No ground assessment reported.';

    const cards = [
        {
            key: 'ground-assessment',
            icon: 'icons/golf-outline.svg',
            title: 'Ground assessment',
            status: traversabilityStatus(ground.safety_to_traverse),
            description: groundDescription
        },
    ];

    if (path.path_type && path.path_type !== 'none') {
        const pathDesc = path.description || '';
        const turnInfo = path.turn_ahead
            ? `Turn ${path.turn_direction || ''} ${path.turn_distance || ''}`.trim()
            : 'No turn detected';
        const pathDescription = pathDesc ? `${pathDesc} (${turnInfo})` : turnInfo;
        const pathStatus = path.turn_ahead
            ? (path.turn_distance === 'immediate' ? 'danger' : path.turn_distance === 'near' ? 'warning' : 'info')
            : 'success';

        cards.push({
            key: 'path-analysis',
            icon: 'icons/analytics-outline.svg',
            title: 'Path analysis',
            status: pathStatus,
            description: pathDescription
        });
    }

    const maintDesc = (maint.description || '').toLowerCase();
    cards.push({
        key: 'maintenance',
        icon: "icons/construct-outline.svg",
        title: 'Maintenance',
        status: maintDesc && maintDesc !== 'none' ? 'warning' : 'success',
        description: maintDesc && maintDesc !== 'none' ? maint.description : 'No maintenance required'
    });

    cards.push({
        key: 'summary',
        icon: 'icons/book-outline.svg',
        title: 'Summary',
        status: 'info',
        description: summaryText || 'No summary reported.'
    });

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

function traversabilityStatus(traversability) {
    const value = String(traversability || '').toLowerCase();
    if (value === 'safe') return 'success';
    if (value === 'caution') return 'warning';
    if (value === 'unsafe') return 'danger';
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

bedrockBtn.addEventListener('click', async () => {
    const model = document.querySelector('input[name="model"]:checked').value;
    await Promise.all([
        runAnalysis(ENDPOINTS.bedrock),
        runOnDeviceAnalysis(ENDPOINTS.local, model)
    ]);
});

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
    if (depthMapAccordion) depthMapAccordion.classList.add('d-none');
    if (detectionInfoHeading) detectionInfoHeading.classList.add('d-none');
    detectionInfo.innerHTML = '';
    ctx.clearRect(0, 0, processedCanvas.width, processedCanvas.height);
    const qa = document.getElementById('qualityAccordion');
    if (qa) qa.classList.add('d-none');
    const dc = document.getElementById('depthMapContainer');
    if (dc) dc.classList.add('d-none');
    const de = document.getElementById('depthMapEmpty');
    if (de) de.classList.remove('d-none');
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

const MAX_IMAGE_DIM = 1920;

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            let { width, height } = img;
            if (width > MAX_IMAGE_DIM || height > MAX_IMAGE_DIM) {
                const scale = MAX_IMAGE_DIM / Math.max(width, height);
                width = Math.round(width * scale);
                height = Math.round(height * scale);
            }
            const c = document.createElement('canvas');
            c.width = width;
            c.height = height;
            c.getContext('2d').drawImage(img, 0, 0, width, height);
            resolve(c.toDataURL('image/jpeg', 0.85));
        };
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
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

            const inCorridor = isDetectionInCorridor(detection.bbox, {
                horizonY,
                imageHeight: img.height,
                centerX: cx,
                bottomLeft,
                bottomRight
            });

            ctx.strokeStyle = inCorridor ? '#ef4444' : '#00ff00';
            ctx.fillStyle = ctx.strokeStyle;
            
            // Draw box
            ctx.strokeRect(x1, y1, width, height);
            
            // Draw label
            const detectionClass = formatDetectionClass(detection.class || detection.label || detection.category || 'object');
            const proximityText = formatProximityLabel(detection.proximity);
            const proxLabel = proximityText ? ` — ${proximityText}` : '';
            const label = `${detectionClass} ${(detection.confidence * 100).toFixed(1)}%${proxLabel}`;
            ctx.fillText(label, x1, y1 - 5);
        });

        processedPlaceholder.classList.add('is-hidden');
        processedSpinner.classList.add('is-hidden');
        processedCanvas.classList.remove('d-none');
        if (depthMapAccordion) depthMapAccordion.classList.remove('d-none');
        if (detectionInfoHeading) {
            detectionInfoHeading.textContent = detections.length ? 'Detected objects' : 'Detection results';
            detectionInfoHeading.classList.remove('d-none');
        }
        detectionInfo.innerHTML = detections.length
            ? detections.map((detection, index) => {
                const detectionClass = formatDetectionClass(detection.class || detection.label || detection.category || 'object');
                const proximityClassMap = {
                    'VERY CLOSE': 'proximity-badge-near',
                    NEARBY: 'proximity-badge-medium',
                    FAR: 'proximity-badge-far'
                };
                const proximityText = formatProximityLabel(detection.proximity);
                const proxBadge = proximityText
                    ? `<span class="badge proximity-badge ${proximityClassMap[proximityText] || 'proximity-badge-far'}">${escapeHtml(proximityText)}</span>`
                    : '';
                const inCorridor = isDetectionInCorridor(detection.bbox, {
                    horizonY,
                    imageHeight: img.height,
                    centerX: cx,
                    bottomLeft,
                    bottomRight
                });
                const inPathBadge = inCorridor
                    ? '<span class="badge bg-danger ms-1">IN PATH</span>'
                    : '';
                return `
                    <div class="detection-item">
                        <span class="detection-label">${detectionClass} ${index + 1} ${proxBadge}${inPathBadge}</span>
                        <span class="detection-confidence">${(detection.confidence * 100).toFixed(1)}%</span>
                    </div>
                `;
            }).join('')
            : '<div class="text-body-secondary">No detections.</div>';
    };
    img.src = imageData;
}

function formatDetectionClass(value) {
    const normalized = String(value || 'object').trim().replace(/[_-]+/g, ' ');
    return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function formatProximityLabel(value) {
    const normalized = String(value || '').trim().toUpperCase();
    if (normalized === 'NEAR' || normalized === 'CLOSE') return 'VERY CLOSE';
    if (normalized === 'MEDIUM') return 'NEARBY';
    return normalized;
}

function isDetectionInCorridor(bbox, corridor) {
    const [x1, , x2, y2] = bbox;
    const bCenterX = (x1 + x2) / 2;
    const bBottomY = y2;

    if (bBottomY <= corridor.horizonY) {
        return false;
    }

    const t = (bBottomY - corridor.horizonY) / (corridor.imageHeight - corridor.horizonY);
    const edgeLeft = corridor.centerX + (corridor.bottomLeft - corridor.centerX) * t;
    const edgeRight = corridor.centerX + (corridor.bottomRight - corridor.centerX) * t;
    return bCenterX > edgeLeft && bCenterX < edgeRight;
}

function formatQualityMetrics(metrics) {
    const m = metrics;
    const entries = [];

    if (m.blur !== undefined) {
        const label = m.blur >= 500 ? 'sharp' : m.blur >= 100 ? 'good' : 'blurry';
        entries.push(['Sharpness', label]);
    }
    if (m.brisque !== undefined) {
        const label = m.brisque < 30 ? 'good' : m.brisque < 60 ? 'fair' : 'poor';
        entries.push(['Image quality', `${label} (brisque=${m.brisque})`]);
    }

    if (!entries.length) return '';

    return `
        <span class="quality-metrics" aria-label="Image quality metrics">
            ${entries.map(([label, value]) => `
                <span class="quality-metric">
                    <span class="quality-metric-label">${label}</span>
                    <span class="quality-metric-value">${escapeHtml(value)}</span>
                </span>
            `).join('')}
        </span>
    `;
}

const MOVING_CLASSES = new Set(['person', 'dog', 'horse', 'sheep', 'cow', 'bird']);

function computeActions(detections, qualityData) {
    const actionsEmpty = document.getElementById('actionsEmpty');
    const actionsResults = document.getElementById('actionsResults');
    const actionsList = document.getElementById('actionsList');

    // Bad image quality → stop
    if (qualityData && !qualityData.sufficient) {
        const html = `
            <div class="analysis-card">
                <div class="analysis-card-header">
                    <div class="analysis-card-title-wrap">
                        <span class="analysis-card-left-icon" aria-hidden="true"><img src="icons/close-outline.svg" alt="" aria-hidden="true"></span>
                        <h3 class="analysis-card-title">STOP</h3>
                    </div>
                    <span class="analysis-status-icon status-danger" aria-hidden="true"><img src="icons/close-outline.svg" alt="" aria-hidden="true" style="transform: scale(1.3);"></span>
                </div>
                <p class="analysis-card-description">Image quality insufficient: ${escapeHtml((qualityData.issues || []).join(', '))}</p>
            </div>`;
        actionsList.innerHTML = html;
        actionsEmpty?.classList.add('d-none');
        actionsResults?.classList.remove('d-none');
        return;
    }

    // No detections → continue
    if (!detections.length) {
        const html = `
            <div class="analysis-card">
                <div class="analysis-card-header">
                    <div class="analysis-card-title-wrap">
                        <span class="analysis-card-left-icon" aria-hidden="true"><img src="icons/checkmark-outline.svg" alt="" aria-hidden="true"></span>
                        <h3 class="analysis-card-title">CONTINUE</h3>
                    </div>
                    <span class="analysis-status-icon status-success" aria-hidden="true"><img src="icons/checkmark-outline.svg" alt="" aria-hidden="true" style="transform: scale(1.3);"></span>
                </div>
                <p class="analysis-card-description">No objects detected — path is clear</p>
            </div>`;
        actionsList.innerHTML = html;
        actionsEmpty?.classList.add('d-none');
        actionsResults?.classList.remove('d-none');
        return;
    }

    const tractorW = parseInt(document.getElementById('tractorWidth')?.value || 90) / 100;
    const horizonPct = parseInt(document.getElementById('horizonLine')?.value || 36) / 100;
    // We need image dimensions — grab from last canvas
    const canvas = document.getElementById('processedCanvas');
    const imgW = canvas?.width || 1;
    const imgH = canvas?.height || 1;
    const horizonY = imgH * horizonPct;
    const cx = imgW / 2;
    const bottomW = imgW * tractorW;
    const corridor = { horizonY, centerX: cx, bottomLeft: cx - bottomW / 2, bottomRight: cx + bottomW / 2, imageHeight: imgH };

    let shouldStop = false;
    let shouldHonk = false;
    let correctLeft = false;
    let correctRight = false;
    const stopReasons = [];
    const honkReasons = [];
    const correctLeftReasons = [];
    const correctRightReasons = [];

    detections.forEach(det => {
        const cls = (det.class || 'object').toLowerCase();
        const prox = (det.proximity || '').toUpperCase();
        const isMoving = MOVING_CLASSES.has(cls);
        const inPath = isDetectionInCorridor(det.bbox, corridor);
        const [x1, , x2] = det.bbox;
        const centerX = (x1 + x2) / 2;
        const isLeftSide = centerX < cx;
        const name = formatDetectionClass(cls).toLowerCase();

        if (isMoving) {
            if (prox === 'VERY CLOSE' || prox === 'NEAR') {
                if (inPath) {
                    shouldStop = true;
                    stopReasons.push(`${name} very close in path`);
                } else {
                    shouldHonk = true;
                    honkReasons.push(`${name} very close`);
                }
            } else if (prox === 'NEARBY' || prox === 'MEDIUM') {
                if (inPath) {
                    shouldStop = true;
                    stopReasons.push(`${name} nearby in path`);
                } else {
                    shouldHonk = true;
                    honkReasons.push(`${name} nearby`);
                }
            } else if (inPath) {
                shouldHonk = true;
                honkReasons.push(`${name} in path ahead`);
            }
        } else {
            // Stationary object
            if (!inPath) {
                // Not in path → continue
            } else if (prox === 'VERY CLOSE' || prox === 'NEAR') {
                shouldStop = true;
                stopReasons.push(`${name} in path, very close`);
            } else if (prox === 'NEARBY' || prox === 'MEDIUM') {
                if (isLeftSide) {
                    correctRight = true;
                    correctRightReasons.push(name);
                } else {
                    correctLeft = true;
                    correctLeftReasons.push(name);
                }
            } else {
                // FAR + in path → continue
            }
        }
    });

    // If corrections in both directions → stop instead
    if (correctLeft && correctRight) {
        shouldStop = true;
        stopReasons.push('obstacles on both sides, cannot dodge');
    }

    const actions = [];
    if (shouldStop) {
        actions.push({ icon: 'icons/close-outline.svg', title: 'STOP', status: 'danger', description: stopReasons.join('; ') });
    }
    if (shouldHonk) {
        actions.push({ icon: 'icons/megaphone-outline.svg', title: 'HONK', status: 'warning', description: honkReasons.join('; ') });
    }
    if (!shouldStop && correctLeft && !correctRight) {
        actions.push({ icon: 'icons/cube-outline.svg', title: 'CORRECT LEFT', status: 'warning', description: `Steer left to avoid ${correctLeftReasons.join(', ')} in path` });
    }
    if (!shouldStop && correctRight && !correctLeft) {
        actions.push({ icon: 'icons/cube-outline.svg', title: 'CORRECT RIGHT', status: 'warning', description: `Steer right to avoid ${correctRightReasons.join(', ')} in path` });
    }
    if (!shouldStop && !shouldHonk && !correctLeft && !correctRight) {
        actions.push({ icon: 'icons/checkmark-outline.svg', title: 'CONTINUE', status: 'success', description: 'Path is clear' });
    }

    // Bedrock-derived actions
    if (lastBedrockData) {
        const ground = lastBedrockData.groundAssessment || {};
        const path = lastBedrockData.pathAnalysis || {};
        const maint = lastBedrockData.maintenance || {};

        if (ground.safety_to_traverse === 'unsafe') {
            actions.push({ icon: 'icons/close-outline.svg', title: 'STOP', status: 'danger', description: `Unsafe ground: ${(ground.hazards || []).join(', ') || ground.surface_type || 'hazardous terrain'}` });
        } else if (ground.safety_to_traverse === 'caution') {
            actions.push({ icon: 'icons/alert-outline.svg', title: 'CAUTION', status: 'warning', description: `Ground conditions: ${(ground.hazards || []).join(', ') || ground.surface_type}` });
        }

        if (path.turn_ahead && path.path_type && path.path_type !== 'none') {
            const dir = path.turn_direction && path.turn_direction !== 'none' ? ` ${path.turn_direction}` : '';
            actions.push({ icon: 'icons/alert-outline.svg', title: `TURN${dir.toUpperCase()}`, status: path.turn_distance === 'immediate' ? 'danger' : 'warning', description: path.description || `Turn${dir} ahead` });
        }

        const maintDesc = (maint.description || '').toLowerCase();
        if (maintDesc && maintDesc !== 'none') {
            actions.push({ icon: 'icons/alert-outline.svg', title: 'MAINTENANCE', status: 'info', description: maint.description });
        }
    }

    const html = actions.map(a => `
        <div class="analysis-card">
            <div class="analysis-card-header">
                <div class="analysis-card-title-wrap">
                    <span class="analysis-card-left-icon" aria-hidden="true">
                        <img src="${a.icon}" alt="" aria-hidden="true">
                    </span>
                    <h3 class="analysis-card-title">${a.title}</h3>
                </div>
                <span class="analysis-status-icon status-${a.status}" aria-hidden="true">
                    <img src="${a.status === 'danger' ? 'icons/close-outline.svg' : a.status === 'warning' ? 'icons/alert-outline.svg' : 'icons/checkmark-outline.svg'}" alt="" aria-hidden="true" style="transform: scale(1.3);">
                </span>
            </div>
            <p class="analysis-card-description">${escapeHtml(a.description)}</p>
        </div>
    `).join('');

    actionsList.innerHTML = html;
    actionsEmpty?.classList.add('d-none');
    actionsResults?.classList.remove('d-none');
}
