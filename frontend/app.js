const ENDPOINTS = {
    local: 'http://localhost:5000/detect',
    sagemaker: 'YOUR_API_GATEWAY_URL' // Replace after deployment
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

uploadBtn.disabled = true;

imageInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];

    if (!file) {
        selectedImage = null;
        selectedImageDataUrl = null;
        uploadBtn.disabled = true;
        resetSelectedPreview();
        resetProcessedState();
        return;
    }

    selectedImage = file;
    uploadBtn.disabled = false;

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
