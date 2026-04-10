const ENDPOINTS = {
    local: 'http://localhost:5000/detect',
    sagemaker: 'YOUR_API_GATEWAY_URL' // Replace after deployment
};

const imageInput = document.getElementById('imageInput');
const uploadBtn = document.getElementById('uploadBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultSection = document.getElementById('resultSection');
const loading = document.getElementById('loading');
const detectionInfo = document.getElementById('detectionInfo');

let selectedImage = null;

imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedImage = file;
        uploadBtn.disabled = false;
    }
});

uploadBtn.addEventListener('click', async () => {
    if (!selectedImage) return;

    uploadBtn.disabled = true;
    loading.style.display = 'block';
    resultSection.style.display = 'none';

    try {
        // Convert image to base64
        const base64Image = await fileToBase64(selectedImage);
        
        // Send to API
        const mode = document.querySelector('input[name="mode"]:checked').value;
        const response = await fetch(ENDPOINTS[mode], {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: base64Image.split(',')[1] // Remove data:image/jpeg;base64, prefix
            })
        });

        const result = await response.json();
        
        // Display results
        displayResults(base64Image, result.detections);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error processing image. Please try again.');
    } finally {
        loading.style.display = 'none';
        uploadBtn.disabled = false;
    }
});

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
        
        // Show info
        detectionInfo.innerHTML = `
            <strong>Detections:</strong> ${detections.length} human(s) detected<br>
            ${detections.map((d, i) => 
                `Human ${i + 1}: Confidence ${(d.confidence * 100).toFixed(1)}%`
            ).join('<br>')}
        `;
        
        resultSection.style.display = 'block';
    };
    img.src = imageData;
}
