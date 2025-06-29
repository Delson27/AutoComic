const form = document.getElementById('uploadForm');
const resultDiv = document.getElementById('result');
const galleryDiv = document.getElementById('gallery');
const loadingDiv = document.getElementById('loading');
let isProcessing = false;
let pollTimer = null;

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('videoFile');
    if (fileInput.files.length === 0) {
        alert('Please select a video file.');
        return;
    }

    // Show loading state and reset UI
    loadingDiv.classList.add('active');
    resultDiv.textContent = '';
    resultDiv.className = 'result';
    galleryDiv.innerHTML = '';
    
    // Add cancel button
    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Cancel Processing';
    cancelBtn.className = 'cancel-btn';
    cancelBtn.onclick = cancelProcessing;
    loadingDiv.appendChild(cancelBtn);

    const formData = new FormData();
    formData.append('video', fileInput.files[0]);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (response.ok) {
            resultDiv.textContent = data.message;
            resultDiv.className = 'result success';
            isProcessing = true;
            
            // Start polling for progress
            await pollProgress();
        } else {
            resultDiv.textContent = 'Error: ' + data.error;
            resultDiv.className = 'result error';
            loadingDiv.classList.remove('active');
        }
    } catch (error) {
        resultDiv.textContent = 'Error: ' + error.message;
        resultDiv.className = 'result error';
        loadingDiv.classList.remove('active');
    }
});

function cancelProcessing() {
    isProcessing = false;
    if (pollTimer) {
        clearTimeout(pollTimer);
    }
    loadingDiv.classList.remove('active');
    resultDiv.textContent = 'Processing cancelled by user';
    resultDiv.className = 'result warning';
}

async function pollProgress() {
    if (!isProcessing) return;

    try {
        const response = await fetch('/progress');
        const data = await response.json();
        
        if (data.error) {
            resultDiv.textContent = 'Error: ' + data.error;
            resultDiv.className = 'result error';
            loadingDiv.classList.remove('active');
            isProcessing = false;
            return;
        }
        
        // Calculate progress metrics
        const processedFrames = data.processed_frames;
        const elapsedTime = data.processing_time;
        const framesPerSecond = processedFrames / elapsedTime;
        
        // Update progress display
        const progressInfo = [
            `Processed: ${processedFrames} frames`,
            `Time: ${elapsedTime.toFixed(1)}s`,
            `Speed: ${framesPerSecond.toFixed(1)} frames/sec`
        ];
        
        resultDiv.innerHTML = progressInfo.join(' | ');
        
        if (data.complete) {
            isProcessing = false;
            loadingDiv.classList.remove('active');
            if (data.frames && data.frames.length > 0) {
                displayFrames(data.frames);
            }
            return;
        }
        
        // Adaptive polling: increase interval for longer processing times
        const pollInterval = Math.min(
            Math.max(1000, Math.floor(elapsedTime * 100)), // Increase interval as time goes on
            5000 // Cap at 5 seconds
        );
        
        // Continue polling with adaptive interval
        pollTimer = setTimeout(pollProgress, pollInterval);
    } catch (error) {
        resultDiv.textContent = 'Error: ' + error.message;
        resultDiv.className = 'result error';
        loadingDiv.classList.remove('active');
        isProcessing = false;
    }
}

function displayFrames(frames) {
    galleryDiv.innerHTML = '';
    frames.forEach((frame, index) => {
        const frameContainer = document.createElement('div');
        frameContainer.className = 'frame-container';
        
        const title = document.createElement('h5');
        title.textContent = `Frame ${index + 1}`;
        
        const framePair = document.createElement('div');
        framePair.className = 'frame-pair';
        
        const originalImg = document.createElement('img');
        originalImg.src = `/frames/original/${frame}`;
        originalImg.alt = `Original Frame ${index + 1}`;
        originalImg.loading = 'lazy';
        
        const cartoonImg = document.createElement('img');
        cartoonImg.src = `/frames/cartoon/cartoon_${frame}`;
        cartoonImg.alt = `Cartoon Frame ${index + 1}`;
        cartoonImg.loading = 'lazy';
        
        framePair.appendChild(originalImg);
        framePair.appendChild(cartoonImg);
        
        frameContainer.appendChild(title);
        frameContainer.appendChild(framePair);
        galleryDiv.appendChild(frameContainer);
    });
} 