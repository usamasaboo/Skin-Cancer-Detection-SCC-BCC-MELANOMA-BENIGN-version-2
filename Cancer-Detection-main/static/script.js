document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultSection = document.getElementById('result-section');
    const topPredictionEl = document.getElementById('top-prediction');
    const barsContainer = document.getElementById('bars-container');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');

    let currentFile = null;

    // Trigger file input
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // File Input Change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Handle File
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (JPG, PNG).');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);

        // Reset results on new image
        resultSection.classList.add('hidden');
    }

    // Remove Image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // prevent triggering dropZone click
        currentFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        analyzeBtn.disabled = true;
        resultSection.classList.add('hidden');
    });

    const heatmapImage = document.getElementById('heatmap-image');
    const downloadBtn = document.getElementById('download-btn');
    let lastPredictions = null;
    let lastSessionId = null;

    // ... (rest of the handleFile and other functions remain)

    // Analyze
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        setLoading(true);

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                lastPredictions = data.predictions;
                lastSessionId = data.session_id;
                displayResults(data);
                updateHistory(); // Refresh history table
            } else {
                alert('Error: ' + data.error);
            }

        } catch (error) {
            console.error(error);
            alert('Something went wrong. Please try again.');
        } finally {
            setLoading(false);
        }
    });

    downloadBtn.addEventListener('click', async () => {
        if (!lastSessionId || !lastPredictions) return;

        try {
            downloadBtn.disabled = true;
            downloadBtn.innerHTML = 'Generating...';

            const response = await fetch(`/download_report/${lastSessionId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ predictions: lastPredictions })
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `Medical_Report_${lastSessionId.substring(0, 8)}.pdf`;
                document.body.appendChild(a);
                a.click();
                a.remove();
            } else {
                alert('Failed to generate report.');
            }
        } catch (error) {
            console.error(error);
            alert('Error downloading report.');
        } finally {
            downloadBtn.disabled = false;
            downloadBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                Download Medical Report
            `;
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            analyzeBtn.disabled = true;
            btnText.classList.add('hidden');
            loader.classList.remove('hidden');
        } else {
            analyzeBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    }

    function displayResults(data) {
        const predictions = data.predictions;
        const heatmapUrl = data.heatmap_url;

        resultSection.classList.remove('hidden');

        // Update heatmap
        heatmapImage.src = heatmapUrl + '?t=' + new Date().getTime(); // Prevent caching

        // Top Prediction
        const top = predictions[0];
        topPredictionEl.innerHTML = `
            <div class="diagnosis-label">Primary Diagnosis</div>
            <div class="diagnosis-value">${top.class.toUpperCase()}</div>
            <div style="color: var(--accent-primary); margin-top: 5px;">${top.probability.toFixed(1)}% Confidence</div>
        `;

        // Bars
        barsContainer.innerHTML = '';
        predictions.forEach(p => {
            const width = p.probability.toFixed(1);
            const html = `
                <div class="bar-item">
                    <div class="bar-label-container">
                        <span>${p.class.toUpperCase()}</span>
                        <span>${width}%</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: 0%"></div>
                    </div>
                </div>
            `;
            const div = document.createElement('div');
            div.innerHTML = html;
            barsContainer.appendChild(div);

            // Texture animation
            setTimeout(() => {
                const fill = div.querySelector('.progress-bar-fill');
                if (fill) fill.style.width = width + '%';
            }, 100);
        });

        // Clinical Indicators
        if (data.clinical_features) {
            const features = data.clinical_features;
            document.getElementById('asym-bar').style.width = (features.asymmetry * 100) + '%';
            document.getElementById('border-bar').style.width = (features.border * 100) + '%';
            document.getElementById('shiny-bar').style.width = (features.shiny * 100) + '%';
            document.getElementById('rough-bar').style.width = (features.roughness * 100) + '%';
            document.getElementById('red-bar').style.width = (features.redness * 100) + '%';
            document.getElementById('ulcer-bar').style.width = (features.ulcer * 100) + '%';
            document.getElementById('multi-bar').style.width = (features.multicolor * 100) + '%';
        }

        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    // --- History Logic ---
    const historyBody = document.getElementById('history-body');
    const refreshHistoryBtn = document.getElementById('refresh-history');

    async function updateHistory() {
        try {
            const response = await fetch('/history');
            const data = await response.json();

            historyBody.innerHTML = '';
            if (data.length === 0) {
                historyBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-secondary);">No analysis history yet.</td></tr>';
                return;
            }

            data.forEach(item => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="color: var(--text-secondary); font-size: 0.85rem;">${item.timestamp}</td>
                    <td style="max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${item.image_name}</td>
                    <td><span class="badge ${item.prediction.toLowerCase()}">${item.prediction}</span></td>
                    <td style="font-weight: 600; color: var(--accent-primary);">${item.confidence}</td>
                `;
                historyBody.appendChild(tr);
            });
        } catch (error) {
            console.error('Error fetching history:', error);
        }
    }

    refreshHistoryBtn.addEventListener('click', updateHistory);

    // Initial load
    updateHistory();
});
