document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('ecg-file-input');
    const statusDiv = document.getElementById('ecg-status');
    const controlsDiv = document.getElementById('ecg-controls');
    const primaryChSelect = document.getElementById('ecg-primary-ch');
    const pairChSelect = document.getElementById('ecg-pair-ch');
    const playBtn = document.getElementById('ecg-btn-play');
    const pauseBtn = document.getElementById('ecg-btn-pause');
    const resetBtn = document.getElementById('ecg-btn-reset');

    const timeSliderContainer = document.getElementById('ecg-time-slider-container');
    const timeSlider = document.getElementById('ecg-time-slider');
    const windowSizeOutput = document.getElementById('window-size-output');

    const nyquistSliderContainer = document.getElementById('ecg-nyquist-slider-container');
    const nyquistSlider = document.getElementById('ecg-nyquist-slider');
    const nyquistFsOutput = document.getElementById('nyquist-fs-output');
    const maxFreqOutput = document.getElementById('max-freq-output');

    const modeButtonsDiv = document.getElementById('ecg-mode-buttons');
    const plotContainer = document.getElementById('ecg-single-plot');
    const plotTitleH5 = document.getElementById('current-plot-title');
    const linearTimeWindow = document.getElementById('linear-time-window');

    // NEW: Detection elements
    const detectionContainer = document.getElementById('ecg-detection-container');
    const detectBtn = document.getElementById('ecg-btn-detect');
    const detectionResults = document.getElementById('ecg-detection-results');
    const detectionContent = document.getElementById('ecg-detection-content');

    let globalECGData = null;
    let streamInterval = null;
    let currentSignalIndex = 0;

    let currentVisualizationMode = 'linear';

    const STEP_SEC = 0.2;
    const INTERVAL_MS = STEP_SEC * 1000;

    // Helper to get CSRF token
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.startsWith(name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    // Signal Processing Utilities
    function resampleSignal(signal, actualFs, simulatedFs) {
        if (simulatedFs >= actualFs) {
            return signal;
        }

        const ratio = actualFs / simulatedFs;
        const resampled = [];
        for (let i = 0; i < signal.length; i += ratio) {
            resampled.push(signal[Math.floor(i)]);
        }
        return resampled;
    }

    function applyLowPassFilter(signal, actualFs, simulatedFs) {
        const cutoffFreq = simulatedFs / 2;
        if (cutoffFreq >= actualFs / 2) {
            return signal;
        }

        const cutoffRatio = cutoffFreq / (actualFs / 2);
        const windowSize = Math.max(3, Math.floor(1 / cutoffRatio / 2) * 2 + 1);

        if (windowSize <= 1) return signal;

        const filteredSignal = [];
        for (let i = 0; i < signal.length; i++) {
            let sum = 0;
            let count = 0;
            for (let j = 0; j < windowSize; j++) {
                const idx = i - Math.floor(windowSize / 2) + j;
                if (idx >= 0 && idx < signal.length) {
                    sum += signal[idx];
                    count++;
                }
            }
            filteredSignal.push(sum / count);
        }
        return filteredSignal;
    }

    function computeRecurrenceMatrix(sig_x, sig_y, bins = 60) {
        if (sig_x.length === 0 || sig_y.length === 0) return Array(bins).fill(0).map(() => Array(bins).fill(0).map(() => 0));

        const normalize = (arr) => {
            const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
            const std = Math.sqrt(arr.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / arr.length) || 1;
            return arr.map(x => Math.max(-3, Math.min(3, (x - mean) / std)));
        };

        const xs = normalize(sig_x);
        const ys = normalize(sig_y);

        const matrix = Array(bins).fill(0).map(() => Array(bins).fill(0));
        const binSize = 6 / bins;

        for (let i = 0; i < xs.length; i++) {
            const xBin = Math.floor((xs[i] + 3) / binSize);
            const yBin = Math.floor((ys[i] + 3) / binSize);
            if (xBin >= 0 && xBin < bins && yBin >= 0 && yBin < bins) {
                matrix[yBin][xBin]++;
            }
        }
        return matrix;
    }

    function xorDetectionPoints(sig_a, sig_b, fs, timeAxis) {
        const hits = [];
        const THRESHOLD_FACTOR = 0.2;
        const maxDiff = Math.max(...sig_a.map((val, i) => Math.abs(val - sig_b[i])));
        const threshold = maxDiff * THRESHOLD_FACTOR;

        for (let i = 0; i < sig_a.length; i++) {
            if (Math.abs(sig_a[i] - sig_b[i]) > threshold) {
                hits.push({ x: timeAxis[i], y: sig_a[i] });
            }
        }
        return hits;
    }

    function updateStatus(message, alertClass = 'alert-info') {
        statusDiv.className = `alert ${alertClass}`;
        statusDiv.innerHTML = message;
    }

    // Drag and Drop
    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => { fileInput.click(); });
    }

    if (dropZone) {
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.style.borderColor = '#58a6ff'; };
        dropZone.ondragleave = () => { dropZone.style.borderColor = '#30363d'; };
        dropZone.ondrop = (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '#30363d';
            handleFiles(Array.from(e.dataTransfer.files));
        };
    }

    if (fileInput) {
        fileInput.onchange = (e) => {
            handleFiles(Array.from(e.target.files));
            fileInput.value = '';
        };
    }
// Replace the entire handleFiles function and setupControls function in ecg_logic.js

function handleFiles(files) {
    if (files.length < 2) {
        updateStatus("⚠️ Please drag and drop both the .dat and .hea files.", 'alert-danger');
        return;
    }

    const datFile = files.find(f => f.name.endsWith('.dat'));
    const heaFile = files.find(f => f.name.endsWith('.hea'));

    if (!datFile || !heaFile) {
        updateStatus("⚠️ Could not find both .dat and .hea files. Both are required.", 'alert-danger');
        return;
    }

    // Hide detection elements when new file is uploaded
    if (detectionContainer) {
        detectionContainer.style.display = 'none';
    }
    if (detectionResults) {
        detectionResults.style.display = 'none';
    }

    updateStatus(`Uploading ${datFile.name} and ${heaFile.name} for server conversion...`, 'alert-warning');

    const formData = new FormData();
    formData.append('dat_file', datFile);
    formData.append('hea_file', heaFile);

    const csrftoken = getCookie('csrftoken');

    fetch('/api/convert_ecg/', {
        method: 'POST',
        headers: { 'X-CSRFToken': csrftoken },
        body: formData,
    })
    .then(async response => {
        if (response.ok) {
            return response.json();
        } else {
            const errorText = await response.text();
            throw new Error(`Server responded with status ${response.status}. Response content: ${errorText.substring(0, 500)}`);
        }
    })
    .then(data => {
        if (data.error) throw new Error(data.error);

        console.log('[ECG] ========== FILE UPLOAD SUCCESS ==========');
        console.log('[ECG] Filename:', data.filename);
        console.log('[ECG] Sampling Frequency (fs):', data.fs, 'Hz');
        console.log('[ECG] Channels:', data.channel_names);
        console.log('[ECG] Duration:', data.duration, 'seconds');
        console.log('[ECG] ==========================================');

        globalECGData = data;

        updateStatus(`✅ ECG Record loaded (${data.fs}Hz). Initializing controls...`, 'alert-success');

        // Setup controls (this will handle button visibility)
        setupControls(data);

        initializePlot(data);
        currentSignalIndex = 0;
        playSignal();

    })
    .catch(error => {
        console.error('Conversion Error:', error);
        updateStatus(`❌ Conversion Failed. Check console for details. Error message: ${error.message}`, 'alert-danger');
        pauseSignal();
    });
}

function setupControls(data) {
    console.log('\n[ECG SETUP] ========== SETTING UP CONTROLS ==========');
    console.log('[ECG SETUP] Received fs:', data.fs, 'Hz');

    primaryChSelect.innerHTML = '';
    pairChSelect.innerHTML = '';
    data.channel_names.forEach((name, index) => {
        const option = `<option value="${index}">${name}</option>`;
        primaryChSelect.innerHTML += option;
        pairChSelect.innerHTML += option;
    });

    if (primaryChSelect.options.length > 0) primaryChSelect.options[0].selected = true;
    pairChSelect.value = data.channel_names.length > 1 ? 1 : 0;

    controlsDiv.style.display = 'flex';
    if (timeSliderContainer) timeSliderContainer.style.display = 'block';
    if (nyquistSliderContainer) nyquistSliderContainer.style.display = 'block';
    if (modeButtonsDiv) modeButtonsDiv.style.display = 'block';

    playBtn.disabled = false;
    pauseBtn.disabled = false;
    if (resetBtn) resetBtn.disabled = false;

    primaryChSelect.onchange = () => updatePlots(false);
    pairChSelect.onchange = () => updatePlots(false);

    if (timeSlider && windowSizeOutput) {
        timeSlider.oninput = () => {
            windowSizeOutput.textContent = parseFloat(timeSlider.value).toFixed(1);
            updatePlots(false);
        };
    }

    // Nyquist Slider
    const maxFs = data.fs;
    const initialSliderFs = Math.min(500, maxFs);
    const maxNyquist = Math.floor(initialSliderFs / 2);

    nyquistSlider.max = maxFs;
    nyquistSlider.value = initialSliderFs;

    if (nyquistFsOutput) nyquistFsOutput.textContent = initialSliderFs.toFixed(0);
    if (maxFreqOutput) maxFreqOutput.textContent = maxNyquist.toFixed(0);

    if (nyquistSlider) {
        nyquistSlider.oninput = () => {
            const simulatedFs = parseFloat(nyquistSlider.value);
            const nyquistLimit = Math.floor(simulatedFs / 2);

            nyquistFsOutput.textContent = simulatedFs.toFixed(0);
            maxFreqOutput.textContent = nyquistLimit.toFixed(0);
        };
    }

    if (resetBtn) {
        resetBtn.onclick = resetSignal;
    }

    document.querySelectorAll('.btn-mode').forEach(button => {
        button.onclick = (e) => {
            document.querySelectorAll('.btn-mode').forEach(btn => btn.classList.remove('active'));
            e.target.classList.add('active');
            currentVisualizationMode = e.target.getAttribute('data-mode');
            updatePlots(false);
        };
    });

    // ========== DETECTION BUTTON LOGIC ==========
    console.log('[ECG SETUP] Checking detection button eligibility...');
    console.log('[ECG SETUP] Detection container exists:', !!detectionContainer);
    console.log('[ECG SETUP] Detection button exists:', !!detectBtn);

    if (!detectionContainer || !detectBtn) {
        console.error('[ECG SETUP] ❌ Detection elements not found in DOM!');
        console.log('[ECG SETUP] ==========================================\n');
        return;
    }

    // Check if sampling frequency is exactly 100Hz
    const is100Hz = (data.fs === 100);

    console.log('[ECG SETUP] Is 100Hz signal?', is100Hz);

    if (is100Hz) {
        // SHOW the detection button
        console.log('[ECG SETUP] ✅ SHOWING detection button (100Hz signal detected)');

        detectionContainer.style.display = 'block';
        detectBtn.disabled = false;

        // Update status message
        setTimeout(() => {
            updateStatus(
                `✅ ECG loaded successfully (${data.fs}Hz, ${data.channel_names.length} channels). ` +
                `<strong>Abnormality detection available!</strong>`,
                'alert-success'
            );
        }, 800);

    } else {
        // HIDE the detection button
        console.log(`[ECG SETUP] ⚠️ HIDING detection button (${data.fs}Hz signal - requires 100Hz)`);

        detectionContainer.style.display = 'none';
        detectionResults.style.display = 'none';

        // Update status message
        setTimeout(() => {
            updateStatus(
                `✅ ECG loaded successfully (${data.fs}Hz, ${data.channel_names.length} channels). ` +
                `<small>Note: Abnormality detection only works with 100Hz PTB-XL signals.</small>`,
                'alert-info'
            );
        }, 800);
    }

    console.log('[ECG SETUP] Detection container display:', detectionContainer.style.display);
    console.log('[ECG SETUP] Detection button disabled:', detectBtn.disabled);
    console.log('[ECG SETUP] ==========================================\n');
}

    // Setup UI Controls
   // Replace the setupControls function in ecg_logic.js with this improved version

    // NEW: Detection Button Handler
    if (detectBtn) {
        detectBtn.onclick = async function() {
            if (!globalECGData) {
                updateStatus('⚠️ No ECG data loaded.', 'alert-warning');
                return;
            }

            detectBtn.disabled = true;
            detectBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            detectionResults.style.display = 'none';

            try {
                const csrftoken = getCookie('csrftoken');

                const response = await fetch('/api/detect_ecg/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': csrftoken
                    },
                    body: JSON.stringify({
                        signals: globalECGData.signals,
                        fs: globalECGData.fs,
                        channel_names: globalECGData.channel_names
                    })
                });

                const result = await response.json();

                if (!response.ok) {
                    throw new Error(result.error || 'Detection failed');
                }

                displayDetectionResults(result);

            } catch (error) {
                console.error('[ECG Detection Error]', error);
                updateStatus(`❌ Detection failed: ${error.message}`, 'alert-danger');
                detectionResults.style.display = 'none';
            } finally {
                detectBtn.disabled = false;
                detectBtn.innerHTML = '<i class="fas fa-heartbeat"></i> Detect Abnormality';
            }
        };
    }

    // NEW: Display Detection Results
    function displayDetectionResults(result) {
        const prediction = result.prediction;
        const isNormal = prediction.is_normal;

        const resultClass = isNormal ? 'detection-result-normal' : 'detection-result-abnormal';
        const statusIcon = isNormal ? '✅' : '⚠️';
        const statusText = isNormal ? 'NORMAL' : 'ABNORMALITY DETECTED';
        const statusColor = isNormal ? '#28a745' : '#dc3545';

        // Build probability bars
        let probabilityHTML = '';
        for (const [className, prob] of Object.entries(prediction.probabilities)) {
            const percentage = (prob * 100).toFixed(1);
            const isHighlighted = className === prediction.predicted_class;

            probabilityHTML += `
                <div class="mb-2">
                    <div class="probability-bar">
                        <span class="probability-label">${className}</span>
                        <div class="probability-fill" style="width: ${percentage}%; ${isHighlighted ? 'background: linear-gradient(90deg, #28a745, #20c997);' : ''}">
                            <span style="color: white; font-size: 12px; font-weight: 600;">${percentage}%</span>
                        </div>
                    </div>
                </div>
            `;
        }

        detectionContent.innerHTML = `
            <div class="${resultClass}">
                <h4 style="color: ${statusColor}; margin-bottom: 15px;">
                    ${statusIcon} ${statusText}
                </h4>
                <div style="margin-bottom: 15px;">
                    <strong style="color: #c9d1d9;">Classification:</strong> 
                    <span style="color: ${statusColor}; font-weight: 600; font-size: 1.1em;">
                        ${prediction.predicted_class}
                    </span>
                </div>
                <div style="margin-bottom: 15px;">
                    <strong style="color: #c9d1d9;">Description:</strong> 
                    <span style="color: #8b949e;">${prediction.description}</span>
                </div>
                <div style="margin-bottom: 15px;">
                    <strong style="color: #c9d1d9;">Confidence:</strong> 
                    <span style="color: #58a6ff; font-weight: 600;">
                        ${(prediction.confidence * 100).toFixed(1)}%
                    </span>
                </div>
            </div>

            <div style="margin-top: 20px;">
                <h6 style="color: #c9d1d9; margin-bottom: 10px;">
                    <i class="fas fa-chart-bar"></i> Class Probabilities
                </h6>
                ${probabilityHTML}
            </div>

            <div style="margin-top: 15px; padding: 10px; background-color: #0d1117; border-radius: 5px;">
                <small style="color: #8b949e;">
                    <strong>Signal Info:</strong> 
                    ${result.signal_info.num_channels} channels, 
                    ${result.signal_info.fs}Hz, 
                    ${result.signal_info.duration_sec.toFixed(2)}s duration
                </small>
            </div>
        `;

        detectionResults.style.display = 'block';
        updateStatus(`✅ Detection complete: ${prediction.predicted_class}`, isNormal ? 'alert-success' : 'alert-warning');
    }

    // Core Streaming Controls
    function resetSignal() {
        if (!globalECGData) return;

        if (streamInterval) clearInterval(streamInterval);

        currentSignalIndex = 0;

        playSignal();

        updateStatus(`Signal reset and streaming with Fs=${nyquistSlider.value} Hz applied.`, 'alert-success');
    }

    playBtn.onclick = playSignal;
    pauseBtn.onclick = pauseSignal;

    function playSignal() {
        if (streamInterval) clearInterval(streamInterval);
        updateStatus("Playing signal stream...", 'alert-success');
        streamInterval = setInterval(updatePlots, INTERVAL_MS);
        updatePlots(false);
    }

    function pauseSignal() {
        if (streamInterval) {
            clearInterval(streamInterval);
            streamInterval = null;
            updateStatus("Signal stream paused.", 'alert-warning');
        }
    }

    function initializePlot(data) {
        const baseLayout = {
            plot_bgcolor: '#0d1117',
            paper_bgcolor: '#161b22',
            font: { color: '#c9d1d9' },
            margin: { t: 40, r: 10, b: 50, l: 50 },
            xaxis: {},
            yaxis: {}
        };

        Plotly.newPlot(plotContainer, [], {...baseLayout, title: 'Loading...', height: 450 });
    }

    // Optimization Helper Function
    function getResampledDataForChannel(chIndex, currentWindowSec) {
        const { fs, signals } = globalECGData;
        const currentSimulatedFs = parseFloat(nyquistSlider.value);
        const s_idx = Math.floor(currentSignalIndex * fs);
        const e_idx = Math.floor((currentSignalIndex + currentWindowSec) * fs);

        const getRawSignalWindow = (chIndex) => signals[chIndex].slice(s_idx, e_idx);
        const getFilteredSignal = (chIndex) => applyLowPassFilter(getRawSignalWindow(chIndex), fs, currentSimulatedFs);
        return resampleSignal(getFilteredSignal(chIndex), fs, currentSimulatedFs);
    }

    // Main Plotting Logic
    function updatePlots(advanceTime = true) {
        if (!globalECGData) return;

        const { fs, duration, channel_names } = globalECGData;
        const primaryChOptions = primaryChSelect.options;
        const selectedChIndices = Array.from(primaryChOptions).filter(opt => opt.selected).map(opt => parseInt(opt.value));

        const primaryChannelForPair = selectedChIndices.length > 0 ? selectedChIndices[0] : 0;
        const pairChIndex = parseInt(pairChSelect.value);

        if (selectedChIndices.length === 0) {
             updateStatus("Please select at least one channel to plot.", 'alert-info');
             return;
        }

        const currentWindowSec = parseFloat(timeSlider.value);
        const simulatedFs = parseFloat(nyquistSlider.value);

        let s_sec = currentSignalIndex;
        let e_sec = currentSignalIndex + currentWindowSec;

        if (e_sec >= duration) {
            e_sec = duration;
            if (s_sec >= duration - STEP_SEC) {
                 currentSignalIndex = 0;
                 s_sec = 0;
                 e_sec = currentWindowSec;
            }
        }

        const sigA_resampled = getResampledDataForChannel(primaryChannelForPair, currentWindowSec);
        const sigB_resampled = getResampledDataForChannel(pairChIndex, currentWindowSec);

        const numSamples = sigA_resampled.length;
        const timeAxis_resampled = Array.from({ length: numSamples }, (_, i) => s_sec + (i / simulatedFs));

        if (advanceTime) {
            const nyquistLimit = simulatedFs / 2;
            console.log(`\n--- ECG Plotting Window ---`);
            console.log(`Effective Fs: ${simulatedFs.toFixed(0)} Hz`);
            console.log(`Window Time: ${currentWindowSec.toFixed(1)} sec`);
            console.log(`Samples Plotted: ${numSamples}`);
            console.log(`Nyquist Limit: ${nyquistLimit.toFixed(1)} Hz`);
        }

        if (linearTimeWindow) linearTimeWindow.textContent = `${s_sec.toFixed(2)}-${e_sec.toFixed(2)}s`;

        let traces = [];
        let layoutUpdates = {};
        let plotTitle = '';

        // Visualization Logic
        if (currentVisualizationMode === 'linear') {
            plotTitle = `1. Linear Waveform (Fs: ${simulatedFs.toFixed(0)} Hz, Samples: ${numSamples})`;

            traces = selectedChIndices.map(chIndex => ({
                x: timeAxis_resampled,
                y: getResampledDataForChannel(chIndex, currentWindowSec),
                mode: 'lines',
                name: channel_names[chIndex],
                type: 'scatter'
            }));
            layoutUpdates = { xaxis: { title: 'Time (s)', color: '#c9d1d9', gridcolor: '#30363d' }, yaxis: { title: 'Amplitude (mV)', color: '#c9d1d9', gridcolor: '#30363d' }, showlegend: true };
            if (plotTitleH5) plotTitleH5.innerHTML = `1. Linear Waveform (<span id="linear-time-window">${s_sec.toFixed(2)}-${e_sec.toFixed(2)}s</span>)`;

        } else if (currentVisualizationMode === 'xor') {
            plotTitle = `2. XOR Detection (Samples: ${numSamples})`;
            const hits = xorDetectionPoints(sigA_resampled, sigB_resampled, simulatedFs, timeAxis_resampled);

            traces = [
                { x: timeAxis_resampled, y: sigA_resampled, mode: 'lines', name: channel_names[primaryChannelForPair] + ' (A)', line: { color: '#58a6ff', width: 1 } },
                { x: hits.map(h => h.x), y: hits.map(h => h.y), mode: 'markers', name: `XOR Hits (${hits.length})`, marker: { color: 'red', size: 8 } }
            ];
            layoutUpdates = { xaxis: { title: 'Time (s)', color: '#c9d1d9', gridcolor: '#30363d' }, yaxis: { title: 'Amplitude (mV)' }, showlegend: true };
            if (plotTitleH5) plotTitleH5.textContent = plotTitle;

        } else if (currentVisualizationMode === 'polar') {
            plotTitle = `3. Polar View (Samples: ${numSamples})`;

            traces = selectedChIndices.map(chIndex => {
                const currentSig = getResampledDataForChannel(chIndex, currentWindowSec);

                const mean = currentSig.reduce((a, b) => a + b, 0) / currentSig.length;
                const r_values = currentSig.map(val => val - mean);
                const theta_values = Array.from({ length: numSamples }, (_, i) => (i / numSamples) * 360);

                return { r: r_values, theta: theta_values, mode: 'lines', type: 'scatterpolar', name: channel_names[chIndex], line: {width: 2} };
            });
            layoutUpdates = { polar: { angularaxis: { rotation: 90, direction: "clockwise" }, radialaxis: { autorange: true } }, xaxis: {}, yaxis: {}, showlegend: true };
            if (plotTitleH5) plotTitleH5.textContent = plotTitle;

        } else if (currentVisualizationMode === 'recurrence') {
            plotTitle = `4. Recurrence Heatmap (Samples: ${numSamples})`;
            const recurrenceMatrix = computeRecurrenceMatrix(sigA_resampled, sigB_resampled);
            traces = [{ z: recurrenceMatrix, type: 'heatmap', colorscale: 'Viridis', x: Array.from({length: 60}, (_, i) => -3 + i * (6/60) + (6/120)), y: Array.from({length: 60}, (_, i) => -3 + i * (6/60) + (6/120)), colorbar: { title: 'Counts', tickfont: { color: '#c9d1d9' } } }];
            layoutUpdates = { xaxis: { title: 'Channel A Normalized Amplitude', color: '#c9d1d9', gridcolor: '#30363d' }, yaxis: { title: 'Channel B Normalized Amplitude', color: '#c9d1d9', gridcolor: '#30363d' }, showlegend: false };
            if (plotTitleH5) plotTitleH5.textContent = plotTitle;
        }

        // Render Plot
        const finalLayout = {
            title: plotTitle,
            plot_bgcolor: '#0d1117',
            paper_bgcolor: '#161b22',
            font: { color: '#c9d1d9' },
            margin: { t: 40, r: 10, b: 50, l: 50 },
            ...layoutUpdates
        };

        Plotly.react(plotContainer, traces, finalLayout);

        if (advanceTime) {
            currentSignalIndex += STEP_SEC;
        }
    }

// Add this debug function to your ecg_logic.js (at the very end, before the closing });)
// You can run this in the browser console after loading a file

function debugDetectionButton() {
    console.log('\n========== ECG DETECTION BUTTON DEBUG ==========');

    // Check if elements exist
    const detectionContainer = document.getElementById('ecg-detection-container');
    const detectBtn = document.getElementById('ecg-btn-detect');
    const detectionResults = document.getElementById('ecg-detection-results');

    console.log('1. DOM Elements:');
    console.log('   - Detection Container:', detectionContainer ? '✅ Found' : '❌ NOT FOUND');
    console.log('   - Detection Button:', detectBtn ? '✅ Found' : '❌ NOT FOUND');
    console.log('   - Results Container:', detectionResults ? '✅ Found' : '❌ NOT FOUND');

    if (detectionContainer) {
        console.log('\n2. Container Styles:');
        console.log('   - display:', detectionContainer.style.display);
        console.log('   - visibility:', detectionContainer.style.visibility || 'not set');
        console.log('   - opacity:', detectionContainer.style.opacity || 'not set');

        const computed = window.getComputedStyle(detectionContainer);
        console.log('\n3. Computed Styles:');
        console.log('   - display:', computed.display);
        console.log('   - visibility:', computed.visibility);
        console.log('   - opacity:', computed.opacity);
        console.log('   - height:', computed.height);

        console.log('\n4. Element Metrics:');
        console.log('   - offsetHeight:', detectionContainer.offsetHeight, '(0 = hidden)');
        console.log('   - offsetWidth:', detectionContainer.offsetWidth, '(0 = hidden)');
    }

    if (detectBtn) {
        console.log('\n5. Button State:');
        console.log('   - disabled:', detectBtn.disabled);
        console.log('   - display:', detectBtn.style.display || 'not set');
    }

    if (globalECGData) {
        console.log('\n6. Loaded ECG Data:');
        console.log('   - Filename:', globalECGData.filename);
        console.log('   - Sampling Frequency:', globalECGData.fs, 'Hz');
        console.log('   - Is 100Hz?:', globalECGData.fs === 100 ? '✅ YES' : '❌ NO');
        console.log('   - Channels:', globalECGData.channel_names.length);
        console.log('   - Duration:', globalECGData.duration, 'seconds');
    } else {
        console.log('\n6. Loaded ECG Data: ❌ No data loaded');
    }

    console.log('\n7. Expected Behavior:');
    if (globalECGData && globalECGData.fs === 100) {
        console.log('   ✅ Button SHOULD be visible (100Hz signal)');
        if (detectionContainer && detectionContainer.style.display === 'none') {
            console.error('   ❌ ERROR: Button is hidden but should be visible!');
        }
    } else if (globalECGData) {
        console.log(`   ⚠️ Button SHOULD be hidden (${globalECGData.fs}Hz signal, needs 100Hz)`);
        if (detectionContainer && detectionContainer.style.display !== 'none') {
            console.error('   ❌ ERROR: Button is visible but should be hidden!');
        }
    }

    console.log('===============================================\n');
}

// Also add this to window so you can call it from console
window.debugDetectionButton = debugDetectionButton;

console.log('[ECG] Debug function loaded. Run debugDetectionButton() in console after loading a file.');
});