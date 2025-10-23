document.addEventListener('DOMContentLoaded', function() {
    // --- Renamed Element IDs ---
    const dropZone = document.getElementById('drop-zone-eeg');
    const fileInput = document.getElementById('eeg-file-input');
    const statusDiv = document.getElementById('eeg-status');
    const controlsDiv = document.getElementById('eeg-controls');
    const primaryChSelect = document.getElementById('eeg-primary-ch');
    const pairChSelect = document.getElementById('eeg-pair-ch');
    const playBtn = document.getElementById('eeg-btn-play');
    const pauseBtn = document.getElementById('eeg-btn-pause');
    const resetBtn = document.getElementById('eeg-btn-reset');

    const timeSliderContainer = document.getElementById('eeg-time-slider-container');
    const timeSlider = document.getElementById('eeg-time-slider');
    const windowSizeOutput = document.getElementById('window-size-output-eeg');

    const nyquistSliderContainer = document.getElementById('eeg-nyquist-slider-container');
    const nyquistSlider = document.getElementById('eeg-nyquist-slider');
    const nyquistFsOutput = document.getElementById('nyquist-fs-output-eeg');
    const maxFreqOutput = document.getElementById('max-freq-output-eeg');

    const modeButtonsDiv = document.getElementById('eeg-mode-buttons');
    const plotContainer = document.getElementById('eeg-single-plot');
    const plotTitleH5 = document.getElementById('current-plot-title-eeg');
    const linearTimeWindow = document.getElementById('linear-time-window-eeg');


    let globalEEGData = null;
    let streamInterval = null;
    let currentSignalIndex = 0;

    let currentVisualizationMode = 'linear';

    const STEP_SEC = 0.2;
    const INTERVAL_MS = STEP_SEC * 1000;

    // Helper to get CSRF token (required for Django POST requests)
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

    // --- Signal Processing Utilities (Unchanged) ---

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

    // --- Drag and Drop and API Call Handlers (Unchanged) ---
    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => { fileInput.click(); });
    }

    if (dropZone) {
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.style.borderColor = '#00B8A9'; };
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

    function handleFiles(files) {
        if (files.length !== 1 || !files[0].name.toLowerCase().endsWith('.set')) {
            updateStatus("⚠️ Please drag and drop a single .set file.", 'alert-danger');
            return;
        }

        const setFile = files[0];

        updateStatus(`Uploading ${setFile.name} for server conversion (MNE)...`, 'alert-warning');

        const formData = new FormData();
        formData.append('file', setFile);

        const csrftoken = getCookie('csrftoken');

        fetch('/api/convert_eeg/', {
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

            globalEEGData = data;

            updateStatus(`✅ EEG Record loaded. FS: ${data.fs} Hz, Channels: ${data.channel_names.length}.`, 'alert-success');
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

    // --- Setup UI Controls and Streaming Controls ---
    function setupControls(data) {
        primaryChSelect.innerHTML = '';
        pairChSelect.innerHTML = '';
        data.channel_names.forEach((name, index) => {
            const option = `<option value="${index}">${name}</option>`;
            primaryChSelect.innerHTML += option;
            pairChSelect.innerHTML += option;
        });

        // --- FIX: Default to selecting ALL channels ---
        for (let i = 0; i < primaryChSelect.options.length; i++) {
            primaryChSelect.options[i].selected = true;
        }

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

        // --- Nyquist Slider Controls ---
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
    }

    // --- Core Streaming Controls ---

    function resetSignal() {
        if (!globalEEGData) return;

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

    // --- Optimization Helper Function ---
    function getResampledDataForChannel(chIndex, currentWindowSec) {
        const { fs, signals } = globalEEGData;
        const currentSimulatedFs = parseFloat(nyquistSlider.value);
        const s_idx = Math.floor(currentSignalIndex * fs);
        const e_idx = Math.floor((currentSignalIndex + currentWindowSec) * fs);

        const getRawSignalWindow = (chIndex) => signals[chIndex].slice(s_idx, e_idx);
        const getFilteredSignal = (chIndex) => applyLowPassFilter(getRawSignalWindow(chIndex), fs, currentSimulatedFs);
        return resampleSignal(getFilteredSignal(chIndex), fs, currentSimulatedFs);
    }


    // --- Main Plotting Logic (Updated Y-Axis) ---
    function updatePlots(advanceTime = true) {
        if (!globalEEGData) return;

        const { fs, duration, channel_names } = globalEEGData;
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


        // --- Console Output (Unchanged) ---
        if (advanceTime) {
            const nyquistLimit = simulatedFs / 2;
            console.log(`\n--- EEG Plotting Window ---`);
            console.log(`Effective Fs: ${simulatedFs.toFixed(0)} Hz`);
            console.log(`Window Time: ${currentWindowSec.toFixed(1)} sec`);
            console.log(`Samples Plotted: ${numSamples}`);
            console.log(`Nyquist Limit: ${nyquistLimit.toFixed(1)} Hz`);
        }

        if (linearTimeWindow) linearTimeWindow.textContent = `${s_sec.toFixed(2)}-${e_sec.toFixed(2)}s`;

        let traces = [];
        let layoutUpdates = {};
        let plotTitle = '';

        // --- DYNAMIC VISUALIZATION LOGIC ---

        if (currentVisualizationMode === 'linear') {
            plotTitle = `1. Linear Waveform (Fs: ${simulatedFs.toFixed(0)} Hz, Samples: ${numSamples})`;

            traces = selectedChIndices.map(chIndex => ({
                x: timeAxis_resampled,
                y: getResampledDataForChannel(chIndex, currentWindowSec),
                mode: 'lines',
                name: channel_names[chIndex],
                type: 'scatter'
            }));
            // Y-Axis unit set to $\mu$V
            layoutUpdates = { xaxis: { title: 'Time (s)', color: '#c9d1d9', gridcolor: '#30363d' }, yaxis: { title: 'Amplitude ($\mu$V)', color: '#c9d1d9', gridcolor: '#30363d' }, showlegend: true };
            if (plotTitleH5) plotTitleH5.innerHTML = `1. Linear Waveform (<span id="linear-time-window-eeg">${s_sec.toFixed(2)}-${e_sec.toFixed(2)}s</span>)`;

        } else if (currentVisualizationMode === 'xor') {
            plotTitle = `2. XOR Detection (Samples: ${numSamples})`;
            const hits = xorDetectionPoints(sigA_resampled, sigB_resampled, simulatedFs, timeAxis_resampled);

            traces = [
                { x: timeAxis_resampled, y: sigA_resampled, mode: 'lines', name: channel_names[primaryChannelForPair] + ' (A)', line: { color: '#00B8A9', width: 1 } },
                { x: hits.map(h => h.x), y: hits.map(h => h.y), mode: 'markers', name: `XOR Hits (${hits.length})`, marker: { color: 'red', size: 8 } }
            ];
            // Y-Axis unit set to $\mu$V
            layoutUpdates = { xaxis: { title: 'Time (s)', color: '#c9d1d9', gridcolor: '#30363d' }, yaxis: { title: 'Amplitude ($\mu$V)' }, showlegend: true };
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

        // --- RENDER THE SELECTED PLOT ---
        const finalLayout = {
            title: plotTitle,
            plot_bgcolor: '#0d1117',
            paper_bgcolor: '#161b22',
            font: { color: '#c9d1d9' },
            margin: { t: 40, r: 10, b: 50, l: 50 },
            ...layoutUpdates
        };

        Plotly.react(plotContainer, traces, finalLayout);

        // Advance the stream only if button is "Play"
        if (advanceTime) {
            currentSignalIndex += STEP_SEC;
        }
    }
});