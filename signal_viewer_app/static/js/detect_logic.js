document.addEventListener('DOMContentLoaded', function() {
    // === DOM ELEMENT REFERENCES ===
    const dropZone = document.getElementById('drop-zone-detect');
    const fileInput = document.getElementById('detect-file-input');
    const analyzeButton = document.getElementById('analyze-audio-button');
    const playButton = document.getElementById('play-audio-button');
    const statusDiv = document.getElementById('detect-status');
    const playerOriginalDiv = document.getElementById('audio-player-original');
    const playerResampledDiv = document.getElementById('audio-player-resampled');
    const timeDomainOriginalDiv = document.getElementById('time-domain-graph-original');
    const timeDomainResampledDiv = document.getElementById('time-domain-graph-resampled');
    const resampledGraphContainer = document.getElementById('resampled-graph-container');
    const predictionDiv = document.getElementById('audio-prediction-output');
    const probabilitiesDiv = document.getElementById('audio-probabilities-graph');
    const debugDiv = document.getElementById('audio-debug-info');
    const nyquistSection = document.getElementById('nyquist-section');
    const resampleSlider = document.getElementById('resample-slider');
    const currentSrDisplay = document.getElementById('current-sr-display');
    const playResampledBtn = document.getElementById('play-resampled-btn');

    // === STATE VARIABLES ===
    let uploadedFile = null;              // User's uploaded audio file
    let audioDataURI = null;              // Base64 data URI of the audio
    let predictionData = null;            // YAMNet classification results
    let originalAudioData = null;         // Full audio samples from analysis
    let currentSampleRate = 16000;        // Current sample rate after processing
    let originalSampleRate = 16000;       // Original file's sample rate
    let originalMaxFrequency = 0;         // Maximum frequency present in audio
    let nyquistFrequencyRequired = 0;     // Minimum required sample rate (2×max_freq)
    let resampledAudioData = null;        // Downsampled audio for aliasing demonstration
    let sliderTimeout = null;             // Debounce timer for slider interactions
    let originalDuration = 0;             // Audio duration in seconds

    // Limit plot points for rendering performance
    const MAX_PLOT_POINTS = 2000;

    // === HELPER FUNCTIONS ===
    
    // Extract CSRF token from cookies for Django POST requests
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

    // Update status message display
    function updateStatus(message, alertClass = 'alert-info') {
        statusDiv.className = `alert ${alertClass}`;
        statusDiv.innerHTML = message;
    }

    // Read file as base64 data URI for transmission to server
    function readFileAsDataURI(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error);
            reader.readAsDataURL(file);
        });
    }

    // === FILE HANDLING ===
    
    // Validate and process uploaded audio file
    async function handleFile(file) {
        const fileName = file.name.toLowerCase();
        
        // Validate file type
        if (!fileName.endsWith('.wav') && !fileName.endsWith('.mp3')) {
            updateStatus("⚠️ Invalid file type. Must be .wav or .mp3.", 'alert-danger');
            analyzeButton.disabled = true;
            playButton.disabled = true;
            return;
        }

        // Warn for large files (server limits to 30 seconds)
        const fileSizeMB = file.size / (1024 * 1024);
        if (fileSizeMB > 10) {
            updateStatus(`⚠️ Warning: Large file (${fileSizeMB.toFixed(1)}MB). Audio will be limited to 30 seconds for analysis.`, 'alert-warning');
        }

        uploadedFile = file;
        updateStatus(`File selected: ${file.name} (${fileSizeMB.toFixed(1)}MB). Reading audio data...`, 'alert-warning');

        try {
            // Convert file to base64 for server transmission
            audioDataURI = await readFileAsDataURI(file);
            updateStatus(`Audio file loaded (${file.name}). Ready for analysis.`, 'alert-info');
            analyzeButton.disabled = false;
            playButton.disabled = false;

            // Reset previous analysis state for new file
            nyquistSection.style.display = 'none';
            resampledGraphContainer.style.display = 'none';
            playerResampledDiv.style.display = 'none';
            originalMaxFrequency = 0;
            nyquistFrequencyRequired = 0;
            originalAudioData = null;
            resampledAudioData = null;
            predictionData = null;
            originalDuration = 0;

        } catch (error) {
            updateStatus(`❌ Error reading file: ${error.message}`, 'alert-danger');
            analyzeButton.disabled = true;
            playButton.disabled = true;
        }
    }

    // === EVENT LISTENERS ===
    
    // Click drop zone to trigger file input
    if (dropZone && fileInput) {
        dropZone.addEventListener('click', (e) => {
            if (e.target === dropZone || e.target.tagName !== 'INPUT') {
                fileInput.click();
            }
        });
    }

    // Drag-and-drop file handling
    if (dropZone) {
        dropZone.ondragover = (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '#00B8A9';
        };

        dropZone.ondragleave = () => {
            dropZone.style.borderColor = '#30363d';
        };

        dropZone.ondrop = (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '#30363d';
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        };
    }

    // File input change handler
    if (fileInput) {
        fileInput.onchange = (e) => {
            if (e.target.files.length) {
                handleFile(e.target.files[0]);
            }
            fileInput.value = ''; // Reset to allow same file re-upload
        };
    }

    // Play original audio button
    playButton.onclick = () => {
        if (!audioDataURI) return;
        playerOriginalDiv.innerHTML = `
            <h5 style="color: #00B8A9;">Original Audio Player:</h5>
            <audio controls autoplay src="${audioDataURI}" style="width: 100%;"></audio>
        `;
    };

    // Analyze audio button - triggers YAMNet classification
    analyzeButton.onclick = async () => {
        await performAnalysis(originalSampleRate);
    };

    // === AUDIO PROCESSING FUNCTIONS ===
    
    // Resample audio while maintaining the same playback duration
    // This demonstrates aliasing by taking fewer samples at a lower rate
    function resampleAudioByDuration(audioData, currentRate, targetRate, duration) {
        // Calculate target sample count: duration (seconds) × sample rate (Hz)
        const targetSamples = Math.floor(duration * targetRate);
        const resampled = new Float32Array(targetSamples);

        // Linear interpolation between samples to maintain audio quality
        for (let i = 0; i < targetSamples; i++) {
            const timePosition = i / targetRate;                    // Time of this sample
            const sourceIndex = timePosition * currentRate;         // Corresponding position in source
            
            const index0 = Math.floor(sourceIndex);
            const index1 = Math.min(index0 + 1, audioData.length - 1);
            const fraction = sourceIndex - index0;

            if (index0 < audioData.length) {
                // Interpolate between adjacent samples
                resampled[i] = audioData[index0] * (1 - fraction) + audioData[index1] * fraction;
            }
        }

        return Array.from(resampled);
    }

    // Reduce data points for plotting performance (keep every Nth sample)
    function downsampleForPlotting(data, maxPoints) {
        if (data.length <= maxPoints) {
            return data;
        }

        const factor = Math.ceil(data.length / maxPoints);
        const downsampled = [];

        for (let i = 0; i < data.length; i += factor) {
            downsampled.push(data[i]);
        }

        return downsampled;
    }

    // Convert Float32 audio array to WAV file format
    function audioArrayToWav(audioData, sampleRate) {
        const numChannels = 1;
        const bitsPerSample = 16;
        const bytesPerSample = bitsPerSample / 8;
        const blockAlign = numChannels * bytesPerSample;

        const dataLength = audioData.length * bytesPerSample;
        const buffer = new ArrayBuffer(44 + dataLength);  // 44-byte WAV header
        const view = new DataView(buffer);

        // Helper to write ASCII strings into buffer
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        // Write WAV file header
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);                      // Subchunk1 size (PCM)
        view.setUint16(20, 1, true);                       // Audio format (1 = PCM)
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true); // Byte rate
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);
        writeString(36, 'data');
        view.setUint32(40, dataLength, true);

        // Write audio samples (convert float [-1,1] to int16)
        let offset = 44;
        for (let i = 0; i < audioData.length; i++) {
            const sample = Math.max(-1, Math.min(1, audioData[i]));
            view.setInt16(offset, sample * 0x7FFF, true);
            offset += 2;
        }

        return buffer;
    }

    // === ANALYSIS FUNCTION ===
    
    // Send audio to server for YAMNet classification and receive results
    async function performAnalysis(targetSampleRate) {
        if (!uploadedFile || !audioDataURI) {
            updateStatus("⚠️ No audio file loaded.", 'alert-warning');
            return;
        }

        analyzeButton.disabled = true;
        updateStatus("Sending audio for YAMNet analysis on server...", 'alert-warning');

        await new Promise(resolve => setTimeout(resolve, 50));

        const csrftoken = getCookie('csrftoken');

        try {
            const requestBody = JSON.stringify({
                audio_data: audioDataURI,
                filename: uploadedFile.name,
                target_sample_rate: targetSampleRate
            });

            // POST audio to Django backend
            const response = await fetch('/api/analyze_audio/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrftoken
                },
                body: requestBody,
            });

            // Handle server errors
            if (!response.ok) {
                const contentType = response.headers.get('content-type');
                let errorMessage;

                if (contentType && contentType.includes('application/json')) {
                    const errorJson = await response.json();
                    errorMessage = errorJson.error || JSON.stringify(errorJson);
                } else {
                    const errorText = await response.text();
                    errorMessage = errorText.substring(0, 500);
                }

                if (response.status === 400) {
                    throw new Error(`Audio file may be too large or too long. Try a shorter clip (max 30 seconds recommended). Details: ${errorMessage}`);
                }

                throw new Error(`Server status ${response.status}. Error: ${errorMessage}`);
            }

            const json_data = await response.json();

            if (json_data.error) {
                throw new Error(json_data.error);
            }

            // Store analysis results
            predictionData = json_data;
            originalAudioData = json_data.waveform;
            currentSampleRate = json_data.sr;
            originalSampleRate = json_data.original_sr;
            originalMaxFrequency = json_data.max_frequency;
            nyquistFrequencyRequired = 2 * originalMaxFrequency;
            resampledAudioData = json_data.waveform;
            originalDuration = originalAudioData.length / currentSampleRate;

            renderResults(json_data);

        } catch (error) {
            console.error('Analysis Error:', error);
            updateStatus(`❌ Analysis Failed. Error: ${error.message}`, 'alert-danger');
        } finally {
            analyzeButton.disabled = false;
        }
    }

    // === VISUALIZATION FUNCTIONS ===
    
    // Plot audio waveform using Plotly (time domain visualization)
    function plotTimeDomain(waveform, sr, divElement, title, showMarkers = false) {
        // Downsample for efficient rendering
        const plotData = downsampleForPlotting(waveform, MAX_PLOT_POINTS);
        const downsampleFactor = Math.ceil(waveform.length / plotData.length);
        const timeAxis = Array.from({ length: plotData.length }, (_, i) => (i * downsampleFactor) / sr);

        const traceConfig = {
            x: timeAxis,
            y: plotData,
            mode: showMarkers ? 'lines+markers' : 'lines',
            line: { color: '#00B8A9', width: 1 },
            name: 'Amplitude'
        };

        // Add markers for low sample rate visualization
        if (showMarkers && plotData.length < 500) {
            traceConfig.marker = {
                size: 4,
                color: '#00B8A9',
                symbol: 'circle'
            };
        }

        const timeDomainLayout = {
            title: title,
            xaxis: {
                title: 'Time (s)',
                color: '#c9d1d9',
                gridcolor: '#30363d'
            },
            yaxis: {
                title: 'Amplitude',
                color: '#c9d1d9',
                gridcolor: '#30363d',
                range: [-1.1, 1.1]
            },
            plot_bgcolor: '#161b22',
            paper_bgcolor: '#161b22',
            font: { color: '#c9d1d9' },
            height: 350,
            hovermode: 'closest',
            margin: { l: 50, r: 30, t: 50, b: 50 }
        };

        Plotly.newPlot(divElement, [traceConfig], timeDomainLayout, {responsive: true});
    }

    // Update Nyquist sampling theorem information display
    function updateNyquistAnalysis(currentRate) {
        document.getElementById('original-sr').textContent = `${originalSampleRate} Hz`;
        document.getElementById('current-sr').textContent = `${currentRate} Hz`;
        document.getElementById('max-freq').textContent = `${originalMaxFrequency.toFixed(0)} Hz`;
        document.getElementById('nyquist-freq').textContent = `${nyquistFrequencyRequired.toFixed(0)} Hz`;

        const statusElement = document.getElementById('sampling-status');

        // Check if current rate satisfies Nyquist criterion (fs >= 2×fmax)
        if (currentRate < nyquistFrequencyRequired) {
            statusElement.textContent = `⚠️ Under-sampling (Current rate < ${nyquistFrequencyRequired.toFixed(0)} Hz)`;
            statusElement.style.color = '#f78166';
        } else {
            statusElement.textContent = `✓ Over-sampling (Current rate ≥ ${nyquistFrequencyRequired.toFixed(0)} Hz)`;
            statusElement.style.color = '#00B8A9';
        }

        nyquistSection.style.display = 'block';
    }

    // Display YAMNet classification results (prediction + probabilities)
    function renderResults(data) {
        const predictionColor = (data.predicted_class === 'Drone') ? '#00B8A9' : '#FFA500';

        predictionDiv.innerHTML = `
            <h4 style="color: ${predictionColor}; font-weight: bold;">
                🧠 Prediction: ${data.predicted_class}
            </h4>
        `;

        // Plot probability distribution bar chart
        const barTrace = {
            x: data.class_names,
            y: data.probabilities.map(p => p * 100),
            type: 'bar',
            marker: {
                color: ['#00B8A9', '#F6416C', '#FFA500']
            }
        };

        const barLayout = {
            title: "Prediction Probabilities (%)",
            xaxis: { title: 'Class', color: '#c9d1d9', gridcolor: '#30363d' },
            yaxis: { title: 'Probability (%)', range: [0, 100], color: '#c9d1d9', gridcolor: '#30363d' },
            plot_bgcolor: '#161b22',
            paper_bgcolor: '#161b22',
            font: { color: '#c9d1d9' },
            height: 300,
            margin: { l: 50, r: 30, t: 50, b: 50 }
        };

        Plotly.newPlot(probabilitiesDiv, [barTrace], barLayout, {responsive: true});

        // Plot waveform without markers for performance
        plotTimeDomain(
            data.waveform,
            data.sr,
            timeDomainOriginalDiv,
            `Original Audio: Time Domain (${data.waveform.length} samples at ${data.sr} Hz, Duration: ${originalDuration.toFixed(2)}s)`,
            false
        );

        updateNyquistAnalysis(data.sr);

        // Display debug information
        debugDiv.innerHTML = `<pre>File: ${data.filename}
Original Sample Rate: ${originalSampleRate} Hz
Current Sample Rate: ${data.sr} Hz
Duration: ${originalDuration.toFixed(2)} seconds
Original Max Frequency: ${originalMaxFrequency.toFixed(2)} Hz
Nyquist Frequency Required (2×F(max)): ${nyquistFrequencyRequired.toFixed(2)} Hz
Predicted: ${data.predicted_class}
Probabilities: ${data.probabilities.map(p => p.toFixed(4)).join(', ')}
Sampling Status: ${data.sr >= nyquistFrequencyRequired ? 'Over-sampling' : 'Under-sampling'}</pre>`;

        updateStatus(`✅ Analysis complete for ${data.filename}.`, 'alert-success');
    }

    // === RESAMPLING SLIDER INTERACTION ===
    
    // Slider handler with debouncing for smooth performance
    resampleSlider.oninput = (e) => {
        const newSampleRate = parseInt(e.target.value);
        currentSrDisplay.textContent = `${newSampleRate} Hz`;

        // Clear previous debounce timer
        if (sliderTimeout) {
            clearTimeout(sliderTimeout);
        }

        // Update status immediately (without waiting for debounce)
        document.getElementById('current-sr').textContent = `${newSampleRate} Hz`;
        const statusElement = document.getElementById('sampling-status');

        if (newSampleRate < nyquistFrequencyRequired) {
            statusElement.textContent = `⚠️ Under-sampling (Current rate < ${nyquistFrequencyRequired.toFixed(0)} Hz)`;
            statusElement.style.color = '#f78166';
        } else {
            statusElement.textContent = `✓ Over-sampling (Current rate ≥ ${nyquistFrequencyRequired.toFixed(0)} Hz)`;
            statusElement.style.color = '#00B8A9';
        }

        // Debounce expensive operations (resampling + plotting)
        sliderTimeout = setTimeout(() => {
            if (originalAudioData && originalDuration > 0) {
                // Resample audio to new rate while maintaining duration
                resampledAudioData = resampleAudioByDuration(
                    originalAudioData,
                    currentSampleRate,
                    newSampleRate,
                    originalDuration
                );

                resampledGraphContainer.style.display = 'block';

                const resampledDuration = resampledAudioData.length / newSampleRate;

                // Show sample markers for low rates to visualize aliasing
                const showMarkers = newSampleRate < 4000;

                plotTimeDomain(
                    resampledAudioData,
                    newSampleRate,
                    timeDomainResampledDiv,
                    `Resampled Audio: Time Domain (${resampledAudioData.length} samples at ${newSampleRate} Hz, Duration: ${resampledDuration.toFixed(2)}s)`,
                    showMarkers
                );
            }
        }, 150); // 150ms debounce delay
    };

    // Play resampled audio button - create WAV from downsampled data
    playResampledBtn.onclick = () => {
        if (!resampledAudioData) {
            updateStatus("⚠️ No resampled audio available.", 'alert-warning');
            return;
        }

        const currentRate = parseInt(resampleSlider.value);
        const resampledDuration = resampledAudioData.length / currentRate;

        // Convert audio array to WAV file and create playable URL
        const wavBuffer = audioArrayToWav(resampledAudioData, currentRate);
        const blob = new Blob([wavBuffer], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);

        playerResampledDiv.style.display = 'block';
        playerResampledDiv.innerHTML = `
            <h5 style="color: #f78166;">Resampled Audio Player (${currentRate} Hz, Duration: ${resampledDuration.toFixed(2)}s):</h5>
            <audio controls autoplay src="${url}" style="width: 100%;"></audio>
        `;

        updateStatus(`Playing audio resampled at ${currentRate} Hz (duration: ${resampledDuration.toFixed(2)}s, original: ${originalDuration.toFixed(2)}s)`, 'alert-info');
    };
});