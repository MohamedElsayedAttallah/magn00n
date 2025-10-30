document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone-cars');
    const fileInput = document.getElementById('cars-file-input');
    const analyzeButton = document.getElementById('analyze-cars-button');
    const predictSpeedButton = document.getElementById('predict-speed-button');
    const playButton = document.getElementById('play-cars-button');
    const statusDiv = document.getElementById('cars-status');
    const playerOriginalDiv = document.getElementById('audio-player-cars-original');
    const playerResampledDiv = document.getElementById('audio-player-cars-resampled');
    const timeDomainOriginalDiv = document.getElementById('time-domain-graph-cars-original');
    const timeDomainResampledDiv = document.getElementById('time-domain-graph-cars-resampled');
    const frequencyDomainOriginalDiv = document.getElementById('frequency-domain-graph-cars');
    const frequencyDomainResampledDiv = document.getElementById('frequency-domain-graph-cars-resampled');
    const resampledGraphContainer = document.getElementById('cars-resampled-graph-container');
    const resampledFreqContainer = document.getElementById('cars-resampled-freq-container');
    const debugDiv = document.getElementById('cars-debug-info');
    const nyquistSection = document.getElementById('nyquist-section-cars');
    const resampleSlider = document.getElementById('cars-resample-slider');
    const currentSrDisplay = document.getElementById('cars-current-sr-display');
    const playResampledBtn = document.getElementById('play-cars-resampled-btn');
    const speedPredictionResult = document.getElementById('speed-prediction-result');
    const predictedSpeedValue = document.getElementById('predicted-speed-value');
    const checkpointName = document.getElementById('checkpoint-name');
    const checkpointVariance = document.getElementById('checkpoint-variance');
    const checkpointMae = document.getElementById('checkpoint-mae');

    // Downsampled Speed Prediction Elements
    const predictDownsampledSpeedBtn = document.getElementById('predict-downsampled-speed-btn');
    const downsampledSpeedResult = document.getElementById('downsampled-speed-result');
    const downsampledSpeedValue = document.getElementById('downsampled-speed-value');
    const downsampledSrDisplay = document.getElementById('downsampled-sr-display');

    let uploadedFile = null;
    let audioDataURI = null;
    let originalAudioData = null;
    let currentSampleRate = 16000;
    let originalSampleRate = 16000;
    let originalMaxFrequency = 0;
    let nyquistFrequencyRequired = 0;
    let resampledAudioData = null;
    let sliderTimeout = null;
    let originalDuration = 0;

    const MAX_PLOT_POINTS = 2000;

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

    function updateStatus(message, alertClass = 'alert-info') {
        statusDiv.className = `alert ${alertClass}`;
        statusDiv.innerHTML = message;
    }

    function readFileAsDataURI(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error);
            reader.readAsDataURL(file);
        });
    }

    async function handleFile(file) {
        const fileName = file.name.toLowerCase();
        if (!fileName.endsWith('.wav') && !fileName.endsWith('.mp3')) {
            updateStatus("⚠️ Invalid file type. Must be .wav or .mp3.", 'alert-danger');
            analyzeButton.disabled = true;
            predictSpeedButton.disabled = true;
            playButton.disabled = true;
            return;
        }

        const fileSizeMB = file.size / (1024 * 1024);
        if (fileSizeMB > 10) {
            updateStatus(`⚠️ Warning: Large file (${fileSizeMB.toFixed(1)}MB). Audio will be limited to 30 seconds for analysis.`, 'alert-warning');
        }

        uploadedFile = file;
        updateStatus(`File selected: ${file.name} (${fileSizeMB.toFixed(1)}MB). Reading audio data...`, 'alert-warning');

        try {
            audioDataURI = await readFileAsDataURI(file);
            updateStatus(`Audio file loaded (${file.name}). Ready for analysis.`, 'alert-info');
            analyzeButton.disabled = false;
            predictSpeedButton.disabled = false;
            playButton.disabled = false;

            // Reset sections
            nyquistSection.style.display = 'none';
            resampledGraphContainer.style.display = 'none';
            resampledFreqContainer.style.display = 'none';
            playerResampledDiv.style.display = 'none';
            speedPredictionResult.style.display = 'none';

            // Reset downsampled speed result
            if (downsampledSpeedResult) {
                downsampledSpeedResult.style.display = 'none';
            }

            originalMaxFrequency = 0;
            nyquistFrequencyRequired = 0;
            originalAudioData = null;
            resampledAudioData = null;
            originalDuration = 0;

        } catch (error) {
            updateStatus(`❌ Error reading file: ${error.message}`, 'alert-danger');
            analyzeButton.disabled = true;
            predictSpeedButton.disabled = true;
            playButton.disabled = true;
        }
    }

    if (dropZone && fileInput) {
        dropZone.addEventListener('click', (e) => {
            if (e.target === dropZone || e.target.tagName !== 'INPUT') {
                fileInput.click();
            }
        });
    }

    if (dropZone) {
        dropZone.ondragover = (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '#FFA500';
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

    if (fileInput) {
        fileInput.onchange = (e) => {
            if (e.target.files.length) {
                handleFile(e.target.files[0]);
            }
            fileInput.value = '';
        };
    }

    playButton.onclick = () => {
        if (!audioDataURI) return;
        playerOriginalDiv.innerHTML = `
            <h5 style="color: #FFA500;">Original Audio Player:</h5>
            <audio controls autoplay src="${audioDataURI}" style="width: 100%;"></audio>
        `;
    };

    analyzeButton.onclick = async () => {
        await performAnalysis(originalSampleRate);
    };

    predictSpeedButton.onclick = async () => {
        await predictSpeed();
    };

    function resampleAudioByDuration(audioData, currentRate, targetRate, duration) {
        const targetSamples = Math.floor(duration * targetRate);
        const resampled = new Float32Array(targetSamples);

        for (let i = 0; i < targetSamples; i++) {
            const timePosition = i / targetRate;
            const sourceIndex = timePosition * currentRate;

            const index0 = Math.floor(sourceIndex);
            const index1 = Math.min(index0 + 1, audioData.length - 1);
            const fraction = sourceIndex - index0;

            if (index0 < audioData.length) {
                resampled[i] = audioData[index0] * (1 - fraction) + audioData[index1] * fraction;
            }
        }

        return Array.from(resampled);
    }

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

    function audioArrayToWav(audioData, sampleRate) {
        const numChannels = 1;
        const bitsPerSample = 16;
        const bytesPerSample = bitsPerSample / 8;
        const blockAlign = numChannels * bytesPerSample;

        const dataLength = audioData.length * bytesPerSample;
        const buffer = new ArrayBuffer(44 + dataLength);
        const view = new DataView(buffer);

        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);
        writeString(36, 'data');
        view.setUint32(40, dataLength, true);

        let offset = 44;
        for (let i = 0; i < audioData.length; i++) {
            const sample = Math.max(-1, Math.min(1, audioData[i]));
            view.setInt16(offset, sample * 0x7FFF, true);
            offset += 2;
        }

        return buffer;
    }

    function computeFFT(audioData, sampleRate) {
        const fft = new Array(audioData.length);
        for (let k = 0; k < audioData.length / 2; k++) {
            let real = 0;
            let imag = 0;
            for (let n = 0; n < audioData.length; n++) {
                const angle = -2 * Math.PI * k * n / audioData.length;
                real += audioData[n] * Math.cos(angle);
                imag += audioData[n] * Math.sin(angle);
            }
            fft[k] = Math.sqrt(real * real + imag * imag);
        }

        const frequencies = [];
        const magnitudes = [];
        for (let i = 0; i < fft.length / 2; i++) {
            frequencies.push(i * sampleRate / audioData.length);
            magnitudes.push(fft[i]);
        }

        return { frequencies, magnitudes };
    }

    async function performAnalysis(targetSampleRate) {
        if (!uploadedFile || !audioDataURI) {
            updateStatus("⚠️ No audio file loaded.", 'alert-warning');
            return;
        }

        analyzeButton.disabled = true;
        updateStatus("Analyzing audio for visualization...", 'alert-warning');

        await new Promise(resolve => setTimeout(resolve, 50));

        const csrftoken = getCookie('csrftoken');

        try {
            const requestBody = JSON.stringify({
                audio_data: audioDataURI,
                filename: uploadedFile.name,
                target_sample_rate: targetSampleRate
            });

            const response = await fetch('/api/analyze_cars/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrftoken
                },
                body: requestBody,
            });

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

            originalAudioData = json_data.waveform;
            currentSampleRate = json_data.sr;
            originalSampleRate = json_data.original_sr;
            originalMaxFrequency = json_data.max_frequency;
            nyquistFrequencyRequired = 2 * originalMaxFrequency;
            resampledAudioData = json_data.waveform;
            originalDuration = originalAudioData.length / currentSampleRate;

            renderResults(json_data);

            // Enable downsampled speed prediction button
            setTimeout(() => {
                if (predictDownsampledSpeedBtn) {
                    predictDownsampledSpeedBtn.disabled = false;
                }
            }, 300);

        } catch (error) {
            console.error('Analysis Error:', error);
            updateStatus(`❌ Analysis Failed. Error: ${error.message}`, 'alert-danger');
        } finally {
            analyzeButton.disabled = false;
        }
    }

    async function predictSpeed() {
        if (!uploadedFile || !audioDataURI) {
            updateStatus("⚠️ No audio file loaded.", 'alert-warning');
            return;
        }

        predictSpeedButton.disabled = true;
        updateStatus("🚗 Predicting vehicle speed...", 'alert-warning');

        const csrftoken = getCookie('csrftoken');

        try {
            const requestBody = JSON.stringify({
                audio_data: audioDataURI,
                filename: uploadedFile.name
            });

            const response = await fetch('/api/predict_speed/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrftoken
                },
                body: requestBody,
            });

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

                throw new Error(`Speed prediction failed (Status ${response.status}): ${errorMessage}`);
            }

            const json_data = await response.json();

            if (json_data.error) {
                throw new Error(json_data.error);
            }

            // Display speed prediction result
            predictedSpeedValue.textContent = `${json_data.predicted_speed_kmh.toFixed(2)} km/h`;
            
            // Update model info for CNN14
            if (json_data.model_name) {
                checkpointName.textContent = json_data.model_name;
            } else {
                checkpointName.textContent = json_data.checkpoint_name || 'CNN14 Model';
            }
            
            // Display model configuration
            if (json_data.model_config) {
                checkpointVariance.textContent = `${json_data.model_config.sample_rate} Hz`;
            } else {
                checkpointVariance.textContent = json_data.checkpoint_variance ? json_data.checkpoint_variance.toFixed(2) : 'N/A';
            }
            
            // Display duration or MAE
            if (json_data.model_config) {
                checkpointMae.textContent = `${json_data.model_config.duration}s clip`;
            } else {
                checkpointMae.textContent = json_data.checkpoint_mae ? `${json_data.checkpoint_mae.toFixed(2)} km/h` : 'N/A';
            }
            
            speedPredictionResult.style.display = 'block';

            // Scroll to result
            speedPredictionResult.scrollIntoView({ behavior: 'smooth', block: 'center' });

            updateStatus(`✅ Speed prediction complete: ${json_data.predicted_speed_kmh.toFixed(2)} km/h`, 'alert-success');

        } catch (error) {
            console.error('Speed Prediction Error:', error);
            updateStatus(`❌ Speed Prediction Failed. Error: ${error.message}`, 'alert-danger');
        } finally {
            predictSpeedButton.disabled = false;
        }
    }

    function plotTimeDomain(waveform, sr, divElement, title, showMarkers = false) {
        const plotData = downsampleForPlotting(waveform, MAX_PLOT_POINTS);
        const downsampleFactor = Math.ceil(waveform.length / plotData.length);
        const timeAxis = Array.from({ length: plotData.length }, (_, i) => (i * downsampleFactor) / sr);

        const traceConfig = {
            x: timeAxis,
            y: plotData,
            mode: showMarkers ? 'lines+markers' : 'lines',
            line: { color: '#FFA500', width: 1 },
            name: 'Amplitude'
        };

        if (showMarkers && plotData.length < 500) {
            traceConfig.marker = {
                size: 4,
                color: '#FFA500',
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

    function plotFrequencyDomain(frequencies, magnitudes, divElement, title) {
        const plotFreqs = downsampleForPlotting(frequencies, MAX_PLOT_POINTS);
        const plotMags = downsampleForPlotting(magnitudes, MAX_PLOT_POINTS);

        const traceConfig = {
            x: plotFreqs,
            y: plotMags,
            mode: 'lines',
            line: { color: '#58a6ff', width: 1 },
            name: 'Magnitude'
        };

        const freqDomainLayout = {
            title: title,
            xaxis: {
                title: 'Frequency (Hz)',
                color: '#c9d1d9',
                gridcolor: '#30363d'
            },
            yaxis: {
                title: 'Magnitude',
                color: '#c9d1d9',
                gridcolor: '#30363d'
            },
            plot_bgcolor: '#161b22',
            paper_bgcolor: '#161b22',
            font: { color: '#c9d1d9' },
            height: 350,
            hovermode: 'closest',
            margin: { l: 50, r: 30, t: 50, b: 50 }
        };

        Plotly.newPlot(divElement, [traceConfig], freqDomainLayout, {responsive: true});
    }

    function updateNyquistAnalysis(currentRate) {
        document.getElementById('cars-original-sr').textContent = `${originalSampleRate} Hz`;
        document.getElementById('cars-current-sr').textContent = `${currentRate} Hz`;
        document.getElementById('cars-max-freq').textContent = `${originalMaxFrequency.toFixed(0)} Hz`;
        document.getElementById('cars-nyquist-freq').textContent = `${nyquistFrequencyRequired.toFixed(0)} Hz`;

        const statusElement = document.getElementById('cars-sampling-status');

        if (currentRate < nyquistFrequencyRequired) {
            statusElement.textContent = `⚠️ Under-sampling (Current rate < ${nyquistFrequencyRequired.toFixed(0)} Hz)`;
            statusElement.style.color = '#f78166';
        } else {
            statusElement.textContent = `✓ Over-sampling (Current rate ≥ ${nyquistFrequencyRequired.toFixed(0)} Hz)`;
            statusElement.style.color = '#FFA500';
        }

        nyquistSection.style.display = 'block';
    }

    function renderResults(data) {
        plotTimeDomain(
            data.waveform,
            data.sr,
            timeDomainOriginalDiv,
            `Original Audio: Time Domain (${data.waveform.length} samples at ${data.sr} Hz, Duration: ${originalDuration.toFixed(2)}s)`,
            false
        );

        plotFrequencyDomain(
            data.fft_frequencies,
            data.fft_magnitudes,
            frequencyDomainOriginalDiv,
            `Original Audio: Frequency Domain (Max Frequency: ${originalMaxFrequency.toFixed(0)} Hz)`
        );

        updateNyquistAnalysis(data.sr);

        debugDiv.innerHTML = `<pre>File: ${data.filename}
Original Sample Rate: ${originalSampleRate} Hz
Current Sample Rate: ${data.sr} Hz
Duration: ${originalDuration.toFixed(2)} seconds
Original Max Frequency: ${originalMaxFrequency.toFixed(2)} Hz
Nyquist Frequency Required (2×F(max)): ${nyquistFrequencyRequired.toFixed(2)} Hz
Sampling Status: ${data.sr >= nyquistFrequencyRequired ? 'Over-sampling' : 'Under-sampling'}</pre>`;

        updateStatus(`✅ Visualization complete for ${data.filename}.`, 'alert-success');
    }

    resampleSlider.oninput = (e) => {
        const newSampleRate = parseInt(e.target.value);
        currentSrDisplay.textContent = `${newSampleRate} Hz`;

        if (sliderTimeout) {
            clearTimeout(sliderTimeout);
        }

        document.getElementById('cars-current-sr').textContent = `${newSampleRate} Hz`;
        const statusElement = document.getElementById('cars-sampling-status');

        if (newSampleRate < nyquistFrequencyRequired) {
            statusElement.textContent = `⚠️ Under-sampling (Current rate < ${nyquistFrequencyRequired.toFixed(0)} Hz)`;
            statusElement.style.color = '#f78166';
        } else {
            statusElement.textContent = `✓ Over-sampling (Current rate ≥ ${nyquistFrequencyRequired.toFixed(0)} Hz)`;
            statusElement.style.color = '#FFA500';
        }

        sliderTimeout = setTimeout(() => {
            if (originalAudioData && originalDuration > 0) {
                resampledAudioData = resampleAudioByDuration(
                    originalAudioData,
                    currentSampleRate,
                    newSampleRate,
                    originalDuration
                );

                resampledGraphContainer.style.display = 'block';
                resampledFreqContainer.style.display = 'block';

                const resampledDuration = resampledAudioData.length / newSampleRate;
                const showMarkers = newSampleRate < 4000;

                plotTimeDomain(
                    resampledAudioData,
                    newSampleRate,
                    timeDomainResampledDiv,
                    `Resampled Audio: Time Domain (${resampledAudioData.length} samples at ${newSampleRate} Hz, Duration: ${resampledDuration.toFixed(2)}s)`,
                    showMarkers
                );

                const fftResult = computeFFT(resampledAudioData.slice(0, Math.min(8192, resampledAudioData.length)), newSampleRate);
                plotFrequencyDomain(
                    fftResult.frequencies,
                    fftResult.magnitudes,
                    frequencyDomainResampledDiv,
                    `Resampled Audio: Frequency Domain (Sample Rate: ${newSampleRate} Hz)`
                );
            }
        }, 150);
    };

    playResampledBtn.onclick = () => {
        if (!resampledAudioData) {
            updateStatus("⚠️ No resampled audio available.", 'alert-warning');
            return;
        }

        const currentRate = parseInt(resampleSlider.value);
        const resampledDuration = resampledAudioData.length / currentRate;

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

    // Predict Speed on Downsampled Audio Button Handler
    if (predictDownsampledSpeedBtn) {
        predictDownsampledSpeedBtn.onclick = async () => {
            if (!originalAudioData || !uploadedFile) {
                updateStatus("⚠️ Please analyze the audio first.", 'alert-warning');
                return;
            }

            const downsampledRate = parseInt(resampleSlider.value);
            predictDownsampledSpeedBtn.disabled = true;
            updateStatus(`🚗 Predicting speed on downsampled audio (${downsampledRate} Hz)...`, 'alert-info');

            try {
                // Step 1: Downsample audio
                console.log(`[DOWNSAMPLE] Downsampling from ${currentSampleRate} Hz to ${downsampledRate} Hz`);
                const downsampledAudio = resampleAudioByDuration(
                    originalAudioData,
                    currentSampleRate,
                    downsampledRate,
                    originalDuration
                );

                // Convert downsampled audio to WAV
                const downsampledWavBuffer = audioArrayToWav(downsampledAudio, downsampledRate);
                const downsampledBlob = new Blob([downsampledWavBuffer], { type: 'audio/wav' });
                
                // Create data URI for downsampled audio
                const reader = new FileReader();
                const downsampledDataURI = await new Promise((resolve) => {
                    reader.onload = () => resolve(reader.result);
                    reader.readAsDataURL(downsampledBlob);
                });

                // Step 2: Predict speed on downsampled audio
                updateStatus(`🚗 Predicting speed on downsampled audio (${downsampledRate} Hz)...`, 'alert-info');

                const csrftoken = getCookie('csrftoken');
                const requestBody = JSON.stringify({
                    audio_data: downsampledDataURI,
                    filename: `downsampled_${downsampledRate}hz_${uploadedFile.name}`
                });

                const response = await fetch('/api/predict_speed/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': csrftoken
                    },
                    body: requestBody,
                });

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

                    throw new Error(`Speed prediction failed (Status ${response.status}): ${errorMessage}`);
                }

                const json_data = await response.json();

                if (json_data.error) {
                    throw new Error(json_data.error);
                }

                // Display downsampled speed prediction result
                downsampledSpeedValue.textContent = `${json_data.predicted_speed_kmh.toFixed(2)} km/h`;
                downsampledSrDisplay.textContent = `${downsampledRate} Hz`;
                downsampledSpeedResult.style.display = 'block';

                // Scroll to result
                downsampledSpeedResult.scrollIntoView({ behavior: 'smooth', block: 'center' });

                updateStatus(
                    `✅ Speed predicted on downsampled audio (${downsampledRate} Hz): ${json_data.predicted_speed_kmh.toFixed(2)} km/h`,
                    'alert-success'
                );

            } catch (error) {
                console.error('Downsampled Speed Prediction Error:', error);
                updateStatus(`❌ Prediction failed: ${error.message}`, 'alert-danger');
            } finally {
                predictDownsampledSpeedBtn.disabled = false;
            }
        };
    }

    function updateAliasingStatus(message, alertClass = 'alert-info') {
        // Legacy function - can be removed if no longer needed
        console.log(`[LEGACY] ${message}`);
    }
});