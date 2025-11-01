document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone-voices');
    const fileInput = document.getElementById('voices-file-input');
    const analyzeButton = document.getElementById('analyze-voices-button');
    const playButton = document.getElementById('play-voices-button');
    const statusDiv = document.getElementById('voices-status');
    const playerOriginalDiv = document.getElementById('audio-player-voices-original');
    const playerResampledDiv = document.getElementById('audio-player-voices-resampled');
    const timeDomainOriginalDiv = document.getElementById('time-domain-graph-voices-original');
    const timeDomainResampledDiv = document.getElementById('time-domain-graph-voices-resampled');
    const resampledGraphContainer = document.getElementById('voices-resampled-graph-container');
    const predictionDiv = document.getElementById('voices-prediction-output');
    const probabilitiesDiv = document.getElementById('voices-probabilities-graph');
    const debugDiv = document.getElementById('voices-debug-info');
    const nyquistSection = document.getElementById('nyquist-section-voices');
    const resampleSlider = document.getElementById('voices-resample-slider');
    const currentSrDisplay = document.getElementById('voices-current-sr-display');
    const playResampledBtn = document.getElementById('play-voices-resampled-btn');

    // Gender Detection Buttons for Aliased and Enhanced Audio
    const detectGenderAliasedBtn = document.getElementById('detect-gender-aliased-btn');
    const detectGenderEnhancedBtn = document.getElementById('detect-gender-enhanced-btn');
    const aliasedGenderResult = document.getElementById('aliased-gender-result');
    const enhancedGenderResult = document.getElementById('enhanced-gender-result');

    // Anti-Aliasing Elements
    const antiAliasingSection = document.getElementById('anti-aliasing-section');
    const applyAntiAliasingBtn = document.getElementById('apply-anti-aliasing-btn');
    const antiAliasingStatus = document.getElementById('anti-aliasing-status');
    const enhancedAudioPlayer = document.getElementById('enhanced-audio-player');
    const enhancedAudioGraphContainer = document.getElementById('enhanced-audio-graph-container');
    const enhancedFrequencyGraphContainer = document.getElementById('enhanced-frequency-graph-container');
    const timeDomainEnhancedDiv = document.getElementById('time-domain-graph-enhanced');
    const frequencyDomainEnhancedDiv = document.getElementById('frequency-domain-graph-enhanced');
    const enhancedAudioStats = document.getElementById('enhanced-audio-stats');

    // CRITICAL: Verify elements exist on page load
    console.log('=== PAGE LOAD VERIFICATION ===');
    console.log('Anti-aliasing section found:', !!antiAliasingSection);
    console.log('Apply button found:', !!applyAntiAliasingBtn);
    console.log('Detect Gender Aliased button found:', !!detectGenderAliasedBtn);
    console.log('Detect Gender Enhanced button found:', !!detectGenderEnhancedBtn);

    if (!antiAliasingSection) {
        console.error('❌ CRITICAL: anti-aliasing-section NOT FOUND in DOM!');
        console.log('Check if the HTML template includes the section correctly.');
    }
    
    if (!detectGenderAliasedBtn) {
        console.error('❌ CRITICAL: detect-gender-aliased-btn NOT FOUND in DOM!');
    }
    
    if (!detectGenderEnhancedBtn) {
        console.error('❌ CRITICAL: detect-gender-enhanced-btn NOT FOUND in DOM!');
    }

    let uploadedFile = null;
    let audioDataURI = null;
    let predictionData = null;
    let originalAudioData = null;
    let currentSampleRate = 16000;
    let originalSampleRate = 16000;
    let originalMaxFrequency = 0;
    let nyquistFrequencyRequired = 0;
    let resampledAudioData = null;
    let sliderTimeout = null;
    let originalDuration = 0;
    let enhancedAudioData = null;

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

    function updateAntiAliasingStatus(message, alertClass = 'alert-info') {
        if (!antiAliasingStatus) {
            console.error('❌ antiAliasingStatus element not found!');
            return;
        }
        antiAliasingStatus.className = `alert ${alertClass} mt-3`;
        antiAliasingStatus.innerHTML = message;
        antiAliasingStatus.style.display = 'block';
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
            playButton.disabled = false;

            // Reset all sections when new file is loaded
            nyquistSection.style.display = 'none';
            resampledGraphContainer.style.display = 'none';
            playerResampledDiv.style.display = 'none';

            // IMPORTANT: Hide anti-aliasing section when new file is loaded
            if (antiAliasingSection) {
                antiAliasingSection.style.display = 'none';
            }
            if (enhancedAudioPlayer) {
                enhancedAudioPlayer.style.display = 'none';
            }
            if (enhancedAudioGraphContainer) {
                enhancedAudioGraphContainer.style.display = 'none';
            }
            if (enhancedFrequencyGraphContainer) {
                enhancedFrequencyGraphContainer.style.display = 'none';
            }
            if (enhancedAudioStats) {
                enhancedAudioStats.style.display = 'none';
            }
            if (antiAliasingStatus) {
                antiAliasingStatus.style.display = 'none';
            }

            originalMaxFrequency = 0;
            nyquistFrequencyRequired = 0;
            originalAudioData = null;
            resampledAudioData = null;
            predictionData = null;
            originalDuration = 0;
            enhancedAudioData = null;

        } catch (error) {
            updateStatus(`❌ Error reading file: ${error.message}`, 'alert-danger');
            analyzeButton.disabled = true;
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
            dropZone.style.borderColor = '#c4b5fd';
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
            <h5 style="color: #c4b5fd;">Original Audio Player:</h5>
            <audio controls autoplay src="${audioDataURI}" style="width: 100%;"></audio>
        `;
    };

    analyzeButton.onclick = async () => {
        await performAnalysis(originalSampleRate);
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

    function predictGenderFromAudio(audioData, sampleRate) {
        let zeroCrossings = 0;
        for (let i = 1; i < audioData.length; i++) {
            if ((audioData[i] >= 0 && audioData[i-1] < 0) || (audioData[i] < 0 && audioData[i-1] >= 0)) {
                zeroCrossings++;
            }
        }

        const estimatedFrequency = (zeroCrossings / 2) / (audioData.length / sampleRate);

        let maleProbability, femaleProbability;

        if (estimatedFrequency < 130) {
            maleProbability = 0.75 + Math.random() * 0.20;
            femaleProbability = 1 - maleProbability;
        } else if (estimatedFrequency > 200) {
            femaleProbability = 0.75 + Math.random() * 0.20;
            maleProbability = 1 - femaleProbability;
        } else {
            const centerPoint = 165;
            const distance = Math.abs(estimatedFrequency - centerPoint);
            if (estimatedFrequency < centerPoint) {
                maleProbability = 0.5 + (distance / centerPoint) * 0.3;
                femaleProbability = 1 - maleProbability;
            } else {
                femaleProbability = 0.5 + (distance / centerPoint) * 0.3;
                maleProbability = 1 - femaleProbability;
            }
        }

        return {
            male: Math.min(0.95, Math.max(0.05, maleProbability)),
            female: Math.min(0.95, Math.max(0.05, femaleProbability)),
            estimatedFrequency: estimatedFrequency
        };
    }
async function performAnalysis(targetSampleRate) {
    if (!uploadedFile || !audioDataURI) {
        updateStatus("⚠️ No audio file loaded.", 'alert-warning');
        return;
    }

    analyzeButton.disabled = true;
    updateStatus("Analyzing voice for gender detection...", 'alert-warning');

    await new Promise(resolve => setTimeout(resolve, 50));

    const csrftoken = getCookie('csrftoken');

    try {
        const requestBody = JSON.stringify({
            audio_data: audioDataURI,
            filename: uploadedFile.name,
            target_sample_rate: targetSampleRate
        });

        const response = await fetch('/api/analyze_voices/', {
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

        predictionData = json_data;
        originalAudioData = json_data.waveform;
        currentSampleRate = json_data.sr;
        originalSampleRate = json_data.original_sr;
        originalMaxFrequency = json_data.max_frequency;
        nyquistFrequencyRequired = 2 * originalMaxFrequency;
        resampledAudioData = json_data.waveform;
        originalDuration = originalAudioData.length / currentSampleRate;

        renderResults(json_data);

        // ✅ SHOW ANTI-ALIASING SECTION - FORCE IT!
        setTimeout(() => {
            console.log('=== FORCING ANTI-ALIASING SECTION VISIBLE ===');

            if (antiAliasingSection) {
                // Remove display:none from inline style
                antiAliasingSection.style.removeProperty('display');
                antiAliasingSection.style.display = 'block';
                antiAliasingSection.style.visibility = 'visible';
                antiAliasingSection.style.opacity = '1';
                antiAliasingSection.style.position = 'relative';
                antiAliasingSection.style.zIndex = '100';

                console.log('✅ Section display:', antiAliasingSection.style.display);
                console.log('✅ Section visibility:', antiAliasingSection.style.visibility);
            } else {
                console.error('❌ antiAliasingSection element is NULL!');
            }

            if (applyAntiAliasingBtn) {
                applyAntiAliasingBtn.disabled = false;
                applyAntiAliasingBtn.style.display = 'inline-block';
                console.log('✅ Button enabled');
            } else {
                console.error('❌ applyAntiAliasingBtn element is NULL!');
            }

            // Scroll into view
            if (antiAliasingSection) {
                antiAliasingSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }, 300);

    } catch (error) {
        console.error('Analysis Error:', error);
        updateStatus(`❌ Analysis Failed. Error: ${error.message}`, 'alert-danger');
    } finally {
        analyzeButton.disabled = false;
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
            line: { color: '#c4b5fd', width: 1 },
            name: 'Amplitude'
        };

        if (showMarkers && plotData.length < 500) {
            traceConfig.marker = {
                size: 4,
                color: '#c4b5fd',
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
        const frequencyTrace = {
            x: frequencies,
            y: magnitudes,
            mode: 'lines',
            line: { color: '#79c0ff', width: 1 },
            name: 'Magnitude'
        };

        const frequencyLayout = {
            title: title,
            xaxis: {
                title: 'Frequency (Hz)',
                color: '#c9d1d9',
                gridcolor: '#30363d',
                type: 'log'
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

        Plotly.newPlot(divElement, [frequencyTrace], frequencyLayout, {responsive: true});
    }

    function updateNyquistAnalysis(currentRate) {
        document.getElementById('voices-original-sr').textContent = `${originalSampleRate} Hz`;
        document.getElementById('voices-current-sr').textContent = `${currentRate} Hz`;
        document.getElementById('voices-max-freq').textContent = `${originalMaxFrequency.toFixed(0)} Hz`;
        document.getElementById('voices-nyquist-freq').textContent = `${nyquistFrequencyRequired.toFixed(0)} Hz`;

        const statusElement = document.getElementById('voices-sampling-status');

        if (currentRate < nyquistFrequencyRequired) {
            statusElement.textContent = `⚠️ Under-sampling (Current rate < ${nyquistFrequencyRequired.toFixed(0)} Hz)`;
            statusElement.style.color = '#f78166';
        } else {
            statusElement.textContent = `✓ Over-sampling (Current rate ≥ ${nyquistFrequencyRequired.toFixed(0)} Hz)`;
            statusElement.style.color = '#c4b5fd';
        }

        nyquistSection.style.display = 'block';
    }

    function renderResults(data) {
        const predictionColor = (data.predicted_gender === 'Male') ? '#58a6ff' : '#f78166';

        predictionDiv.innerHTML = `
            <h4 style="color: ${predictionColor}; font-weight: bold;">
                🎤 Detected Gender: ${data.predicted_gender}
            </h4>
            <p class="text-secondary">Confidence: ${(data.confidence * 100).toFixed(1)}%</p>
        `;

        const barTrace = {
            x: ['Male', 'Female'],
            y: [data.male_probability * 100, data.female_probability * 100],
            type: 'bar',
            marker: {
                color: ['#58a6ff', '#f78166']
            }
        };

        const barLayout = {
            title: "Gender Prediction Probabilities (%)",
            xaxis: { title: 'Gender', color: '#c9d1d9', gridcolor: '#30363d' },
            yaxis: { title: 'Probability (%)', range: [0, 100], color: '#c9d1d9', gridcolor: '#30363d' },
            plot_bgcolor: '#161b22',
            paper_bgcolor: '#161b22',
            font: { color: '#c9d1d9' },
            height: 300,
            margin: { l: 50, r: 30, t: 50, b: 50 }
        };

        Plotly.newPlot(probabilitiesDiv, [barTrace], barLayout, {responsive: true});

        plotTimeDomain(
            data.waveform,
            data.sr,
            timeDomainOriginalDiv,
            `Original Audio: Time Domain (${data.waveform.length} samples at ${data.sr} Hz, Duration: ${originalDuration.toFixed(2)}s)`,
            false
        );

        updateNyquistAnalysis(data.sr);

        debugDiv.innerHTML = `<pre>File: ${data.filename}
Original Sample Rate: ${originalSampleRate} Hz
Current Sample Rate: ${data.sr} Hz
Duration: ${originalDuration.toFixed(2)} seconds
Original Max Frequency: ${originalMaxFrequency.toFixed(2)} Hz
Nyquist Frequency Required (2×F(max)): ${nyquistFrequencyRequired.toFixed(2)} Hz
Predicted Gender: ${data.predicted_gender}
Confidence: ${(data.confidence * 100).toFixed(1)}%
Male Probability: ${(data.male_probability * 100).toFixed(1)}%
Female Probability: ${(data.female_probability * 100).toFixed(1)}%
Sampling Status: ${data.sr >= nyquistFrequencyRequired ? 'Over-sampling' : 'Under-sampling'}</pre>`;

        updateStatus(`✅ Analysis complete for ${data.filename}.`, 'alert-success');
    }

    resampleSlider.oninput = (e) => {
        const newSampleRate = parseInt(e.target.value);
        currentSrDisplay.textContent = `${newSampleRate} Hz`;

        if (sliderTimeout) {
            clearTimeout(sliderTimeout);
        }

        document.getElementById('voices-current-sr').textContent = `${newSampleRate} Hz`;
        const statusElement = document.getElementById('voices-sampling-status');

        if (newSampleRate < nyquistFrequencyRequired) {
            statusElement.textContent = `⚠️ Under-sampling (Current rate < ${nyquistFrequencyRequired.toFixed(0)} Hz)`;
            statusElement.style.color = '#f78166';
        } else {
            statusElement.textContent = `✓ Over-sampling (Current rate ≥ ${nyquistFrequencyRequired.toFixed(0)} Hz)`;
            statusElement.style.color = '#c4b5fd';
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

                const resampledDuration = resampledAudioData.length / newSampleRate;
                const showMarkers = newSampleRate < 4000;

                plotTimeDomain(
                    resampledAudioData,
                    newSampleRate,
                    timeDomainResampledDiv,
                    `Resampled Audio: Time Domain (${resampledAudioData.length} samples at ${newSampleRate} Hz, Duration: ${resampledDuration.toFixed(2)}s)`,
                    showMarkers
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

        // Enable aliased gender detection button after resampled audio is created
        console.log('🔍 DEBUG: Attempting to enable aliased gender button...');
        console.log('Button exists:', !!detectGenderAliasedBtn);
        console.log('Button element:', detectGenderAliasedBtn);
        
        if (detectGenderAliasedBtn) {
            // Try multiple methods to enable the button
            detectGenderAliasedBtn.disabled = false;
            detectGenderAliasedBtn.removeAttribute('disabled');
            
            // Also try direct DOM manipulation as backup
            setTimeout(() => {
                const btn = document.getElementById('detect-gender-aliased-btn');
                if (btn) {
                    btn.disabled = false;
                    btn.removeAttribute('disabled');
                    btn.classList.remove('disabled');
                    console.log('✅ Button force-enabled via setTimeout');
                    console.log('   Final disabled state:', btn.disabled);
                    console.log('   Has disabled attr:', btn.hasAttribute('disabled'));
                }
            }, 100);
            
            console.log('✅ Aliased gender detection button enabled (resampled audio ready)');
        } else {
            console.error('❌ detectGenderAliasedBtn is null or undefined!');
        }
    };

    // Anti-Aliasing Button Handler
    if (applyAntiAliasingBtn) {
        applyAntiAliasingBtn.onclick = async () => {
            // ✅ This button applies anti-aliasing to the resampled audio from the slider
            if (!resampledAudioData || !uploadedFile) {
                updateAntiAliasingStatus("⚠️ Please move the slider and play resampled audio first.", 'alert-warning');
                return;
            }

            applyAntiAliasingBtn.disabled = true;
            updateAntiAliasingStatus("🔄 Applying anti-aliasing enhancement to resampled audio... This may take a moment.", 'alert-info');

            const csrftoken = getCookie('csrftoken');
            const resampledRate = parseInt(resampleSlider.value);

            try {
                // Convert resampled audio to WAV and create data URI
                const resampledWavBuffer = audioArrayToWav(resampledAudioData, resampledRate);
                const resampledBlob = new Blob([resampledWavBuffer], { type: 'audio/wav' });
                
                const reader = new FileReader();
                const resampledDataURI = await new Promise((resolve) => {
                    reader.onload = () => resolve(reader.result);
                    reader.readAsDataURL(resampledBlob);
                });

                console.log(`[ANTI_ALIASING] Applying anti-aliasing to resampled audio at ${resampledRate} Hz (${resampledAudioData.length} samples)`);

                const requestBody = JSON.stringify({
                    audio_data: resampledDataURI,
                    filename: `antialiased_${resampledRate}hz_${uploadedFile.name}`,
                    sample_rate: resampledRate
                });

                const response = await fetch('/api/apply_anti_aliasing/', {
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

                    throw new Error(`Anti-aliasing failed (Status ${response.status}): ${errorMessage}`);
                }

                const json_data = await response.json();

                if (json_data.error) {
                    throw new Error(json_data.error);
                }

                enhancedAudioData = json_data;

                // Display enhanced audio player
                const enhancedAudioSrc = `data:audio/wav;base64,${json_data.enhanced_audio_b64}`;
                enhancedAudioPlayer.style.display = 'block';
                enhancedAudioPlayer.innerHTML = `
                    <h5 style="color: #79c0ff;">Enhanced Audio Player (After Anti-Aliasing):</h5>
                    <audio controls src="${enhancedAudioSrc}" style="width: 100%;"></audio>
                    <div class="mt-2">
                        <a href="${enhancedAudioSrc}" download="${json_data.filename}" class="btn btn-sm btn-outline-info">
                            💾 Download Enhanced Audio
                        </a>
                    </div>
                `;

                // Plot enhanced audio time domain
                enhancedAudioGraphContainer.style.display = 'block';
                plotTimeDomain(
                    json_data.waveform,
                    json_data.sr,
                    timeDomainEnhancedDiv,
                    `Enhanced Audio: Time Domain (${json_data.samples} samples at ${json_data.sr} Hz, Duration: ${json_data.duration.toFixed(2)}s)`,
                    false
                );

                // Plot enhanced audio frequency domain
                if (json_data.fft_frequencies && json_data.fft_magnitudes) {
                    enhancedFrequencyGraphContainer.style.display = 'block';
                    plotFrequencyDomain(
                        json_data.fft_frequencies,
                        json_data.fft_magnitudes,
                        frequencyDomainEnhancedDiv,
                        `Enhanced Audio: Frequency Domain (Max Frequency: ${json_data.max_frequency.toFixed(0)} Hz)`
                    );
                }

                // Display enhanced audio statistics
                enhancedAudioStats.style.display = 'block';
                document.getElementById('enhanced-duration').textContent = `${json_data.duration.toFixed(2)}s`;
                document.getElementById('enhanced-sr').textContent = `${json_data.sr} Hz`;
                document.getElementById('enhanced-max-freq').textContent = `${json_data.max_frequency.toFixed(0)} Hz`;

                updateAntiAliasingStatus("✅ Anti-aliasing enhancement complete! Listen to the enhanced audio above.", 'alert-success');

                // Enable gender detection button for enhanced audio
                console.log('🔍 DEBUG: Attempting to enable enhanced gender button...');
                if (detectGenderEnhancedBtn) {
                    detectGenderEnhancedBtn.disabled = false;
                    detectGenderEnhancedBtn.removeAttribute('disabled');
                    
                    // Also try direct DOM manipulation as backup
                    setTimeout(() => {
                        const btn = document.getElementById('detect-gender-enhanced-btn');
                        if (btn) {
                            btn.disabled = false;
                            btn.removeAttribute('disabled');
                            btn.classList.remove('disabled');
                            console.log('✅ Enhanced button force-enabled via setTimeout');
                            console.log('   Final disabled state:', btn.disabled);
                            console.log('   Has disabled attr:', btn.hasAttribute('disabled'));
                        }
                    }, 100);
                    
                    console.log('✅ Enhanced gender detection button enabled');
                } else {
                    console.error('❌ detectGenderEnhancedBtn element is NULL!');
                }

            } catch (error) {
                console.error('Anti-Aliasing Error:', error);
                updateAntiAliasingStatus(`❌ Anti-aliasing failed: ${error.message}`, 'alert-danger');
            } finally {
                applyAntiAliasingBtn.disabled = false;
            }
        };
    } else {
        console.error('❌ applyAntiAliasingBtn not found - cannot attach event handler!');
    }

    // Detect Gender on Aliased/Downsampled Audio Button Handler
    if (detectGenderAliasedBtn) {
        detectGenderAliasedBtn.onclick = async () => {
            // ✅ This button sends the resampled audio (from slider) to gender detection
            // The slider creates resampledAudioData at the target sample rate
            if (!resampledAudioData || !uploadedFile) {
                updateStatus("⚠️ Please move the slider and play resampled audio first.", 'alert-warning');
                return;
            }

            const downsampledRate = parseInt(resampleSlider.value);
            detectGenderAliasedBtn.disabled = true;
            
            // Show loading indicator
            aliasedGenderResult.style.display = 'block';
            document.getElementById('aliased-gender-value').innerHTML = '<div class="spinner-border text-warning" role="status"><span class="visually-hidden">Loading...</span></div>';
            document.getElementById('aliased-gender-confidence').textContent = 'Analyzing...';
            document.getElementById('aliased-male-prob').textContent = '-';
            document.getElementById('aliased-female-prob').textContent = '-';
            aliasedGenderResult.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            updateStatus(`🎤 Detecting gender on resampled audio (${downsampledRate} Hz)... Please wait.`, 'alert-info');

            try {
                // Use pre-computed resampled audio from slider
                // This audio has already been processed with the target sample rate
                console.log(`[GENDER_ALIASED] Using resampled audio at ${downsampledRate} Hz (${resampledAudioData.length} samples)`);

                // Convert resampled audio to WAV
                const resampledWavBuffer = audioArrayToWav(resampledAudioData, downsampledRate);
                const resampledBlob = new Blob([resampledWavBuffer], { type: 'audio/wav' });
                
                // Create data URI
                const reader = new FileReader();
                const resampledDataURI = await new Promise((resolve) => {
                    reader.onload = () => resolve(reader.result);
                    reader.readAsDataURL(resampledBlob);
                });

                // Detect gender on resampled audio
                const csrftoken = getCookie('csrftoken');
                const requestBody = JSON.stringify({
                    audio_data: resampledDataURI,
                    filename: `resampled_${downsampledRate}hz_${uploadedFile.name}`,
                    target_sample_rate: downsampledRate
                });

                console.log(`[GENDER_ALIASED] Sending resampled audio to gender detection API`);
                const response = await fetch('/api/analyze_voices/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': csrftoken
                    },
                    body: requestBody,
                });

                if (!response.ok) {
                    const errorJson = await response.json();
                    throw new Error(errorJson.error || 'Gender detection failed');
                }

                const genderData = await response.json();

                // Display aliased gender result
                document.getElementById('aliased-gender-value').textContent = genderData.predicted_gender;
                document.getElementById('aliased-gender-confidence').textContent = `${(genderData.confidence * 100).toFixed(1)}%`;
                document.getElementById('aliased-male-prob').textContent = `${(genderData.male_probability * 100).toFixed(1)}%`;
                document.getElementById('aliased-female-prob').textContent = `${(genderData.female_probability * 100).toFixed(1)}%`;
                
                updateStatus(
                    `✅ Gender detected on resampled audio (${downsampledRate} Hz): ${genderData.predicted_gender} (${(genderData.confidence * 100).toFixed(1)}%)`,
                    'alert-success'
                );

            } catch (error) {
                console.error('Gender Detection (Aliased) Error:', error);
                updateStatus(`❌ Gender detection failed: ${error.message}`, 'alert-danger');
                aliasedGenderResult.style.display = 'none';
            } finally {
                detectGenderAliasedBtn.disabled = false;
            }
        };
    }

    // Detect Gender on Enhanced/Anti-Aliased Audio Button Handler
    if (detectGenderEnhancedBtn) {
        detectGenderEnhancedBtn.onclick = async () => {
            if (!enhancedAudioData || !enhancedAudioData.enhanced_audio_b64) {
                updateStatus("⚠️ Please apply anti-aliasing first.", 'alert-warning');
                return;
            }

            detectGenderEnhancedBtn.disabled = true;
            
            // Show loading indicator
            enhancedGenderResult.style.display = 'block';
            document.getElementById('enhanced-gender-value').innerHTML = '<div class="spinner-border text-info" role="status"><span class="visually-hidden">Loading...</span></div>';
            document.getElementById('enhanced-gender-confidence').textContent = 'Analyzing...';
            document.getElementById('enhanced-male-prob').textContent = '-';
            document.getElementById('enhanced-female-prob').textContent = '-';
            enhancedGenderResult.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            updateStatus(`🎤 Detecting gender on enhanced audio... Please wait.`, 'alert-info');

            try {
                // Use the enhanced audio data URI
                const enhancedDataURI = `data:audio/wav;base64,${enhancedAudioData.enhanced_audio_b64}`;

                // Detect gender on enhanced audio
                const csrftoken = getCookie('csrftoken');
                const requestBody = JSON.stringify({
                    audio_data: enhancedDataURI,
                    filename: `enhanced_${uploadedFile.name}`,
                    target_sample_rate: enhancedAudioData.sr
                });

                const response = await fetch('/api/analyze_voices/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': csrftoken
                    },
                    body: requestBody,
                });

                if (!response.ok) {
                    const errorJson = await response.json();
                    throw new Error(errorJson.error || 'Gender detection failed');
                }

                const genderData = await response.json();

                // Display enhanced gender result
                document.getElementById('enhanced-gender-value').textContent = genderData.predicted_gender;
                document.getElementById('enhanced-gender-confidence').textContent = `${(genderData.confidence * 100).toFixed(1)}%`;
                document.getElementById('enhanced-male-prob').textContent = `${(genderData.male_probability * 100).toFixed(1)}%`;
                document.getElementById('enhanced-female-prob').textContent = `${(genderData.female_probability * 100).toFixed(1)}%`;
                
                updateStatus(
                    `✅ Gender detected on enhanced audio: ${genderData.predicted_gender} (${(genderData.confidence * 100).toFixed(1)}%)`,
                    'alert-success'
                );

            } catch (error) {
                console.error('Gender Detection (Enhanced) Error:', error);
                updateStatus(`❌ Gender detection failed: ${error.message}`, 'alert-danger');
                enhancedGenderResult.style.display = 'none';
            } finally {
                detectGenderEnhancedBtn.disabled = false;
            }
        };
    }
});