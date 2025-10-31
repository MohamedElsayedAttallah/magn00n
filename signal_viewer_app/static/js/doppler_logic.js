document.addEventListener('DOMContentLoaded', function() {
    console.log('=== Doppler Page Initialization Started ===');

    // Get all DOM elements with error checking
    const fileNameInput = document.getElementById('doppler-file-name');
    const velocityInput = document.getElementById('doppler-velocity');
    const frequencyInput = document.getElementById('doppler-frequency');
    const generateBtn = document.getElementById('doppler-btn');
    const statusDiv = document.getElementById('doppler-status');
    const playerOriginalDiv = document.getElementById('doppler-audio-player-original');
    const playerResampledDiv = document.getElementById('doppler-audio-player-resampled');
    const timeDomainOriginalDiv = document.getElementById('doppler-time-domain-graph-original');
    const timeDomainResampledDiv = document.getElementById('doppler-time-domain-graph-resampled');
    const resampledGraphContainer = document.getElementById('doppler-resampled-graph-container');
    const debugDiv = document.getElementById('doppler-debug-info');
    const nyquistSection = document.getElementById('doppler-nyquist-section');
    const resampleSlider = document.getElementById('doppler-resample-slider');
    const currentSrDisplay = document.getElementById('doppler-current-sr-display');
    const playResampledBtn = document.getElementById('doppler-play-resampled-btn');

    console.log('Button element check:', {
        playResampledBtn: !!playResampledBtn,
        resampleSlider: !!resampleSlider
    });

    // State variables
    let originalAudioData = null;
    let currentSampleRate = 44100;
    let originalSampleRate = 44100;
    let originalMaxFrequency = 0;
    let nyquistFrequencyRequired = 0;
    let resampledAudioData = null;
    let sliderTimeout = null;
    let originalDuration = 0;
    let audioDataURI = null;

    const MAX_PLOT_POINTS = 2000;

    // Verify critical elements exist
    if (!generateBtn) {
        console.error('CRITICAL: Generate button not found!');
        alert('Page loading error: Generate button not found. Please refresh the page.');
        return;
    }

    if (!statusDiv) {
        console.error('CRITICAL: Status div not found!');
        alert('Page loading error: Status display not found. Please refresh the page.');
        return;
    }

    console.log('✓ All critical DOM elements found');

    // Helper Functions
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
        if (statusDiv) {
            statusDiv.className = `alert ${alertClass}`;
            statusDiv.innerHTML = message;
            statusDiv.style.display = 'block';
            console.log(`Status: ${message}`);
        }
    }

    // Resample audio to maintain the SAME DURATION but at different sample rate
    // This creates aliasing artifacts when downsampling below Nyquist frequency
    function resampleAudioByDuration(audioData, currentRate, targetRate, duration) {
        // Calculate how many samples we need at target rate to maintain same duration
        // Duration = samples / sampleRate, so samples = duration * sampleRate
        const targetSamples = Math.floor(duration * targetRate);
        const resampled = new Float32Array(targetSamples);

        console.log(`\n=== RESAMPLING AUDIO ===`);
        console.log(`Input: ${audioData.length} samples @ ${currentRate}Hz (duration: ${(audioData.length/currentRate).toFixed(3)}s)`);
        console.log(`Target: ${targetSamples} samples @ ${targetRate}Hz (duration: ${duration.toFixed(3)}s)`);
        console.log(`Downsample ratio: ${(audioData.length/targetSamples).toFixed(3)}x`);

        // Simple downsampling by taking every Nth sample (causes aliasing!)
        const step = audioData.length / targetSamples;
        
        for (let i = 0; i < targetSamples; i++) {
            const sourceIndex = Math.floor(i * step);
            resampled[i] = audioData[Math.min(sourceIndex, audioData.length - 1)];
        }

        console.log(`✓ Resampling complete: ${resampled.length} samples @ ${targetRate}Hz`);
        console.log(`   Playback duration will be: ${(resampled.length/targetRate).toFixed(3)}s`);
        return Array.from(resampled);
    }

    function downsampleForPlotting(data, maxPoints) {
        if (data.length <= maxPoints) return data;
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

    function plotTimeDomain(waveform, sr, divElement, title, showMarkers = false) {
        if (!divElement) {
            console.error('plotTimeDomain: divElement is null');
            return;
        }

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
            traceConfig.marker = { size: 4, color: '#c4b5fd', symbol: 'circle' };
        }

        const layout = {
            title: title,
            xaxis: { title: 'Time (s)', color: '#c9d1d9', gridcolor: '#30363d' },
            yaxis: { title: 'Amplitude', color: '#c9d1d9', gridcolor: '#30363d', range: [-1.1, 1.1] },
            plot_bgcolor: '#161b22',
            paper_bgcolor: '#161b22',
            font: { color: '#c9d1d9' },
            height: 350,
            hovermode: 'closest',
            margin: { l: 50, r: 30, t: 50, b: 50 }
        };

        Plotly.newPlot(divElement, [traceConfig], layout, {responsive: true});
    }

    function updateNyquistAnalysis(currentRate) {
        const updates = {
            'doppler-original-sr': `${originalSampleRate} Hz`,
            'doppler-current-sr': `${currentRate} Hz`,
            'doppler-max-freq': `${originalMaxFrequency.toFixed(0)} Hz`,
            'doppler-nyquist-freq': `${nyquistFrequencyRequired.toFixed(0)} Hz`
        };

        for (const [id, text] of Object.entries(updates)) {
            const elem = document.getElementById(id);
            if (elem) elem.textContent = text;
        }

        const statusElement = document.getElementById('doppler-sampling-status');
        if (statusElement) {
            if (currentRate < nyquistFrequencyRequired) {
                statusElement.textContent = `⚠️ Under-sampling (Current rate < ${nyquistFrequencyRequired.toFixed(0)} Hz) - ALIASING WILL OCCUR`;
                statusElement.style.color = '#f78166';
            } else {
                statusElement.textContent = `✓ Over-sampling (Current rate ≥ ${nyquistFrequencyRequired.toFixed(0)} Hz) - No Aliasing`;
                statusElement.style.color = '#c4b5fd';
            }
        }

        if (nyquistSection) nyquistSection.style.display = 'block';
    }

    function renderResults(data) {
        console.log('Rendering results...');

        if (!data || !data.waveform_data) {
            console.error('Invalid data:', data);
            updateStatus('❌ Invalid data received from server', 'alert-danger');
            return;
        }

        // Store data
        originalAudioData = data.waveform_data;  // Downsampled for plotting
        currentSampleRate = data.sr;
        originalSampleRate = data.sr;
        originalMaxFrequency = data.max_frequency;
        nyquistFrequencyRequired = 2 * originalMaxFrequency;
        resampledAudioData = data.waveform_data;  // Initially same as original (downsampled)
        audioDataURI = `data:audio/wav;base64,${data.audio_b64}`;
        
        // IMPORTANT: Calculate duration from the ACTUAL audio file, not downsampled data
        // The waveform_data is downsampled for plotting, but the WAV contains full audio
        // For a 5-second file at 44100 Hz, we have 220500 samples, but waveform_data only has ~10000
        originalDuration = 5.0;  // FIXED_DURATION from server
        
        // For resampling, we'll need the full audio data, so let's decode the WAV
        console.log('Decoding full audio from WAV...');
        fetch(audioDataURI)
            .then(response => response.arrayBuffer())
            .then(arrayBuffer => {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                return audioContext.decodeAudioData(arrayBuffer);
            })
            .then(audioBuffer => {
                // Extract the full audio samples
                const fullAudioData = audioBuffer.getChannelData(0);
                originalAudioData = Array.from(fullAudioData);
                resampledAudioData = Array.from(fullAudioData);
                originalDuration = audioBuffer.duration;
                
                console.log(`✓ Full audio decoded: ${originalAudioData.length} samples at ${originalSampleRate} Hz = ${originalDuration.toFixed(3)}s`);
            })
            .catch(error => {
                console.error('Error decoding audio:', error);
                // Fallback to downsampled data
                console.warn('Using downsampled waveform data for resampling');
            });

        console.log(`Original audio (downsampled for plot): ${data.waveform_data.length} samples at ${originalSampleRate} Hz`);
        console.log(`Actual duration: ${originalDuration.toFixed(3)}s`);

        // Update original audio player
        if (playerOriginalDiv) {
            try {
                playerOriginalDiv.innerHTML = `
                    <h5 style="color: #c4b5fd;">Original Audio Player (${originalSampleRate} Hz):</h5>
                    <audio controls src="${audioDataURI}" style="width: 100%;"></audio>
                `;
                console.log('✓ Audio player updated');
            } catch (error) {
                console.error('Error updating audio player:', error);
            }
        }

        // Plot original audio
        if (timeDomainOriginalDiv) {
            try {
                plotTimeDomain(
                    data.waveform_data,
                    data.sr,
                    timeDomainOriginalDiv,
                    `Original Audio: Time Domain (${data.waveform_data.length} samples at ${data.sr} Hz, Duration: ${originalDuration.toFixed(2)}s)`,
                    false
                );
                console.log('✓ Waveform plotted');
            } catch (error) {
                console.error('Error plotting waveform:', error);
            }
        }

        // Update Nyquist analysis
        updateNyquistAnalysis(data.sr);

        // Update slider
        if (resampleSlider) resampleSlider.value = data.sr;
        if (currentSrDisplay) currentSrDisplay.textContent = `${data.sr} Hz`;

        // Update debug info
        if (debugDiv) {
            debugDiv.innerHTML = `<pre>File: ${data.output_file_name}
Original Sample Rate: ${originalSampleRate} Hz
Current Sample Rate: ${data.sr} Hz
Duration: ${originalDuration.toFixed(2)} seconds
Original Max Frequency: ${originalMaxFrequency.toFixed(2)} Hz
Nyquist Frequency Required (2×F(max)): ${nyquistFrequencyRequired.toFixed(2)} Hz
Velocity: ${data.velocity_kmh} km/h
Base Frequency: ${data.base_frequency} Hz
Doppler Shift (Approaching): ${data.doppler_shift_approach.toFixed(2)} Hz
Doppler Shift (Receding): ${data.doppler_shift_recede.toFixed(2)} Hz
Sampling Status: ${data.sr >= nyquistFrequencyRequired ? 'Over-sampling' : 'Under-sampling'}</pre>`;
        }

        updateStatus(`✅ Doppler effect generated successfully! File: ${data.output_file_name}`, 'alert-success');
        console.log('✓ Results rendered successfully');
    }

    // Optimized slider with debouncing - same as detect_logic.js
    if (resampleSlider) {
        resampleSlider.oninput = (e) => {
            const newSampleRate = parseInt(e.target.value);
            if (currentSrDisplay) currentSrDisplay.textContent = `${newSampleRate} Hz`;

            // Clear previous timeout
            if (sliderTimeout) {
                clearTimeout(sliderTimeout);
            }

            // Update status immediately
            const currentSrElem = document.getElementById('doppler-current-sr');
            if (currentSrElem) currentSrElem.textContent = `${newSampleRate} Hz`;

            const statusElement = document.getElementById('doppler-sampling-status');
            if (statusElement) {
                if (newSampleRate < nyquistFrequencyRequired) {
                    statusElement.textContent = `⚠️ Under-sampling (Current rate < ${nyquistFrequencyRequired.toFixed(0)} Hz) - ALIASING WILL OCCUR`;
                    statusElement.style.color = '#f78166';
                } else {
                    statusElement.textContent = `✓ Over-sampling (Current rate ≥ ${nyquistFrequencyRequired.toFixed(0)} Hz) - No Aliasing`;
                    statusElement.style.color = '#c4b5fd';
                }
            }

            // Debounce the heavy resampling and plotting operation
            sliderTimeout = setTimeout(() => {
                if (originalAudioData && originalDuration > 0) {
                    console.log(`\n=== RESAMPLING TO ${newSampleRate} Hz ===`);

                    // FIXED: Resample based on duration to maintain the same length
                    resampledAudioData = resampleAudioByDuration(
                        originalAudioData,
                        currentSampleRate,
                        newSampleRate,
                        originalDuration
                    );

                    if (resampledGraphContainer) resampledGraphContainer.style.display = 'block';

                    // Calculate actual duration to verify
                    const resampledDuration = resampledAudioData.length / newSampleRate;

                    // Show markers only for very low sample rates to visualize downsampling
                    const showMarkers = newSampleRate < 4000;

                    if (timeDomainResampledDiv) {
                        plotTimeDomain(
                            resampledAudioData,
                            newSampleRate,
                            timeDomainResampledDiv,
                            `Resampled Audio: Time Domain (${resampledAudioData.length} samples at ${newSampleRate} Hz, Duration: ${resampledDuration.toFixed(2)}s)`,
                            showMarkers
                        );
                    }

                    console.log(`✓ Resampled waveform plotted`);
                    
                    // Update status to show resampling is complete
                    if (newSampleRate < nyquistFrequencyRequired) {
                        updateStatus(`✓ Audio resampled to ${newSampleRate} Hz - Ready to play with ALIASING! Click "Play Resampled Audio" button.`, 'alert-warning');
                    } else {
                        updateStatus(`✓ Audio resampled to ${newSampleRate} Hz - Ready to play (No aliasing at this rate).`, 'alert-info');
                    }
                }
            }, 150); // 150ms debounce delay
        };
    }

    // Play resampled button handler
    if (playResampledBtn) {
        playResampledBtn.onclick = () => {
            console.log(`\n=== PLAY RESAMPLED BUTTON CLICKED ===`);
            console.log(`resampledAudioData exists: ${!!resampledAudioData}`);
            console.log(`resampledAudioData length: ${resampledAudioData ? resampledAudioData.length : 'N/A'}`);
            console.log(`originalAudioData exists: ${!!originalAudioData}`);
            
            if (!resampledAudioData || resampledAudioData.length === 0) {
                console.error('❌ No resampled audio data available!');
                updateStatus("⚠️ No resampled audio available. Please generate Doppler audio first.", 'alert-warning');
                return;
            }

            if (!resampleSlider) {
                console.error('❌ Resample slider not found!');
                updateStatus("⚠️ Error: Slider not found.", 'alert-danger');
                return;
            }

            const currentRate = parseInt(resampleSlider.value);
            
            console.log(`\n=== PLAYING RESAMPLED AUDIO ===`);
            console.log(`Target sample rate: ${currentRate} Hz`);
            console.log(`Original sample rate: ${originalSampleRate} Hz`);
            console.log(`Original duration: ${originalDuration.toFixed(3)}s`);
            console.log(`Resampled data samples: ${resampledAudioData.length}`);

            // Calculate the duration when played at the target rate
            const resampledDuration = resampledAudioData.length / currentRate;
            console.log(`Duration when played at ${currentRate}Hz: ${resampledDuration.toFixed(3)}s`);

            // Create WAV at the target sample rate (this will preserve the aliasing effect)
            console.log(`Creating WAV buffer...`);
            let wavBuffer;
            try {
                wavBuffer = audioArrayToWav(resampledAudioData, currentRate);
                console.log(`✓ WAV buffer created, size: ${wavBuffer.byteLength} bytes`);
            } catch (error) {
                console.error('❌ Error creating WAV:', error);
                updateStatus("❌ Error creating audio file.", 'alert-danger');
                return;
            }

            const blob = new Blob([wavBuffer], { type: 'audio/wav' });
            const url = URL.createObjectURL(blob);
            console.log(`✓ Blob URL created: ${url.substring(0, 50)}...`);

            if (playerResampledDiv) {
                playerResampledDiv.style.display = 'block';

                const aliasingWarning = currentRate < nyquistFrequencyRequired
                    ? `<span style="color: #f78166;">⚠️ ALIASING EFFECT PRESENT</span>`
                    : `<span style="color: #c4b5fd;">✓ No Aliasing</span>`;

                playerResampledDiv.innerHTML = `
                    <h5 style="color: #f78166;">Resampled Audio Player (${currentRate} Hz, Duration: ${resampledDuration.toFixed(2)}s) ${aliasingWarning}</h5>
                    <audio controls autoplay src="${url}" style="width: 100%;"></audio>
                    <p class="text-secondary mt-2" style="font-size: 0.9rem;">
                        <strong>Info:</strong> ${resampledAudioData.length} samples @ ${currentRate}Hz = ${resampledDuration.toFixed(2)}s playback
                    </p>
                `;
            }

            const statusMsg = currentRate < nyquistFrequencyRequired
                ? `🔊 Playing aliased audio at ${currentRate} Hz - Listen for ALIASING artifacts! Duration: ${resampledDuration.toFixed(2)}s`
                : `🔊 Playing resampled audio at ${currentRate} Hz - No aliasing. Duration: ${resampledDuration.toFixed(2)}s`;

            updateStatus(statusMsg, currentRate < nyquistFrequencyRequired ? 'alert-warning' : 'alert-info');
            
            console.log(`✓ Audio player created and playing`);
        };
        console.log('✓ Play Resampled button onclick handler attached');
    } else {
        console.warn('⚠️ Play Resampled button not found - handler not attached');
    }

    // MAIN GENERATE BUTTON HANDLER
    generateBtn.onclick = async () => {
        console.log('=== Generate Button Clicked ===');

        const fileName = fileNameInput ? fileNameInput.value.trim() : '';
        const velocity = velocityInput ? parseFloat(velocityInput.value) : 0;
        const frequency = frequencyInput ? parseFloat(frequencyInput.value) : 0;

        console.log('Inputs:', { fileName, velocity, frequency });

        // Validation
        if (!fileName) {
            updateStatus("⚠️ Please enter a file name.", 'alert-danger');
            return;
        }
        if (!velocity || velocity <= 0) {
            updateStatus("⚠️ Please enter a valid velocity (positive number).", 'alert-danger');
            return;
        }
        if (!frequency || frequency <= 0) {
            updateStatus("⚠️ Please enter a valid frequency (positive number).", 'alert-danger');
            return;
        }

        // Disable button
        generateBtn.disabled = true;
        updateStatus("🔄 Generating Doppler audio on server... Please wait.", 'alert-warning');

        // Reset UI
        if (nyquistSection) nyquistSection.style.display = 'none';
        if (resampledGraphContainer) resampledGraphContainer.style.display = 'none';
        if (playerResampledDiv) playerResampledDiv.style.display = 'none';
        if (playerOriginalDiv) playerOriginalDiv.innerHTML = '<h5 style="color: #c4b5fd;">Original Audio Player:</h5>';
        if (timeDomainOriginalDiv) timeDomainOriginalDiv.innerHTML = '';
        if (timeDomainResampledDiv) timeDomainResampledDiv.innerHTML = '';
        if (debugDiv) debugDiv.innerHTML = '';

        originalMaxFrequency = 0;
        nyquistFrequencyRequired = 0;
        originalAudioData = null;
        resampledAudioData = null;
        originalDuration = 0;

        const csrftoken = getCookie('csrftoken');

        try {
            console.log('Sending request...');

            const response = await fetch('/api/generate_doppler/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrftoken
                },
                body: JSON.stringify({
                    file_name: fileName,
                    velocity: velocity,
                    frequency: frequency
                })
            });

            console.log('Response status:', response.status);

            if (!response.ok) {
                let errorMessage = `Server error: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.error || errorMessage;
                } catch (e) {
                    const errorText = await response.text();
                    errorMessage = errorText || errorMessage;
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();
            console.log('Data received successfully');

            if (data.error) {
                throw new Error(data.error);
            }

            renderResults(data);

        } catch (error) {
            console.error('Error:', error);
            updateStatus(`❌ Error: ${error.message}`, 'alert-danger');
        } finally {
            generateBtn.disabled = false;
        }
    };

    // Initial status
    updateStatus("Enter parameters and click 'Generate Doppler Audio' to start.", 'alert-info');
    console.log('=== Doppler Page Initialization Complete ===');
});