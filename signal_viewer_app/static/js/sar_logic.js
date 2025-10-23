document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone-sar');
    const fileInput = document.getElementById('sar-file-input');
    const loadButton = document.getElementById('sar_load_button');
    const statusDiv = document.getElementById('sar-status');
    const imageDiv = document.getElementById('sar_output_image');
    const debugDiv = document.getElementById('sar_debug_info');

    let uploadedFile = null;

    function updateStatus(message, alertClass = 'alert-info') {
        statusDiv.className = `alert ${alertClass}`;
        statusDiv.innerHTML = message;
    }

    // Helper to get CSRF token (Unchanged)
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

    // --- FILE HANDLING LOGIC (Unchanged) ---

    function processDroppedFiles(files) {
        const file = files[0];
        if (!file) return;

        const fileName = file.name.toLowerCase();
        if (!fileName.endsWith('.tiff') && !fileName.endsWith('.tif')) {
            updateStatus("⚠️ Invalid file type. Please upload a .tiff or .tif file.", 'alert-danger');
            loadButton.disabled = true;
            return;
        }

        uploadedFile = file;
        updateStatus(`File selected: ${file.name}. Click 'Load and Process SAR Image'.`, 'alert-info');
        loadButton.disabled = false;
        debugDiv.innerHTML = '';
        imageDiv.innerHTML = '';
    }

    // 1. Click-to-Upload Handler (Standard programmatic click)
    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => {
             fileInput.click();
        });
    }

    // 2. Drag-and-Drop Visuals & Handler
    if (dropZone) {
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.style.borderColor = '#F6416C'; };
        dropZone.ondragleave = () => { dropZone.style.borderColor = '#30363d'; };
        dropZone.ondrop = (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '#30363d';
            processDroppedFiles(Array.from(e.dataTransfer.files));
        };
    }

    // 3. File Input Change Handler
    if (fileInput) {
        fileInput.onchange = (e) => {
            if (e.target.files.length > 0) {
                 processDroppedFiles(Array.from(e.target.files));
                 fileInput.value = '';
            }
        };
    }

    // --- BUTTON HANDLER: Load SAR Data (With Loading Effect) ---
    loadButton.addEventListener('click', async () => {
        if (!uploadedFile) {
            updateStatus("⚠️ No file is currently loaded.", 'alert-danger');
            return;
        }

        // --- START LOADING EFFECT ---
        const originalButtonText = loadButton.innerHTML;
        loadButton.disabled = true;
        loadButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> Processing...';
        updateStatus("Processing SAR data on server... This may take a moment.", 'alert-warning');

        await new Promise(resolve => setTimeout(resolve, 50));

        const formData = new FormData();
        formData.append('file', uploadedFile);

        const csrftoken = getCookie('csrftoken');

        try {
            const response = await fetch('/api/process_sar/', {
                method: 'POST',
                headers: { 'X-CSRFToken': csrftoken },
                body: formData,
            });

            const contentType = response.headers.get("content-type");

            if (response.ok && contentType && contentType.includes("application/json")) {
                const json_data = await response.json();

                if (json_data.error) {
                    throw new Error(json_data.error);
                }

                // Success Path
                imageDiv.innerHTML = `<img src="data:image/png;base64,${json_data.image_b64}" style="max-width: 100%; border-radius: 6px;" alt="Processed SAR Image"/>`;
                debugDiv.innerHTML = `<pre>${json_data.debug_info}</pre>`;
                updateStatus("✅ SAR image successfully generated and displayed.", 'alert-success');

            } else {
                // Handle non-JSON or server error response
                const errorText = await response.text();
                throw new Error(`Server status ${response.status}. Full response check required. Content: ${errorText.substring(0, 500)}`);
            }
        } catch (error) {
            // Error Path
            console.error('SAR Processing Error:', error);
            debugDiv.innerHTML = `<pre style="color:red;">Error: ${error.message}</pre>`;
            updateStatus(`❌ SAR Processing Failed. Check console for details.`, 'alert-danger');
        } finally {
             // --- END LOADING EFFECT ---
             loadButton.disabled = false;
             loadButton.innerHTML = originalButtonText; // Restore original button text
        }
    });
});