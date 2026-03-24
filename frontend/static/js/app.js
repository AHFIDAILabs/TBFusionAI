/**
 * TBFusionAI - Main JavaScript Application
 * Version: 3.0 - Complete Fix for Recording and Upload Issues
 * 
 * FIXES:
 * 1. WebM recording support
 * 2. Proper File object creation
 * 3. Fixed double-click upload issue
 * 4. FAQ accordion working
 * 5. Better error messages
 */

// ============================================================================
// Global Variables
// ============================================================================

let currentAudioFile = null;
let predictionResult = null;
let mediaRecorder = null;
let audioChunks = [];

// API Base URL
const API_BASE_URL = window.location.origin + '/api/v1';

// ============================================================================
// Utility Functions
// ============================================================================

function showLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.style.display = 'flex';
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.style.display = 'none';
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    document.body.appendChild(notification);
    setTimeout(() => notification.classList.add('show'), 100);
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function validateAudioFile(file) {
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/webm'];
    const maxSize = 10 * 1024 * 1024; // 10MB
    
    if (!validTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|ogg|webm)$/i)) {
        return {
            valid: false,
            error: 'Invalid file type. Please upload a WAV, MP3, OGG, or WebM file.'
        };
    }
    if (file.size > maxSize) {
        return { valid: false, error: 'File size exceeds 10MB limit.' };
    }
    if (file.size === 0) {
        return { valid: false, error: 'File is empty.' };
    }
    return { valid: true };
}

// ============================================================================
// Prediction Page Functions
// ============================================================================

function initPredictionPage() {
    const uploadArea = document.getElementById('audioUploadArea');
    const audioFileInput = document.getElementById('audioFile');
    const predictionForm = document.getElementById('predictionForm');
    
    if (!uploadArea || !audioFileInput || !predictionForm) return;

    const startRecordBtn = document.getElementById('startRecordBtn');
    const stopRecordBtn = document.getElementById('stopRecordBtn');
    const removeAudioBtn = document.getElementById('removeAudio');
    const newPredictionBtn = document.getElementById('newPredictionBtn');
    const printResultsBtn = document.getElementById('printResultsBtn');

    // Recording buttons
    if (startRecordBtn && stopRecordBtn) {
        startRecordBtn.addEventListener('click', startRecording);
        stopRecordBtn.addEventListener('click', stopRecording);
    }
    
    // Upload area click
    uploadArea.addEventListener('click', () => audioFileInput.click());
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary-color)';
        uploadArea.style.backgroundColor = 'var(--lighter)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--gray-lighter)';
        uploadArea.style.backgroundColor = 'transparent';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--gray-lighter)';
        const files = e.dataTransfer.files;
        if (files.length > 0) handleAudioFileSelect(files[0]);
    });
    
    // CRITICAL FIX: Use 'change' event only ONCE
    audioFileInput.addEventListener('change', handleFileInputChange, { once: false });
    
    // Remove audio button
    if (removeAudioBtn) {
        removeAudioBtn.addEventListener('click', (e) => {
            e.preventDefault();
            removeAudioFile();
        });
    }
    
    // Form submission
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        await handlePredictionSubmit();
    });
    
    // New prediction button
    if (newPredictionBtn) {
        newPredictionBtn.addEventListener('click', resetPredictionForm);
    }
    
    // Print button
    if (printResultsBtn) {
        printResultsBtn.addEventListener('click', () => window.print());
    }
}

// ============================================================================
// File Input Handler - FIXED
// ============================================================================

function handleFileInputChange(e) {
    // CRITICAL FIX: Handle the file selection properly
    if (e.target.files && e.target.files.length > 0) {
        handleAudioFileSelect(e.target.files[0]);
    }
}

// ============================================================================
// Audio Recording Functions - FIXED
// ============================================================================

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
        
        // Try WAV first, fallback to WebM
        let options = {};
        if (MediaRecorder.isTypeSupported('audio/wav')) {
            options = { mimeType: 'audio/wav' };
        } else if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
            options = { mimeType: 'audio/webm;codecs=opus' };
        } else if (MediaRecorder.isTypeSupported('audio/webm')) {
            options = { mimeType: 'audio/webm' };
        }
        
        mediaRecorder = new MediaRecorder(stream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        // Store stream for cleanup
        mediaRecorder.stream = stream;

        mediaRecorder.start();
        console.log('✓ Recording started with format:', mediaRecorder.mimeType);
        
        document.getElementById('startRecordBtn').classList.add('hidden');
        document.getElementById('stopRecordBtn').classList.remove('hidden');
        
        showNotification('Recording... cough clearly into the microphone', 'info');
        
    } catch (err) {
        console.error('❌ Microphone access error:', err);
        showNotification('Microphone access denied. Please allow microphone access.', 'error');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        
        // CRITICAL FIX: Define onstop HERE, not in startRecording
        mediaRecorder.onstop = async () => {
            try {
                // Determine format
                const mimeType = mediaRecorder.mimeType || 'audio/webm';
                const extension = mimeType.includes('webm') ? 'webm' : 'wav';
                
                // Create Blob
                const audioBlob = new Blob(audioChunks, { type: mimeType });
                
                // CRITICAL: Create proper File object
                const timestamp = Date.now();
                const file = new File(
                    [audioBlob], 
                    `recorded_cough_${timestamp}.${extension}`,
                    { 
                        type: mimeType,
                        lastModified: timestamp
                    }
                );
                
                console.log('✓ Recording stopped. File created:', {
                    name: file.name,
                    size: file.size,
                    type: file.type,
                    extension: extension
                });
                
                // Handle file
                handleAudioFileSelect(file);
                
                // Cleanup media tracks
                if (mediaRecorder.stream) {
                    mediaRecorder.stream.getTracks().forEach(track => track.stop());
                }
                
                showNotification('Recording saved successfully!', 'success');
                
            } catch (error) {
                console.error('❌ Error processing recording:', error);
                showNotification('Failed to process recording. Please try again.', 'error');
            }
        };
        
        document.getElementById('startRecordBtn').classList.remove('hidden');
        document.getElementById('stopRecordBtn').classList.add('hidden');
    }
}

function handleAudioFileSelect(file) {
    const validation = validateAudioFile(file);
    if (!validation.valid) {
        showNotification(validation.error, 'error');
        return;
    }
    
    currentAudioFile = file;
    console.log('✓ Audio file selected:', {
        name: file.name,
        size: formatFileSize(file.size),
        type: file.type
    });
    
    document.getElementById('audioUploadArea').style.display = 'none';
    document.getElementById('audioPreview').style.display = 'block';
    document.getElementById('audioFileName').textContent = `${file.name} (${formatFileSize(file.size)})`;
    
    // Create preview
    const audioPlayer = document.getElementById('audioPlayer');
    const url = URL.createObjectURL(file);
    audioPlayer.src = url;
    
    // Clear the file input to allow re-selection of the same file
    const audioFileInput = document.getElementById('audioFile');
    if (audioFileInput) {
        audioFileInput.value = '';
    }
}

function removeAudioFile() {
    currentAudioFile = null;
    
    // Revoke object URL to free memory
    const audioPlayer = document.getElementById('audioPlayer');
    if (audioPlayer.src) {
        URL.revokeObjectURL(audioPlayer.src);
        audioPlayer.src = '';
    }
    
    document.getElementById('audioUploadArea').style.display = 'block';
    document.getElementById('audioPreview').style.display = 'none';
    document.getElementById('audioFile').value = '';
    
    // Reset recording buttons
    document.getElementById('startRecordBtn').classList.remove('hidden');
    document.getElementById('stopRecordBtn').classList.add('hidden');
}

// ============================================================================
// Prediction Submission - FIXED
// ============================================================================

async function handlePredictionSubmit() {
    const form = document.getElementById('predictionForm');
    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }
    
    if (!currentAudioFile) {
        showNotification('Please record or upload a cough sample', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const formData = new FormData(form);
        
        // CRITICAL: Remove any existing audio_file and add our file
        formData.delete('audio_file');
        formData.append('audio_file', currentAudioFile, currentAudioFile.name);

        console.log('📤 Submitting prediction:', {
            audioFile: currentAudioFile.name,
            audioSize: formatFileSize(currentAudioFile.size),
            audioType: currentAudioFile.type
        });

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData,
            mode: 'cors'
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || result.message || 'Prediction failed');
        }
        
        console.log('✓ Prediction successful:', result);
        
        predictionResult = result;
        displayPredictionResults(result);
        showNotification('Analysis completed!', 'success');
        
    } catch (error) {
        console.error('❌ Prediction error:', error);
        showNotification(error.message || 'Prediction failed. Please try again.', 'error');
    } finally {
        hideLoading();
    }
}

function displayPredictionResults(result) {
    const resultsPanel = document.getElementById('resultsPanel');
    const isTBPositive = result.prediction_class === 1;
    
    resultsPanel.style.display = 'block';
    resultsPanel.scrollIntoView({ behavior: 'smooth' });
    
    document.getElementById('predictionBadge').className = `prediction-badge ${isTBPositive ? 'tb-positive' : 'tb-negative'}`;
    document.getElementById('badgeIcon').innerHTML = isTBPositive ? '<i class="fas fa-exclamation-triangle"></i>' : '<i class="fas fa-check-circle"></i>';
    document.getElementById('predictionText').textContent = result.prediction;
    document.getElementById('probabilityText').textContent = `Probability: ${(result.probability * 100).toFixed(1)}%`;
    
    document.getElementById('confidenceValue').textContent = result.confidence_level;
    document.getElementById('confidenceFill').style.width = `${(result.confidence * 100).toFixed(0)}%`;
    document.getElementById('recommendationText').textContent = result.recommendation;
    
    const specImg = document.getElementById('spectrogramImage');
    if (result.spectrogram_base64) {
        document.getElementById('spectrogramCard').style.display = 'block';
        specImg.src = `data:image/png;base64,${result.spectrogram_base64}`;
    } else {
        document.getElementById('spectrogramCard').style.display = 'none';
    }
}

function resetPredictionForm() {
    document.getElementById('predictionForm').reset();
    removeAudioFile();
    document.getElementById('resultsPanel').style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================================
// FAQ Accordion - FIXED
// ============================================================================

function initFAQ() {
    console.log('🔧 Initializing FAQ accordion...');
    
    const faqItems = document.querySelectorAll('.faq-item');
    console.log(`✓ Found ${faqItems.length} FAQ items`);
    
    if (faqItems.length === 0) {
        console.warn('⚠️ No FAQ items found. Not on FAQ page or elements missing.');
        return;
    }
    
    faqItems.forEach((item, index) => {
        const question = item.querySelector('.faq-question');
        const answer = item.querySelector('.faq-answer');
        
        if (!question) {
            console.error(`❌ FAQ item ${index} missing .faq-question`);
            return;
        }
        
        if (!answer) {
            console.error(`❌ FAQ item ${index} missing .faq-answer`);
            return;
        }
        
        // Remove existing listeners by cloning
        const newQuestion = question.cloneNode(true);
        question.replaceWith(newQuestion);
        
        // Add click listener
        newQuestion.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            console.log(`📌 Clicked FAQ item ${index + 1}`);
            
            const isCurrentlyActive = item.classList.contains('active');
            
            // Close all other items
            faqItems.forEach(otherItem => {
                if (otherItem !== item) {
                    otherItem.classList.remove('active');
                }
            });
            
            // Toggle current item
            if (isCurrentlyActive) {
                item.classList.remove('active');
                console.log(`➖ Closed FAQ item ${index + 1}`);
            } else {
                item.classList.add('active');
                console.log(`➕ Opened FAQ item ${index + 1}`);
            }
        });
        
        console.log(`✓ FAQ item ${index + 1} initialized`);
    });
    
    console.log('✅ FAQ accordion fully initialized!');
}

// ============================================================================
// API Health Check
// ============================================================================

async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        if (data.status === 'healthy' && data.model_loaded) {
            console.log('✓ API is healthy and models loaded');
            return true;
        } else {
            console.warn('⚠️ API running but models not loaded');
            return false;
        }
    } catch (error) {
        console.error('✗ API health check failed:', error);
        return false;
    }
}

// ============================================================================
// Navigation
// ============================================================================

function initNavigation() {
    const navToggle = document.getElementById('navToggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (!navToggle || !navMenu) return;
    
    navToggle.addEventListener('click', () => {
        navMenu.classList.toggle('active');
        navToggle.classList.toggle('active');
    });
    
    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!navToggle.contains(e.target) && !navMenu.contains(e.target)) {
            navMenu.classList.remove('active');
            navToggle.classList.remove('active');
        }
    });
    
    // Close menu when clicking link
    navMenu.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('active');
            navToggle.classList.remove('active');
        });
    });
}

// ============================================================================
// Initialization
// ============================================================================

function initApp() {
    console.log('🚀 Initializing TBFusionAI...');
    
    // Core functionality
    initNavigation();
    initPredictionPage();
    initFAQ();
    
    // Check API
    checkAPIHealth();
    
    console.log('✓ TBFusionAI initialized successfully');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}

// Export for global access
window.TBFusionAI = {
    showLoading,
    hideLoading,
    showNotification,
    checkAPIHealth,
    initFAQ
};



// /**
//  * TBFusionAI - Main JavaScript Application
//  * Handles all frontend interactions and API calls
//  * Version: 2.0 - Fixed recording, FAQ accordion, error handling
//  */

// // ============================================================================
// // Global Variables
// // ============================================================================

// let currentAudioFile = null;
// let predictionResult = null;
// let mediaRecorder = null;
// let audioChunks = [];

// // API Base URL
// const API_BASE_URL = window.location.origin + '/api/v1';

// // ============================================================================
// // Utility Functions
// // ============================================================================

// function showLoading() {
//     const overlay = document.getElementById('loadingOverlay');
//     if (overlay) overlay.style.display = 'flex';
// }

// function hideLoading() {
//     const overlay = document.getElementById('loadingOverlay');
//     if (overlay) overlay.style.display = 'none';
// }

// function showNotification(message, type = 'info') {
//     const notification = document.createElement('div');
//     notification.className = `notification notification-${type}`;
//     notification.innerHTML = `
//         <div class="notification-content">
//             <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
//             <span>${message}</span>
//         </div>
//     `;
//     document.body.appendChild(notification);
//     setTimeout(() => notification.classList.add('show'), 100);
//     setTimeout(() => {
//         notification.classList.remove('show');
//         setTimeout(() => notification.remove(), 300);
//     }, 5000);
// }

// function formatFileSize(bytes) {
//     if (bytes === 0) return '0 Bytes';
//     const k = 1024;
//     const sizes = ['Bytes', 'KB', 'MB', 'GB'];
//     const i = Math.floor(Math.log(bytes) / Math.log(k));
//     return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
// }

// function validateAudioFile(file) {
//     const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/webm'];
//     const maxSize = 10 * 1024 * 1024; // 10MB
    
//     if (!validTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|ogg|webm)$/i)) {
//         return {
//             valid: false,
//             error: 'Invalid file type. Please upload a WAV, MP3, OGG, or WebM file.'
//         };
//     }
//     if (file.size > maxSize) {
//         return { valid: false, error: 'File size exceeds 10MB limit.' };
//     }
//     return { valid: true };
// }

// // ============================================================================
// // Prediction Page Functions
// // ============================================================================

// function initPredictionPage() {
//     const uploadArea = document.getElementById('audioUploadArea');
//     const audioFileInput = document.getElementById('audioFile');
//     const predictionForm = document.getElementById('predictionForm');
    
//     if (!uploadArea || !audioFileInput || !predictionForm) return;

//     const startRecordBtn = document.getElementById('startRecordBtn');
//     const stopRecordBtn = document.getElementById('stopRecordBtn');
//     const removeAudioBtn = document.getElementById('removeAudio');
//     const newPredictionBtn = document.getElementById('newPredictionBtn');
//     const printResultsBtn = document.getElementById('printResultsBtn');

//     if (startRecordBtn && stopRecordBtn) {
//         startRecordBtn.addEventListener('click', startRecording);
//         stopRecordBtn.addEventListener('click', stopRecording);
//     }
    
//     uploadArea.addEventListener('click', () => audioFileInput.click());
    
//     uploadArea.addEventListener('dragover', (e) => {
//         e.preventDefault();
//         uploadArea.style.borderColor = 'var(--primary-color)';
//         uploadArea.style.backgroundColor = 'var(--lighter)';
//     });
    
//     uploadArea.addEventListener('dragleave', () => {
//         uploadArea.style.borderColor = 'var(--gray-lighter)';
//         uploadArea.style.backgroundColor = 'transparent';
//     });
    
//     uploadArea.addEventListener('drop', (e) => {
//         e.preventDefault();
//         uploadArea.style.borderColor = 'var(--gray-lighter)';
//         const files = e.dataTransfer.files;
//         if (files.length > 0) handleAudioFileSelect(files[0]);
//     });
    
//     audioFileInput.addEventListener('change', (e) => {
//         if (e.target.files.length > 0) handleAudioFileSelect(e.target.files[0]);
//     });
    
//     if (removeAudioBtn) {
//         removeAudioBtn.addEventListener('click', (e) => {
//             e.preventDefault();
//             removeAudioFile();
//         });
//     }
    
//     predictionForm.addEventListener('submit', async (e) => {
//         e.preventDefault();
//         await handlePredictionSubmit();
//     });
    
//     if (newPredictionBtn) {
//         newPredictionBtn.addEventListener('click', resetPredictionForm);
//     }
    
//     if (printResultsBtn) {
//         printResultsBtn.addEventListener('click', () => window.print());
//     }
// }

// // ============================================================================
// // Audio Recording Functions - FIXED
// // ============================================================================

// async function startRecording() {
//     try {
//         const stream = await navigator.mediaDevices.getUserMedia({ 
//             audio: {
//                 channelCount: 1,
//                 sampleRate: 16000,
//                 echoCancellation: true,
//                 noiseSuppression: true
//             }
//         });
        
//         // Determine best audio format
//         const options = MediaRecorder.isTypeSupported('audio/wav') 
//             ? { mimeType: 'audio/wav' }
//             : MediaRecorder.isTypeSupported('audio/webm') 
//             ? { mimeType: 'audio/webm' }
//             : {};
        
//         mediaRecorder = new MediaRecorder(stream, options);
//         audioChunks = [];

//         mediaRecorder.ondataavailable = (event) => {
//             if (event.data.size > 0) {
//                 audioChunks.push(event.data);
//             }
//         };

//         // Store stream reference for cleanup
//         mediaRecorder.stream = stream;

//         mediaRecorder.start();
//         console.log('✓ Recording started with format:', mediaRecorder.mimeType);
        
//         document.getElementById('startRecordBtn').classList.add('hidden');
//         document.getElementById('stopRecordBtn').classList.remove('hidden');
        
//         showNotification('Recording... cough clearly into the microphone', 'info');
        
//     } catch (err) {
//         console.error('Microphone access error:', err);
//         showNotification('Microphone access denied. Please allow microphone access.', 'error');
//     }
// }

// function stopRecording() {
//     if (mediaRecorder && mediaRecorder.state !== 'inactive') {
//         mediaRecorder.stop();
        
//         // CRITICAL FIX: Define onstop handler HERE, not in startRecording
//         mediaRecorder.onstop = async () => {
//             try {
//                 // Determine actual format
//                 const mimeType = mediaRecorder.mimeType || 'audio/wav';
//                 const extension = mimeType.includes('webm') ? 'webm' : 'wav';
                
//                 // Create proper Blob
//                 const audioBlob = new Blob(audioChunks, { type: mimeType });
                
//                 // CRITICAL: Create proper File object with all required properties
//                 const timestamp = new Date().getTime();
//                 const file = new File(
//                     [audioBlob], 
//                     `recorded_cough_${timestamp}.${extension}`,
//                     { 
//                         type: mimeType,
//                         lastModified: timestamp
//                     }
//                 );
                
//                 console.log('✓ Recording stopped. File created:', {
//                     name: file.name,
//                     size: file.size,
//                     type: file.type,
//                     lastModified: new Date(file.lastModified).toISOString()
//                 });
                
//                 // Handle the file
//                 handleAudioFileSelect(file);
                
//                 // Clean up: Stop all media tracks
//                 if (mediaRecorder.stream) {
//                     mediaRecorder.stream.getTracks().forEach(track => track.stop());
//                 }
                
//                 showNotification('Recording saved successfully!', 'success');
                
//             } catch (error) {
//                 console.error('Error processing recording:', error);
//                 showNotification('Failed to process recording. Please try again.', 'error');
//             }
//         };
        
//         document.getElementById('startRecordBtn').classList.remove('hidden');
//         document.getElementById('stopRecordBtn').classList.add('hidden');
//     }
// }

// function handleAudioFileSelect(file) {
//     const validation = validateAudioFile(file);
//     if (!validation.valid) {
//         showNotification(validation.error, 'error');
//         return;
//     }
    
//     currentAudioFile = file;
//     console.log('✓ Audio file selected:', {
//         name: file.name,
//         size: formatFileSize(file.size),
//         type: file.type
//     });
    
//     document.getElementById('audioUploadArea').style.display = 'none';
//     document.getElementById('audioPreview').style.display = 'block';
//     document.getElementById('audioFileName').textContent = `${file.name} (${formatFileSize(file.size)})`;
//     document.getElementById('audioPlayer').src = URL.createObjectURL(file);
// }

// function removeAudioFile() {
//     currentAudioFile = null;
//     document.getElementById('audioUploadArea').style.display = 'block';
//     document.getElementById('audioPreview').style.display = 'none';
//     document.getElementById('audioFile').value = '';
//     document.getElementById('audioPlayer').src = '';
    
//     // Reset recording buttons
//     document.getElementById('startRecordBtn').classList.remove('hidden');
//     document.getElementById('stopRecordBtn').classList.add('hidden');
// }

// // ============================================================================
// // Prediction Submission
// // ============================================================================

// async function handlePredictionSubmit() {
//     const form = document.getElementById('predictionForm');
//     if (!form.checkValidity()) {
//         form.reportValidity();
//         return;
//     }
    
//     if (!currentAudioFile) {
//         showNotification('Please record or upload a cough sample', 'error');
//         return;
//     }
    
//     showLoading();
    
//     try {
//         const formData = new FormData(form);
        
//         // CRITICAL: Remove any existing audio_file and add our file
//         formData.delete('audio_file');
//         formData.append('audio_file', currentAudioFile, currentAudioFile.name);

//         console.log('📤 Submitting prediction with:', {
//             audioFile: currentAudioFile.name,
//             audioSize: formatFileSize(currentAudioFile.size),
//             audioType: currentAudioFile.type
//         });

//         const response = await fetch(`${API_BASE_URL}/predict`, {
//             method: 'POST',
//             body: formData,
//             mode: 'cors'
//         });
        
//         const result = await response.json();
        
//         if (!response.ok) {
//             throw new Error(result.detail || result.message || 'Prediction failed');
//         }
        
//         console.log('✓ Prediction successful:', result);
        
//         predictionResult = result;
//         displayPredictionResults(result);
//         showNotification('Analysis completed!', 'success');
        
//     } catch (error) {
//         console.error('❌ Prediction error:', error);
//         showNotification(error.message || 'Prediction failed. Please try again.', 'error');
//     } finally {
//         hideLoading();
//     }
// }

// function displayPredictionResults(result) {
//     const resultsPanel = document.getElementById('resultsPanel');
//     const isTBPositive = result.prediction_class === 1;
    
//     resultsPanel.style.display = 'block';
//     resultsPanel.scrollIntoView({ behavior: 'smooth' });
    
//     document.getElementById('predictionBadge').className = `prediction-badge ${isTBPositive ? 'tb-positive' : 'tb-negative'}`;
//     document.getElementById('badgeIcon').innerHTML = isTBPositive ? '<i class="fas fa-exclamation-triangle"></i>' : '<i class="fas fa-check-circle"></i>';
//     document.getElementById('predictionText').textContent = result.prediction;
//     document.getElementById('probabilityText').textContent = `Probability: ${(result.probability * 100).toFixed(1)}%`;
    
//     document.getElementById('confidenceValue').textContent = result.confidence_level;
//     document.getElementById('confidenceFill').style.width = `${(result.confidence * 100).toFixed(0)}%`;
//     document.getElementById('recommendationText').textContent = result.recommendation;
    
//     const specImg = document.getElementById('spectrogramImage');
//     if (result.spectrogram_base64) {
//         document.getElementById('spectrogramCard').style.display = 'block';
//         specImg.src = `data:image/png;base64,${result.spectrogram_base64}`;
//     } else {
//         document.getElementById('spectrogramCard').style.display = 'none';
//     }
// }

// function resetPredictionForm() {
//     document.getElementById('predictionForm').reset();
//     removeAudioFile();
//     document.getElementById('resultsPanel').style.display = 'none';
//     window.scrollTo({ top: 0, behavior: 'smooth' });
// }

// // ============================================================================
// // FAQ Accordion - FIXED
// // ============================================================================

// function initFAQ() {
//     console.log('🔧 Initializing FAQ accordion...');
    
//     const faqItems = document.querySelectorAll('.faq-item');
//     console.log(`✓ Found ${faqItems.length} FAQ items`);
    
//     if (faqItems.length === 0) {
//         console.warn('⚠️ No FAQ items found. Not on FAQ page or elements missing.');
//         return;
//     }
    
//     faqItems.forEach((item, index) => {
//         const question = item.querySelector('.faq-question');
//         const answer = item.querySelector('.faq-answer');
        
//         if (!question) {
//             console.error(`❌ FAQ item ${index} missing .faq-question`);
//             return;
//         }
        
//         if (!answer) {
//             console.error(`❌ FAQ item ${index} missing .faq-answer`);
//             return;
//         }
        
//         // Remove existing listeners by cloning
//         const newQuestion = question.cloneNode(true);
//         question.replaceWith(newQuestion);
        
//         // Add click listener
//         newQuestion.addEventListener('click', function(e) {
//             e.preventDefault();
//             e.stopPropagation();
            
//             console.log(`📌 Clicked FAQ item ${index + 1}`);
            
//             const isCurrentlyActive = item.classList.contains('active');
            
//             // Close all other items
//             faqItems.forEach(otherItem => {
//                 if (otherItem !== item) {
//                     otherItem.classList.remove('active');
//                 }
//             });
            
//             // Toggle current item
//             if (isCurrentlyActive) {
//                 item.classList.remove('active');
//                 console.log(`➖ Closed FAQ item ${index + 1}`);
//             } else {
//                 item.classList.add('active');
//                 console.log(`➕ Opened FAQ item ${index + 1}`);
//             }
//         });
        
//         console.log(`✓ FAQ item ${index + 1} initialized`);
//     });
    
//     console.log('✅ FAQ accordion fully initialized!');
// }

// // ============================================================================
// // API Health Check
// // ============================================================================

// async function checkAPIHealth() {
//     try {
//         const response = await fetch(`${API_BASE_URL}/health`);
//         const data = await response.json();
//         if (data.status === 'healthy' && data.model_loaded) {
//             console.log('✓ API is healthy and models loaded');
//             return true;
//         } else {
//             console.warn('⚠️ API running but models not loaded');
//             return false;
//         }
//     } catch (error) {
//         console.error('✗ API health check failed:', error);
//         return false;
//     }
// }

// // ============================================================================
// // Navigation
// // ============================================================================

// function initNavigation() {
//     const navToggle = document.getElementById('navToggle');
//     const navMenu = document.querySelector('.nav-menu');
    
//     if (!navToggle || !navMenu) return;
    
//     navToggle.addEventListener('click', () => {
//         navMenu.classList.toggle('active');
//         navToggle.classList.toggle('active');
//     });
    
//     // Close menu when clicking outside
//     document.addEventListener('click', (e) => {
//         if (!navToggle.contains(e.target) && !navMenu.contains(e.target)) {
//             navMenu.classList.remove('active');
//             navToggle.classList.remove('active');
//         }
//     });
    
//     // Close menu when clicking link
//     navMenu.querySelectorAll('.nav-link').forEach(link => {
//         link.addEventListener('click', () => {
//             navMenu.classList.remove('active');
//             navToggle.classList.remove('active');
//         });
//     });
// }

// // ============================================================================
// // Initialization
// // ============================================================================

// function initApp() {
//     console.log('🚀 Initializing TBFusionAI...');
    
//     // Core functionality
//     initNavigation();
//     initPredictionPage();
//     initFAQ();
    
//     // Check API
//     checkAPIHealth();
    
//     console.log('✓ TBFusionAI initialized successfully');
// }

// // Initialize when DOM is ready
// if (document.readyState === 'loading') {
//     document.addEventListener('DOMContentLoaded', initApp);
// } else {
//     initApp();
// }

// // Export for global access
// window.TBFusionAI = {
//     showLoading,
//     hideLoading,
//     showNotification,
//     checkAPIHealth,
//     initFAQ
// };

// /**
//  * TBFusionAI - Main JavaScript Application
//  * Handles all frontend interactions and API calls
//  */

// // ============================================================================
// // Global Variables
// // ============================================================================

// let currentAudioFile = null;
// let predictionResult = null;
// let mediaRecorder = null;
// let audioChunks = [];

// // API Base URL
// const API_BASE_URL = window.location.origin + '/api/v1';

// // ============================================================================
// // Utility Functions
// // ============================================================================

// function showLoading() {
//     const overlay = document.getElementById('loadingOverlay');
//     if (overlay) overlay.style.display = 'flex';
// }

// function hideLoading() {
//     const overlay = document.getElementById('loadingOverlay');
//     if (overlay) overlay.style.display = 'none';
// }

// function showNotification(message, type = 'info') {
//     const notification = document.createElement('div');
//     notification.className = `notification notification-${type}`;
//     notification.innerHTML = `
//         <div class="notification-content">
//             <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
//             <span>${message}</span>
//         </div>
//     `;
//     document.body.appendChild(notification);
//     setTimeout(() => notification.classList.add('show'), 100);
//     setTimeout(() => {
//         notification.classList.remove('show');
//         setTimeout(() => notification.remove(), 300);
//     }, 5000);
// }

// function formatFileSize(bytes) {
//     if (bytes === 0) return '0 Bytes';
//     const k = 1024;
//     const sizes = ['Bytes', 'KB', 'MB', 'GB'];
//     const i = Math.floor(Math.log(bytes) / Math.log(k));
//     return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
// }

// function validateAudioFile(file) {
//     const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/webm'];
//     const maxSize = 10 * 1024 * 1024; // 10MB
    
//     if (!validTypes.includes(file.type) && !file.name.endsWith('.wav')) {
//         return {
//             valid: false,
//             error: 'Invalid file type. Please upload a WAV, MP3, or OGG file.'
//         };
//     }
//     if (file.size > maxSize) {
//         return { valid: false, error: 'File size exceeds 10MB limit.' };
//     }
//     return { valid: true };
// }

// // ============================================================================
// // Prediction Page Functions
// // ============================================================================

// function initPredictionPage() {
//     const uploadArea = document.getElementById('audioUploadArea');
//     const audioFileInput = document.getElementById('audioFile');
//     const predictionForm = document.getElementById('predictionForm');
    
//     if (!uploadArea || !audioFileInput || !predictionForm) return;

//     const startRecordBtn = document.getElementById('startRecordBtn');
//     const stopRecordBtn = document.getElementById('stopRecordBtn');
//     const removeAudioBtn = document.getElementById('removeAudio');
//     const newPredictionBtn = document.getElementById('newPredictionBtn');
//     const printResultsBtn = document.getElementById('printResultsBtn');

//     if (startRecordBtn && stopRecordBtn) {
//         startRecordBtn.addEventListener('click', startRecording);
//         stopRecordBtn.addEventListener('click', stopRecording);
//     }
    
//     uploadArea.addEventListener('click', () => audioFileInput.click());
    
//     uploadArea.addEventListener('dragover', (e) => {
//         e.preventDefault();
//         uploadArea.style.borderColor = 'var(--primary-color)';
//         uploadArea.style.backgroundColor = 'var(--lighter)';
//     });
    
//     uploadArea.addEventListener('dragleave', () => {
//         uploadArea.style.borderColor = 'var(--gray-lighter)';
//         uploadArea.style.backgroundColor = 'transparent';
//     });
    
//     uploadArea.addEventListener('drop', (e) => {
//         e.preventDefault();
//         uploadArea.style.borderColor = 'var(--gray-lighter)';
//         const files = e.dataTransfer.files;
//         if (files.length > 0) handleAudioFileSelect(files[0]);
//     });
    
//     audioFileInput.addEventListener('change', (e) => {
//         if (e.target.files.length > 0) handleAudioFileSelect(e.target.files[0]);
//     });
    
//     if (removeAudioBtn) {
//         removeAudioBtn.addEventListener('click', (e) => {
//             e.preventDefault();
//             removeAudioFile();
//         });
//     }
    
//     predictionForm.addEventListener('submit', async (e) => {
//         e.preventDefault();
//         await handlePredictionSubmit();
//     });
    
//     if (newPredictionBtn) {
//         newPredictionBtn.addEventListener('click', resetPredictionForm);
//     }
    
//     if (printResultsBtn) {
//         printResultsBtn.addEventListener('click', () => window.print());
//     }
// }

// async function startRecording() {
//     try {
//         const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
//         mediaRecorder = new MediaRecorder(stream);
//         audioChunks = [];

//         mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);
//         mediaRecorder.onstop = () => {
//             const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
//             const file = new File([audioBlob], "recorded_cough.wav", { type: 'audio/wav' });
//             handleAudioFileSelect(file);
//             stream.getTracks().forEach(track => track.stop());
//         };

//         mediaRecorder.start();
//         document.getElementById('startRecordBtn').classList.add('hidden');
//         document.getElementById('stopRecordBtn').classList.remove('hidden');
//         showNotification('Recording... cough clearly into the mic.', 'info');
//     } catch (err) {
//         showNotification('Microphone access denied.', 'error');
//     }
// }

// function stopRecording() {
//     if (mediaRecorder && mediaRecorder.state !== 'inactive') {
//         mediaRecorder.stop();
//         document.getElementById('startRecordBtn').classList.remove('hidden');
//         document.getElementById('stopRecordBtn').classList.add('hidden');
//     }
// }

// function handleAudioFileSelect(file) {
//     const validation = validateAudioFile(file);
//     if (!validation.valid) {
//         showNotification(validation.error, 'error');
//         return;
//     }
//     currentAudioFile = file;
//     document.getElementById('audioUploadArea').style.display = 'none';
//     document.getElementById('audioPreview').style.display = 'block';
//     document.getElementById('audioFileName').textContent = `${file.name} (${formatFileSize(file.size)})`;
//     document.getElementById('audioPlayer').src = URL.createObjectURL(file);
// }

// function removeAudioFile() {
//     currentAudioFile = null;
//     document.getElementById('audioUploadArea').style.display = 'block';
//     document.getElementById('audioPreview').style.display = 'none';
//     document.getElementById('audioFile').value = '';
//     document.getElementById('audioPlayer').src = '';
// }

// async function handlePredictionSubmit() {
//     const form = document.getElementById('predictionForm');
//     if (!form.checkValidity()) {
//         form.reportValidity();
//         return;
//     }
    
//     if (!currentAudioFile) {
//         showNotification('Please record or upload a cough sample', 'error');
//         return;
//     }
    
//     showLoading();
    
//     try {
//         const formData = new FormData(form);
//         // Ensure the backend receives exactly what it expects in routes.py
//         formData.delete('audio_file'); 
//         formData.append('audio_file', currentAudioFile, currentAudioFile.name || 'recorded_cough.wav');

//         const response = await fetch(`${API_BASE_URL}/predict`, {
//             method: 'POST',
//             body: formData,
//             mode: 'cors'
//         });
        
//         const result = await response.json();
        
//         if (!response.ok) {
//             throw new Error(result.detail || result.message || 'Prediction failed');
//         }
        
//         predictionResult = result;
//         displayPredictionResults(result);
//         showNotification('Analysis completed!', 'success');
        
//     } catch (error) {
//         console.error('Prediction error:', error);
//         showNotification(error.message, 'error');
//     } finally {
//         hideLoading();
//     }
// }

// function displayPredictionResults(result) {
//     const resultsPanel = document.getElementById('resultsPanel');
//     const isTBPositive = result.prediction_class === 1;
    
//     resultsPanel.style.display = 'block';
//     resultsPanel.scrollIntoView({ behavior: 'smooth' });
    
//     document.getElementById('predictionBadge').className = `prediction-badge ${isTBPositive ? 'tb-positive' : 'tb-negative'}`;
//     document.getElementById('badgeIcon').innerHTML = isTBPositive ? '<i class="fas fa-exclamation-triangle"></i>' : '<i class="fas fa-check-circle"></i>';
//     document.getElementById('predictionText').textContent = result.prediction;
//     document.getElementById('probabilityText').textContent = `Probability: ${(result.probability * 100).toFixed(1)}%`;
    
//     document.getElementById('confidenceValue').textContent = result.confidence_level;
//     document.getElementById('confidenceFill').style.width = `${(result.confidence * 100).toFixed(0)}%`;
//     document.getElementById('recommendationText').textContent = result.recommendation;
    
//     const specImg = document.getElementById('spectrogramImage');
//     if (result.spectrogram_base64) {
//         document.getElementById('spectrogramCard').style.display = 'block';
//         specImg.src = `data:image/png;base64,${result.spectrogram_base64}`;
//     } else {
//         document.getElementById('spectrogramCard').style.display = 'none';
//     }
// }

// function resetPredictionForm() {
//     document.getElementById('predictionForm').reset();
//     removeAudioFile();
//     document.getElementById('resultsPanel').style.display = 'none';
//     window.scrollTo({ top: 0, behavior: 'smooth' });
// }

// // ============================================================================
// // FAQ & Health
// // ============================================================================

// function initFAQ() {
//     document.querySelectorAll('.faq-item').forEach(item => {
//         item.querySelector('.faq-question').addEventListener('click', () => {
//             item.classList.toggle('active');
//         });
//     });
// }

// async function checkAPIHealth() {
//     try {
//         const response = await fetch(`${API_BASE_URL}/health`);
//         const data = await response.json();
//         if (data.status === 'healthy' && data.model_loaded) {
//             console.log('✓ API is healthy');
//             return true;
//         }
//     } catch (error) {
//         console.error('✗ API health check failed');
//     }
//     return false;
// }

// // ============================================================================
// // Initialization
// // ============================================================================

// function initApp() {
//     console.log('🚀 Initializing TBFusionAI...');
//     initPredictionPage();
//     initFAQ();
//     checkAPIHealth();
//     console.log('✓ TBFusionAI initialized successfully');
// }

// document.addEventListener('DOMContentLoaded', initApp);


// // ============ FAQ ACCORDION FIX ============
// /**
//  * FAQ Accordion Initialization
//  * CRITICAL: This must be called when the FAQ page loads
//  */
// function initFAQ() {
//     console.log('🔧 Starting FAQ accordion initialization...');
    
//     // Wait for DOM to be ready
//     if (document.readyState === 'loading') {
//         document.addEventListener('DOMContentLoaded', initFAQAccordion);
//     } else {
//         initFAQAccordion();
//     }
// }

// function initFAQAccordion() {
//     const faqItems = document.querySelectorAll('.faq-item');
    
//     console.log(`✓ Found ${faqItems.length} FAQ items`);
    
//     if (faqItems.length === 0) {
//         console.warn('⚠️ No FAQ items found! Check HTML structure.');
//         return;
//     }
    
//     faqItems.forEach((item, index) => {
//         const question = item.querySelector('.faq-question');
//         const answer = item.querySelector('.faq-answer');
        
//         if (!question) {
//             console.error(`❌ FAQ item ${index} missing .faq-question`);
//             return;
//         }
        
//         if (!answer) {
//             console.error(`❌ FAQ item ${index} missing .faq-answer`);
//             return;
//         }
        
//         // Remove any existing listeners
//         question.replaceWith(question.cloneNode(true));
//         const newQuestion = item.querySelector('.faq-question');
        
//         newQuestion.addEventListener('click', function(e) {
//             e.preventDefault();
//             e.stopPropagation();
            
//             console.log(`📌 Clicked FAQ item ${index + 1}`);
            
//             const isCurrentlyActive = item.classList.contains('active');
            
//             // Close all other items
//             faqItems.forEach(otherItem => {
//                 if (otherItem !== item) {
//                     otherItem.classList.remove('active');
//                 }
//             });
            
//             // Toggle current item
//             if (isCurrentlyActive) {
//                 item.classList.remove('active');
//                 console.log(`➖ Closed FAQ item ${index + 1}`);
//             } else {
//                 item.classList.add('active');
//                 console.log(`➕ Opened FAQ item ${index + 1}`);
//             }
//         });
        
//         console.log(`✓ Initialized FAQ item ${index + 1}`);
//     });
    
//     console.log('✅ FAQ accordion fully initialized!');
// }

// // Auto-initialize on page load
// if (typeof window !== 'undefined') {
//     window.initFAQ = initFAQ;
    
//     // If we're on the FAQ page, initialize immediately
//     if (window.location.pathname.includes('faq')) {
//         console.log('📄 FAQ page detected, auto-initializing...');
//         initFAQ();
//     }
// }

// // /**
// //  * Initialize FAQ accordion functionality
// //  */
// // function initFAQ() {
// //     const faqItems = document.querySelectorAll('.faq-item');
    
// //     faqItems.forEach(item => {
// //         const question = item.querySelector('.faq-question');
        
// //         question.addEventListener('click', () => {
// //             // Toggle current item
// //             const isActive = item.classList.contains('active');
            
// //             // Optional: Close other items (comment out if you want multiple open)
// //             faqItems.forEach(otherItem => {
// //                 if (otherItem !== item) {
// //                     otherItem.classList.remove('active');
// //                 }
// //             });
            
// //             // Toggle clicked item
// //             if (isActive) {
// //                 item.classList.remove('active');
// //             } else {
// //                 item.classList.add('active');
// //             }
// //         });
// //     });
// // }



// // /**
// //  * TBFusionAI - Main JavaScript Application
// //  * Handles all frontend interactions and API calls
// //  */

// // // ============================================================================
// // // Global Variables
// // // ============================================================================

// // let currentAudioFile = null;
// // let predictionResult = null;
// // let mediaRecorder = null;
// // let audioChunks = [];

// // // API Base URL
// // //const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';
// // const API_BASE_URL = window.location.origin + '/api/v1';

// // // ============================================================================
// // // Utility Functions
// // // ============================================================================

// // /**
// //  * Show loading overlay
// //  */
// // function showLoading() {
// //     const overlay = document.getElementById('loadingOverlay');
// //     if (overlay) {
// //         overlay.style.display = 'flex';
// //     }
// // }

// // /**
// //  * Hide loading overlay
// //  */
// // function hideLoading() {
// //     const overlay = document.getElementById('loadingOverlay');
// //     if (overlay) {
// //         overlay.style.display = 'none';
// //     }
// // }

// // /**
// //  * Show notification
// //  */
// // function showNotification(message, type = 'info') {
// //     // Create notification element
// //     const notification = document.createElement('div');
// //     notification.className = `notification notification-${type}`;
// //     notification.innerHTML = `
// //         <div class="notification-content">
// //             <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
// //             <span>${message}</span>
// //         </div>
// //     `;
    
// //     // Add to body
// //     document.body.appendChild(notification);
    
// //     // Show notification
// //     setTimeout(() => {
// //         notification.classList.add('show');
// //     }, 100);
    
// //     // Remove after 5 seconds
// //     setTimeout(() => {
// //         notification.classList.remove('show');
// //         setTimeout(() => {
// //             notification.remove();
// //         }, 300);
// //     }, 5000);
// // }

// // /**
// //  * Format file size
// //  */
// // function formatFileSize(bytes) {
// //     if (bytes === 0) return '0 Bytes';
// //     const k = 1024;
// //     const sizes = ['Bytes', 'KB', 'MB', 'GB'];
// //     const i = Math.floor(Math.log(bytes) / Math.log(k));
// //     return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
// // }

// // /**
// //  * Validate audio file
// //  */
// // function validateAudioFile(file) {
// //     const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg'];
// //     const maxSize = 10 * 1024 * 1024; // 10MB
    
// //     if (!validTypes.includes(file.type)) {
// //         return {
// //             valid: false,
// //             error: 'Invalid file type. Please upload a WAV, MP3, or OGG file.'
// //         };
// //     }
    
// //     if (file.size > maxSize) {
// //         return {
// //             valid: false,
// //             error: 'File size exceeds 10MB limit.'
// //         };
// //     }
    
// //     return { valid: true };
// // }

// // /**
// //  * Debounce function
// //  */
// // function debounce(func, wait) {
// //     let timeout;
// //     return function executedFunction(...args) {
// //         const later = () => {
// //             clearTimeout(timeout);
// //             func(...args);
// //         };
// //         clearTimeout(timeout);
// //         timeout = setTimeout(later, wait);
// //     };
// // }

// // // ============================================================================
// // // Navigation
// // // ============================================================================

// // /**
// //  * Initialize navigation
// //  */
// // function initNavigation() {
// //     const navToggle = document.getElementById('navToggle');
// //     const navMenu = document.querySelector('.nav-menu');
    
// //     if (navToggle && navMenu) {
// //         navToggle.addEventListener('click', () => {
// //             navMenu.classList.toggle('active');
// //             navToggle.classList.toggle('active');
// //         });
        
// //         // Close menu when clicking outside
// //         document.addEventListener('click', (e) => {
// //             if (!navToggle.contains(e.target) && !navMenu.contains(e.target)) {
// //                 navMenu.classList.remove('active');
// //                 navToggle.classList.remove('active');
// //             }
// //         });
        
// //         // Close menu when clicking on a link
// //         const navLinks = navMenu.querySelectorAll('.nav-link');
// //         navLinks.forEach(link => {
// //             link.addEventListener('click', () => {
// //                 navMenu.classList.remove('active');
// //                 navToggle.classList.remove('active');
// //             });
// //         });
// //     }
// // }

// // // ============================================================================
// // // Prediction Page Functions
// // // ============================================================================

// // /**
// //  * Initialize prediction page
// //  */
// // function initPredictionPage() {
// //     const uploadArea = document.getElementById('audioUploadArea');
// //     const audioFileInput = document.getElementById('audioFile');
// //     const predictionForm = document.getElementById('predictionForm');
    
// //     // Check if we are actually on the prediction page first
// //     if (!uploadArea || !audioFileInput || !predictionForm) {
// //         return; 
// //     }

// //     const startRecordBtn = document.getElementById('startRecordBtn');
// //     const stopRecordBtn = document.getElementById('stopRecordBtn');
// //     const removeAudioBtn = document.getElementById('removeAudio');
// //     const newPredictionBtn = document.getElementById('newPredictionBtn');
// //     const printResultsBtn = document.getElementById('printResultsBtn');

// //     // Recording listeners
// //     if (startRecordBtn && stopRecordBtn) {
// //         startRecordBtn.addEventListener('click', startRecording);
// //         stopRecordBtn.addEventListener('click', stopRecording);
// //     }
    
// //     // Audio upload handling
// //     uploadArea.addEventListener('click', () => {
// //         audioFileInput.click();
// //     });
    
// //     // Drag and drop
// //     uploadArea.addEventListener('dragover', (e) => {
// //         e.preventDefault();
// //         uploadArea.style.borderColor = 'var(--primary-color)';
// //         uploadArea.style.backgroundColor = 'var(--lighter)';
// //     });
    
// //     uploadArea.addEventListener('dragleave', () => {
// //         uploadArea.style.borderColor = 'var(--gray-lighter)';
// //         uploadArea.style.backgroundColor = 'transparent';
// //     });
    
// //     uploadArea.addEventListener('drop', (e) => {
// //         e.preventDefault();
// //         uploadArea.style.borderColor = 'var(--gray-lighter)';
// //         uploadArea.style.backgroundColor = 'transparent';
        
// //         const files = e.dataTransfer.files;
// //         if (files.length > 0) {
// //             handleAudioFileSelect(files[0]);
// //         }
// //     });
    
// //     // File input change
// //     audioFileInput.addEventListener('change', (e) => {
// //         if (e.target.files.length > 0) {
// //             handleAudioFileSelect(e.target.files[0]);
// //         }
// //     });
    
// //     // Remove audio
// //     if (removeAudioBtn) {
// //         removeAudioBtn.addEventListener('click', (e) => {
// //             e.preventDefault();
// //             removeAudioFile();
// //         });
// //     }
    
// //     // Form submission
// //     predictionForm.addEventListener('submit', async (e) => {
// //         e.preventDefault();
// //         await handlePredictionSubmit();
// //     });
    
// //     // New prediction
// //     if (newPredictionBtn) {
// //         newPredictionBtn.addEventListener('click', () => {
// //             resetPredictionForm();
// //         });
// //     }
    
// //     // Print results
// //     if (printResultsBtn) {
// //         printResultsBtn.addEventListener('click', () => {
// //             window.print();
// //         });
// //     }
// // }

// // /**
// //  * Start audio recording
// //  */
// // async function startRecording() {
// //     try {
// //         const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
// //         mediaRecorder = new MediaRecorder(stream);
// //         audioChunks = [];

// //         mediaRecorder.ondataavailable = (event) => {
// //             audioChunks.push(event.data);
// //         };

// //         mediaRecorder.onstop = () => {
// //             const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
// //             const file = new File([audioBlob], "recorded_cough.wav", { type: 'audio/wav' });
// //             handleAudioFileSelect(file);
            
// //             // Stop all tracks to release microphone
// //             stream.getTracks().forEach(track => track.stop());
// //         };

// //         mediaRecorder.start();
        
// //         // Toggle UI
// //         document.getElementById('startRecordBtn').classList.add('hidden');
// //         document.getElementById('stopRecordBtn').classList.remove('hidden');
// //         showNotification('Recording started... cough clearly into the mic.', 'info');
// //     } catch (err) {
// //         console.error("Error accessing microphone:", err);
// //         showNotification('Microphone access denied or not supported.', 'error');
// //     }
// // }

// // /**
// //  * Stop audio recording
// //  */
// // function stopRecording() {
// //     if (mediaRecorder && mediaRecorder.state !== 'inactive') {
// //         mediaRecorder.stop();
// //         document.getElementById('startRecordBtn').classList.remove('hidden');
// //         document.getElementById('stopRecordBtn').classList.add('hidden');
// //     }
// // }

// // /**
// //  * Handle audio file selection
// //  */
// // function handleAudioFileSelect(file) {
// //     const validation = validateAudioFile(file);
    
// //     if (!validation.valid) {
// //         showNotification(validation.error, 'error');
// //         return;
// //     }
    
// //     currentAudioFile = file;
    
// //     // Update UI
// //     const uploadArea = document.getElementById('audioUploadArea');
// //     const audioPreview = document.getElementById('audioPreview');
// //     const audioFileName = document.getElementById('audioFileName');
// //     const audioPlayer = document.getElementById('audioPlayer');
    
// //     uploadArea.style.display = 'none';
// //     audioPreview.style.display = 'block';
    
// //     audioFileName.textContent = `${file.name} (${formatFileSize(file.size)})`;
    
// //     // Set audio player source
// //     const audioURL = URL.createObjectURL(file);
// //     audioPlayer.src = audioURL;
// // }

// // /**
// //  * Remove audio file
// //  */
// // /**
// //  * Remove audio file
// //  */
// // function removeAudioFile() {
// //     currentAudioFile = null;
    
// //     const uploadArea = document.getElementById('audioUploadArea');
// //     const audioPreview = document.getElementById('audioPreview');
// //     const audioFileInput = document.getElementById('audioFile');
// //     const audioPlayer = document.getElementById('audioPlayer');
    
// //     // Reset Recording UI buttons
// //     document.getElementById('startRecordBtn').classList.remove('hidden');
// //     document.getElementById('stopRecordBtn').classList.add('hidden');
    
// //     // Reset Upload UI
// //     uploadArea.style.display = 'block';
// //     audioPreview.style.display = 'none';
// //     audioFileInput.value = '';
// //     audioPlayer.src = '';
// // }

// // /**
// //  * Handle prediction form submission
// //  */
// // async function handlePredictionSubmit() {
// //     // Validate form
// //     const form = document.getElementById('predictionForm');
// //     if (!form.checkValidity()) {
// //         form.reportValidity();
// //         return;
// //     }
    
// //     // Check if we have an audio file (either recorded or uploaded)
// //     if (!currentAudioFile) {
// //         showNotification('Please record or upload a cough audio sample', 'error');
// //         return;
// //     }
    
// //     showLoading();
    
// // try {
// //         // Prepare form data from the HTML form
// //         const formData = new FormData(form);
        
// //         // Remove duplicate 'file' field if it exists
// //         formData.delete('file');

// //         // MANUALLY ATTACH AUDIO:
// //         // This ensures that recorded blobs are sent as 'audio_file'
// //         // even if the hidden file input is empty.
// //         formData.set('audio_file', currentAudioFile, currentAudioFile.name || 'recorded_cough.wav');
        
// //         // Make API request with explicit CORS mode
// //         const response = await fetch(`${API_BASE_URL}/predict`, {
// //             method: 'POST',
// //             body: formData,
// //             mode: 'cors'
// //         });
        
// //         hideLoading();
        
// //         if (!response.ok) {
// //             // Try to parse error message from server
// //             let errorMessage = 'Prediction failed';
// //             try {
// //                 const errorData = await response.json();
// //                 errorMessage = errorData.message || errorMessage;
// //             } catch (e) {
// //                 errorMessage = `Server error: ${response.status}`;
// //             }
// //             throw new Error(errorMessage);
// //         }
        
// //         const result = await response.json();
// //         predictionResult = result;
        
// //         // Display results
// //         displayPredictionResults(result);
        
// //         showNotification('Analysis completed successfully!', 'success');
        
// //     } catch (error) {
// //         hideLoading();
// //         console.error('Prediction error:', error);
        
// //         // Detailed error for the user
// //         const displayError = error.message.includes('fetch') 
// //             ? 'Connection failed. Please ensure the API server is running at ' + API_BASE_URL 
// //             : error.message;
            
// //         showNotification(displayError, 'error');
// //     }
// // }
// // /**
// //  * Handle prediction form submission
// //  */
// // // async function handlePredictionSubmit() {
// // //     // Validate form
// // //     const form = document.getElementById('predictionForm');
// // //     if (!form.checkValidity()) {
// // //         form.reportValidity();
// // //         return;
// // //     }
    
// // //     if (!currentAudioFile) {
// // //         showNotification('Please upload an audio file', 'error');
// // //         return;
// // //     }
    
// // //     showLoading();
    
// // //     try {
// // //         // Prepare form data
// // //         const formData = new FormData(form);
        
// // //         // Make API request
// // //         const response = await fetch(`${API_BASE_URL}/predict`, {
// // //             method: 'POST',
// // //             body: formData
// // //         });
        
// // //         hideLoading();
        
// // //         if (!response.ok) {
// // //             const error = await response.json();
// // //             throw new Error(error.message || 'Prediction failed');
// // //         }
        
// // //         const result = await response.json();
// // //         predictionResult = result;
        
// // //         // Display results
// // //         displayPredictionResults(result);
        
// // //         showNotification('Analysis completed successfully!', 'success');
        
// // //     } catch (error) {
// // //         hideLoading();
// // //         console.error('Prediction error:', error);
// // //         showNotification(error.message || 'Failed to analyze audio. Please try again.', 'error');
// // //     }
// // // }

// // /**
// //  * Display prediction results
// //  */
// // function displayPredictionResults(result) {
// //     const resultsPanel = document.getElementById('resultsPanel');
// //     const predictionBadge = document.getElementById('predictionBadge');
// //     const badgeIcon = document.getElementById('badgeIcon');
// //     const predictionText = document.getElementById('predictionText');
// //     const probabilityText = document.getElementById('probabilityText');
// //     const confidenceValue = document.getElementById('confidenceValue');
// //     const confidenceFill = document.getElementById('confidenceFill');
// //     const recommendationText = document.getElementById('recommendationText');
// //     const spectrogramCard = document.getElementById('spectrogramCard');
// //     const spectrogramImage = document.getElementById('spectrogramImage');
    
// //     // Show results panel
// //     resultsPanel.style.display = 'block';
// //     resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
// //     // Update prediction badge
// //     const isTBPositive = result.prediction_class === 1;
// //     predictionBadge.className = `prediction-badge ${isTBPositive ? 'tb-positive' : 'tb-negative'}`;
    
// //     badgeIcon.innerHTML = isTBPositive 
// //         ? '<i class="fas fa-exclamation-triangle"></i>'
// //         : '<i class="fas fa-check-circle"></i>';
    
// //     predictionText.textContent = result.prediction;
// //     probabilityText.textContent = `Probability: ${(result.probability * 100).toFixed(1)}%`;
    
// //     // Update confidence
// //     confidenceValue.textContent = result.confidence_level;
// //     const confidencePercent = (result.confidence * 100).toFixed(0);
// //     confidenceFill.style.width = `${confidencePercent}%`;
    
// //     // Update recommendation
// //     recommendationText.textContent = result.recommendation;
    
// //     // Update spectrogram if available
// //     if (result.spectrogram_base64) {
// //         spectrogramCard.style.display = 'block';
// //         spectrogramImage.src = `data:image/png;base64,${result.spectrogram_base64}`;
// //     } else {
// //         spectrogramCard.style.display = 'none';
// //     }
// // }

// // /**
// //  * Reset prediction form
// //  */
// // function resetPredictionForm() {
// //     const form = document.getElementById('predictionForm');
// //     const resultsPanel = document.getElementById('resultsPanel');
    
// //     form.reset();
// //     removeAudioFile();
// //     resultsPanel.style.display = 'none';
// //     predictionResult = null;
    
// //     // Scroll to form
// //     form.scrollIntoView({ behavior: 'smooth', block: 'start' });
// // }

// // // ============================================================================
// // // FAQ Page Functions
// // // ============================================================================

// // /**
// //  * Initialize FAQ page
// //  */
// // function initFAQ() {
// //     const faqItems = document.querySelectorAll('.faq-item');
    
// //     if (faqItems.length === 0) {
// //         return; // Not on FAQ page
// //     }
    
// //     faqItems.forEach(item => {
// //         const question = item.querySelector('.faq-question');
        
// //         question.addEventListener('click', () => {
// //             const isActive = item.classList.contains('active');
            
// //             // Close all other items
// //             faqItems.forEach(otherItem => {
// //                 if (otherItem !== item) {
// //                     otherItem.classList.remove('active');
// //                 }
// //             });
            
// //             // Toggle current item
// //             if (isActive) {
// //                 item.classList.remove('active');
// //             } else {
// //                 item.classList.add('active');
// //             }
// //         });
// //     });
// // }

// // // ============================================================================
// // // API Health Check
// // // ============================================================================

// // /**
// //  * Check API health
// //  */
// // async function checkAPIHealth() {
// //     try {
// //         const response = await fetch(`${API_BASE_URL}/health`);
// //         const data = await response.json();
        
// //         if (data.status === 'healthy' && data.model_loaded) {
// //             console.log('✓ API is healthy and models are loaded');
// //             return true;
// //         } else {
// //             console.warn('⚠ API is running but models may not be loaded');
// //             return false;
// //         }
// //     } catch (error) {
// //         console.error('✗ API health check failed:', error);
// //         return false;
// //     }
// // }

// // // ============================================================================
// // // Model Information
// // // ============================================================================

// // /**
// //  * Get model information
// //  */
// // async function getModelInfo() {
// //     try {
// //         const response = await fetch(`${API_BASE_URL}/model/info`);
// //         const data = await response.json();
        
// //         console.log('Model Information:', data);
// //         return data;
// //     } catch (error) {
// //         console.error('Failed to get model info:', error);
// //         return null;
// //     }
// // }

// // // ============================================================================
// // // Smooth Scrolling
// // // ============================================================================

// // /**
// //  * Initialize smooth scrolling for anchor links
// //  */
// // function initSmoothScrolling() {
// //     document.querySelectorAll('a[href^="#"]').forEach(anchor => {
// //         anchor.addEventListener('click', function (e) {
// //             const href = this.getAttribute('href');
            
// //             if (href === '#') return;
            
// //             e.preventDefault();
            
// //             const target = document.querySelector(href);
// //             if (target) {
// //                 target.scrollIntoView({
// //                     behavior: 'smooth',
// //                     block: 'start'
// //                 });
// //             }
// //         });
// //     });
// // }

// // // ============================================================================
// // // Animation on Scroll
// // // ============================================================================

// // /**
// //  * Initialize scroll animations
// //  */
// // function initScrollAnimations() {
// //     const observerOptions = {
// //         threshold: 0.1,
// //         rootMargin: '0px 0px -50px 0px'
// //     };
    
// //     const observer = new IntersectionObserver((entries) => {
// //         entries.forEach(entry => {
// //             if (entry.isIntersecting) {
// //                 entry.target.classList.add('animate-in');
// //                 observer.unobserve(entry.target);
// //             }
// //         });
// //     }, observerOptions);
    
// //     // Observe elements
// //     document.querySelectorAll('.feature-card, .step-item, .faq-item').forEach(el => {
// //         observer.observe(el);
// //     });
// // }

// // // ============================================================================
// // // Form Validation Enhancement
// // // ============================================================================

// // /**
// //  * Add real-time form validation
// //  */
// // function enhanceFormValidation() {
// //     const form = document.getElementById('predictionForm');
    
// //     if (!form) return;
    
// //     const inputs = form.querySelectorAll('input[required], select[required]');
    
// //     inputs.forEach(input => {
// //         input.addEventListener('blur', () => {
// //             if (!input.checkValidity()) {
// //                 input.classList.add('invalid');
// //             } else {
// //                 input.classList.remove('invalid');
// //             }
// //         });
        
// //         input.addEventListener('input', () => {
// //             if (input.classList.contains('invalid') && input.checkValidity()) {
// //                 input.classList.remove('invalid');
// //             }
// //         });
// //     });
// // }

// // // ============================================================================
// // // Keyboard Shortcuts
// // // ============================================================================

// // /**
// //  * Initialize keyboard shortcuts
// //  */
// // function initKeyboardShortcuts() {
// //     document.addEventListener('keydown', (e) => {
// //         // Ctrl/Cmd + K: Focus search (if exists)
// //         if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
// //             e.preventDefault();
// //             const searchInput = document.querySelector('input[type="search"]');
// //             if (searchInput) searchInput.focus();
// //         }
        
// //         // Escape: Close modals/overlays
// //         if (e.key === 'Escape') {
// //             const navMenu = document.querySelector('.nav-menu');
// //             if (navMenu && navMenu.classList.contains('active')) {
// //                 navMenu.classList.remove('active');
// //             }
// //         }
// //     });
// // }

// // // ============================================================================
// // // Print Styles
// // // ============================================================================

// // /**
// //  * Prepare page for printing
// //  */
// // function preparePrint() {
// //     window.addEventListener('beforeprint', () => {
// //         // Hide elements that shouldn't be printed
// //         document.querySelectorAll('.no-print').forEach(el => {
// //             el.style.display = 'none';
// //         });
// //     });
    
// //     window.addEventListener('afterprint', () => {
// //         // Restore hidden elements
// //         document.querySelectorAll('.no-print').forEach(el => {
// //             el.style.display = '';
// //         });
// //     });
// // }

// // // ============================================================================
// // // Performance Monitoring
// // // ============================================================================

// // /**
// //  * Log performance metrics
// //  */
// // function logPerformanceMetrics() {
// //     if (window.performance && window.performance.timing) {
// //         window.addEventListener('load', () => {
// //             setTimeout(() => {
// //                 const perfData = window.performance.timing;
// //                 const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
// //                 const connectTime = perfData.responseEnd - perfData.requestStart;
// //                 const renderTime = perfData.domComplete - perfData.domLoading;
                
// //                 console.log('Performance Metrics:');
// //                 console.log(`  Page Load Time: ${pageLoadTime}ms`);
// //                 console.log(`  Connection Time: ${connectTime}ms`);
// //                 console.log(`  Render Time: ${renderTime}ms`);
// //             }, 0);
// //         });
// //     }
// // }

// // // ============================================================================
// // // Error Handling
// // // ============================================================================

// // /**
// //  * Global error handler
// //  */
// // function initErrorHandling() {
// //     window.addEventListener('error', (e) => {
// //         console.error('Global error:', e.error);
// //         // You can send errors to a logging service here
// //     });
    
// //     window.addEventListener('unhandledrejection', (e) => {
// //         console.error('Unhandled promise rejection:', e.reason);
// //         // You can send errors to a logging service here
// //     });
// // }

// // // ============================================================================
// // // Initialize Application
// // // ============================================================================

// // /**
// //  * Main initialization function
// //  */
// // function initApp() {
// //     console.log('🚀 Initializing TBFusionAI...');
    
// //     // Core functionality
// //     initNavigation();
// //     initSmoothScrolling();
// //     initKeyboardShortcuts();
// //     initErrorHandling();
// //     preparePrint();
    
// //     // Page-specific functionality
// //     initPredictionPage();
// //     initFAQ();
    
// //     // Enhancements
// //     enhanceFormValidation();
// //     initScrollAnimations();
    
// //     // API check
// //     checkAPIHealth();
    
// //     // Performance monitoring (only in development)
// //     if (window.location.hostname === 'localhost') {
// //         logPerformanceMetrics();
// //     }
    
// //     console.log('✓ TBFusionAI initialized successfully');
// // }

// // /**
// //  * Initialize FAQ accordion functionality
// //  */
// // function initFAQ() {
// //     const faqItems = document.querySelectorAll('.faq-item');
    
// //     faqItems.forEach(item => {
// //         const question = item.querySelector('.faq-question');
        
// //         question.addEventListener('click', () => {
// //             // Close other items (optional - remove if you want multiple open)
// //             faqItems.forEach(otherItem => {
// //                 if (otherItem !== item && otherItem.classList.contains('active')) {
// //                     otherItem.classList.remove('active');
// //                 }
// //             });
            
// //             // Toggle current item
// //             item.classList.toggle('active');
// //         });
// //     });
// // }
// // // ============================================================================
// // // Document Ready
// // // ============================================================================

// // // Initialize when DOM is ready
// // if (document.readyState === 'loading') {
// //     document.addEventListener('DOMContentLoaded', initApp);
// // } else {
// //     initApp();
// // }

// // // Export functions for use in other scripts
// // window.TBFusionAI = {
// //     showLoading,
// //     hideLoading,
// //     showNotification,
// //     checkAPIHealth,
// //     getModelInfo
// // };