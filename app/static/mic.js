// static/mic.js
// Microphone recording + transcription logic, self-contained and safe.
// Uses window.API_URL if available, otherwise falls back to localhost.

(function() {
  const API_URL = window.API_URL || 'http://127.0.0.1:5000';

  function initMic() {
    let mediaRecorder = null;
    let audioChunks = [];
    const micButton = document.getElementById('mic-button');
    const chatInput = document.getElementById('chat-input');

    if (!micButton) return;

    micButton.addEventListener('click', async () => {
      // If already recording, stop
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        micButton.classList.remove('bg-red-400');
        micButton.classList.add('bg-lavender_soft');
        return;
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Audio recording not supported in this browser.');
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = e => {
          if (e.data && e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
          try {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');

            micButton.disabled = true;
            micButton.classList.add('opacity-50');

            const res = await fetch(`${API_URL}/transcribe`, { method: 'POST', body: formData });
            if (!res.ok) throw new Error('Transcription failed');

            const data = await res.json();
            if (data && data.transcript && chatInput) {
              chatInput.value = data.transcript;
              chatInput.focus();
            }
          } catch (err) {
            console.error('Transcription error:', err);
            alert('Transcription failed.');
          } finally {
            micButton.disabled = false;
            micButton.classList.remove('opacity-50');
          }
        };

        // start recording
        mediaRecorder.start();
        micButton.classList.remove('bg-lavender_soft');
        micButton.classList.add('bg-red-400');

      } catch (err) {
        console.error('Could not access microphone:', err);
        alert('Could not access microphone.');
      }
    });
  }

  // Ensure DOM is ready before hooking elements
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMic);
  } else {
    initMic();
  }
})();
