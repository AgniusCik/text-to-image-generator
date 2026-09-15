async function receiveInput(event) {
    event.preventDefault();

    const promptInput = document.getElementById('prompt');
    const userInput = promptInput.value;
    const statusEl = document.getElementById('status');
    const resultImg = document.getElementById('result-image');
    const submitBtn = document.getElementById('submit-btn');

    // validate input length
    if (userInput.length > 100 || userInput.length < 1) {
        statusEl.textContent = 'Prompt is not within the range of 1-100 characters';
        return;
    }

    // clear any previous error/status text and previous image
    statusEl.textContent = '';
    resultImg.src = '';

    // enter loading state
    statusEl.textContent = 'Generating your image... this may take a minute.';
    submitBtn.disabled = true; // prevent double-submits while waiting
    submitBtn.textContent = 'Generating...';

    try {
        const response = await fetch('http://127.0.0.1:8000/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ caption: userInput }),
        });

        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }

        const data = await response.json();
        resultImg.src = data.image;
        statusEl.textContent = 'Done!';
    } catch (error) {
        console.error('Error generating image:', error);
        statusEl.textContent = 'Something went wrong generating the image. Please try again.';
    } finally {
        // always leave loading state, whether it succeeded or failed
        submitBtn.disabled = false;
        submitBtn.textContent = 'Generate';
    }
}

document.getElementById('form').addEventListener('submit', receiveInput);