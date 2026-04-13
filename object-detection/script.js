// Variables for HTML elements
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");
const startBtn = document.getElementById("startBtn");

let model; // Object Detection Model

// Status helper
function setStatus(message) {
    statusText.innerHTML = `<div class="new">${message}</div>`;
}

// Init / Load model
async function init() {

    // Set backend (use webgl to speed up processing)
    await tf.setBackend("webgl");

    // Load COCO-SSD model
    model = await cocoSsd.load();

    // Set status and enable the start button
    setStatus("Model loaded!");
    startBtn.disabled = false;
}
init();


// Start camera when #startBtn is clicked
startBtn.onclick = async () => {
    // Get camera stream
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    // Wait for video metadata to load
    video.onloadedmetadata = () => {
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Set status and disable the start button and sample buttons
        setStatus("Camera started!");
        startBtn.disabled = true;

        // Start the detection loop
        runLoop();
    };
};


//  Function to draw predictions on canvas
async function drawPredictions(predictions) {

    // Save the current canvas state
    ctx.save();

    // Flip the canvas horizontally
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0);

    // Set styles for drawing boxes and labels
    ctx.strokeStyle = "#E31C79";
    ctx.fillStyle = "#E31C79";
    ctx.lineWidth = 2;
    ctx.font = "16px Arial";

    // Draw predictions
    predictions.forEach(pred => {
        // If prediction score is above threshold
        if (pred.score > 0.6) {
            // Get bounding box coordinates
            const [x, y, width, height] = pred.bbox;

            // Draw box
            ctx.strokeRect(x, y, width, height);

            // Draw label (flipped)
            ctx.save();
            ctx.scale(-1, 1);
            ctx.fillText(
                `${pred.class} (${Math.round(pred.score * 100)}%)`,
                -x - width, // adjust position in flipped space
                y > 10 ? y - 5 : 10
            );
            // Restore the context
            ctx.restore();
        }
    });
    // Restore the context
    ctx.restore();
}

// Main loop to run object detection
async function runLoop() {
    // Estimate objects in the video frame
    const predictions = await model.detect(video);
    // Draw the predictions on the canvas
    await drawPredictions(predictions);
    // This continues the detection loop
    requestAnimationFrame(runLoop);
}

