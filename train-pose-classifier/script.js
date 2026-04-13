// Variables for HTML elements
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");
const startBtn = document.getElementById("startBtn");

let detector;        // Pose detector
let lastPose = null; // Last detected pose

// Status helper
function setStatus(message) {
    statusText.innerHTML = `<div class="new">${message}</div>`;
}

// Init / Load the model
async function init() {

    // Set backend (use webgl to speed up processing)
    await tf.setBackend("webgl");

    // Load the pose detection model
    detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.BlazePose,
        { runtime: "tfjs", modelType: "full" }
    );

    // Set status and enable the start button
    setStatus("Model loaded!");
    startBtn.disabled = false;

    // Enable load button if saved model exists
    if (localStorage.getItem("pose-labels")) {
        document.getElementById("loadBtn").disabled = false;
    }
}
// Call init function
init();


// Start the camera when #startBtn is clicked
startBtn.onclick = async () => {
    // Get camera stream
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    // Camera flipped/mirror
    video.style.transform = "scaleX(-1)";

    // Wait for video metadata to load
    video.addEventListener("loadedmetadata", () => {
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Set status and disable the start button and sample buttons
        setStatus("Camera started!");
        startBtn.disabled = true;
        document.querySelectorAll('.sampleBtn').forEach(btn => btn.disabled = false);

        // Start the detection loop
        runLoop();
    });
};


//  Variables for training
let trainingData = [];
let labels = [];
let labelMap = [];
let model;

// Convert keypoints to normalised array for training
function keypointsToArray(keypoints) {
    return keypoints.flatMap(kp => [
        kp.x / video.videoWidth,
        kp.y / video.videoHeight
    ]);
}

// Add samples for training when a button with .sampleBtn is held down
document.querySelectorAll(".sampleBtn").forEach(btn => {
    btn.onmousedown = () => {
        // Capture a sample ever 100ms
        btn.interval = setInterval(() => addSample(btn.dataset.label), 100);
    };
    // Stop capture samples when mouse is released
    btn.onmouseup = () => clearInterval(btn.interval);
    btn.onmouseleave = () => clearInterval(btn.interval);
});
// Function to add a training sample
function addSample(label) {
    if (!lastPose) return;
    // Prepare the training sample with the last detected pose keypoints normalised and the provided label
    trainingData.push(keypointsToArray(lastPose.keypoints));
    labels.push(label);
    // Update the sample counts and status
    setStatus(`Added: ${label}`);
    updateCounts();
    // Enable the train button
    document.getElementById("trainBtn").disabled = false;
}

// Update sample counts on the buttons
function updateCounts() {
    document.querySelectorAll('button[data-label]').forEach(btn => {
        const label = btn.dataset.label;
        const count = labels.filter(l => l === label).length; // Get the current sample count for the label from the labels array
        btn.textContent = `Add ${label.charAt(0).toUpperCase() + label.slice(1)} (${count} Samples)`;
    });
}


// Train Model when the #trainBtn button is clicked
document.getElementById("trainBtn").onclick = trainModel;
// Function to train the model
async function trainModel() {
    // Convert training data and labels to tensors
    const xs = tf.tensor2d(trainingData);
    // Normalise the labels to the range [0, 1]
    labelMap = [...new Set(labels)];
    const ys = tf.tensor2d(
        labels.map(l => labelMap.map(x => x === l ? 1 : 0))
    );
    // Define the model architecture
    model = tf.sequential(); // This creates a new sequential model (a sequential model is a linear stack of layers)
    // Input layer with 64 units and ReLU activation (This layer learns to extract features from the input)
    model.add(tf.layers.dense({ inputShape: [xs.shape[1]], units: 64, activation: "relu" }));
    // Hidden layer with 32 units and ReLU activation (This layer learns to combine features)
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    // Output layer with units equal to the number of classes and softmax activation (This layer produces the final class probabilities)
    model.add(tf.layers.dense({ units: labelMap.length, activation: "softmax" }));

    // Compile the model (This configures the model for training)
    model.compile({
        optimizer: "adam", // This is the optimization algorithm used to minimize the loss function
        loss: "categoricalCrossentropy", // This is the loss function used for multi-class classification
        metrics: ["accuracy"] // This measures the accuracy of the model during training
    });

    // Train the model (This is where the model learns from the training data)
    await model.fit(xs, ys, { epochs: 20 }); // This trains the model for 20 epochs

    // Set the status and enable the save button
    setStatus("Model trained!");
    document.getElementById("saveBtn").disabled = false;

    // Clean up tensors
    xs.dispose();
    ys.dispose();
}

// Function to predict the class of a pose
function predictClass(keypoints) {
    // Check if the model is loaded
    if (!model) return null;

    // Prepare the input tensor
    const input = tf.tensor2d([keypointsToArray(keypoints)]);
    // Make a prediction
    const prediction = model.predict(input);
    // Get the prediction data
    const data = prediction.dataSync();
    // Find the class with the highest probability
    const index = data.indexOf(Math.max(...data));
    // Map the index to the corresponding label
    const result = {
        label: labelMap[index],
        confidence: Math.round(data[index] * 100)
    };

    // Dispose of the tensors
    input.dispose();
    prediction.dispose();

    return result;
}

// Function to draw the detected pose on the canvas
function drawPose(pose) {
    // Save the current canvas state
    ctx.save();

    // Flip the canvas horizontally to match camera view
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0);

    // Set the fill and stroke styles
    ctx.fillStyle = "#E31C79";
    ctx.strokeStyle = "#FF3EB5";

    // Draw skeleton
    const pairs = poseDetection.util.getAdjacentPairs(
        poseDetection.SupportedModels.BlazePose
    );
    pairs.forEach(([i, j]) => {
        const a = pose.keypoints[i];
        const b = pose.keypoints[j];
        // Check if both keypoints are detected
        if (a.score > 0.5 && b.score > 0.5) {
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
        }
    });

    // Draw each keypoint
    pose.keypoints.forEach(kp => {
        if (kp.score > 0.5) {
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 5, 0, Math.PI * 2);
            ctx.fill();
        }
    });

    // Draw the predicted class label and prediction confidence
    if (model) {
        // Make a prediction
        const prediction = predictClass(pose.keypoints);
        // Draw the label and confidence on the canvas
        ctx.scale(-1, 1);
        ctx.font = "30px Arial";
        ctx.fillText(prediction.label + " (" + prediction.confidence + "%)", -canvas.width + 10, 40);

    }

    // Restore the canvas state
    ctx.restore();
}

// Main loop to estimate poses and draw them on the canvas
async function runLoop() {
    // Estimate poses from the video
    const poses = await detector.estimatePoses(video);
    // Check if any poses were detected and draw the first detected pose
    if (poses.length > 0) {
        lastPose = poses[0];
        drawPose(lastPose);
    }
    // This continues the detection loop
    requestAnimationFrame(runLoop);
}


// Save model to local storage
document.getElementById("saveBtn").onclick = saveModel;
async function saveModel() {
    await model.save("localstorage://pose-model");
    localStorage.setItem("pose-labels", JSON.stringify(labelMap));
    setStatus("Model saved!");
    // Enable the load button
    document.getElementById("loadBtn").disabled = false;
}

// Load model from local storage
document.getElementById("loadBtn").onclick = loadModel;
async function loadModel() {
    model = await tf.loadLayersModel("localstorage://pose-model");
    labelMap = JSON.parse(localStorage.getItem("pose-labels")) || [];
    setStatus("Model loaded!");
}

// Reset the model and UI
document.getElementById("resetBtn").onclick = resetModel;
async function resetModel() {
    // Reset the model data and labels
    model = null;
    labelMap = [];
    trainingData = [];
    labels = [];
    // Remove model from local storage if one exists
    if(localStorage.getItem("pose-labels")) {
        await tf.io.removeModel("localstorage://pose-model");
        localStorage.removeItem("pose-labels");
    }
    // Reset the UI and buttons
    document.getElementById("saveBtn").disabled = true;
    document.getElementById("loadBtn").disabled = true;
    updateCounts();
    // Set status message
    setStatus("Model reset!");
}
