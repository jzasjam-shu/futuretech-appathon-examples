// ===============================
// Elements
// ===============================
const statusText = document.getElementById("status");
const dataPreview = document.getElementById("dataPreview");

const trainBtn = document.getElementById("trainBtn");
const saveBtn = document.getElementById("saveBtn");
const loadBtn = document.getElementById("loadBtn");
const resetBtn = document.getElementById("resetBtn");

const predictBtn = document.getElementById("predictBtn");
const hpInput = document.getElementById("hpInput");
const weightInput = document.getElementById("weightInput");
const cylInput = document.getElementById("cylInput");
const predictionOutput = document.getElementById("predictionOutput");


// ===============================
// Variables
// ===============================
let data = [];  // Raw data
let model;      // TensorFlow.js model

let inputMin, inputMax; // Input feature min/max values
let labelMin, labelMax; // Label min/max values


// ===============================
// Status helper (sets status message)
// ===============================
function setStatus(msg) {
    statusText.innerHTML = `<div class="new">${msg}</div>`;
}


// ===============================
// Init / Load the data
// ===============================
async function init() {
    // Set backend (use webgl to speed up processing)
    await tf.setBackend("webgl");

    // Data JSON url
    const URL = "https://storage.googleapis.com/tfjs-tutorials/carsData.json";
    // Fetch the data
    const raw = await fetch(URL).then(res => res.json());
    // Map the data to a more usable format and filter out any invalid entries
    data = raw.map(d => ({
        hp: d.Horsepower,
        weight: d.Weight_in_lbs,
        cyl: d.Cylinders,
        mpg: d.Miles_per_Gallon
    })).filter(d =>
        d.hp != null && d.weight != null && d.cyl != null && d.mpg != null
    );

    // Update the data preview (first 100 entries)
    dataPreview.innerHTML = data.slice(0, 100)
        .map(d => `HP:${d.hp}, W:${d.weight}, C:${d.cyl} → MPG:${d.mpg}`)
        .join("<br>");
    
    // Set status and enable the train button
    setStatus(`Data loaded  - Total Rows: ${data.length}`);
    trainBtn.disabled = false;

    // Enable load button if saved model exists
    if (localStorage.getItem("cars-norm")) {
        loadBtn.disabled = false;
    }
}
// Call init function
init();

// ===============================
// Train Model (when the #trainBtn button is clicked)
// ===============================
trainBtn.onclick = trainModel;

// Function to train the model
async function trainModel() {
    // Show status
    setStatus("Preparing data...");

    // Prepare data and labels for training
    const inputs = data.map(d => [d.hp, d.weight, d.cyl]);
    const labels = data.map(d => d.mpg);
    // Create tensors from the data and labels
    const xs = tf.tensor2d(inputs);
    const ys = tf.tensor2d(labels, [labels.length, 1]);

    // Normalise (per feature)
    inputMin = xs.min(0);
    inputMax = xs.max(0);

    labelMin = ys.min();
    labelMax = ys.max();

    // Scale the data to the range [0, 1]
    // uses sub and div to make a sum like (x - min) / (max - min)
    const normXs = xs.sub(inputMin).div(inputMax.sub(inputMin)); 
    const normYs = ys.sub(labelMin).div(labelMax.sub(labelMin)); 

    // Create the model (sequential is a linear stack of layers)
    model = tf.sequential();
    // Create the input layer
    model.add(tf.layers.dense({
        inputShape: [3],    // 3 represents the number of input features (HP, Weight, Cylinders)
        units: 50,          // This is the number of neurons in the layer
        activation: "relu"  // This is the activation function used in the layer
    }));
    // Create the output layer (units is the number of output features)
    model.add(tf.layers.dense({ units: 1 }));
    // Compile the model to configure the learning process
    model.compile({
        optimizer: tf.train.adam(), // This is the optimization algorithm used to minimise the loss function
        loss: "meanSquaredError"    // This is the loss function used for regression
    });

    // Set status
    setStatus("Training...");

    // Train the model
    await model.fit(normXs, normYs, {
        epochs: 50,     // Number of training iterations
        batchSize: 32   // Number of samples per iteration
    });

    // Set status and enable buttons
    setStatus("Model trained!");
    predictBtn.disabled = false;
    saveBtn.disabled = false;

    // Dispose of tensors
    xs.dispose();
    ys.dispose();
    normXs.dispose();
    normYs.dispose();
};


// ===============================
// Predict (make a prediction and display the result when predictBtn is clicked)
// ===============================
predictBtn.onclick = predict;

// Function to make a prediction based on user input
function predict(){
    if (!model) return;

    const hp = parseFloat(hpInput.value);
    const weight = parseFloat(weightInput.value);
    const cyl = parseFloat(cylInput.value);

    const input = tf.tensor2d([[hp, weight, cyl]]);

    const normInput = input.sub(inputMin).div(inputMax.sub(inputMin));

    const pred = model.predict(normInput);

    const unNorm = pred
        .mul(labelMax.sub(labelMin))
        .add(labelMin);

    const value = unNorm.dataSync()[0];

    predictionOutput.innerText = `Estimated MPG: ${value.toFixed(1)}`;

    input.dispose();
    pred.dispose();
    unNorm.dispose();
};


// ===============================
// Save Model (when saveBtn is clicked)
// ===============================
saveBtn.onclick = saveModel;

// Function to save model to local browser storage
async function saveModel() {
    // Save the model in local storage
    await model.save("localstorage://cars-model");

    // Save the normalisation data in local storage (we need this to make predictions with the model later)
    localStorage.setItem("cars-model-norm", JSON.stringify({
        inputMin: inputMin.arraySync(),
        inputMax: inputMax.arraySync(),
        labelMin: labelMin.dataSync()[0],
        labelMax: labelMax.dataSync()[0]
    }));

    // Set status message and enable the load button
    setStatus("Model saved!");
    loadBtn.disabled = false;
};


// ===============================
// Load Model (when loadBtn is clicked)
// ===============================
loadBtn.onclick = loadModel;

// Function to load model from browser's local storage
async function loadModel() {
    // Load the model from local storage
    model = await tf.loadLayersModel("localstorage://cars-model");

    // Load the normalisation data from local storage
    const norm = JSON.parse(localStorage.getItem("cars-model-norm"));

    // Restore the normalisation tensors
    inputMin = tf.tensor1d(norm.inputMin);
    inputMax = tf.tensor1d(norm.inputMax);
    labelMin = tf.scalar(norm.labelMin);
    labelMax = tf.scalar(norm.labelMax);

    // Set status message
    setStatus("Model loaded!");
    predictBtn.disabled = false;
};


// ===============================
// Reset the model and UI (when resetBtn is clicked)
// ===============================
resetBtn.onclick = resetModel;

// Function to reset the model and UI
async function resetModel() {
    // Reset the model 
    model = null;
    // Reset the training data and labels
    trainingData = [];
    labels = [];
    // Clear the prediction output
    predictionOutput.innerText = "";
    // Reset the UI elements
    trainBtn.disabled = false;
    predictBtn.disabled = true;
    saveBtn.disabled = true;
    loadBtn.disabled = true;
    // Remove model and normalisation data from local storage if they exist
    try {
        await tf.io.removeModel("localstorage://cars-multi");
        localStorage.removeItem("cars-norm");
    } catch (e) {}
    // Set status message
    setStatus("Reset complete!");
};

