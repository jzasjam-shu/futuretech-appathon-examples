import * as DICTIONARY from './dictionary.js';

// ===============================
// Elements
// ===============================
const postBtn = document.getElementById('post');
const commentBox = document.getElementById('comment');
const commentsList = document.getElementById('commentsList');
const statusText = document.getElementById('status');

// ===============================
// Configuration
// ===============================
const MODEL_URL = './model.json';
const SPAM_THRESHOLD = 0.75; // Threshold for classifying a comment as spam (eg 75% chance)
const ENCODING_LENGTH = 20; // The fixed length for input encoding (must match model input shape)

// ===============================
// Variables
// ===============================
let model;  // The TensorFlow.js model for spam detection

// ===============================
// Status helper (sets status message)
// ===============================
function setStatus(message) {
    statusText.innerHTML = `<div class="new">${message}</div>`;
}

// ===============================
// Init / Load the model
// ===============================
async function init() {
    // Load the TensorFlow.js model
    model = await tf.loadLayersModel(MODEL_URL);
    // Set status message
    setStatus("Model loaded!");
}
// Call init function
init();

// ===============================
// Data Preprocessing
// ===============================
// Tokenizer (the function that converts text to tensor)
function tokenize(wordArray) {

    // Initialise the result array with the start token
    let result = [DICTIONARY.START];

    // Convert each word to its corresponding encoding
    for (let i = 0; i < wordArray.length && i < ENCODING_LENGTH - 1; i++) {
        let encoding = DICTIONARY.LOOKUP[wordArray[i]];
        result.push(encoding ?? DICTIONARY.UNKNOWN);
    }
    // Pad the sequence of words to match the fixed encoding length
    while (result.length < ENCODING_LENGTH) {
        result.push(DICTIONARY.PAD);
    }

    // Convert the result array to a tensor and return
    return tf.tensor([result]);
}

// ===============================
// Handle comment post (when postBtn is clicked)
// ===============================
postBtn.onclick = handleCommentPost;
// Function to handle comment posting
async function handleCommentPost() {

    // Get comment text and trim whitespace
    const text = commentBox.innerText.trim();
    // Check if comment is empty
    if (!text) return;

    // Tokenize and encode the comment
    const tokens = tokenize(
        text.toLowerCase().replace(/[^\w\s]/g, ' ').split(' ')
    );

    // Create a list item for the comment display
    const li = document.createElement('li');

    // Get comment spam prediction
    const isSpam = await predictSpam(tokens);
    // If spam, add spam class
    if (isSpam) {
        li.classList.add('spam');
    }

    // Build comment display
    const p = document.createElement('p');
    p.innerText = text;
    const time = document.createElement('span');
    time.innerText = 'Comment at ' + new Date().toLocaleString();

    // Add elements to the list item
    li.appendChild(time);
    li.appendChild(p);

    // Add list item to comments list
    commentsList.prepend(li);

    // Reset UI
    commentBox.innerText = '';
    // Set status message
    setStatus(isSpam ? "Spam detected!" : "Comment posted");
}

// ===============================
// Predict if a comment is spam
// ===============================
// Function to predict if a comment is spam based on the input tensor 
async function predictSpam(inputTensor) {

    // Make prediction
    const prediction = model.predict(inputTensor);
    // Get prediction data
    const data = await prediction.data();
    // Get spam score
    const spamScore = data[1];

    // Clean up tensors
    inputTensor.dispose();
    prediction.dispose();

    // Return whether the comment is spam
    return spamScore > SPAM_THRESHOLD;
}