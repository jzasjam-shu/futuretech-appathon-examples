import * as DICTIONARY from './dictionary.js';

// DOM
const postBtn = document.getElementById('post');
const commentBox = document.getElementById('comment');
const commentsList = document.getElementById('commentsList');
const statusText = document.getElementById('status');

// Constants
const MODEL_URL = './model.json';
const SPAM_THRESHOLD = 0.75;
const ENCODING_LENGTH = 20;

// State
let model;
let currentUserName = 'Anonymous';

// Status helper
function setStatus(message) {
    statusText.innerHTML = `<div class="new">${message}</div>`;
}

// Init
async function init() {
    model = await tf.loadLayersModel(MODEL_URL);
    setStatus("Model loaded!");
}
init();


// Handle comment post
postBtn.onclick = handleCommentPost;

async function handleCommentPost() {

    const text = commentBox.innerText.trim();
    if (!text) return;

    postBtn.classList.add('processing');
    commentBox.classList.add('processing');

    const tokens = tokenize(
        text.toLowerCase().replace(/[^\w\s]/g, ' ').split(' ')
    );

    const li = document.createElement('li');

    const isSpam = await predictSpam(tokens);

    if (isSpam) {
        li.classList.add('spam');
    }

    // Build comment UI
    const p = document.createElement('p');
    p.innerText = text;

    const name = document.createElement('span');
    name.className = 'username';
    name.innerText = currentUserName;

    const time = document.createElement('span');
    time.className = 'timestamp';
    time.innerText = ' at ' + new Date().toLocaleString();

    li.appendChild(name);
    li.appendChild(time);
    li.appendChild(p);

    commentsList.prepend(li);

    // Reset UI
    commentBox.innerText = '';
    postBtn.classList.remove('processing');
    commentBox.classList.remove('processing');

    setStatus(isSpam ? "Spam detected!" : "Comment posted");
}


// Predict function 
async function predictSpam(inputTensor) {

    const prediction = model.predict(inputTensor);
    const data = await prediction.data();

    const spamScore = data[1];

    // Clean up tensors
    inputTensor.dispose();
    prediction.dispose();

    return spamScore > SPAM_THRESHOLD;
}


// Tokenizer
function tokenize(wordArray) {

    let result = [DICTIONARY.START];

    for (let i = 0; i < wordArray.length && i < ENCODING_LENGTH - 1; i++) {
        let encoding = DICTIONARY.LOOKUP[wordArray[i]];
        result.push(encoding ?? DICTIONARY.UNKNOWN);
    }

    while (result.length < ENCODING_LENGTH) {
        result.push(DICTIONARY.PAD);
    }

    return tf.tensor([result]);
}