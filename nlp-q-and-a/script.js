// ===============================
// Elements
// ===============================
const contextInput = document.getElementById("context");
const questionInput = document.getElementById("question");
const answerDiv = document.getElementById("answer");

const statusText = document.getElementById("status");
const askBtn = document.getElementById("askBtn");
const resetBtn = document.getElementById("resetBtn");

// ===============================
// Variables
// ===============================
let model; // The Q&A model

// ===============================
// Status helper (sets status message)
// ===============================
function setStatus(message) {
    statusText.innerHTML = `<div class="new">${message}</div>`;
}

// ===============================
// Init / Load the model
// ===============================
// Set initial context text
contextInput.value = "Sheffield Hallam University is one of the UK’s largest and most diverse universities: a community of approximately 31,000 students, nearly 4,000 staff and 345,000 alumni around the globe.  Our mission is simple: we transform lives.  We are an award-winning university, recently receiving Gold in the Teaching Excellence Framework for outstanding support for student success and progression. We provide people from all backgrounds with the opportunity to acquire the skills, knowledge and experience to succeed at whatever they choose to do.     As one of the UK’s largest and most progressive universities, our teaching, research and partnerships are characterised by a focus on real world impact - addressing the health, economic and social challenges facing society today. We are ambitious for our university, our students, our colleagues, our partners, our city and our region. Our vision is to be the world's leading applied university; showing what a university genuinely focused on transforming lives can achieve.  Who we are Founded in 1843 as the Sheffield School of Design, Sheffield Hallam has exercised a powerful impact on the city, region, nation and world. Today we are one of the UK’s largest and most diverse universities: a community of approximately 31,000 students, nearly 4,000 staff and 345,000 alumni around the world.   Our students The strength of our student cohort is in its diversity - matched by our commitment as a university to providing opportunity: Around 31,000 undergraduate and postgraduate students A cohort of over 4,500 international students from across the globe Over 50% come from within 25 miles of our main campus 71% of our students come from a family without parents participating in higher education 25% of our students come from neighbourhoods with the lowest rates of young people progressing to higher education, more than double the sector average of 12% 95% of our graduates are in work or further study fifteen months after graduating (2022/23 Graduate Outcomes Survey).   Our teaching Our committed staff provide award-winning teaching and student support for our diverse population of students. We were awarded Gold in the 2023 Teaching Excellence Framework, which recognised our outstanding support for student success both during and after their studies. We provide our students with a curriculum designed for collaboration and cross-disciplinary learning, courses which are designed with industry experts, embedded work experience opportunities, digital skills pathways, and placement years. This includes a guarantee that all of our students will gain a placement during their degree through our award-winning Highly Skilled Employability strategy This commitment to applied learning and teaching means: We are one of the biggest trainers of healthcare staff for the NHS in the country We train around 1,000 teachers a year (across degree stages) - with around 600 entering the workforce each year (60% of which in South Yorkshire) We are home to the largest modern business school in the country, with more than 7,000 students from 100 countries, studying at five locations around the globe. Our National Centre of Excellence for Degree Apprenticeships has the biggest portfolio of degree apprenticeships in the UK Gold award from the government's Teaching Excellence Framework Five stars in the QS Star rating, achieving top marks in the categories of teaching, employability, facilities, innovation and inclusiveness We were named Outstanding Entrepreneurial University at the THE Awards 2021 Our research Our research and innovation work is characterised by a focus on real-world impact. Working across disciplines in world-class facilities, our academics tackle challenges facing society today. Our research focuses on three key themes: driving future economies, enabling healthier lives and building stronger communities. We have an international reputation in areas such as materials science, art and design, sports science and engineering, biomedicine, and economic and social research – all delivered through our research institutes. This contributes to our standing in the Research Excellence Framework (REF): In REF 2021, 72% of our research submitted was rated world-class or internationally excellent (REF) Our place We are proudly a university of place - locally, nationally and internationally. This commitment to place - and with it, opportunity - means we are a leader in widening access to higher education to students from disadvantaged and under-represented backgrounds. We have admitted more students from neighbourhoods with historically low numbers going to university than any other provider in the UK for years in a row Our widening participation team works with more than 1,100 educational settings and engages with more than 60,000 young people in our region every year Alongside the University of Sheffield, we run the Higher Education Progression Partnership South Yorkshire (HeppSY), to support young people most at risk of missing out on higher education Whilst the university is grounded in its city and region, we are also proud of our national and global reach. Our brand new campus, Sheffield Hallam University in London, opens in 2026 Our global community of international students, staff and partnerships spans 120 countries We have partnerships with more than 100 universities in countries around the world In 2019, we announced our first formal strategic international partnership with La Trobe University, Melbourne, Australia. This provides international learning experiences, collaborative research opportunities and sharing of good practice and approaches to higher education across borders.";
// Set initial question text
questionInput.value = "What are common research topics at Hallam?";
answerDiv.innerHTML = "";

// Init / Load model
async function init() {
    // Set status message
    setStatus("Loading model, this may take a few moments...");

    // Set backend (use webgl to speed up processing)
    await tf.setBackend("webgl");

    // Load the Q&A model
    model = await qna.load();

    // Set status message and enable ask button
    setStatus("Model loaded!");
    askBtn.disabled = false;
}
// Call init function
init();

// ===============================
// Search for answers (when askBtn is clicked)
// ===============================
askBtn.onclick = runQuery;

// Function to run the query
async function runQuery() {

    // Check if model is loaded
    if (!model) return;

    // Show searching status
    setStatus("Searching...");

    // Get question and context
    const question = questionInput.value;
    const context = contextInput.value;

    // Validate input is not empty
    if (!question || !context) {
        // Show error status
        setStatus("Please enter question and context");
        return;
    }

    // Find answers using the model
    const answers = await model.findAnswers(question, context);

    // Show results
    displayAnswers(answers);
}

// ===============================
// Prediction 
// ===============================

// Function to display answers
function displayAnswers(answers) {

    // If no answers found
    if (answers.length === 0) {
        // Display message and status
        answerDiv.innerHTML = "No answers found.";
        setStatus("No answers found.");
        return;
    }

    // Display answers format the output as the answer text along with the confidence score (rounded to 3 decimal places)
    answerDiv.innerHTML = "";
    for(const answer of answers) {
        answerDiv.innerHTML += `${answer.text} (score: ${answer.score.toFixed(3)})<br>`;
    }

    // Set status message
    setStatus("Done!");
}