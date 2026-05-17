/* ==========================================================================
   Aegeas Predict JS - Serverless Inference & Dynamic Dashboard Orchestrator
   ========================================================================== */

// Global Model Bundle Reference
let modelBundle = null;

// Tab Management
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    loadModelBundle();
});

function initTabs() {
    const navItems = document.querySelectorAll(".nav-item");
    const tabContents = document.querySelectorAll(".tab-content");
    const pageTitle = document.getElementById("page-title");

    const tabTitleMap = {
        "simulator": "Client Subscription Simulator",
        "scoreboard": "Model Scoreboard & Offline Metrics",
        "visuals": "Visual Model Analysis",
        "info": "Project Methodology Details"
    };

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetTab = item.getAttribute("data-tab");

            // Toggle Nav Active Class
            navItems.forEach(nav => nav.classList.remove("active"));
            item.classList.add("active");

            // Toggle Tab Contents Visibility
            tabContents.forEach(content => {
                content.classList.remove("active");
                if (content.id === `tab-${targetTab}`) {
                    content.classList.add("active");
                }
            });

            // Update Page Header Title
            if (pageTitle && tabTitleMap[targetTab]) {
                pageTitle.textContent = tabTitleMap[targetTab];
            }
        });
    });
}

// Fetch Model Bundle and Initialize UI
async function loadModelBundle() {
    console.log("[Inference] Fetching model bundle JSON...");
    try {
        const response = await fetch("assets/model_bundle.json");
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        modelBundle = await response.json();
        console.log("[Inference] Model bundle loaded successfully!");
        
        // 1. Populate category dropdowns
        populateDropdowns();

        // 2. Populate metric scoreboard
        populateScoreboard();

        // 3. Register input listeners for simulator
        registerSimulatorListeners();

        // 4. Initial prediction calculations
        calculatePredictions();

    } catch (error) {
        console.error("[Inference] ERROR loading model bundle:", error);
    }
}

// Dynamic Option Compiler for Select elements
function populateDropdowns() {
    if (!modelBundle || !modelBundle.categorical) return;

    for (const [col, categories] of Object.entries(modelBundle.categorical)) {
        const select = document.getElementById(`input-${col}`);
        if (select) {
            select.innerHTML = "";
            categories.forEach(cat => {
                const opt = document.createElement("option");
                opt.value = cat;
                // Capitalize option text beautifully, convert '.' to space
                let displayVal = cat.charAt(0).toUpperCase() + cat.slice(1);
                if (displayVal.endsWith(".")) {
                    displayVal = displayVal.slice(0, -1);
                }
                opt.textContent = displayVal;
                select.appendChild(opt);
            });

            // Set sensible default selected values matching standard distributions
            if (col === "job") select.value = "management";
            if (col === "marital") select.value = "married";
            if (col === "education") select.value = "secondary";
            if (col === "contact") select.value = "cellular";
            if (col === "month") select.value = "may";
            if (col === "poutcome") select.value = "unknown";
        }
    }
}

// Offline Metrics Renderer
function populateScoreboard() {
    if (!modelBundle || !modelBundle.metrics) return;

    const metrics = modelBundle.metrics;

    // Helper to format values as percentages
    const fmtPct = (val) => (val * 100).toFixed(2) + "%";

    // 1. Accuracy cards
    document.getElementById("score-lr-accuracy").textContent = fmtPct(metrics["Logistic Regression"]["Accuracy"]);
    document.getElementById("score-gb-accuracy").textContent = fmtPct(metrics["Gradient Boosting"]["Accuracy"]);

    // 2. F1 Score cards
    document.getElementById("score-lr-f1").textContent = fmtPct(metrics["Logistic Regression"]["F1-Score"]);
    document.getElementById("score-gb-f1").textContent = fmtPct(metrics["Gradient Boosting"]["F1-Score"]);

    // 3. ROC-AUC cards
    document.getElementById("score-lr-auc").textContent = fmtPct(metrics["Logistic Regression"]["ROC-AUC"]);
    document.getElementById("score-gb-auc").textContent = fmtPct(metrics["Gradient Boosting"]["ROC-AUC"]);

    // 4. Performance Summary Table cells
    document.getElementById("tbl-lr-acc").textContent = fmtPct(metrics["Logistic Regression"]["Accuracy"]);
    document.getElementById("tbl-gb-acc").textContent = fmtPct(metrics["Gradient Boosting"]["Accuracy"]);
    
    document.getElementById("tbl-lr-prec").textContent = fmtPct(metrics["Logistic Regression"]["Precision"]);
    document.getElementById("tbl-gb-prec").textContent = fmtPct(metrics["Gradient Boosting"]["Precision"]);

    document.getElementById("tbl-lr-rec").textContent = fmtPct(metrics["Logistic Regression"]["Recall"]);
    document.getElementById("tbl-gb-rec").textContent = fmtPct(metrics["Gradient Boosting"]["Recall"]);

    document.getElementById("tbl-lr-f1").textContent = fmtPct(metrics["Logistic Regression"]["F1-Score"]);
    document.getElementById("tbl-gb-f1").textContent = fmtPct(metrics["Gradient Boosting"]["F1-Score"]);

    document.getElementById("tbl-lr-auc").textContent = fmtPct(metrics["Logistic Regression"]["ROC-AUC"]);
    document.getElementById("tbl-gb-auc").textContent = fmtPct(metrics["Gradient Boosting"]["ROC-AUC"]);

    // 5. Highlights/Advantage column calculations
    calculateScoreboardAdvantages(metrics);
}

function calculateScoreboardAdvantages(metrics) {
    const diff = (metrics["Gradient Boosting"]["Accuracy"] - metrics["Logistic Regression"]["Accuracy"]) * 100;
    const gbAdv = diff > 0;
    const absDiff = Math.abs(diff).toFixed(1);

    document.getElementById("adv-acc").innerHTML = gbAdv 
        ? `<span class="badge gb-badge">Gradient Boosting (+${absDiff}%)</span>` 
        : `<span class="badge lr-badge">Logistic Regression (+${absDiff}%)</span>`;

    const precDiff = metrics["Gradient Boosting"]["Precision"] - metrics["Logistic Regression"]["Precision"];
    document.getElementById("adv-prec").innerHTML = precDiff > 0 
        ? `<span class="badge gb-badge">Gradient Boosting</span>` 
        : `<span class="badge lr-badge">Logistic Regression</span>`;

    const recDiff = metrics["Gradient Boosting"]["Recall"] - metrics["Logistic Regression"]["Recall"];
    document.getElementById("adv-rec").innerHTML = recDiff > 0 
        ? `<span class="badge gb-badge">Gradient Boosting</span>` 
        : `<span class="badge lr-badge">Logistic Regression</span>`;

    const f1Diff = metrics["Gradient Boosting"]["F1-Score"] - metrics["Logistic Regression"]["F1-Score"];
    document.getElementById("adv-f1").innerHTML = f1Diff > 0 
        ? `<span class="badge gb-badge">Gradient Boosting</span>` 
        : `<span class="badge lr-badge">Logistic Regression</span>`;

    const aucDiff = metrics["Gradient Boosting"]["ROC-AUC"] - metrics["Logistic Regression"]["ROC-AUC"];
    document.getElementById("adv-auc-row").innerHTML = aucDiff > 0 
        ? `<span class="badge gb-badge">Gradient Boosting</span>` 
        : `<span class="badge lr-badge">Logistic Regression</span>`;
}

// Bind sliders and form handlers
function registerSimulatorListeners() {
    const form = document.getElementById("simulator-form");
    if (!form) return;

    // Listeners for slider real-time numeric value displays
    const bindings = [
        { sliderId: "input-age", displayId: "age-val", suffix: "" },
        { sliderId: "input-duration", displayId: "duration-val", suffix: "s" },
        { sliderId: "input-campaign", displayId: "campaign-val", suffix: "" },
        { sliderId: "input-previous", displayId: "previous-val", suffix: "" },
        { sliderId: "input-day", displayId: "day-val", suffix: "" }
    ];

    bindings.forEach(bind => {
        const slider = document.getElementById(bind.sliderId);
        const display = document.getElementById(bind.displayId);
        if (slider && display) {
            slider.addEventListener("input", (e) => {
                display.textContent = e.target.value + bind.suffix;
                calculatePredictions();
            });
        }
    });

    // Listen to changes on other form controls
    const numInputs = ["input-balance", "input-pdays"];
    numInputs.forEach(id => {
        const inp = document.getElementById(id);
        if (inp) {
            inp.addEventListener("input", calculatePredictions);
        }
    });

    const selects = ["input-job", "input-marital", "input-education", "input-contact", "input-month", "input-poutcome"];
    selects.forEach(id => {
        const sel = document.getElementById(id);
        if (sel) {
            sel.addEventListener("change", calculatePredictions);
        }
    });

    const radios = document.querySelectorAll('input[type="radio"]');
    radios.forEach(rad => {
        rad.addEventListener("change", calculatePredictions);
    });
}

// Read simulator state and evaluate equations
function calculatePredictions() {
    if (!modelBundle) return;

    // 1. Gather all inputs from the DOM
    const rawInputs = {};

    // Numerical attributes
    rawInputs["age"] = parseFloat(document.getElementById("input-age").value) || 35;
    rawInputs["balance"] = parseFloat(document.getElementById("input-balance").value) || 0;
    rawInputs["duration"] = parseFloat(document.getElementById("input-duration").value) || 0;
    rawInputs["campaign"] = parseFloat(document.getElementById("input-campaign").value) || 1;
    rawInputs["pdays"] = parseFloat(document.getElementById("input-pdays").value) || -1;
    rawInputs["previous"] = parseFloat(document.getElementById("input-previous").value) || 0;
    rawInputs["day"] = parseFloat(document.getElementById("input-day").value) || 15;

    // Categorical select fields
    rawInputs["job"] = document.getElementById("input-job").value;
    rawInputs["marital"] = document.getElementById("input-marital").value;
    rawInputs["education"] = document.getElementById("input-education").value;
    rawInputs["contact"] = document.getElementById("input-contact").value;
    rawInputs["month"] = document.getElementById("input-month").value;
    rawInputs["poutcome"] = document.getElementById("input-poutcome").value;

    // Binary radio groups
    rawInputs["default"] = document.querySelector('input[name="default"]:checked').value;
    rawInputs["housing"] = document.querySelector('input[name="housing"]:checked').value;
    rawInputs["loan"] = document.querySelector('input[name="loan"]:checked').value;

    // 2. Perform Real-Time Preprocessing & Scaling (Standardization + One-Hot)
    // Align features dynamically according to exported preprocessor bundle schema
    const features = [];
    
    for (const name of modelBundle.feature_names) {
        if (modelBundle.numerical[name]) {
            // Standard Scale: (X - mean) / std
            const rawVal = rawInputs[name];
            const mean = modelBundle.numerical[name].mean;
            const std = modelBundle.numerical[name].scale;
            const scaled = (rawVal - mean) / std;
            features.push(scaled);
        } else {
            // One-Hot Encoding
            // Name matches: "categoricalFeatureName_categoryValue" (e.g. "job_retired")
            let encodedVal = 0;
            for (const catCol of Object.keys(modelBundle.categorical)) {
                if (name.startsWith(catCol + "_")) {
                    const catVal = name.substring(catCol.length + 1);
                    const userVal = rawInputs[catCol];
                    if (userVal === catVal) {
                        encodedVal = 1;
                    }
                    break;
                }
            }
            features.push(encodedVal);
        }
    }

    // 3. Compute Logistic Regression Prediction Probability
    // equation: z = intercept + coeff_0 * feat_0 + coeff_1 * feat_1 + ...
    // prob = 1 / (1 + e^-z)
    const lrCoeffs = modelBundle.logistic_regression.coefficients;
    const lrIntercept = modelBundle.logistic_regression.intercept;
    let lrZ = lrIntercept;
    for (let i = 0; i < features.length; i++) {
        lrZ += lrCoeffs[i] * features[i];
    }
    const lrProb = 1.0 / (1.0 + Math.exp(-lrZ));

    // 4. Compute Gradient Boosting Decision Tree Probability
    // equation: score = init_value + learning_rate * sum(tree_prediction)
    // prob = 1 / (1 + e^-score)
    const gbInit = modelBundle.gradient_boosting.init_value;
    const gbLR = modelBundle.gradient_boosting.learning_rate;
    const gbTrees = modelBundle.gradient_boosting.trees;
    
    let gbScore = gbInit;
    for (const tree of gbTrees) {
        gbScore += gbLR * traverseDecisionTree(tree, features);
    }
    const gbProb = 1.0 / (1.0 + Math.exp(-gbScore));

    // 5. Animate gauges and updates displays
    updateGauge("lr-progress", "lr-val-text", lrProb, false);
    updateGauge("gb-progress", "gb-val-text", gbProb, true);

    // 6. Update positive/negative risk card tags
    updateConclusion("lr-conclusion", lrProb);
    updateConclusion("gb-conclusion", gbProb);

    // 7. Update Live Risk Assessments
    updateLiveRiskAssessments(rawInputs);
}

// Recursively traverse a decision node matching feature threshold splits
function traverseDecisionTree(node, features) {
    if (node.leaf) {
        return node.value;
    }
    const val = features[node.feature_idx];
    if (val <= node.threshold) {
        return traverseDecisionTree(node.left, features);
    } else {
        return traverseDecisionTree(node.right, features);
    }
}

// Percentage progress dial animator
function updateGauge(gaugeId, textId, prob, isGB) {
    const gauge = document.getElementById(gaugeId);
    const textText = document.getElementById(textId);
    if (!gauge || !textText) return;

    const percentage = (prob * 100).toFixed(1);
    textText.textContent = percentage + "%";

    // Set conic-gradient degree
    const deg = Math.round(prob * 360);
    const fillCol = isGB ? "var(--secondary)" : "var(--primary)";
    gauge.style.background = `conic-gradient(${fillCol} ${deg}deg, rgba(255, 255, 255, 0.05) 0deg)`;
}

// Conclusion Card Status Indicator
function updateConclusion(elId, prob) {
    const el = document.getElementById(elId);
    if (!el) return;

    // Standard baseline for positive campaign target
    if (prob >= 0.5) {
        el.className = "prediction-conclusion subscribe";
        el.innerHTML = `<i class="fa-solid fa-circle-check"></i> <span>Subscribes (Highly Likely)</span>`;
    } else if (prob >= 0.22) {
        // Since dataset positive baseline is only ~11.7%, a probability > 22% is a solid lead
        el.className = "prediction-conclusion subscribe";
        el.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> <span>Strong Target Lead</span>`;
    } else {
        el.className = "prediction-conclusion no-subscribe";
        el.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> <span>Unlikely to subscribe</span>`;
    }
}

// Dynamically compute key indicators on live simulator updates
function updateLiveRiskAssessments(inputs) {
    // 1. Call Duration Assessment
    const duration = inputs.duration;
    let durStatusText = "Low Impact";
    let durStatusClass = "negative";
    let durPercent = 20;
    let durColor = "var(--coral)";

    if (duration > 500) {
        durStatusText = "Extremely High (Optimal)";
        durStatusClass = "positive";
        durPercent = 95;
        durColor = "var(--secondary)";
    } else if (duration > 200) {
        durStatusText = "Moderate Engagement";
        durStatusClass = "neuter";
        durPercent = 60;
        durColor = "var(--primary)";
    }

    updateRiskFactor("bar-duration", "status-duration", durStatusText, durStatusClass, durPercent, durColor);

    // 2. Financial Balance Level Assessment
    const balance = inputs.balance;
    let balStatusText = "Low Reserves";
    let balStatusClass = "negative";
    let balPercent = 30;
    let balColor = "var(--coral)";

    if (balance > 4000) {
        balStatusText = "High Assets (A-Tier)";
        balStatusClass = "positive";
        balPercent = 90;
        balColor = "var(--secondary)";
    } else if (balance >= 1000) {
        balStatusText = "Stable Balance";
        balStatusClass = "neuter";
        balPercent = 65;
        balColor = "var(--primary)";
    } else if (balance >= 0) {
        balStatusText = "Low Reserves";
        balStatusClass = "neuter";
        balPercent = 45;
        balColor = "var(--primary)";
    }

    updateRiskFactor("bar-balance", "status-balance", balStatusText, balStatusClass, balPercent, balColor);

    // 3. Housing Debt Assessment
    const housing = inputs.housing;
    let houseStatusText = "Debt Burden";
    let houseStatusClass = "negative";
    let housePercent = 40;
    let houseColor = "var(--coral)";

    if (housing === "no") {
        houseStatusText = "No Debt (Optimal)";
        houseStatusClass = "positive";
        housePercent = 95;
        houseColor = "var(--secondary)";
    }

    updateRiskFactor("bar-housing", "status-housing", houseStatusText, houseStatusClass, housePercent, houseColor);
}

// UI helper to update individual risk factor rows
function updateRiskFactor(barId, statusId, text, styleClass, percent, color) {
    const bar = document.getElementById(barId);
    const status = document.getElementById(statusId);
    if (bar && status) {
        status.textContent = text;
        status.className = `factor-status ${styleClass}`;
        bar.style.width = `${percent}%`;
        bar.style.backgroundColor = color;
    }
}
