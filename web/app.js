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
            if (col === "job") select.value = "admin.";
            if (col === "marital") select.value = "married";
            if (col === "education") select.value = "university.degree";
            if (col === "contact") select.value = "cellular";
            if (col === "month") select.value = "may";
            if (col === "day_of_week") select.value = "mon";
            if (col === "poutcome") select.value = "nonexistent";
        }
    }
}

// Offline Metrics Renderer
function populateScoreboard() {
    if (!modelBundle || !modelBundle.metrics) return;

    const metrics = modelBundle.metrics;

    // Helper to format values as percentages
    const fmtPct = (val) => val !== undefined ? (val * 100).toFixed(2) + "%" : "--";

    // 1. Accuracy cards
    if (document.getElementById("score-lr-accuracy")) document.getElementById("score-lr-accuracy").textContent = fmtPct(metrics["Logistic Regression"]["Accuracy"]);
    if (document.getElementById("score-knn-accuracy")) document.getElementById("score-knn-accuracy").textContent = fmtPct(metrics["K-Nearest Neighbors"]["Accuracy"]);
    if (document.getElementById("score-nb-accuracy")) document.getElementById("score-nb-accuracy").textContent = fmtPct(metrics["Naive Bayes"]["Accuracy"]);
    if (document.getElementById("score-rf-accuracy")) document.getElementById("score-rf-accuracy").textContent = fmtPct(metrics["Random Forest"]["Accuracy"]);
    if (document.getElementById("score-gb-accuracy")) document.getElementById("score-gb-accuracy").textContent = fmtPct(metrics["Gradient Boosting"]["Accuracy"]);

    // 2. F1 Score cards
    if (document.getElementById("score-lr-f1")) document.getElementById("score-lr-f1").textContent = fmtPct(metrics["Logistic Regression"]["F1-Score"]);
    if (document.getElementById("score-knn-f1")) document.getElementById("score-knn-f1").textContent = fmtPct(metrics["K-Nearest Neighbors"]["F1-Score"]);
    if (document.getElementById("score-nb-f1")) document.getElementById("score-nb-f1").textContent = fmtPct(metrics["Naive Bayes"]["F1-Score"]);
    if (document.getElementById("score-rf-f1")) document.getElementById("score-rf-f1").textContent = fmtPct(metrics["Random Forest"]["F1-Score"]);
    if (document.getElementById("score-gb-f1")) document.getElementById("score-gb-f1").textContent = fmtPct(metrics["Gradient Boosting"]["F1-Score"]);

    // 3. ROC-AUC cards
    if (document.getElementById("score-lr-auc")) document.getElementById("score-lr-auc").textContent = fmtPct(metrics["Logistic Regression"]["ROC-AUC"]);
    if (document.getElementById("score-knn-auc")) document.getElementById("score-knn-auc").textContent = fmtPct(metrics["K-Nearest Neighbors"]["ROC-AUC"]);
    if (document.getElementById("score-nb-auc")) document.getElementById("score-nb-auc").textContent = fmtPct(metrics["Naive Bayes"]["ROC-AUC"]);
    if (document.getElementById("score-rf-auc")) document.getElementById("score-rf-auc").textContent = fmtPct(metrics["Random Forest"]["ROC-AUC"]);
    if (document.getElementById("score-gb-auc")) document.getElementById("score-gb-auc").textContent = fmtPct(metrics["Gradient Boosting"]["ROC-AUC"]);

    // 4. Performance Summary Table cells
    const models = {
        "lr": "Logistic Regression",
        "knn": "K-Nearest Neighbors",
        "nb": "Naive Bayes",
        "rf": "Random Forest",
        "gb": "Gradient Boosting"
    };

    for (const [code, name] of Object.entries(models)) {
        if (metrics[name]) {
            if (document.getElementById(`tbl-${code}-acc`)) document.getElementById(`tbl-${code}-acc`).textContent = fmtPct(metrics[name]["Accuracy"]);
            if (document.getElementById(`tbl-${code}-prec`)) document.getElementById(`tbl-${code}-prec`).textContent = fmtPct(metrics[name]["Precision"]);
            if (document.getElementById(`tbl-${code}-rec`)) document.getElementById(`tbl-${code}-rec`).textContent = fmtPct(metrics[name]["Recall"]);
            if (document.getElementById(`tbl-${code}-f1`)) document.getElementById(`tbl-${code}-f1`).textContent = fmtPct(metrics[name]["F1-Score"]);
            if (document.getElementById(`tbl-${code}-auc`)) document.getElementById(`tbl-${code}-auc`).textContent = fmtPct(metrics[name]["ROC-AUC"]);
        }
    }

    // 5. Highlights/Advantage column calculations
    calculateScoreboardAdvantages(metrics);
}

function calculateScoreboardAdvantages(metrics) {
    const models = ["Logistic Regression", "K-Nearest Neighbors", "Naive Bayes", "Random Forest", "Gradient Boosting"];
    
    const badgeClasses = {
        "Logistic Regression": "lr-badge",
        "K-Nearest Neighbors": "knn-badge",
        "Naive Bayes": "nb-badge",
        "Random Forest": "rf-badge",
        "Gradient Boosting": "gb-badge"
    };

    // Find best for a given metric
    const getBestModel = (metricKey) => {
        let bestModel = models[0];
        let maxVal = -1;
        for (const model of models) {
            if (metrics[model] && metrics[model][metricKey] > maxVal) {
                maxVal = metrics[model][metricKey];
                bestModel = model;
            }
        }
        return bestModel;
    };

    const bestAcc = getBestModel("Accuracy");
    if (document.getElementById("best-acc")) {
        document.getElementById("best-acc").innerHTML = `<span class="badge ${badgeClasses[bestAcc]}">${bestAcc}</span>`;
    }

    const bestPrec = getBestModel("Precision");
    if (document.getElementById("best-prec")) {
        document.getElementById("best-prec").innerHTML = `<span class="badge ${badgeClasses[bestPrec]}">${bestPrec}</span>`;
    }

    const bestRec = getBestModel("Recall");
    if (document.getElementById("best-rec")) {
        document.getElementById("best-rec").innerHTML = `<span class="badge ${badgeClasses[bestRec]}">${bestRec}</span>`;
    }

    const bestF1 = getBestModel("F1-Score");
    if (document.getElementById("best-f1")) {
        document.getElementById("best-f1").innerHTML = `<span class="badge ${badgeClasses[bestF1]}">${bestF1}</span>`;
    }

    const bestAuc = getBestModel("ROC-AUC");
    if (document.getElementById("best-auc-row")) {
        document.getElementById("best-auc-row").innerHTML = `<span class="badge ${badgeClasses[bestAuc]}">${bestAuc}</span>`;
    }
}

// Bind sliders and form handlers
function registerSimulatorListeners() {
    const form = document.getElementById("simulator-form");
    if (!form) return;

    // Listeners for slider real-time numeric value displays
    const bindings = [
        { sliderId: "input-age", displayId: "age-val", suffix: "" },
        { sliderId: "input-campaign", displayId: "campaign-val", suffix: "" },
        { sliderId: "input-previous", displayId: "previous-val", suffix: "" },
        { sliderId: "input-euribor3m", displayId: "euribor3m-val", suffix: "%" },
        { sliderId: "input-emp_var_rate", displayId: "emp_var_rate-val", suffix: "" },
        { sliderId: "input-cons_price_idx", displayId: "cons_price_idx-val", suffix: "" },
        { sliderId: "input-cons_conf_idx", displayId: "cons_conf_idx-val", suffix: "" },
        { sliderId: "input-nr_employed", displayId: "nr_employed-val", suffix: "" }
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
    const numInputs = ["input-pdays"];
    numInputs.forEach(id => {
        const inp = document.getElementById(id);
        if (inp) {
            inp.addEventListener("input", calculatePredictions);
        }
    });

    const selects = ["input-job", "input-marital", "input-education", "input-contact", "input-month", "input-day_of_week", "input-poutcome"];
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
    rawInputs["campaign"] = parseFloat(document.getElementById("input-campaign").value) || 1;
    rawInputs["pdays"] = parseFloat(document.getElementById("input-pdays").value) || 999;
    rawInputs["previous"] = parseFloat(document.getElementById("input-previous").value) || 0;
    rawInputs["euribor3m"] = parseFloat(document.getElementById("input-euribor3m").value) || 4.86;
    rawInputs["emp.var.rate"] = parseFloat(document.getElementById("input-emp_var_rate").value) || 1.1;
    rawInputs["cons.price.idx"] = parseFloat(document.getElementById("input-cons_price_idx").value) || 93.99;
    rawInputs["cons.conf.idx"] = parseFloat(document.getElementById("input-cons_conf_idx").value) || -36.4;
    rawInputs["nr.employed"] = parseFloat(document.getElementById("input-nr_employed").value) || 5191;

    // Categorical select fields
    rawInputs["job"] = document.getElementById("input-job").value;
    rawInputs["marital"] = document.getElementById("input-marital").value;
    rawInputs["education"] = document.getElementById("input-education").value;
    rawInputs["contact"] = document.getElementById("input-contact").value;
    rawInputs["month"] = document.getElementById("input-month").value;
    rawInputs["day_of_week"] = document.getElementById("input-day_of_week").value;
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
    // 1. Euribor 3M Rate Assessment
    const euribor = inputs["euribor3m"];
    let eurStatusText = "Moderate Rate";
    let eurStatusClass = "neuter";
    let eurPercent = 60;
    let eurColor = "var(--primary)";

    if (euribor > 4.5) {
        eurStatusText = "High Rate (Restrictive)";
        eurStatusClass = "negative";
        eurPercent = 30;
        eurColor = "var(--coral)";
    } else if (euribor < 2.0) {
        eurStatusText = "Low Rate (Expansionary)";
        eurStatusClass = "positive";
        eurPercent = 90;
        eurColor = "var(--secondary)";
    }

    updateRiskFactor("bar-euribor", "status-euribor", eurStatusText, eurStatusClass, eurPercent, eurColor);

    // 2. Consumer Confidence Index Assessment
    const confidence = inputs["cons.conf.idx"];
    let confStatusText = "Stable Sentiment";
    let confStatusClass = "neuter";
    let confPercent = 65;
    let confColor = "var(--primary)";

    if (confidence < -45.0) {
        confStatusText = "Pessimistic Sentiment";
        confStatusClass = "negative";
        confPercent = 30;
        confColor = "var(--coral)";
    } else if (confidence > -35.0) {
        confStatusText = "Optimistic Sentiment";
        confStatusClass = "positive";
        confPercent = 90;
        confColor = "var(--secondary)";
    }

    updateRiskFactor("bar-confidence", "status-confidence", confStatusText, confStatusClass, confPercent, confColor);

    // 3. Housing Debt Assessment
    const housing = inputs.housing;
    let houseStatusText = "Active Housing Loan";
    let houseStatusClass = "negative";
    let housePercent = 45;
    let houseColor = "var(--coral)";

    if (housing === "no") {
        houseStatusText = "No Debt (Optimal)";
        houseStatusClass = "positive";
        housePercent = 95;
        houseColor = "var(--secondary)";
    } else if (housing === "unknown") {
        houseStatusText = "Unknown Debt Status";
        houseStatusClass = "neuter";
        housePercent = 60;
        houseColor = "var(--primary)";
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
