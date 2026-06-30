// Road Accident Severity Predictor — Frontend JS

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(el => el.classList.remove("active"));
  document.getElementById("tab-" + name).classList.add("active");
  event.currentTarget.classList.add("active");
}

// ── Algorithm hints ───────────────────────────────────────────────────────────
const ALGO_HINTS = {
  rf:  "Ensemble of 200 CART decision trees (Bagging). No feature scaling required.",
  svm: "RBF kernel SVM — One-vs-One multi-class. Uses StandardScaler internally.",
  nn:  "MLP: Input(9)→128→64→32→Output(3). ReLU + Adam + Early stopping."
};

function updateAlgoHint() {
  const m = document.getElementById("model").value;
  document.getElementById("algoHint").textContent = ALGO_HINTS[m];
}

// ── Safety tips ───────────────────────────────────────────────────────────────
const TIPS = {
  Slight:  "✅ Low-risk scenario. Standard road safety precautions apply.",
  Serious: "⚠️ Moderate-high risk. Reduce speed, increase following distance, and stay alert.",
  Fatal:   "🚨 Extremely dangerous conditions. Avoid travel if possible. If unavoidable, drive very slowly and alert emergency services if needed."
};

// ── Prediction ────────────────────────────────────────────────────────────────
async function predict() {
  const btn = document.getElementById("predictBtn");
  btn.textContent = "Predicting…";
  btn.disabled = true;

  const payload = {
    model:        document.getElementById("model").value,
    hour:         document.getElementById("hour").value,
    speed:        document.getElementById("speed").value,
    weather:      document.getElementById("weather").value,
    road_type:    document.getElementById("road_type").value,
    light:        document.getElementById("light").value,
    vehicles:     document.getElementById("vehicles").value,
    alcohol:      document.getElementById("alcohol").checked ? "1" : "0",
    seatbelt:     document.getElementById("seatbelt").checked ? "1" : "0",
    road_surface: document.getElementById("road_surface").value,
  };

  try {
    const res  = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.error) { alert("Error: " + data.error); return; }

    // Show result
    document.getElementById("placeholder").classList.add("hidden");
    document.getElementById("resultContent").classList.remove("hidden");

    // Badge
    const badge = document.getElementById("severityBadge");
    badge.textContent = data.severity.toUpperCase();
    badge.style.background = data.color + "18";
    badge.style.color       = data.color;
    badge.style.border      = "2.5px solid " + data.color;

    // Bars (animate after short delay)
    setTimeout(() => {
      setBar("barSlight",  "pctSlight",  data.probabilities.Slight);
      setBar("barSerious", "pctSerious", data.probabilities.Serious);
      setBar("barFatal",   "pctFatal",   data.probabilities.Fatal);
    }, 80);

    // Algorithm info
    const a = data.algo;
    document.getElementById("algoUsedBox").innerHTML =
      `<strong>Model used:</strong> ${a.name}<br/>
       <strong>Type:</strong> ${a.type}<br/>
       <strong>Config:</strong> ${a.key}<br/>
       <em>${a.note}</em>`;

    document.getElementById("tipBox").textContent = TIPS[data.severity];

  } catch (e) {
    alert("Connection error — is Flask running? (python app.py)");
  } finally {
    btn.textContent = "Predict Severity →";
    btn.disabled = false;
  }
}

function setBar(barId, pctId, value) {
  document.getElementById(barId).style.width = value + "%";
  document.getElementById(pctId).textContent = value + "%";
}
