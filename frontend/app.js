const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadArea = document.getElementById("uploadArea");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const progress = document.getElementById("progress");
const preview = document.getElementById("preview");
const previewImg = document.getElementById("previewImg");
const resultSection = document.getElementById("result");
const jsonResult = document.getElementById("jsonResult");

// API base (override by setting window.API_BASE in the browser console)
const API_BASE = window.API_BASE || "http://localhost:5000";

let confidenceChart = null;

function show(el) {
  el.style.display = "block";
}

function hide(el) {
  el.style.display = "none";
}

// Drag and drop functionality
uploadArea.addEventListener("click", () => fileInput.click());

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    fileInput.files = files;
    handleFileSelect();
  }
});

// File selection handler
fileInput.addEventListener("change", handleFileSelect);

function handleFileSelect() {
  const file = fileInput.files[0];
  if (file) {
    fileName.textContent = file.name;
    fileInfo.classList.remove("hidden");
    uploadBtn.disabled = false;
  }
}

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    alert("Choisissez un fichier d'abord.");
    return;
  }

  // Preview if image
  if (file.type && file.type.startsWith("image/")) {
    previewImg.src = URL.createObjectURL(file);
    show(preview);
  } else {
    hide(preview);
  }

  // Prepare upload
  const form = new FormData();
  form.append("file", file);

  hide(resultSection);
  progress.classList.remove("hidden");

  try {
    const resp = await fetch(`${API_BASE}/api/classify`, {
      method: "POST",
      body: form,
    });
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`Server error ${resp.status}: ${txt}`);
    }
    const data = await resp.json();

    // Display results with visualization
    displayResults(data);
    show(resultSection);
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    jsonResult.textContent = "Erreur: " + e.toString();
    show(resultSection);
  } finally {
    progress.classList.add("hidden");
  }
});

function displayResults(data) {
  // Display raw JSON
  jsonResult.textContent = JSON.stringify(data, null, 2);
  
  console.log("=== displayResults called ===");
  console.log("data:", data);

  // Extract main classification (from actual backend response)
  const finalClass = data.classification || "Unknown";
  const finalScore = data.confidence_globale || 0;
  const confidencePercent = Math.round(finalScore * 100);

  console.log("finalClass:", finalClass, "finalScore:", finalScore, "percent:", confidencePercent);

  // Update classification badge
  const classNameEl = document.getElementById("className");
  const confPercentEl = document.getElementById("confidencePercent");
  
  if (classNameEl) classNameEl.textContent = formatClassName(finalClass);
  if (confPercentEl) confPercentEl.textContent = confidencePercent + "%";

  // Update module scores from actual backend response
  const cvScore = data.confidence_cv || 0;
  const nlpScore = data.confidence_nlp || 0;
  const gabaritsScore = data.confidence_gabarit || 0;

  console.log("cvScore:", cvScore, "nlpScore:", nlpScore, "gabaritsScore:", gabaritsScore);

  // Update CV module
  const scoreCV = document.getElementById("scoreCV");
  const classCV = document.getElementById("classCV");
  if (scoreCV) scoreCV.textContent = Math.round(cvScore * 100) + "%";
  if (classCV) classCV.textContent = formatClassName(finalClass);
  const moduleCV = document.getElementById("moduleCV");
  if (moduleCV) {
    if (cvScore > 0.6) moduleCV.classList.add("high-score");
    else moduleCV.classList.remove("high-score");
  }

  // Update NLP module
  const scoreNLP = document.getElementById("scoreNLP");
  const classNLP = document.getElementById("classNLP");
  if (scoreNLP) scoreNLP.textContent = Math.round(nlpScore * 100) + "%";
  if (classNLP) classNLP.textContent = formatClassName(finalClass);
  const moduleNLP = document.getElementById("moduleNLP");
  if (moduleNLP) {
    if (nlpScore > 0.6) moduleNLP.classList.add("high-score");
    else moduleNLP.classList.remove("high-score");
  }

  // Update Gabarits module
  const scoreGabarits = document.getElementById("scoreGabarits");
  const classGabarits = document.getElementById("classGabarits");
  if (scoreGabarits) scoreGabarits.textContent = Math.round(gabaritsScore * 100) + "%";
  if (classGabarits) classGabarits.textContent = formatClassName(finalClass);
  const moduleGabarits = document.getElementById("moduleGabarits");
  if (moduleGabarits) {
    if (gabaritsScore > 0.6) moduleGabarits.classList.add("high-score");
    else moduleGabarits.classList.remove("high-score");
  }

  // Update processing info
  if (data.processing_times?.total) {
    const processingTimeEl = document.getElementById("processingTime");
    if (processingTimeEl) processingTimeEl.textContent = data.processing_times.total.toFixed(2) + "s";
  }
  
  if (data.metadata?.ocr_confidence !== undefined) {
    const ocrConfEl = document.getElementById("ocrConfidence");
    if (ocrConfEl) ocrConfEl.textContent = Math.round(data.metadata.ocr_confidence * 100) + "%";
  }
  
  if (data.should_reject !== undefined) {
    const status = data.should_reject ? "❌ Rejeté" : "✅ Accepté";
    const color = data.should_reject ? "#e74c3c" : "#27ae60";
    const rejectEl = document.getElementById("shouldReject");
    if (rejectEl) {
      rejectEl.textContent = status;
      rejectEl.style.color = color;
    }
  }
  
  const pagesCountEl = document.getElementById("pagesCount");
  if (pagesCountEl) pagesCountEl.textContent = "1";

  // Create confidence chart
  createConfidenceChart(data);
}

function updateModuleBox(moduleName, prediction, boxId, scoreId, classId) {
  if (!prediction) {
    document.getElementById(scoreId).textContent = "-";
    document.getElementById(classId).textContent = "-";
    return;
  }

  const score = prediction.score || 0;
  const scorePercent = Math.round(score * 100);
  const predictedClass = formatClassName(prediction.class || "Unknown");

  document.getElementById(scoreId).textContent = `${scorePercent}%`;
  document.getElementById(classId).textContent = predictedClass;

  // Highlight high scores
  const box = document.getElementById(boxId);
  if (score > 0.6) {
    box.classList.add("high-score");
  } else {
    box.classList.remove("high-score");
  }
}

function createConfidenceChart(data) {
  const ctx = document.getElementById("confidenceChart");
  if (!ctx) return;

  // Use all_scores from backend response
  const allScores = data.all_scores || {};
  const labels = Object.keys(allScores).map(formatClassName);

  // Create datasets for each module based on scores
  const cvScore = data.confidence_cv || 0;
  const nlpScore = data.confidence_nlp || 0;
  const gabaritsScore = data.confidence_gabarit || 0;
  const fusionScore = data.confidence_globale || 0;

  // Distribute scores proportionally across classes
  const cvData = Object.values(allScores).map(score => Math.round(score * cvScore * 100));
  const nlpData = Object.values(allScores).map(score => Math.round(score * nlpScore * 100));
  const gabaritsData = Object.values(allScores).map(score => Math.round(score * gabaritsScore * 100));
  const fusionData = Object.values(allScores).map(score => Math.round(score * fusionScore * 100));

  // Destroy previous chart if exists
  if (confidenceChart) {
    confidenceChart.destroy();
  }

  confidenceChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Vision (CV)",
          data: cvData,
          backgroundColor: "rgba(102, 126, 234, 0.7)",
          borderColor: "rgba(102, 126, 234, 1)",
          borderWidth: 1,
        },
        {
          label: "Texte (NLP)",
          data: nlpData,
          backgroundColor: "rgba(118, 75, 162, 0.7)",
          borderColor: "rgba(118, 75, 162, 1)",
          borderWidth: 1,
        },
        {
          label: "Layout (Gabarits)",
          data: gabaritsData,
          backgroundColor: "rgba(82, 211, 130, 0.7)",
          borderColor: "rgba(82, 211, 130, 1)",
          borderWidth: 1,
        },
        {
          label: "Fusion (Final)",
          data: fusionData,
          backgroundColor: "rgba(255, 159, 64, 0.7)",
          borderColor: "rgba(255, 159, 64, 1)",
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
          labels: {
            font: { size: 12, weight: "600" },
            padding: 15,
            usePointStyle: true,
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: {
            callback: (value) => `${value}%`,
            font: { size: 11 },
          },
          grid: {
            color: "rgba(0, 0, 0, 0.05)",
          },
        },
        x: {
          ticks: {
            font: { size: 11, weight: "500" },
          },
          grid: { display: false },
        },
      },
    },
  });
}
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Vision (CV)",
          data: cvData,
          backgroundColor: "rgba(102, 126, 234, 0.7)",
          borderColor: "rgba(102, 126, 234, 1)",
          borderWidth: 1,
        },
        {
          label: "Texte (NLP)",
          data: nlpData,
          backgroundColor: "rgba(118, 75, 162, 0.7)",
          borderColor: "rgba(118, 75, 162, 1)",
          borderWidth: 1,
        },
        {
          label: "Layout (Gabarits)",
          data: gabaritsData,
          backgroundColor: "rgba(82, 211, 130, 0.7)",
          borderColor: "rgba(82, 211, 130, 1)",
          borderWidth: 1,
        },
        {
          label: "Fusion (Final)",
          data: fusionData,
          backgroundColor: "rgba(255, 159, 64, 0.7)",
          borderColor: "rgba(255, 159, 64, 1)",
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
          labels: {
            font: { size: 12, weight: "600" },
            padding: 15,
            usePointStyle: true,
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: {
            callback: (value) => `${value}%`,
            font: { size: 11 },
          },
          grid: {
            color: "rgba(0, 0, 0, 0.05)",
          },
        },
        x: {
          ticks: {
            font: { size: 11, weight: "500" },
          },
          grid: { display: false },
        },
      },
    },
  });
}

function formatClassName(className) {
  if (!className) return "Unknown";
  return className
    .replace(/_/g, " ")
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

// Copy JSON to clipboard
document.getElementById("copyJsonBtn").addEventListener("click", () => {
  const text = jsonResult.textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById("copyJsonBtn");
    const originalText = btn.textContent;
    btn.textContent = "✅ Copié!";
    setTimeout(() => {
      btn.textContent = originalText;
    }, 2000);
  });
});
