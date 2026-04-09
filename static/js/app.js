const state = {
  currentPrediction: null,
  chart: null
};

const elements = {
  themeToggle: document.getElementById("themeToggle"),
  messageInput: document.getElementById("messageInput"),
  predictBtn: document.getElementById("predictBtn"),
  loading: document.getElementById("loading"),
  predictError: document.getElementById("predictError"),
  resultCard: document.getElementById("resultCard"),
  resultLabel: document.getElementById("resultLabel"),
  resultConfidence: document.getElementById("resultConfidence"),
  highlightedText: document.getElementById("highlightedText"),
  suggestionText: document.getElementById("suggestionText"),
  explanationText: document.getElementById("explanationText"),
  chatInput: document.getElementById("chatInput"),
  chatBtn: document.getElementById("chatBtn"),
  chatOutput: document.getElementById("chatOutput"),
  csvFile: document.getElementById("csvFile"),
  bulkBtn: document.getElementById("bulkBtn"),
  bulkStatus: document.getElementById("bulkStatus"),
  totalChecked: document.getElementById("totalChecked"),
  modelAccuracy: document.getElementById("modelAccuracy"),
  avgConfidence: document.getElementById("avgConfidence"),
  historyList: document.getElementById("historyList"),
  ratioChart: document.getElementById("ratioChart")
};

function setTheme(theme) {
  document.body.classList.toggle("dark", theme === "dark");
  localStorage.setItem("theme", theme);
}

function initializeTheme() {
  const saved = localStorage.getItem("theme");
  if (saved) {
    setTheme(saved);
    return;
  }
  setTheme(window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}

function setLoading(active) {
  elements.loading.classList.toggle("hidden", !active);
  elements.predictBtn.disabled = active;
}

function setError(message) {
  elements.predictError.textContent = message || "";
  elements.predictError.classList.toggle("hidden", !message);
}

function renderPrediction(prediction) {
  state.currentPrediction = prediction;
  elements.resultCard.classList.remove("hidden");

  const label = prediction.label === "spam" ? "Spam" : "Not Spam";
  elements.resultLabel.innerHTML =
    `${label}<span class="tag ${prediction.label}">${(prediction.spam_probability * 100).toFixed(1)}% spam risk</span>`;
  elements.resultConfidence.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;
  elements.highlightedText.innerHTML = prediction.highlighted_text;
  elements.suggestionText.textContent = prediction.suggestion;
  elements.explanationText.textContent = prediction.explanation;
}

function renderHistory(items) {
  if (!items.length) {
    elements.historyList.innerHTML = "<div class='muted'>No history yet.</div>";
    return;
  }
  elements.historyList.innerHTML = items
    .slice()
    .reverse()
    .slice(0, 30)
    .map((item) => {
      const date = new Date(item.timestamp).toLocaleString();
      return `
        <div class="history-item">
          <div><strong>${item.message.slice(0, 90)}</strong></div>
          <div>${date} <span class="tag ${item.label}">${item.label.toUpperCase()}</span></div>
          <div>Confidence: ${(item.confidence * 100).toFixed(1)}%</div>
        </div>
      `;
    })
    .join("");
}

function renderChart(spamCount, hamCount) {
  if (state.chart) {
    state.chart.destroy();
  }
  state.chart = new Chart(elements.ratioChart, {
    type: "pie",
    data: {
      labels: ["Spam", "Ham"],
      datasets: [
        {
          data: [spamCount, hamCount],
          backgroundColor: ["#ef4444", "#22c55e"]
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          labels: {
            color: getComputedStyle(document.body).getPropertyValue("--text").trim()
          }
        }
      }
    }
  });
}

async function refreshAnalytics() {
  const response = await fetch("/analyze");
  const data = await response.json();
  elements.totalChecked.textContent = data.total_messages_checked;
  elements.modelAccuracy.textContent = `${(data.model_accuracy * 100).toFixed(1)}%`;
  elements.avgConfidence.textContent = `${(data.average_confidence * 100).toFixed(1)}%`;
  renderChart(data.spam_count, data.ham_count);
}

async function refreshHistory() {
  const response = await fetch("/history");
  const data = await response.json();
  renderHistory(data.history || []);
}

async function onPredict() {
  const message = elements.messageInput.value.trim();
  if (!message) {
    setError("Please enter a message first.");
    return;
  }
  setError("");
  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }
    renderPrediction(data.prediction);
    await Promise.all([refreshAnalytics(), refreshHistory()]);
  } catch (error) {
    setError(error.message || "Something went wrong.");
  } finally {
    setLoading(false);
  }
}

async function onChat() {
  const question = elements.chatInput.value.trim();
  if (!question) return;

  elements.chatOutput.textContent = "Thinking...";
  const context = state.currentPrediction || {};

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, context })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Unable to fetch answer.");
    elements.chatOutput.textContent = data.answer;
  } catch (error) {
    elements.chatOutput.textContent = error.message || "Failed to get assistant response.";
  }
}

async function onBulkUpload() {
  const file = elements.csvFile.files[0];
  if (!file) {
    elements.bulkStatus.textContent = "Please choose a CSV file.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  elements.bulkBtn.disabled = true;
  elements.bulkStatus.textContent = "Processing CSV...";

  try {
    const response = await fetch("/bulk_predict", {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Bulk prediction failed.");
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "bulk_predictions.csv";
    anchor.click();
    window.URL.revokeObjectURL(url);
    elements.bulkStatus.textContent = "Done. Results downloaded.";
    await Promise.all([refreshAnalytics(), refreshHistory()]);
  } catch (error) {
    elements.bulkStatus.textContent = error.message || "Bulk upload failed.";
  } finally {
    elements.bulkBtn.disabled = false;
  }
}

function bindEvents() {
  elements.themeToggle.addEventListener("click", () => {
    const next = document.body.classList.contains("dark") ? "light" : "dark";
    setTheme(next);
    refreshAnalytics().catch(() => {});
  });
  elements.predictBtn.addEventListener("click", onPredict);
  elements.chatBtn.addEventListener("click", onChat);
  elements.bulkBtn.addEventListener("click", onBulkUpload);
}

async function initialize() {
  initializeTheme();
  bindEvents();
  await Promise.all([refreshAnalytics(), refreshHistory()]);
}

initialize().catch((err) => {
  console.error("Failed to initialize app:", err);
});
