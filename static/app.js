/**
 * static/app.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Kisan Saathi - Frontend Controller
 * Multilingual Voice/Chat, Document Vault, Task Stepper, and PMFBY Execution.
 */

// State
let currentSessionId = localStorage.getItem("kisan_saathi_session_id") || "sess_" + Math.random().toString(36).substring(2, 10);
localStorage.setItem("kisan_saathi_session_id", currentSessionId);

let currentLanguage = "hi";
let isTtsEnabled = true;
let isRecording = false;
let recognition = null;
let currentPendingDocData = null;

// Speech Synthesis
const synth = window.speechSynthesis;

// Locale mappings for Web Speech API
const LANG_LOCALES = {
  hi: "hi-IN",
  en: "en-IN",
  gar: "hi-IN",
  pa: "pa-IN",
  mr: "mr-IN",
  bn: "bn-IN",
  gu: "gu-IN",
  ta: "ta-IN",
  te: "te-IN",
  kn: "kn-IN",
  ml: "ml-IN",
  or: "or-IN",
  as: "as-IN",
};

// ─────────────────────────────────────────────────────────────────────────────
// INITIALIZATION
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initLanguageSelector();
  initSpeechRecognition();
  initChatInputs();
  initDocumentUpload();
  loadServices();
  loadVault();
  loadTasks();

  document.getElementById("runDemoBtn").addEventListener("click", runQuickDemoWorkflow);
});

// ─────────────────────────────────────────────────────────────────────────────
// TAB SWITCHING
// ─────────────────────────────────────────────────────────────────────────────
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetPane = tab.getAttribute("data-tab");
      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
      });
      document.getElementById(targetPane).classList.add("active");
    });
  });
}

function switchTab(tabId) {
  const tabBtn = document.querySelector(`.nav-tab[data-tab="${tabId}"]`);
  if (tabBtn) tabBtn.click();
}

// ─────────────────────────────────────────────────────────────────────────────
// LANGUAGE & SPEECH SYNTHESIS (TTS)
// ─────────────────────────────────────────────────────────────────────────────
function initLanguageSelector() {
  const select = document.getElementById("langSelect");
  select.addEventListener("change", (e) => {
    currentLanguage = e.target.value;
    if (recognition) {
      recognition.lang = LANG_LOCALES[currentLanguage] || "hi-IN";
    }
  });

  const ttsBtn = document.getElementById("ttsToggleBtn");
  ttsBtn.addEventListener("click", () => {
    isTtsEnabled = !isTtsEnabled;
    document.getElementById("ttsIcon").textContent = isTtsEnabled ? "🔊" : "🔇";
    if (!isTtsEnabled && synth) {
      synth.cancel();
    }
  });
}

function speakText(text) {
  if (!isTtsEnabled || !synth) return;
  synth.cancel(); // Stop any ongoing speech

  // Strip markdown, asterisks, brackets, or emojis for clean voice
  const cleanSpeech = text
    .replace(/[#*`_~]/g, "")
    .replace(/[\[\]\(\)\{\}]/g, "")
    .replace(/[🌾📋👨‍🌾🌱📍📄⚠️✅❌🎉]/g, "")
    .trim();

  const utterance = new SpeechSynthesisUtterance(cleanSpeech);
  utterance.lang = LANG_LOCALES[currentLanguage] || "hi-IN";
  utterance.rate = 0.95; // Slightly slower for better farmer comprehension
  utterance.pitch = 1.0;

  synth.speak(utterance);
}

// ─────────────────────────────────────────────────────────────────────────────
// SPEECH-TO-TEXT (STT / MICROPHONE)
// ─────────────────────────────────────────────────────────────────────────────
function initSpeechRecognition() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    console.warn("Web Speech API not supported in this browser.");
    return;
  }

  recognition = new SpeechRec();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = LANG_LOCALES[currentLanguage] || "hi-IN";

  recognition.onstart = () => {
    isRecording = true;
    updateMicButtons(true);
    const hint = document.getElementById("heroMicHint");
    if (hint) hint.textContent = "🎙️ आपकी आवाज़ सुनी जा रही है... बोलें";
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    document.getElementById("chatInput").value = transcript;
    switchTab("tab-chat");
    sendChatMessage(transcript);
  };

  recognition.onerror = (event) => {
    console.error("Speech recognition error:", event.error);
    stopRecording();
  };

  recognition.onend = () => {
    stopRecording();
  };

  // Attach to Hero and Chat mic buttons
  document.getElementById("heroMicBtn").addEventListener("click", toggleRecording);
  document.getElementById("chatMicBtn").addEventListener("click", toggleRecording);
}

function toggleRecording() {
  if (!recognition) {
    alert("आपके ब्राउज़र में आवाज़ रिकॉर्डिंग समर्थित नहीं है। कृपया टाइप करें।");
    return;
  }
  if (isRecording) {
    recognition.stop();
  } else {
    recognition.lang = LANG_LOCALES[currentLanguage] || "hi-IN";
    try {
      recognition.start();
    } catch (e) {
      console.warn("Mic start error:", e);
    }
  }
}

function stopRecording() {
  isRecording = false;
  updateMicButtons(false);
  const hint = document.getElementById("heroMicHint");
  if (hint) hint.textContent = "माइक दबाएं और अपनी समस्या बोलें";
}

function updateMicButtons(recording) {
  const heroBtn = document.getElementById("heroMicBtn");
  const chatBtn = document.getElementById("chatMicBtn");
  if (heroBtn) heroBtn.classList.toggle("listening", recording);
  if (chatBtn) chatBtn.classList.toggle("listening", recording);
}

// ─────────────────────────────────────────────────────────────────────────────
// CHAT & WORKSPACE
// ─────────────────────────────────────────────────────────────────────────────
function initChatInputs() {
  const input = document.getElementById("chatInput");
  const sendBtn = document.getElementById("chatSendBtn");

  sendBtn.addEventListener("click", () => {
    const text = input.value.trim();
    if (text) {
      sendChatMessage(text);
      input.value = "";
    }
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const text = input.value.trim();
      if (text) {
        sendChatMessage(text);
        input.value = "";
      }
    }
  });
}

function startQuickAction(prompt) {
  switchTab("tab-chat");
  sendChatMessage(prompt);
}

function sendQuickPrompt(prompt) {
  sendChatMessage(prompt);
}

async function sendChatMessage(query) {
  appendUserMessage(query);

  // Show typing placeholder
  const placeholderId = appendAssistantMessage("🌾 किसान साथी विचार कर रहा है...");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        session_id: currentSessionId,
        language: currentLanguage,
      }),
    });

    if (!res.ok) {
      throw new Error(`Server returned HTTP ${res.status}`);
    }

    const data = await res.json();

    // Replace placeholder with real response
    updateAssistantMessage(placeholderId, data.reply_text);

    // Speak aloud in farmer's language
    speakText(data.speech_text || data.reply_text);

    // Render task stepper if a plan was generated
    if (data.plan && data.plan.steps && data.plan.steps.length > 0) {
      renderTaskStepper(data.plan);
    }

    // Refresh tasks list
    loadTasks();
  } catch (err) {
    console.error("Chat error:", err);
    updateAssistantMessage(placeholderId, "⚠️ सर्वर से संपर्क नहीं हो सका। कृपया पुनः प्रयास करें।");
  }
}

function appendUserMessage(text) {
  const container = document.getElementById("chatMessages");
  const div = document.createElement("div");
  div.className = "message-bubble user";
  div.innerHTML = `
    <div class="bubble-header">
      <strong>आप (Farmer)</strong>
      <span class="time-stamp">अभी</span>
    </div>
    <div class="bubble-body">${escapeHtml(text)}</div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function appendAssistantMessage(text) {
  const container = document.getElementById("chatMessages");
  const id = "msg_" + Math.random().toString(36).substring(2, 8);
  const div = document.createElement("div");
  div.id = id;
  div.className = "message-bubble assistant";
  div.innerHTML = `
    <div class="bubble-header">
      <span class="agent-avatar">🌾</span>
      <strong>किसान साथी AI</strong>
      <span class="time-stamp">अभी</span>
    </div>
    <div class="bubble-body">${formatAgentReply(text)}</div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return id;
}

function updateAssistantMessage(elementId, text) {
  const msgEl = document.getElementById(elementId);
  if (msgEl) {
    const body = msgEl.querySelector(".bubble-body");
    if (body) {
      body.innerHTML = formatAgentReply(text);
    }
  }
  const container = document.getElementById("chatMessages");
  container.scrollTop = container.scrollHeight;
}

function formatAgentReply(text) {
  return escapeHtml(text)
    .replace(/\n\n/g, "<br><br>")
    .replace(/\n/g, "<br>");
}

function escapeHtml(text) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
  return String(text).replace(/[&<>"']/g, m => map[m]);
}

// ─────────────────────────────────────────────────────────────────────────────
// TASK STEPPER
// ─────────────────────────────────────────────────────────────────────────────
function renderTaskStepper(plan) {
  const box = document.getElementById("taskStepperBox");
  const stepsContainer = document.getElementById("stepperSteps");
  const goalEl = document.getElementById("stepperGoal");

  goalEl.textContent = plan.goal || "कार्य योजना";
  stepsContainer.innerHTML = "";

  plan.steps.forEach(step => {
    const row = document.createElement("div");
    row.className = "step-row";

    let icon = step.step_id;
    if (step.status === "completed") icon = "✓";

    row.innerHTML = `
      <div class="step-num ${step.status}">${icon}</div>
      <div class="step-info">
        <strong>${escapeHtml(step.title)}</strong>
        <span>${escapeHtml(step.description)}</span>
      </div>
    `;
    stepsContainer.appendChild(row);
  });

  box.classList.remove("hidden");
}

// ─────────────────────────────────────────────────────────────────────────────
// SERVICES CATALOG
// ─────────────────────────────────────────────────────────────────────────────
async function loadServices() {
  try {
    const res = await fetch("/api/services");
    const data = await res.json();
    const container = document.getElementById("servicesList");
    container.innerHTML = "";

    data.services.forEach(srv => {
      const item = document.createElement("div");
      item.className = "service-list-item";
      item.innerHTML = `
        <div class="sli-icon">🌾</div>
        <div class="sli-content">
          <h3>${srv.hindi_title}</h3>
          <p>${srv.description}</p>
          <div class="sli-deadlines">⏳ ${srv.deadlines}</div>
        </div>
        <button class="btn btn-primary" onclick="startQuickAction('${srv.title} योजना का लाभ कैसे लें?')">
          शुरू करें
        </button>
      `;
      container.appendChild(item);
    });
  } catch (err) {
    console.error("Failed to load services:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// DOCUMENT VAULT & OCR UPLOAD
// ─────────────────────────────────────────────────────────────────────────────
function initDocumentUpload() {
  const fileInput = document.getElementById("fileUploadInput");
  const dropZone = document.getElementById("dropZone");

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      uploadFile(e.target.files[0]);
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });
}

async function uploadFile(file) {
  const progress = document.getElementById("uploadProgress");
  progress.classList.remove("hidden");

  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", currentSessionId);
  formData.append("doc_type", "farmer_identity_or_land");

  try {
    const res = await fetch("/api/documents/upload", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Upload failed HTTP ${res.status}`);
    }

    const data = await res.json();
    progress.classList.add("hidden");

    // Open OCR Review Modal so farmer verifies extracted data
    openDocReviewModal(data.document);
    loadVault();
  } catch (err) {
    progress.classList.add("hidden");
    alert("दस्तावेज़ अपलोड में समस्या: " + err.message);
  }
}

async function loadVault() {
  try {
    const res = await fetch(`/api/documents/vault?session_id=${currentSessionId}`);
    const data = await res.json();
    const container = document.getElementById("vaultDocsList");

    if (!data.documents || data.documents.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <span>📄</span>
          <p>अभी कोई दस्तावेज़ अपलोड नहीं हुआ है। ऊपर बटन दबाकर अपलोड करें।</p>
        </div>
      `;
      return;
    }

    container.innerHTML = "";
    data.documents.forEach(doc => {
      const ext = doc.extracted_data;
      const card = document.createElement("div");
      card.className = "doc-card";
      card.innerHTML = `
        <div class="doc-card-header">
          <strong>${escapeHtml(doc.original_filename)}</strong>
          <span class="doc-badge">${doc.status === "verified" ? "✓ सत्यापित" : "समीक्षाधीन"}</span>
        </div>
        <div class="doc-field-row">
          <span>किसान:</span>
          <strong>${escapeHtml(ext.farmer_name)}</strong>
        </div>
        <div class="doc-field-row">
          <span>आधार:</span>
          <strong>•••• •••• ${escapeHtml(ext.aadhaar_last4)}</strong>
        </div>
        <div class="doc-field-row">
          <span>खसरा संख्या:</span>
          <strong>${escapeHtml(ext.khasra_plot_no)} (${ext.land_area_hectares} ha)</strong>
        </div>
        <div class="doc-field-row">
          <span>बैंक IFSC:</span>
          <strong>${escapeHtml(ext.ifsc_code)} (खाता: ••${escapeHtml(ext.bank_account_last4)})</strong>
        </div>
        <div style="margin-top:10px;">
          <button class="btn btn-secondary" style="width:100%; font-size:0.85rem;" onclick='openDocReviewModal(${JSON.stringify(doc)})'>
            समीक्षा / संपादित करें (Review)
          </button>
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error("Error loading vault:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// DOCUMENT OCR REVIEW MODAL
// ─────────────────────────────────────────────────────────────────────────────
function openDocReviewModal(doc) {
  currentPendingDocData = doc;
  const ext = doc.extracted_data;

  document.getElementById("reviewFarmerName").value = ext.farmer_name || "रामचन्द्र वर्मा";
  document.getElementById("reviewAadhaarLast4").value = ext.aadhaar_last4 || "7823";
  document.getElementById("reviewCropName").value = ext.crop_name || "बाजरा (Bajra)";
  document.getElementById("reviewKhasra").value = ext.khasra_plot_no || "142/2";
  document.getElementById("reviewArea").value = ext.land_area_hectares || 1.85;
  document.getElementById("reviewIfsc").value = ext.ifsc_code || "SBIN0001234";
  document.getElementById("reviewAccountLast4").value = ext.bank_account_last4 || "9876";

  document.getElementById("docReviewModal").classList.remove("hidden");
}

function closeDocReviewModal() {
  document.getElementById("docReviewModal").classList.add("hidden");
}

function proceedToApplicationFromReview() {
  closeDocReviewModal();

  // Gather verified fields
  const appData = {
    session_id: currentSessionId,
    farmer_name: document.getElementById("reviewFarmerName").value.trim(),
    aadhaar_last4: document.getElementById("reviewAadhaarLast4").value.trim(),
    crop_name: document.getElementById("reviewCropName").value.trim(),
    khasra_plot_no: document.getElementById("reviewKhasra").value.trim(),
    land_area_hectares: parseFloat(document.getElementById("reviewArea").value) || 1.85,
    ifsc_code: document.getElementById("reviewIfsc").value.trim(),
    bank_account_last4: document.getElementById("reviewAccountLast4").value.trim(),
    state: "Rajasthan",
    district: "Barmer",
    sub_district: "Sindhari",
    village: "Sindhari",
  };

  // Open OTP & Confirmation modal
  openOtpModal(appData);
}

// ─────────────────────────────────────────────────────────────────────────────
// OTP & APPLICATION EXECUTION
// ─────────────────────────────────────────────────────────────────────────────
let activeApplicationData = null;

async function openOtpModal(appData) {
  activeApplicationData = appData;
  const summaryEl = document.getElementById("otpAppSummary");

  summaryEl.innerHTML = `
    <strong>किसान का नाम:</strong> ${escapeHtml(appData.farmer_name)}<br>
    <strong>फसल:</strong> ${escapeHtml(appData.crop_name)} | <strong>खसरा संख्या:</strong> ${escapeHtml(appData.khasra_plot_no)}<br>
    <strong>रकबा:</strong> ${appData.land_area_hectares} हेक्टेयर | <strong>बैंक IFSC:</strong> ${escapeHtml(appData.ifsc_code)}<br>
    <strong>क्षेत्र:</strong> ${escapeHtml(appData.village)}, ${escapeHtml(appData.district)}, ${escapeHtml(appData.state)}<br>
    <strong>अनुमानित किसान प्रीमियम (2%):</strong> ₹${Math.round(appData.land_area_hectares * 960)} (लगभग)
  `;

  document.getElementById("otpInputField").value = "";
  document.getElementById("otpModal").classList.remove("hidden");

  // Call backend to initialize application in StateManager & LangGraph
  try {
    await fetch("/api/insurance/start-application", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(appData),
    });
  } catch (err) {
    console.error("Start app error:", err);
  }
}

function closeOtpModal() {
  document.getElementById("otpModal").classList.add("hidden");
}

async function submitApplicationOtp() {
  const otpInput = document.getElementById("otpInputField");
  const otpVal = otpInput.value.trim();

  if (!otpVal || otpVal.length !== 6 || !/^\d+$/.test(otpVal)) {
    alert("कृपया सही 6-अंकों का OTP दर्ज करें।");
    return;
  }

  const btn = document.getElementById("submitOtpBtn");
  btn.disabled = true;
  btn.textContent = "सत्यापन एवं सबमिशन हो रहा है...";

  try {
    const res = await fetch("/api/insurance/submit-otp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: currentSessionId,
        otp: otpVal,
      }),
    });

    const result = await res.json();
    btn.disabled = false;
    btn.textContent = "पुष्टि करें और सबमिट करें (Submit Application)";

    if (result.success) {
      closeOtpModal();
      openReceiptModal(result);
      loadTasks();
    } else {
      alert("सबमिशन विफल: " + (result.error || "कृपया OTP जांचें।"));
    }
  } catch (err) {
    btn.disabled = false;
    btn.textContent = "पुष्टि करें और सबमिट करें (Submit Application)";
    alert("सबमिशन में त्रुटि: " + err.message);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// RECEIPT MODAL
// ─────────────────────────────────────────────────────────────────────────────
function openReceiptModal(result) {
  document.getElementById("receiptRefNo").textContent = "आवेदन संख्या: " + result.application_reference_no;
  document.getElementById("receiptMessageText").textContent = result.farmer_message;

  const wmNotice = document.getElementById("simulationWatermarkNotice");
  if (result.is_simulation) {
    wmNotice.classList.remove("hidden");
  } else {
    wmNotice.classList.add("hidden");
  }

  document.getElementById("receiptModal").classList.remove("hidden");

  // Speak congratulatory completion
  speakText(result.farmer_message);
}

function closeReceiptModal() {
  document.getElementById("receiptModal").classList.add("hidden");
}

// ─────────────────────────────────────────────────────────────────────────────
// QUICK DEMO WORKFLOW
// ─────────────────────────────────────────────────────────────────────────────
function runQuickDemoWorkflow() {
  const demoData = {
    session_id: currentSessionId,
    farmer_name: "रामचन्द्र वर्मा (Ramchandra Verma)",
    aadhaar_last4: "7823",
    crop_name: "बाजरा (Bajra)",
    khasra_plot_no: "142/2",
    land_area_hectares: 1.85,
    ifsc_code: "SBIN0001234",
    bank_account_last4: "9876",
    state: "Rajasthan",
    district: "Barmer",
    sub_district: "Sindhari",
    village: "Sindhari",
  };

  openOtpModal(demoData);
  // Auto-fill demo OTP
  setTimeout(() => {
    document.getElementById("otpInputField").value = "123456";
  }, 600);
}

// ─────────────────────────────────────────────────────────────────────────────
// TASKS HISTORY
// ─────────────────────────────────────────────────────────────────────────────
async function loadTasks() {
  try {
    const res = await fetch(`/api/tasks?session_id=${currentSessionId}`);
    const data = await res.json();
    const container = document.getElementById("tasksList");

    if (!data.tasks || data.tasks.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <span>📋</span>
          <p>कोई सक्रिय कार्य नहीं है। होम स्क्रीन से सेवा शुरू करें।</p>
        </div>
      `;
      return;
    }

    container.innerHTML = "";
    data.tasks.forEach(t => {
      const card = document.createElement("div");
      card.className = "service-card";
      card.style.marginBottom = "10px";
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <h4>${escapeHtml(t.category.toUpperCase().replace(/_/g, " "))}</h4>
          <span class="badge-action">${t.status}</span>
        </div>
        <p style="margin:6px 0;"><strong>अनुरोध:</strong> ${escapeHtml(t.query)}</p>
        <span style="font-size:0.8rem; color:#6b7280;">आईडी: ${t.task_id}</span>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error("Error loading tasks:", err);
  }
}

