"use strict";

/* JanSetu AI — Module 6 frontend. Vanilla JS, no build step. */

const API = {
  voice: "/api/voice",
  checkDoc: "/api/check-doc",
  generatePdf: "/api/generate-pdf",
  reset: "/api/reset",
};

const FIELD_ORDER = [
  ["scheme_type", "योजना / Scheme"],
  ["applicant_name", "नाम / Name"],
  ["father_or_husband_name", "पिता या पति"],
  ["age", "उम्र / Age"],
  ["annual_income_inr", "वार्षिक आय"],
  ["district", "जिला / District"],
  ["address", "पता / Address"],
  ["grievance_summary", "शिकायत / Grievance"],
];

const SCHEME_LABELS = {
  old_age_pension: "वृद्धावस्था पेंशन",
  income_certificate: "आय प्रमाण पत्र",
  caste_certificate: "जाति प्रमाण पत्र",
  pm_kisan: "पीएम किसान",
  jansunwai: "जनसुनवाई",
  unknown: "अज्ञात",
};

const MISSING_LABELS = {
  applicant_name: "नाम",
  father_or_husband_name: "पिता या पति का नाम",
  age: "उम्र",
  annual_income_inr: "वार्षिक आय",
  district: "जिला",
  address: "पता",
  grievance_summary: "शिकायत का विवरण",
};

const SESSION_KEY = "jansetu.session";

const $ = (id) => document.getElementById(id);
const chat = $("chat");
const textInput = $("textInput");
const sendBtn = $("sendBtn");
const micBtn = $("micBtn");
const attachBtn = $("attachBtn");
const docFile = $("docFile");
const docType = $("docType");
const pdfBtn = $("pdfBtn");
const resetBtn = $("resetBtn");

const welcomeNode = document.querySelector(".bubble.welcome");

let sessionId = readSessionId();
let busy = false;
let turns = 0;
let lastReady = false;
let recorder = null;
let chunks = [];
let recordingTimer = null;
let recordingStartedAt = 0;

/* ----------------------------------------------------------------- session */

function newSessionId() {
  if (window.crypto && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return "s-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
}

function readSessionId() {
  try {
    const stored = sessionStorage.getItem(SESSION_KEY);
    if (stored) return stored;
    const created = newSessionId();
    sessionStorage.setItem(SESSION_KEY, created);
    return created;
  } catch {
    return newSessionId();
  }
}

/* ----------------------------------------------------------------- helpers */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function clockTime() {
  return new Date().toLocaleTimeString("hi-IN", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function scrollToEnd() {
  chat.scrollTop = chat.scrollHeight;
}

function appendChat(node) {
  chat.append(node);
  scrollToEnd();
  return node;
}

function addUserBubble(text) {
  const bubble = el("div", "bubble user");
  bubble.append(el("p", null, text), el("span", "meta", clockTime()));
  return appendChat(bubble);
}

function addBotBubble(text, audioUrl) {
  const bubble = el("div", "bubble bot");
  bubble.append(el("p", null, text));
  if (audioUrl) {
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = audioUrl;
    bubble.append(audio);
  }
  bubble.append(el("span", "meta", clockTime()));
  return appendChat(bubble);
}

function addTyping() {
  const bubble = el("div", "bubble bot");
  const dots = el("span", "typing");
  dots.append(el("i"), el("i"), el("i"));
  bubble.append(dots);
  return appendChat(bubble);
}

function addNote(kind, icon, title, detail, options = {}) {
  const note = el("div", "note " + kind);
  note.append(el("span", "note-icon", icon));

  const body = el("div");
  body.append(el("strong", null, title));
  if (detail) {
    const paragraph = el("div", null, detail);
    paragraph.style.marginTop = "2px";
    body.append(paragraph);
  }
  if (options.meta) body.append(el("div", "note-detail", options.meta));
  if (options.link) {
    const link = el("a", null, options.link.text);
    link.href = options.link.href;
    link.target = "_blank";
    link.rel = "noopener";
    if (options.link.download) link.download = options.link.download;
    link.style.display = "inline-block";
    link.style.marginTop = "4px";
    body.append(link);
  }

  note.append(body);
  return appendChat(note);
}

let toastTimer = null;

function toast(message, kind = "") {
  const node = $("toast");
  node.textContent = message;
  node.className = "toast " + kind;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    node.hidden = true;
  }, 3800);
}

async function readError(response) {
  try {
    const data = await response.json();
    const detail = data.detail;
    if (typeof detail === "string") return detail;
    if (detail) return JSON.stringify(detail);
  } catch {
    /* fall through */
  }
  return response.status + " " + response.statusText;
}

async function postForm(url, formData) {
  const response = await fetch(url, { method: "POST", body: formData });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

/* -------------------------------------------------------------- busy state */

function syncComposer() {
  sendBtn.disabled = busy;
  micBtn.disabled = busy;
  attachBtn.disabled = busy;
  pdfBtn.disabled = busy || !lastReady;
  pdfBtn.classList.toggle("ready", lastReady && !busy);
}

function setBusy(value) {
  busy = value;
  syncComposer();
}

function setPdfReady(value) {
  lastReady = Boolean(value);
  syncComposer();
}

/* ------------------------------------------------------------- inspector UI */

function renderSlots(slots) {
  const list = $("slotList");
  list.textContent = "";

  if (!slots) {
    const empty = el("div", "doc-body empty", "पहला संदेश भेजें — निकाले गए slots यहाँ दिखेंगे।");
    list.append(empty);
    $("slotCount").textContent = "0/0";
    $("missingWrap").textContent = "";
    return;
  }

  let shown = 0;
  let filled = 0;

  for (const [key, label] of FIELD_ORDER) {
    if (key === "grievance_summary" && slots.scheme_type !== "jansunwai" && !slots.grievance_summary) {
      continue;
    }

    shown += 1;
    const raw = slots[key];
    let value = raw;
    let has = raw !== null && raw !== undefined && String(raw).trim() !== "";
    if (key === "scheme_type" && raw === "unknown") has = false;

    if (has) {
      filled += 1;
      if (key === "scheme_type") value = SCHEME_LABELS[raw] || raw;
      else if (key === "age") value = `${raw} वर्ष`;
      else if (key === "annual_income_inr") {
        value = "₹" + Number(raw).toLocaleString("en-IN");
      }
    }

    const row = el("div", "slot-row" + (has ? "" : " missing"));
    row.append(el("dt", null, label), el("dd", null, has ? String(value) : ""));
    list.append(row);
  }

  $("slotCount").textContent = `${filled}/${shown}`;

  const wrap = $("missingWrap");
  wrap.textContent = "";
  const missing = Array.isArray(slots.missing_fields) ? slots.missing_fields : [];
  if (missing.length) {
    wrap.append(el("span", "missing-label", "अभी बाकी"));
    for (const key of missing) {
      wrap.append(el("span", "badge-missing", MISSING_LABELS[key] || key));
    }
  }
}

function renderEligibility(eligibility, citations) {
  const badge = $("eligBadge");
  const reasons = $("reasonList");
  const chips = $("citationWrap");
  const docs = $("docsWrap");

  reasons.textContent = "";
  chips.textContent = "";
  docs.textContent = "";

  if (!eligibility) {
    badge.textContent = "—";
    badge.className = "badge neutral";
    return;
  }

  if (eligibility.is_eligible === true) {
    badge.textContent = "पात्र / Eligible";
    badge.className = "badge ok";
  } else if (eligibility.is_eligible === false) {
    badge.textContent = "अपात्र / Not eligible";
    badge.className = "badge bad";
  } else {
    badge.textContent = "जानकारी अपेक्षित";
    badge.className = "badge warn";
  }

  for (const reason of eligibility.reasons || []) {
    reasons.append(el("li", null, reason));
  }

  const seen = new Set();

  // Show the rule the verdict was actually grounded on first, then any extra
  // chunks this utterance retrieved, so the two can never be confused.
  const verdict = [...(eligibility.source_citations || [])];
  const retrieved = [...new Set((citations || []).map((c) => `${c.source_file}#${c.section}`))];
  const extras = retrieved.filter((text) => !verdict.includes(text));

  const addGroup = (label, items) => {
    if (!items.length) return;
    chips.append(el("div", "chips-label", label));
    for (const text of items) {
      if (seen.has(text)) continue;
      seen.add(text);
      const source = (citations || []).find(
        (c) => `${c.source_file}#${c.section}` === text
      );
      const chip = el("span", "cite", text);
      if (source && source.content) chip.title = source.content;
      chips.append(chip);
    }
  };

  addGroup("निर्णय के स्रोत", verdict);
  addGroup("इस टर्न में मिले", extras);

  const required = eligibility.required_documents || [];
  if (required.length) {
    docs.append(el("div", "docs-title", "आवश्यक दस्तावेज़"));
    for (const document of required) {
      docs.append(el("div", "doc-item", document));
    }
  }
}

function renderDoc(result) {
  const badge = $("docBadge");
  const body = $("docBody");
  body.textContent = "";
  body.classList.remove("empty");

  if (!result) {
    badge.textContent = "No scan";
    badge.className = "badge neutral";
    body.classList.add("empty");
    body.textContent = "अभी कोई दस्तावेज़ जाँचा नहीं गया। पेपरक्लिप दबाकर फोटो भेजें।";
    return;
  }

  if (result.is_blurry) {
    badge.textContent = "Blurry";
    badge.className = "badge warn";
  } else if (result.is_valid) {
    badge.textContent = "Pass";
    badge.className = "badge ok";
  } else {
    badge.textContent = "Fail";
    badge.className = "badge bad";
  }

  const grid = el("div", "stat-grid");
  const addStat = (key, value) => {
    const stat = el("div", "stat");
    stat.append(el("span", "k", key), el("span", "v", value));
    grid.append(stat);
  };

  addStat("blur_score", Number(result.blur_score).toFixed(2));
  addStat(
    "name_match",
    result.name_match_score === null || result.name_match_score === undefined
      ? "—"
      : `${Number(result.name_match_score).toFixed(1)}%`
  );
  addStat("extracted name", result.extracted_name || "—");
  addStat(
    "extracted income",
    result.extracted_income_inr === null || result.extracted_income_inr === undefined
      ? "—"
      : "₹" + Number(result.extracted_income_inr).toLocaleString("en-IN")
  );
  body.append(grid);

  const warningClass = result.is_valid ? "" : result.is_blurry ? " warn" : " bad";
  body.append(el("div", "doc-warning" + warningClass, result.hindi_warning));
}

function resetInspector() {
  renderSlots(null);
  renderEligibility(null, []);
  renderDoc(null);
}

/* ---------------------------------------------------------------- one turn */

async function runVoiceTurn({ text, audio, voiceMessage }) {
  if (busy) return;

  const userBubble = text
    ? addUserBubble(text)
    : addUserBubble("🎙️ वॉइस संदेश भेजा…");

  setBusy(true);
  const typing = addTyping();

  const form = new FormData();
  form.append("session_id", sessionId);
  if (text) form.append("text", text);
  if (audio) form.append("audio", audio, audio.name || "voice.webm");

  try {
    const data = await postForm(API.voice, form);
    typing.remove();

    if (voiceMessage && data.transcript) {
      userBubble.querySelector("p").textContent = "🎙️ " + data.transcript;
      scrollToEnd();
    }

    addBotBubble(data.bot_reply_hindi, data.tts_audio_url);
    renderSlots(data.slots);
    renderEligibility(data.eligibility, data.rag_citations);

    turns += 1;
    $("turnCount").textContent = turns + (turns === 1 ? " turn" : " turns");

    const wasReady = lastReady;
    setPdfReady(Boolean(data.ready_for_pdf));
    if (!wasReady && lastReady) {
      addNote(
        "ok",
        "✓",
        "सभी जानकारी पूरी है",
        "अब नीचे से pre-filled आवेदन PDF बना सकते हैं।"
      );
    }
  } catch (error) {
    typing.remove();
    addNote("bad", "!", "संदेश नहीं भेजा जा सका", error.message);
    toast("सर्वर से संपर्क नहीं हो सका", "bad");
  } finally {
    setBusy(false);
  }
}

function sendText(raw) {
  const text = (raw || "").trim();
  if (!text || busy) return;
  textInput.value = "";
  runVoiceTurn({ text });
}

/* ------------------------------------------------------------ voice capture */

function supportedMimeType() {
  if (typeof MediaRecorder === "undefined") return "";
  const options = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  for (const type of options) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

function startRecordingTimer() {
  recordingStartedAt = Date.now();
  $("recordingTime").textContent = "0:00";
  $("recordingBar").hidden = false;
  clearInterval(recordingTimer);
  recordingTimer = setInterval(() => {
    const seconds = Math.floor((Date.now() - recordingStartedAt) / 1000);
    $("recordingTime").textContent =
      Math.floor(seconds / 60) + ":" + String(seconds % 60).padStart(2, "0");
  }, 250);
}

function stopRecordingTimer() {
  clearInterval(recordingTimer);
  recordingTimer = null;
  $("recordingBar").hidden = true;
}

async function toggleRecording() {
  if (busy) return;

  if (recorder && recorder.state === "recording") {
    recorder.stop();
    return;
  }

  if (!navigator.mediaDevices || typeof MediaRecorder === "undefined") {
    toast("इस ब्राउज़र में वॉइस रिकॉर्डिंग उपलब्ध नहीं है", "bad");
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    toast("माइक्रोफ़ोन की अनुमति नहीं मिली", "bad");
    return;
  }

  const mimeType = supportedMimeType();
  const options = mimeType ? { mimeType } : undefined;
  const active = new MediaRecorder(stream, options);
  chunks = [];

  active.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) chunks.push(event.data);
  };

  active.onstop = async () => {
    stream.getTracks().forEach((track) => track.stop());
    stopRecordingTimer();
    micBtn.classList.remove("recording");
    micBtn.setAttribute("aria-pressed", "false");
    recorder = null;

    const type = active.mimeType || mimeType || "audio/webm";
    const blob = new Blob(chunks, { type });
    chunks = [];

    if (blob.size < 1200) {
      toast("रिकॉर्डिंग बहुत छोटी थी — कृपया फिर बोलें", "bad");
      return;
    }

    const name = type.includes("mp4") || type.includes("m4a") ? "voice.m4a" : "voice.webm";
    await runVoiceTurn({ audio: new File([blob], name, { type }), voiceMessage: true });
  };

  recorder = active;
  active.start();
  micBtn.classList.add("recording");
  micBtn.setAttribute("aria-pressed", "true");
  startRecordingTimer();
  toast("रिकॉर्डिंग शुरू — रोकने के लिए माइक दबाएँ");
}

/* ------------------------------------------------------------- doc pre-check */

async function uploadDocument(file) {
  if (!file || busy) return;

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("document_type", docType.value);
  form.append("file", file, file.name);

  addUserBubble("📎 दस्तावेज़: " + file.name);
  setBusy(true);
  const typing = addTyping();

  try {
    const result = await postForm(API.checkDoc, form);
    typing.remove();
    renderDoc(result);

    const kind = result.is_valid ? "ok" : result.is_blurry ? "warn" : "bad";
    const icon = result.is_valid ? "✓" : "!";
    const title = result.is_valid
      ? "दस्तावेज़ सत्यापित"
      : result.is_blurry
        ? "फोटो धुंधली है"
        : "दस्तावेज़ में अंतर";
    const match =
      result.name_match_score === null || result.name_match_score === undefined
        ? "नाम मिलान उपलब्ध नहीं"
        : "name match " + Number(result.name_match_score).toFixed(1) + "%";
    const meta = `blur ${Number(result.blur_score).toFixed(2)} · ${match}`;

    addNote(kind, icon, title, result.hindi_warning, { meta });
  } catch (error) {
    typing.remove();
    addNote("bad", "!", "दस्तावेज़ जाँच नहीं हो सकी", error.message);
  } finally {
    docFile.value = "";
    setBusy(false);
  }
}

/* -------------------------------------------------------------------- PDF */

async function generatePdf() {
  if (busy || !lastReady) return;

  setBusy(true);
  const typing = addTyping();

  try {
    const data = await postJson(API.generatePdf, { session_id: sessionId });
    typing.remove();

    addNote(
      "ok",
      "✓",
      "Pre-filled आवेदन तैयार है",
      "आवेदन ID: " + data.application_id,
      { link: { href: data.pdf_url, text: "PDF खोलें / डाउनलोड करें", download: data.filename } }
    );

    const anchor = document.createElement("a");
    anchor.href = data.pdf_url;
    anchor.download = data.filename;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();

    toast("PDF डाउनलोड शुरू हो गया");
  } catch (error) {
    typing.remove();
    addNote("bad", "!", "PDF नहीं बन सका", error.message);
  } finally {
    setBusy(false);
  }
}

/* ------------------------------------------------------------------ reset */

async function resetDemo() {
  if (busy) return;

  try {
    const form = new FormData();
    form.append("session_id", sessionId);
    await postForm(API.reset, form);
  } catch {
    /* a fresh local session is still the right outcome */
  }

  sessionId = newSessionId();
  try {
    sessionStorage.setItem(SESSION_KEY, sessionId);
  } catch {
    /* private mode: keep the in-memory id */
  }

  turns = 0;
  $("turnCount").textContent = "0 turns";

  chat.textContent = "";
  const day = el("div", "day-chip");
  day.append(el("span", null, "आज"));
  chat.append(day);
  if (welcomeNode) chat.append(welcomeNode);

  setPdfReady(false);
  resetInspector();
  textInput.value = "";
  scrollToEnd();
  toast("डेमो रीसेट हो गया — नई शुरुआत करें");
}

/* ------------------------------------------------------------------- wiring */

sendBtn.addEventListener("click", () => sendText(textInput.value));

textInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendText(textInput.value);
  }
});

micBtn.addEventListener("click", toggleRecording);

attachBtn.addEventListener("click", () => docFile.click());

docFile.addEventListener("change", () => {
  if (docFile.files && docFile.files[0]) uploadDocument(docFile.files[0]);
});

pdfBtn.addEventListener("click", generatePdf);
resetBtn.addEventListener("click", resetDemo);

chat.addEventListener("click", (event) => {
  const chip = event.target.closest(".chip");
  if (chip && chip.dataset.prompt) sendText(chip.dataset.prompt);
});

window.addEventListener("beforeunload", () => {
  if (recorder && recorder.state === "recording") recorder.stop();
});

/* ------------------------------------------------------------------- boot */

resetInspector();
setPdfReady(false);
syncComposer();
textInput.focus();
