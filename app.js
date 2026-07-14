const wg = window.webgazer;

import { updateFaceAnalysis, resetFaceAnalysis, getDebugInfo, startCalibrationSampling, stopCalibrationSampling } from "./faceAnalyzer.js";
import {
  splitIntoSections,
  explainParagraph,
  clearExplanationCache,
  isLocalLlmEnabled,
  prewarmLocalLlm,
} from "./llmExplainer.js";


//DOM references
const bodyEl = document.body;
const landingPage = document.getElementById("landing-page");
const workspacePage = document.getElementById("workspace-page");
const tryNowBtn = document.getElementById("try-now-btn");
const backHomeBtn = document.getElementById("back-home-btn");
const debugToggleBtn = document.getElementById("debug-toggle-btn");
const debugPanel = document.getElementById("debug-panel");

const textInput = document.getElementById("text-input");
const renderTextBtn = document.getElementById("render-text-btn");
const clearTextBtn = document.getElementById("clear-text-btn");

const startTrackingBtn = document.getElementById("start-tracking-btn");
const stopTrackingBtn = document.getElementById("stop-tracking-btn");
const calibrateBtn = document.getElementById("calibrate-btn");

const readerContent = document.getElementById("reader-content");
const readerScrollArea = document.getElementById("reader-scroll-area");
const readerFontSizeSlider = document.getElementById("reader-font-size-slider");
const readerFontSizeValue = document.getElementById("reader-font-size-value");
const readerFontDecreaseBtn = document.getElementById("reader-font-decrease-btn");
const readerFontIncreaseBtn = document.getElementById("reader-font-increase-btn");
const focusModeToggleBtn = document.getElementById("focus-mode-toggle-btn");
const readerProgressLabel = document.getElementById("reader-progress-label");
const readingProgressFill = document.getElementById("reading-progress-fill");
const readerTrackingStatus = document.getElementById("reader-tracking-status");
const readerTrackingStatusLabel = document.getElementById("reader-tracking-status-label");
const readerEditTextBtn = document.getElementById("reader-edit-text-btn");
const prepareView = document.getElementById("prepare-view");
const readingView = document.getElementById("reading-view");
const themeOptionButtons = [...document.querySelectorAll("[data-theme-option]")];

const webcamStatus  = document.getElementById("webcam-status");
const trackingStatus = document.getElementById("tracking-status");
const au4StatusEl   = document.getElementById("au4-status");
const au7StatusEl   = document.getElementById("au7-status");

const exportLogBtn = document.getElementById("export-log-btn");
const resetDataBtn = document.getElementById("reset-data-btn");

const gazeDot = document.getElementById("gaze-dot");

const calibrationLayer = document.getElementById("calibration-layer");
const calibrationPoints = document.getElementById("calibration-points");
const calibrationTitle = document.getElementById("calibration-title");
const calibrationDescription = document.getElementById("calibration-description");
const calibrationStatus = document.getElementById("calibration-status");
const closeCalibrationBtn = document.getElementById("close-calibration-btn");
const retryFaceCalibrationBtn = document.getElementById("retry-face-calibration-btn");


//State
const MAX_DELTA_MS = 250;
const GAZE_TOLERANCE_PX = 24;
const SMOOTH_ALPHA = 0.1;

// Confusion detection thresholds
const CONFUSION_DWELL_MS = 2000;
const CONFUSION_SOFT_DWELL_MS = 3500;
const CONFUSION_HOLD_STRONG_MS = 700;
const CONFUSION_HOLD_SOFT_MS = 1200;
const CONFUSION_GLOBAL_COOLDOWN_MS = 20000;
const FONT_SIZE_MIN = 18;
const FONT_SIZE_MAX = 48;
const FONT_SIZE_STEP = 2;
const DEFAULT_FONT_SIZE = 28;

let lineRegistry = [];
let lineMap = new Map();
let preparedParagraphs = [];
let pendingReaderRerenderFrame = 0;
let focusModeEnabled = true;

let isTracking = false;
let webgazerStarted = false;
let calibrationPhase = "idle"; // idle / gaze / face / complete / failed
let faceCalibrationTimeout = 0;

let lastTick = null;
let currentLineKey = null;
let currentParagraphIndex = null;
let currentParagraphEnteredAt = null;
let currentParagraphDwellMs = 0;
let confusionHoldMs = 0;
let lastExplanationAt = -Infinity;
let smoothX = null;
let smoothY = null;

// Confusion state
let faceState = null;
let explainedParagraphs = new Map(); 
const pendingExplanationParagraphs = new Set();
const activeNotes = new Map();
let noteRequestSequence = 0;

// Session log
const sessionLog = [];
const gazeLog = [];
const faceLog = [];
let sessionStartTime = performance.now();
let lastFaceLogAt = -Infinity;
let previousFaceLogState = null;

function makeLineKey(paragraphIndex, lineIndex) {
  return `${paragraphIndex}:${lineIndex}`;
}

function formatMs(ms) {
  if (!Number.isFinite(ms)) return "0 ms";

  if (ms < 1000) {
    return `${Math.round(ms)} ms`;
  }

  return `${(ms / 1000).toFixed(1)} s`;
}

function setStatus(element, text, active = false) {
  element.textContent = text;
  element.classList.toggle("is-active", active);
  element.classList.toggle("is-inactive", !active);
  updateReaderTrackingStatus();
}

function readStoredPreference(key, fallback) {
  try {
    const stored = window.localStorage.getItem(key);
    return stored === null ? fallback : stored;
  } catch {
    return fallback;
  }
}

function storePreference(key, value) {
  try {
    window.localStorage.setItem(key, String(value));
  } catch {
  }
}

function clampFontSize(value) {
  const numericValue = Number(value);
  const clamped = Math.min(FONT_SIZE_MAX, Math.max(FONT_SIZE_MIN, numericValue));
  return Math.round(clamped / FONT_SIZE_STEP) * FONT_SIZE_STEP;
}

function updateReaderTrackingStatus() {
  if (!readerTrackingStatus || !readerTrackingStatusLabel) return;

  readerTrackingStatus.classList.toggle("is-active", isTracking);
  readerTrackingStatus.classList.toggle("is-inactive", !isTracking);
  readerTrackingStatusLabel.textContent = isTracking
    ? "Eye-tracker attivo"
    : "Eye-tracker spento";
}

function setTheme(theme) {
  const validTheme = ["light", "cream", "dark"].includes(theme) ? theme : "cream";
  workspacePage.dataset.theme = validTheme;
  themeOptionButtons.forEach((button) => {
    const isSelected = button.dataset.themeOption === validTheme;
    button.classList.toggle("is-selected", isSelected);
    button.setAttribute("aria-pressed", String(isSelected));
  });
  storePreference("lume.theme", validTheme);
}

function setFocusMode(enabled) {
  focusModeEnabled = Boolean(enabled);
  readerContent.classList.toggle("is-focus-mode", focusModeEnabled);
  focusModeToggleBtn.classList.toggle("is-active", focusModeEnabled);
  focusModeToggleBtn.setAttribute("aria-checked", String(focusModeEnabled));
  storePreference("lume.focusMode", focusModeEnabled);
}

function setWorkspaceMode(mode) {
  const isReading = mode === "reading";
  prepareView.hidden = isReading;
  readingView.hidden = !isReading;
  workspacePage.dataset.workspaceMode = mode;

  if (isReading) {
    window.requestAnimationFrame(() => {
      updateReadingProgress();
      schedulePreparedTextRerender();
    });
  }
}

function updateReadingStatus(paragraphIndex = currentParagraphIndex) {
  if (!preparedParagraphs.length) {
    readerProgressLabel.textContent = "Prepara il testo";
    return;
  }

  const visibleParagraph = Number.isInteger(paragraphIndex) ? paragraphIndex + 1 : 1;
  readerProgressLabel.textContent = `Paragrafo ${visibleParagraph} di ${preparedParagraphs.length}`;
}

function updateReadingProgress() {
  const scrollableHeight = readerScrollArea.scrollHeight - readerScrollArea.clientHeight;
  const progress = scrollableHeight > 0 ? readerScrollArea.scrollTop / scrollableHeight : 0;
  readingProgressFill.style.transform = `scaleX(${Math.min(1, Math.max(0, progress))})`;
}

function setScreen(screen) {
  const onWorkspace = screen === "workspace";
  bodyEl.dataset.screen = screen;
  bodyEl.classList.toggle("is-workspace-screen", onWorkspace);
  landingPage.hidden = onWorkspace;
  workspacePage.hidden = !onWorkspace;
}

function setDebugOpen(forceOpen) {
  debugPanel.hidden = !forceOpen;
  debugToggleBtn.classList.toggle("is-active", forceOpen);
  debugToggleBtn.setAttribute("aria-expanded", String(forceOpen));
}

function updateAUIndicators() {
  const dbg = getDebugInfo();
  if (!dbg.ready) {
    const label = dbg.calibrationState === "failed" ? "errore" : "calibra";
    au4StatusEl.textContent = label;
    au7StatusEl.textContent = label;
    au4StatusEl.className = "status-badge";
    au7StatusEl.className = "status-badge";
    return;
  }
  const au4On = faceState?.au4Active ?? false;
  const au4Support = faceState?.au4Supportive ?? false;
  const au7On = faceState?.au7Active ?? false;
  au4StatusEl.textContent = au4On ? "attivo" : au4Support ? "supporto" : dbg.au4Delta ?? "—";
  au7StatusEl.textContent = au7On ? "attivo" : dbg.au7Delta ?? "—";
  au4StatusEl.className = "status-badge" + (au4On || au4Support ? " is-active" : "");
  au7StatusEl.className = "status-badge" + (au7On ? " is-active" : "");
}

function resetMetricsOnly({ clearNotes = false, resetFace = false, resetLogs = false } = {}) {
  lineRegistry.forEach((line) => {
    line.dwellMs = 0;
    line.samples = 0;
    line.element.classList.remove("is-current", "is-read");
    line.paragraphElement.classList.remove("is-current", "is-confused");
  });

  currentLineKey = null;
  currentParagraphIndex = null;
  currentParagraphEnteredAt = null;
  currentParagraphDwellMs = 0;
  confusionHoldMs = 0;
  lastExplanationAt = -Infinity;
  lastTick = null;

  explainedParagraphs.clear();
  pendingExplanationParagraphs.clear();
  faceState = null;
  if (clearNotes) {
    activeNotes.clear();
  }
  renderAllMarginSlots();
  clearExplanationCache();
  if (resetFace) resetFaceAnalysis();
  if (resetLogs) {
    sessionLog.length = 0;
    gazeLog.length = 0;
    faceLog.length = 0;
    sessionStartTime = performance.now();
    lastFaceLogAt = -Infinity;
    previousFaceLogState = null;
  }
  updateReadingStatus();
}

function resetReaderPlaceholder() {
  if (pendingReaderRerenderFrame) {
    window.cancelAnimationFrame(pendingReaderRerenderFrame);
    pendingReaderRerenderFrame = 0;
  }

  preparedParagraphs = [];
  readerContent.innerHTML = `
    <div class="reader-placeholder">
      <strong>Incolla un testo e premi “Prepara testo”.</strong>
      <span>Qui comparirà la lettura organizzata in paragrafi.</span>
    </div>
  `;

  lineRegistry = [];
  lineMap = new Map();

  resetMetricsOnly({ clearNotes: true, resetLogs: true });
  updateReadingProgress();
}


//Text rendering
function syncReaderFontSizeControls(fontSizePx) {
  const safeSize = clampFontSize(fontSizePx);
  const label = `${safeSize} px`;

  readerFontSizeSlider.value = String(safeSize);
  readerFontSizeValue.value = label;
  readerFontSizeValue.textContent = label;
}

function getReaderFontSizePx() {
  const computedFontSize = parseFloat(window.getComputedStyle(readerContent).fontSize);
  if (Number.isFinite(computedFontSize)) {
    return Math.round(computedFontSize);
  }

  return Number(readerFontSizeSlider.value) || DEFAULT_FONT_SIZE;
}

function setReaderFontSize(fontSizePx) {
  const safeSize = clampFontSize(fontSizePx);
  document.documentElement.style.setProperty("--reader-font-size", `${safeSize}px`);
  syncReaderFontSizeControls(safeSize);
  storePreference("lume.fontSize", safeSize);
}

function renderPreparedText() {
  if (!preparedParagraphs.length) {
    resetReaderPlaceholder();
    return;
  }

  resetMetricsOnly({ resetLogs: true });
  readerContent.innerHTML = "";
  lineRegistry = [];
  lineMap = new Map();
  renderParagraphs(preparedParagraphs);
  setFocusMode(focusModeEnabled);
  updateReadingStatus();
  updateReadingProgress();
}

function schedulePreparedTextRerender() {
  if (!preparedParagraphs.length || readingView.hidden) return;

  if (pendingReaderRerenderFrame) {
    window.cancelAnimationFrame(pendingReaderRerenderFrame);
  }

  pendingReaderRerenderFrame = window.requestAnimationFrame(() => {
    pendingReaderRerenderFrame = 0;
    renderPreparedText();
  });
}

function getReaderTextWidth() {
  const probeRow = document.createElement("div");
  const probeText = document.createElement("p");
  const probeMargin = document.createElement("aside");
  probeRow.className = "reading-row reading-row--measurement";
  probeText.className = "paragraph-box";
  probeMargin.className = "margin-slot";
  probeRow.append(probeText, probeMargin);
  readerContent.appendChild(probeRow);
  const width = probeText.getBoundingClientRect().width;
  probeRow.remove();
  return width || readerContent.clientWidth;
}

function getCanvasFontFromElement(element) {
  const style = window.getComputedStyle(element);

  return [
    style.fontStyle,
    style.fontVariant,
    style.fontWeight,
    style.fontSize,
    style.fontFamily,
  ].join(" ");
}

function splitParagraphIntoVisualLines(paragraphText) {
  const availableWidth = getReaderTextWidth();
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  ctx.font = getCanvasFontFromElement(readerContent);

  const words = paragraphText.trim().split(/\s+/);
  const lines = [];

  let currentLine = "";

  words.forEach((word) => {
    const candidate = currentLine ? `${currentLine} ${word}` : word;
    const candidateWidth = ctx.measureText(candidate).width;

    if (candidateWidth > availableWidth && currentLine) {
      lines.push(currentLine);
      currentLine = word;
    } else {
      currentLine = candidate;
    }
  });

  if (currentLine) {
    lines.push(currentLine);
  }

  return lines;
}

async function renderText() {
  const rawText = textInput.value.trim();

  if (!rawText) {
    resetReaderPlaceholder();
    return;
  }

  renderTextBtn.disabled = true;
  renderTextBtn.textContent = "Analisi testo…";

  try {
    await document.fonts.load(`400 ${getReaderFontSizePx()}px Atkinson Hyperlegible`);

    preparedParagraphs = await splitIntoSections(rawText);
    activeNotes.clear();
    setWorkspaceMode("reading");
    renderPreparedText();
    readerScrollArea.scrollTop = 0;

    if (isLocalLlmEnabled()) {
      prewarmLocalLlm().catch((error) => {
        console.warn("Local LLM warm-up failed:", error);
      });
    }
  } catch (error) {
    console.error("Text split failed:", error);
    alert(error?.message || "Errore durante la divisione del testo.");
  } finally {
    renderTextBtn.disabled = false;
    renderTextBtn.textContent = "Prepara testo";
  }
}

function renderParagraphs(paragraphs) {
  const marginHeading = document.createElement("div");
  marginHeading.className = "reading-margin-heading-row";
  marginHeading.innerHTML = `
    <span aria-hidden="true"></span>
    <div class="margin-column-header"><span>Margine · spiegazioni</span><i aria-hidden="true"></i></div>
  `;
  readerContent.appendChild(marginHeading);

  paragraphs.forEach((paragraphText, paragraphIndex) => {
    const row = document.createElement("div");
    row.className = "reading-row";
    row.dataset.paragraphIndex = String(paragraphIndex);

    const box = document.createElement("p");
    box.className = "paragraph-box";
    box.dataset.paragraphIndex = String(paragraphIndex);

    const visualLines = splitParagraphIntoVisualLines(paragraphText);

    visualLines.forEach((lineText, lineIndex) => {
      const lineElement = document.createElement("span");
      lineElement.className = "reader-line";
      lineElement.textContent = lineText;
      lineElement.dataset.paragraphIndex = String(paragraphIndex);
      lineElement.dataset.lineIndex = String(lineIndex);

      box.appendChild(lineElement);

      const key = makeLineKey(paragraphIndex, lineIndex);
      lineRegistry.push({ key, paragraphIndex, lineIndex, text: lineText, element: lineElement, paragraphElement: box, dwellMs: 0, samples: 0 });
      lineMap.set(key, lineRegistry[lineRegistry.length - 1]);
    });

    const explainButton = document.createElement("button");
    explainButton.type = "button";
    explainButton.className = "paragraph-explain-button";
    explainButton.dataset.paragraphIndex = String(paragraphIndex);
    explainButton.setAttribute("aria-controls", `margin-note-${paragraphIndex}`);
    explainButton.setAttribute("aria-expanded", "false");
    explainButton.setAttribute("aria-label", `Spiega il paragrafo ${paragraphIndex + 1}`);
    explainButton.innerHTML = `
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M12 3a7.5 7.5 0 0 0-4.8 13.26V20l3.25-1.63A7.5 7.5 0 1 0 12 3Z"></path>
        <path d="M9.4 10a2.65 2.65 0 0 1 5.13.94c0 1.78-2.53 1.84-2.53 3.17"></path>
        <path d="M12 16.55h.01"></path>
      </svg>
      <span>Spiegami</span>
    `;
    explainButton.addEventListener("click", () => requestManualExplanation(paragraphIndex));
    box.appendChild(explainButton);

    const marginSlot = document.createElement("aside");
    marginSlot.className = "margin-slot";
    marginSlot.dataset.paragraphIndex = String(paragraphIndex);
    marginSlot.setAttribute("aria-label", `Nota del paragrafo ${paragraphIndex + 1}`);

    row.append(box, marginSlot);
    readerContent.appendChild(row);
    renderMarginSlot(paragraphIndex);
  });
}

function renderAllMarginSlots() {
  preparedParagraphs.forEach((_, paragraphIndex) => renderMarginSlot(paragraphIndex));
}

function renderMarginSlot(paragraphIndex) {
  const slot = readerContent.querySelector(`.margin-slot[data-paragraph-index="${paragraphIndex}"]`);
  if (!slot) return;

  const note = activeNotes.get(paragraphIndex);
  slot.replaceChildren();
  slot.classList.toggle("margin-slot--empty", !note);

  if (!note) {
    syncExplanationButton(paragraphIndex);
    return;
  }

  const noteElement = document.createElement("article");
  noteElement.className = "margin-note" + (note.loading ? " is-loading" : "");
  noteElement.id = `margin-note-${paragraphIndex}`;
  noteElement.setAttribute("role", "status");
  noteElement.tabIndex = -1;

  const header = document.createElement("div");
  header.className = "margin-note__header";
  const label = document.createElement("span");
  label.className = "margin-note__label";
  label.innerHTML = "<i aria-hidden=\"true\"></i>Spiegazione";
  const closeButton = document.createElement("button");
  closeButton.type = "button";
  closeButton.className = "margin-note__close";
  closeButton.textContent = "×";
  closeButton.setAttribute("aria-label", `Chiudi la spiegazione del paragrafo ${paragraphIndex + 1}`);
  closeButton.addEventListener("click", () => closeMarginNote(paragraphIndex));
  header.append(label, closeButton);

  const content = document.createElement("p");
  content.className = "margin-note__content";
  content.textContent = note.content;

  const footer = document.createElement("footer");
  footer.className = "margin-note__footer";
  footer.innerHTML = `<i aria-hidden="true"></i><span>${note.trigger}</span>`;

  noteElement.append(header, content, footer);
  slot.appendChild(noteElement);
  syncExplanationButton(paragraphIndex);
}

function syncExplanationButton(paragraphIndex) {
  const button = readerContent.querySelector(
    `.paragraph-explain-button[data-paragraph-index="${paragraphIndex}"]`
  );
  if (!button) return;

  const note = activeNotes.get(paragraphIndex);
  const label = button.querySelector("span");
  const isLoading = Boolean(note?.loading) || pendingExplanationParagraphs.has(paragraphIndex);

  button.disabled = isLoading;
  button.classList.toggle("is-active", Boolean(note));
  button.setAttribute("aria-expanded", String(Boolean(note)));

  if (label) {
    label.textContent = isLoading
      ? "Sto spiegando…"
      : note
        ? "Spiegazione aperta"
        : "Spiegami";
  }
}

function requestManualExplanation(paragraphIndex) {
  const openNote = activeNotes.get(paragraphIndex);

  if (openNote) {
    const noteElement = document.getElementById(`margin-note-${paragraphIndex}`);
    noteElement?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    noteElement?.focus({ preventScroll: true });
    return;
  }

  triggerExplanation(paragraphIndex, { source: "manual" });
}

function updateMarginNoteText(paragraphIndex, text) {
  const contentEl = readerContent.querySelector(
    `.margin-slot[data-paragraph-index="${paragraphIndex}"] .margin-note__content`
  );
  if (!contentEl) {
    renderMarginSlot(paragraphIndex);
    return;
  }
  contentEl.textContent = text;
}

function closeMarginNote(paragraphIndex) {
  activeNotes.delete(paragraphIndex);
  explainedParagraphs.delete(paragraphIndex);
  lineRegistry
    .filter((line) => line.paragraphIndex === paragraphIndex)
    .forEach((line) => line.paragraphElement.classList.remove("is-confused"));
  renderMarginSlot(paragraphIndex);
}


//Line detection
function findLineFromGaze(x, y) {
  const readerRect = readerContent.getBoundingClientRect();

  const insideReaderX =
    x >= readerRect.left - GAZE_TOLERANCE_PX &&
    x <= readerRect.right + GAZE_TOLERANCE_PX;

  const insideReaderY =
    y >= readerRect.top - GAZE_TOLERANCE_PX &&
    y <= readerRect.bottom + GAZE_TOLERANCE_PX;

  if (!insideReaderX || !insideReaderY) {
    return null;
  }

  let bestLine = null;
  let bestDistance = Infinity;

  lineRegistry.forEach((line) => {
    const rect = line.element.getBoundingClientRect();

    const verticalMatch =
      y >= rect.top - GAZE_TOLERANCE_PX &&
      y <= rect.bottom + GAZE_TOLERANCE_PX;

    if (!verticalMatch) return;

    const centerY = rect.top + rect.height / 2;
    const distance = Math.abs(y - centerY);

    if (distance < bestDistance) {
      bestDistance = distance;
      bestLine = line;
    }
  });

  return bestLine;
}

function updateCurrentLineVisuals(activeLine) {
  lineRegistry.forEach((line) => {
    line.element.classList.remove("is-current", "is-read");
    line.paragraphElement.classList.remove("is-current");
  });

  readerContent.classList.toggle("has-active-paragraph", Boolean(activeLine));

  if (activeLine) {
    activeLine.element.classList.add("is-current");
    activeLine.paragraphElement.classList.add("is-current");
    updateReadingStatus(activeLine.paragraphIndex);
  }
}

function makeFaceLogRecord(now, type = "face-frame") {
  const debug = getDebugInfo();
  const quality = faceState?.quality ?? debug.quality ?? {};
  const baseline = debug.baselines;

  return {
    t_ms: Math.round(now - sessionStartTime),
    type,
    paragraph: currentParagraphIndex,
    ready: Boolean(faceState?.ready ?? debug.ready),
    rEAR: faceState?.rEAR ?? debug.rEAR ?? null,
    lEAR: faceState?.lEAR ?? debug.lEAR ?? null,
    meanEAR: faceState?.meanEAR ?? debug.meanEAR ?? null,
    rEARDelta: faceState?.rEARDelta ?? debug.rEARDelta ?? null,
    lEARDelta: faceState?.lEARDelta ?? debug.lEARDelta ?? null,
    meanEARDelta: faceState?.meanEARDelta ?? debug.meanEARDelta ?? null,
    au4Raw: faceState?.au4Raw ?? debug.au4Raw ?? null,
    au4Delta: faceState?.au4Delta ?? debug.au4Delta ?? null,
    au7Delta: faceState?.au7Delta ?? debug.au7Delta ?? null,
    au4Supportive: Boolean(faceState?.au4Supportive),
    au4Active: Boolean(faceState?.au4Active),
    au7Active: Boolean(faceState?.au7Active),
    au7Candidate: Boolean(faceState?.au7Candidate ?? debug.au7Candidate),
    au7BlockedBy: faceState?.au7BlockedBy ?? debug.au7BlockedBy ?? [],
    au7TimerMs: faceState?.au7TimerMs ?? debug.au7TimerMs ?? 0,
    blinkState: faceState?.blinkState ?? debug.blinkState ?? "open",
    blinkCandidate: Boolean(faceState?.blinkCandidate ?? debug.blinkCandidate),
    blinkConfirmed: Boolean(faceState?.blinkConfirmed ?? debug.blinkConfirmed),
    blinkBlocksAU7: Boolean(faceState?.blinkBlocksAU7 ?? debug.blinkBlocksAU7),
    blinkCandidateAgeMs: faceState?.blinkCandidateAgeMs ?? debug.blinkCandidateAgeMs ?? 0,
    blinkSuppressionRemainingMs: debug.blinkSuppressionRemainingMs ?? 0,
    qualitySuspended: Boolean(faceState?.qualitySuspended ?? debug.qualitySuspended),
    invalidForMs: faceState?.invalidForMs ?? debug.invalidForMs ?? 0,
    quality: {
      ok: Boolean(quality.ok),
      reasons: quality.reasons ?? [],
      rollDeg: quality.rollDeg ?? null,
      eyeWidthRatio: quality.eyeWidthRatio ?? null,
      faceScaleRatio: quality.faceScaleRatio ?? null,
      jitter: quality.jitter ?? null,
    },
    baseline: {
      au4: baseline?.au4 ?? null,
      rEAR: baseline?.rEAR ?? null,
      lEAR: baseline?.lEAR ?? null,
    },
  };
}

function logFaceAnalysis(now) {
  const debug = getDebugInfo();
  const current = {
    ready: Boolean(faceState?.ready ?? debug.ready),
    qualityOk: Boolean(faceState?.quality?.ok ?? debug.quality?.ok),
    qualitySuspended: Boolean(faceState?.qualitySuspended ?? debug.qualitySuspended),
    blinkCandidate: Boolean(faceState?.blinkCandidate ?? debug.blinkCandidate),
    blinkConfirmed: Boolean(faceState?.blinkConfirmed ?? debug.blinkConfirmed),
    blinkBlocksAU7: Boolean(faceState?.blinkBlocksAU7 ?? debug.blinkBlocksAU7),
    au4Supportive: Boolean(faceState?.au4Supportive),
    au4Active: Boolean(faceState?.au4Active),
    au7Candidate: Boolean(faceState?.au7Candidate ?? debug.au7Candidate),
    au7Active: Boolean(faceState?.au7Active),
    confusedLikely: Boolean(faceState?.confusedLikely),
    confusedStrong: Boolean(faceState?.confusedStrong),
  };

  if (now - lastFaceLogAt >= 100) {
    faceLog.push(makeFaceLogRecord(now));
    lastFaceLogAt = now;
  }

  if (!previousFaceLogState && current.ready) {
    faceLog.push(makeFaceLogRecord(now, "baseline-ready"));
  }

  if (previousFaceLogState) {
    if (!previousFaceLogState.ready && current.ready) {
      faceLog.push(makeFaceLogRecord(now, "baseline-ready"));
    }
    if (previousFaceLogState.qualityOk !== current.qualityOk) {
      faceLog.push(makeFaceLogRecord(now, current.qualityOk ? "quality-valid" : "quality-invalid"));
    }
    if (previousFaceLogState.qualitySuspended !== current.qualitySuspended) {
      faceLog.push(makeFaceLogRecord(now, current.qualitySuspended ? "quality-suspended" : "quality-resumed"));
    }
    if (previousFaceLogState.blinkBlocksAU7 !== current.blinkBlocksAU7) {
      faceLog.push(makeFaceLogRecord(now, `blink-blocks-au7-${current.blinkBlocksAU7 ? "on" : "off"}`));
    }
    if (!previousFaceLogState.blinkCandidate && current.blinkCandidate) {
      faceLog.push(makeFaceLogRecord(now, "blink-candidate-start"));
    }
    if (!previousFaceLogState.blinkConfirmed && current.blinkConfirmed) {
      faceLog.push(makeFaceLogRecord(now, "blink-start"));
    }
    if (previousFaceLogState.blinkCandidate && !current.blinkCandidate && !current.blinkConfirmed) {
      faceLog.push(makeFaceLogRecord(now, "blink-candidate-cancelled"));
    }
    if (previousFaceLogState.blinkConfirmed && !current.blinkConfirmed) {
      faceLog.push(makeFaceLogRecord(now, "blink-end"));
    }

    [
      ["au4Supportive", "au4-supportive"],
      ["au4Active", "au4-active"],
      ["au7Candidate", "au7-candidate"],
      ["au7Active", "au7-active"],
      ["confusedLikely", "confused-likely"],
      ["confusedStrong", "confused-strong"],
    ].forEach(([key, event]) => {
      if (previousFaceLogState[key] !== current[key]) {
        faceLog.push(makeFaceLogRecord(now, `${event}-${current[key] ? "on" : "off"}`));
      }
    });
  }

  previousFaceLogState = current;
}

function logParagraphEnter(paragraphId, now) {
  currentParagraphEnteredAt = now;
  sessionLog.push({
    type: "paragraph-enter",
    paragraphId,
    t_ms: Math.round(now - sessionStartTime),
  });
}

function logParagraphLeave(paragraphId, now) {
  if (!Number.isInteger(paragraphId)) return;
  sessionLog.push({
    type: "paragraph-leave",
    paragraphId,
    durationMs: Math.max(0, Math.round(now - (currentParagraphEnteredAt ?? now))),
    t_ms: Math.round(now - sessionStartTime),
  });
  currentParagraphEnteredAt = null;
}

//Gaze listener
function handleGaze(data) {
  if (!isTracking || !data) return;

  const now = performance.now();

  if (lastTick === null) {
    lastTick = now;
    return;
  }

  const delta = Math.min(now - lastTick, MAX_DELTA_MS);
  lastTick = now;

  smoothX = smoothX === null ? data.x : SMOOTH_ALPHA * data.x + (1 - SMOOTH_ALPHA) * smoothX;
  smoothY = smoothY === null ? data.y : SMOOTH_ALPHA * data.y + (1 - SMOOTH_ALPHA) * smoothY;

  const x = smoothX;
  const y = smoothY;

  updateGazeDot(x, y);
 
  faceState = updateFaceAnalysis(wg);
  logFaceAnalysis(now);
  syncFaceCalibrationUi();
  updateAUIndicators();

  const activeLine = findLineFromGaze(x, y);

  if (!activeLine) {
    logParagraphLeave(currentParagraphIndex, now);
    currentLineKey = null;
    currentParagraphIndex = null;
    currentParagraphDwellMs = 0;
    confusionHoldMs = 0;

    gazeLog.push({
      t_ms: Math.round(performance.now() - sessionStartTime),
      x: data.x,
      y: data.y,
      paragraph: currentParagraphIndex,
    });

    updateCurrentLineVisuals(null);
    return;
  }

  if (currentParagraphIndex !== activeLine.paragraphIndex) {
    logParagraphLeave(currentParagraphIndex, now);
    currentParagraphIndex = activeLine.paragraphIndex;
    logParagraphEnter(currentParagraphIndex, now);
    currentParagraphDwellMs = 0;
    confusionHoldMs = 0;
  }

  activeLine.dwellMs += delta;
  activeLine.samples += 1;
  currentParagraphDwellMs += delta;
  currentLineKey = activeLine.key;

  gazeLog.push({
    t_ms: Math.round(performance.now() - sessionStartTime),
    x: data.x,
    y: data.y,
    paragraph: currentParagraphIndex,
  });

  updateCurrentLineVisuals(activeLine);

  if (faceState?.confused) {
    confusionHoldMs += delta;
  } else {
    confusionHoldMs = 0;
  }

  if (pendingExplanationParagraphs.size === 0 && faceState?.confused) {
    const pIdx = activeLine.paragraphIndex;
    const alreadyExplained = explainedParagraphs.has(pIdx);
    const globalCooldownOk = now - lastExplanationAt > CONFUSION_GLOBAL_COOLDOWN_MS;
    const dwellThreshold = faceState.confusedStrong
      ? CONFUSION_DWELL_MS
      : CONFUSION_SOFT_DWELL_MS;
    const holdThreshold = faceState.confusedStrong
      ? CONFUSION_HOLD_STRONG_MS
      : CONFUSION_HOLD_SOFT_MS;

    if (
      !alreadyExplained &&
      globalCooldownOk &&
      currentParagraphDwellMs > dwellThreshold &&
      confusionHoldMs > holdThreshold
    ) {
      triggerExplanation(pIdx, { source: "confusion" });
    }
  }
}

function updateGazeDot(x, y) {
  gazeDot.style.left = `${x}px`;
  gazeDot.style.top = `${y}px`;
}

function safeWebGazerCall(fn) {
  try {
    fn();
  } catch (e) {
    console.warn("WebGazer call failed:", e);
  }
}

async function startTracking() {
  try {
    if (!window.isSecureContext) {
      throw new Error(
        "Contesto non sicuro: usa http://localhost:5173 oppure HTTPS."
      );
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("getUserMedia non disponibile nel browser.");
    }

    const testStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });

    testStream.getTracks().forEach((track) => track.stop());

    if (!webgazerStarted) {
      wg.setRegression("ridge");
      try {
        wg.setTracker("TFFacemesh");
      } catch (trackerError) {
        console.warn(
          "TFFacemesh non disponibile, uso il tracker di default.",
          trackerError
        );
      }

      try {
        wg.applyKalmanFilter(true);
      } catch (filterError) {
        console.warn("Kalman filter non applicato.", filterError);
      }

      wg.setGazeListener(handleGaze);

      await wg.begin();

      safeWebGazerCall(() => wg.removeMouseEventListeners());

      try {
        wg.showVideoPreview(false);
        wg.showPredictionPoints(false);
      } catch (viewError) {
        console.warn("Preview/prediction points non modificati.", viewError);
      }

      webgazerStarted = true;
      setStatus(webcamStatus, "attiva", true);
    } else {
      try {
        wg.resume();
      } catch (resumeError) {
        console.warn("Resume fallito, continuo comunque.", resumeError);
      }

      wg.setGazeListener(handleGaze);
    }

    isTracking = true;
    lastTick = null;

    setStatus(trackingStatus, "attivo", true);
    return true;
  } catch (error) {
    console.error("ERRORE REALE WEBGAZER:", error);

    setStatus(webcamStatus, "errore", false);
    setStatus(trackingStatus, "fermo", false);

    alert(error.message || "Errore sconosciuto nell'avvio di WebGazer.");
    return false;
  }
}

function stopTracking() {
  isTracking = false;
  lastTick = null;
  smoothX = null;
  smoothY = null;

  if (webgazerStarted) {
    safeWebGazerCall(() => wg.pause());
  }

  gazeDot.classList.remove("is-visible");

  setStatus(trackingStatus, "fermo", false);
}


//Calibration
async function openCalibration() {
  if (!webgazerStarted) {
    const started = await startTracking();
    if (!started) {
      return;
    }
  } else if (!isTracking) {
    const resumed = await startTracking();
    if (!resumed) {
      return;
    }
  }

  clearFaceCalibrationTimeout();
  resetFaceAnalysis();
  faceState = null;
  updateAUIndicators();
  calibrationPhase = "gaze";

  calibrationLayer.classList.add("is-visible");
  calibrationLayer.setAttribute("aria-hidden", "false");
  setGazeCalibrationCopy();
  renderCalibrationPoints();
}

function closeCalibration() {
  clearFaceCalibrationTimeout();
  if (calibrationPhase === "face") {
    stopCalibrationSampling();
  }
  calibrationPhase = "idle";
  calibrationLayer.classList.remove("is-visible");
  calibrationLayer.setAttribute("aria-hidden", "true");
  calibrationPoints.innerHTML = "";
}

async function enterWorkspaceAndStart() {
  setScreen("workspace");
  setWorkspaceMode("prepare");
  textInput.focus({ preventScroll: false });
  await openCalibration();
}

function goBackHome() {
  closeCalibration();
  stopTracking();
  setDebugOpen(false);
  setScreen("landing");
}

function renderCalibrationPoints() {
  calibrationPoints.innerHTML = "";

  const points = [
    { x: 15, y: 18 },
    { x: 50, y: 18 },
    { x: 85, y: 18 },

    { x: 15, y: 50 },
    { x: 50, y: 50 },
    { x: 85, y: 50 },

    { x: 15, y: 82 },
    { x: 50, y: 82 },
    { x: 85, y: 82 },
  ];

  points.forEach((point, index) => {
    const pointButton = document.createElement("button");

    pointButton.type = "button";
    pointButton.className = "calibration-point";
    pointButton.style.left = `${point.x}%`;
    pointButton.style.top = `${point.y}%`;
    pointButton.dataset.clicks = "0";
    pointButton.textContent = "0/5";
    pointButton.setAttribute("aria-label", `Calibration point ${index + 1}`);

    pointButton.addEventListener("click", (event) => {
      handleCalibrationClick(event, pointButton);
    });

    calibrationPoints.appendChild(pointButton);
  });
}

function handleCalibrationClick(event, pointButton) {
  const rect = pointButton.getBoundingClientRect();

  const x = rect.left + rect.width / 2;
  const y = rect.top + rect.height / 2;

  safeWebGazerCall(() => {
    if (typeof wg.recordScreenPosition === "function") {
      wg.recordScreenPosition(x, y, "click");
    }
  });

  const clicks = Number(pointButton.dataset.clicks) + 1;

  pointButton.dataset.clicks = String(clicks);
  pointButton.textContent = `${Math.min(clicks, 5)}/5`;

  if (clicks >= 5) {
    pointButton.classList.add("is-done");
    pointButton.disabled = true;
  }

  const allDone = [...calibrationPoints.querySelectorAll(".calibration-point")]
    .every((point) => point.classList.contains("is-done"));

  if (allDone) {
    setTimeout(() => {
      if (calibrationPhase === "gaze") startFaceCalibration();
    }, 500);
  }
}

function clearFaceCalibrationTimeout() {
  if (faceCalibrationTimeout) {
    clearTimeout(faceCalibrationTimeout);
    faceCalibrationTimeout = 0;
  }
}

function setGazeCalibrationCopy() {
  calibrationTitle.textContent = "Allinea lo sguardo";
  calibrationDescription.textContent = "Guarda ogni punto e cliccalo 5 volte. Dopo passerai alla breve calibrazione del viso.";
  calibrationStatus.textContent = "";
  closeCalibrationBtn.textContent = "Chiudi";
  retryFaceCalibrationBtn.hidden = true;
}

function startFaceCalibration() {
  clearFaceCalibrationTimeout();
  calibrationPhase = "face";
  calibrationPoints.innerHTML = "";
  calibrationTitle.textContent = "Calibrazione del viso";
  calibrationDescription.textContent = "Guarda il punto centrale, mantieni il viso frontale e rilassato, con gli occhi aperti normalmente.";
  calibrationStatus.textContent = "Attendi un istante: preparo la raccolta dei campioni…";
  closeCalibrationBtn.textContent = "Annulla";
  retryFaceCalibrationBtn.hidden = true;

  startCalibrationSampling();
  faceCalibrationTimeout = window.setTimeout(() => {
    if (calibrationPhase !== "face") return;
    stopCalibrationSampling();
    syncFaceCalibrationUi();
  }, 6100);
}

function syncFaceCalibrationUi() {
  if (calibrationPhase !== "face") return;

  const dbg = getDebugInfo();
  const calibration = dbg.calibration;
  if (dbg.calibrationState === "ready") {
    clearFaceCalibrationTimeout();
    calibrationPhase = "complete";
    calibrationStatus.textContent = `Calibrazione completata (${calibration.accepted} campioni validi).`;
    closeCalibrationBtn.textContent = "Continua";
    window.setTimeout(() => {
      if (calibrationPhase === "complete") closeCalibration();
    }, 450);
    return;
  }

  if (dbg.calibrationState === "failed") {
    clearFaceCalibrationTimeout();
    calibrationPhase = "failed";
    calibrationStatus.textContent = "Campioni insufficienti o instabili. Mantieni il viso frontale, gli occhi aperti e riprova.";
    closeCalibrationBtn.textContent = "Chiudi";
    retryFaceCalibrationBtn.hidden = false;
    return;
  }

  if (dbg.calibrationState === "settling") {
    calibrationStatus.textContent = "Mantieni il viso rilassato: la raccolta inizia tra un istante…";
    return;
  }

  const discarded = calibration.discarded ? `, ${calibration.discarded} scartati` : "";
  calibrationStatus.textContent = `Campioni validi: ${calibration.accepted}/${calibration.targetSamples}${discarded}.`;
}


//Confusion explanation
async function triggerExplanation(paragraphIndex, { source = "confusion" } = {}) {
  if (pendingExplanationParagraphs.has(paragraphIndex)) return;

  const triggerTime = performance.now();
  const requestId = ++noteRequestSequence;
  const wasManuallyRequested = source === "manual";
  const triggerLabel = wasManuallyRequested
    ? "Spiegazione richiesta da te"
    : "Rilevata esitazione durante la lettura";

  pendingExplanationParagraphs.add(paragraphIndex);
  if (!wasManuallyRequested) {
    lastExplanationAt = triggerTime;
  }
  explainedParagraphs.set(paragraphIndex, triggerTime);

  sessionLog.push({
    t_ms: Math.round(triggerTime - sessionStartTime),
    paragraph: paragraphIndex,
    type: wasManuallyRequested
      ? "manual-explanation"
      : faceState?.confusedStrong
        ? "strong"
        : "likely",
    source,
    au4_delta: wasManuallyRequested ? undefined : faceState?.au4Delta?.toFixed(3),
    au7_delta: wasManuallyRequested ? undefined : faceState?.au7Delta?.toFixed(3),
  });

  const paragraphText = lineRegistry
    .filter((l) => l.paragraphIndex === paragraphIndex)
    .map((l) => l.text)
    .join(" ");

  const box = lineRegistry.find((l) => l.paragraphIndex === paragraphIndex)?.paragraphElement;
  if (!wasManuallyRequested) {
    box?.classList.add("is-confused");
  }

  activeNotes.set(paragraphIndex, {
    content: isLocalLlmEnabled()
      ? "Analisi in corso… Può richiedere un pò di tempo."
      : "Analisi in corso…",
    loading: true,
    trigger: triggerLabel,
    requestId,
  });
  renderMarginSlot(paragraphIndex);

  try {
    const explanation = await explainParagraph(paragraphText, (partialText) => {
      const note = activeNotes.get(paragraphIndex);
      if (note?.requestId !== requestId) return;
      activeNotes.set(paragraphIndex, { ...note, content: partialText });
      updateMarginNoteText(paragraphIndex, partialText);
    });
    if (activeNotes.get(paragraphIndex)?.requestId === requestId) {
      activeNotes.set(paragraphIndex, {
        content: explanation,
        loading: false,
        trigger: triggerLabel,
        requestId,
      });
    }
  } catch (err) {
    if (activeNotes.get(paragraphIndex)?.requestId === requestId) {
      activeNotes.set(paragraphIndex, {
        content: `⚠ ${err.message}`,
        loading: false,
        trigger: "La spiegazione non è disponibile",
        requestId,
      });
    }
  } finally {
    pendingExplanationParagraphs.delete(paragraphIndex);
    if (activeNotes.get(paragraphIndex)?.requestId === requestId) {
      renderMarginSlot(paragraphIndex);
    } else {
      syncExplanationButton(paragraphIndex);
    }
  }
}


//Events
readerFontSizeSlider.addEventListener("input", () => {
  setReaderFontSize(Number(readerFontSizeSlider.value));
  schedulePreparedTextRerender();
});

readerFontDecreaseBtn.addEventListener("click", () => {
  setReaderFontSize(getReaderFontSizePx() - FONT_SIZE_STEP);
  schedulePreparedTextRerender();
});

readerFontIncreaseBtn.addEventListener("click", () => {
  setReaderFontSize(getReaderFontSizePx() + FONT_SIZE_STEP);
  schedulePreparedTextRerender();
});

focusModeToggleBtn.addEventListener("click", () => {
  setFocusMode(!focusModeEnabled);
});

themeOptionButtons.forEach((button) => {
  button.addEventListener("click", () => setTheme(button.dataset.themeOption));
});

readerEditTextBtn.addEventListener("click", () => {
  setWorkspaceMode("prepare");
  textInput.focus({ preventScroll: true });
});

readerScrollArea.addEventListener("scroll", updateReadingProgress, { passive: true });
window.addEventListener("resize", schedulePreparedTextRerender);

exportLogBtn.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify({
    sessionLog,
    gazeLog,
    faceLog,
  }, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `lume_log_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

tryNowBtn.addEventListener("click", () => {
  enterWorkspaceAndStart().catch((error) => {
    console.error("Workspace start failed:", error);
  });
});

backHomeBtn.addEventListener("click", goBackHome);

debugToggleBtn.addEventListener("click", () => {
  setDebugOpen(debugPanel.hidden);
});

renderTextBtn.addEventListener("click", renderText);

clearTextBtn.addEventListener("click", () => {
  textInput.value = "";
  resetReaderPlaceholder();
});

startTrackingBtn.addEventListener("click", startTracking);
stopTrackingBtn.addEventListener("click", stopTracking);

calibrateBtn.addEventListener("click", openCalibration);
closeCalibrationBtn.addEventListener("click", closeCalibration);
retryFaceCalibrationBtn.addEventListener("click", startFaceCalibration);

resetDataBtn.addEventListener("click", () => {
  resetMetricsOnly({ clearNotes: true, resetFace: true, resetLogs: true });
});

window.addEventListener("beforeunload", () => {
  safeWebGazerCall(() => wg.end());
});


//Initial UI state
setScreen("landing");
setDebugOpen(false);
setTheme(readStoredPreference("lume.theme", "cream"));
setReaderFontSize(clampFontSize(readStoredPreference("lume.fontSize", DEFAULT_FONT_SIZE)));
setFocusMode(readStoredPreference("lume.focusMode", "true") !== "false");
setStatus(webcamStatus, "spenta", false);
setStatus(trackingStatus, "fermo", false);

if (isLocalLlmEnabled()) {
  window.setTimeout(() => {
    prewarmLocalLlm().catch((error) => {
      console.warn("Local LLM warm-up failed:", error);
    });
  }, 1200);
}
