/*
  Geometric facial-signal proxies inspired by AU4 (brow lowering) and AU7
  (lid tightening). This module is intentionally local-only: it operates on
  WebGazer/TFFaceMesh landmarks and does not classify FACS Action Units.
 */

const LM = {
  BROW_R_INNER: 107,
  BROW_L_INNER: 336,

  R_OUTER: 33, R_UP1: 160, R_UP2: 158,
  R_INNER: 133, R_LO1: 153, R_LO2: 144,
  L_OUTER: 263, L_UP1: 387, L_UP2: 385,
  L_INNER: 362, L_LO1: 380, L_LO2: 373,

  NOSE_TIP: 1,
};

const REQUIRED_LANDMARKS = [...new Set(Object.values(LM))];

const CALIBRATION_SETTLE_MS = 550;
const CALIBRATION_SAMPLE_INTERVAL_MS = 100;
const CALIBRATION_TARGET_SAMPLES = 35;
const CALIBRATION_MIN_SAMPLES = 25;
const CALIBRATION_MAX_MS = 6000;
const TRIM_FRACTION = 0.15;
const ROBUST_Z_LIMIT = 3.5;

const MAX_TIME_STEP_MS = 100;
const AU4_SUPPORT_MS = 350;
const AU4_ACTIVE_MS = 450;

const AU7_ACTIVE_MS = 300;
const DELTA_SMOOTH_TAU_MS = 140;

const ENABLE_NEUTRAL_DRIFT = false;
const DRIFT_TAU_MS = 120000;
const NEUTRAL_HOLD_MS = 1500;

const AU4_SUPPORT_DELTA = 0.009;
const AU4_ACTIVE_DELTA = 0.016;
const AU4_MAX_VALID_DELTA = 0.100;
const AU7_DELTA = 0.014;
const AU7_EYE_MIN_FRACTION = 0.45;
const AU7_MAX_ASYMMETRY = 0.035;

const BLINK_CLOSING_RATIO = 0.72;
const BLINK_RECOVERY_RATIO = 0.84;
const BLINK_MIN_CLOSE_MS = 80;
const BLINK_MAX_CLOSE_MS = 350;
const BLINK_SUSTAINED_MS = 400;
const BLINK_SUPPRESSION_MS = 280;
const EYES_TOO_CLOSED_RATIO = 0.38;
const INVALID_GRACE_MS = 120;

const MIN_FACE_SCALE_PX = 40;
const MAX_FACE_SCALE_PX = 600;
const MIN_FACE_SCALE_RATIO = 0.72;
const MAX_FACE_SCALE_RATIO = 1.35;
const MIN_EYE_WIDTH_RATIO = 0.68;
const MAX_EYE_WIDTH_RATIO = 1.45;
const MAX_EYE_WIDTH_RATIO_CHANGE = 0.25;
const MAX_ROLL_DEG = 18;
const MAX_PITCH_PROXY_DELTA = 0.11;
const MAX_JITTER_RATIO = 0.035;
const MIN_OPEN_EAR = 0.16;
const CALIBRATION_MIN_EAR = 0.19;

const ROBUST_FLOORS = {
  au4: 0.003,
  rEAR: 0.006,
  lEAR: 0.006,
  faceScale: 2,
  eyeWidthR: 1,
  eyeWidthL: 1,
  eyeWidthRatio: 0.015,
  rollDeg: 1,
  pitchProxy: 0.01,
  jitter: 0.003,
};

let baselineAU4 = null;
let baselineREAR = null;
let baselineLEAR = null;
let baselineEAR = null;
let qualityBaseline = null;
let robustBaseline = null;

let smoothAU4Delta = null;
let smoothAU7Delta = null;
let smoothREARDelta = null;
let smoothLEARDelta = null;

let au4SupportForMs = 0;
let au4ActiveForMs = 0;
let au7ActiveForMs = 0;
let neutralForMs = 0;
let lastUpdateAt = null;
let lastLandmarkSnapshot = null;
let invalidSince = null;
let blinkState = createBlinkState();
let lastReliableResult = null;

let calibration = createCalibrationState();

let _debug = createDebug();

export const getDebugInfo = () => _debug;

function nowMs() {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function isFinitePoint(point) {
  return Array.isArray(point)
    && Number.isFinite(point[0])
    && Number.isFinite(point[1])
    && (point.length < 3 || Number.isFinite(point[2]));
}

function hasRequiredLandmarks(pts) {
  return Array.isArray(pts)
    && REQUIRED_LANDMARKS.every((index) => isFinitePoint(pts[index]));
}

export function dist2d(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.hypot(dx, dy);
}

function normalize2d(vector) {
  const length = Math.hypot(vector.x, vector.y);
  if (!Number.isFinite(length) || length <= 0) return null;
  return { x: vector.x / length, y: vector.y / length };
}

function dot2d(a, b) {
  return a.x * b.x + a.y * b.y;
}

function clampTimeDelta(now) {
  const dt = lastUpdateAt === null ? 0 : now - lastUpdateAt;
  lastUpdateAt = now;
  return Math.min(Math.max(Number.isFinite(dt) ? dt : 0, 0), MAX_TIME_STEP_MS);
}

function timeAlpha(dtMs, tauMs) {
  return 1 - Math.exp(-dtMs / tauMs);
}

function timeEma(previous, next, dtMs, tauMs) {
  if (previous === null || !Number.isFinite(previous)) return next;
  const alpha = timeAlpha(dtMs, tauMs);
  return alpha * next + (1 - alpha) * previous;
}

export function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function mad(values, center = median(values)) {
  if (center === null) return null;
  return median(values.map((value) => Math.abs(value - center)));
}

export function trimmedMean(values, fraction = TRIM_FRACTION) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const trim = Math.floor(sorted.length * fraction);
  const kept = sorted.slice(trim, sorted.length - trim);
  const source = kept.length ? kept : sorted;
  return source.reduce((sum, value) => sum + value, 0) / source.length;
}

function rounded(value, digits = 4) {
  return Number.isFinite(value) ? Number(value.toFixed(digits)) : null;
}

export function computeAU4Raw(pts) {
  const sides = [
    { brow: pts[LM.BROW_R_INNER], outer: pts[LM.R_OUTER], inner: pts[LM.R_INNER] },
    { brow: pts[LM.BROW_L_INNER], outer: pts[LM.L_OUTER], inner: pts[LM.L_INNER] },
  ];

  const values = sides.map(({ brow, outer, inner }) => {
    const eyeAxis = normalize2d({ x: inner[0] - outer[0], y: inner[1] - outer[1] });
    const eyeWidth = dist2d(outer, inner);
    if (!eyeAxis || eyeWidth <= 0) return null;

    let downAxis = { x: -eyeAxis.y, y: eyeAxis.x };
    if (downAxis.y < 0) downAxis = { x: -downAxis.x, y: -downAxis.y };

    const browToEye = { x: inner[0] - brow[0], y: inner[1] - brow[1] };
    return dot2d(browToEye, downAxis) / eyeWidth;
  });

  return values.every(Number.isFinite) ? (values[0] + values[1]) / 2 : null;
}

export function computeEAR(pts) {
  const rWidth = dist2d(pts[LM.R_OUTER], pts[LM.R_INNER]);
  const lWidth = dist2d(pts[LM.L_OUTER], pts[LM.L_INNER]);
  if (rWidth <= 0 || lWidth <= 0) return null;

  const rEAR = (
    dist2d(pts[LM.R_UP1], pts[LM.R_LO2])
    + dist2d(pts[LM.R_UP2], pts[LM.R_LO1])
  ) / (2 * rWidth);
  const lEAR = (
    dist2d(pts[LM.L_UP1], pts[LM.L_LO2])
    + dist2d(pts[LM.L_UP2], pts[LM.L_LO1])
  ) / (2 * lWidth);

  return Number.isFinite(rEAR) && Number.isFinite(lEAR)
    ? { rEAR, lEAR, meanEAR: (rEAR + lEAR) / 2 }
    : null;
}

function meanPoint(...points) {
  return [
    points.reduce((sum, point) => sum + point[0], 0) / points.length,
    points.reduce((sum, point) => sum + point[1], 0) / points.length,
  ];
}

export function computeMetrics(pts) {
  const ear = computeEAR(pts);
  const au4 = computeAU4Raw(pts);
  if (!ear || !Number.isFinite(au4)) return null;

  const eyeWidthR = dist2d(pts[LM.R_OUTER], pts[LM.R_INNER]);
  const eyeWidthL = dist2d(pts[LM.L_OUTER], pts[LM.L_INNER]);
  const eyeMidR = meanPoint(pts[LM.R_OUTER], pts[LM.R_INNER]);
  const eyeMidL = meanPoint(pts[LM.L_OUTER], pts[LM.L_INNER]);
  const eyeMidpoint = meanPoint(eyeMidR, eyeMidL);
  const interocularAxis = normalize2d({ x: eyeMidL[0] - eyeMidR[0], y: eyeMidL[1] - eyeMidR[1] });
  const faceScale = dist2d(eyeMidR, eyeMidL);

  if (!interocularAxis || faceScale <= 0 || eyeWidthR <= 0 || eyeWidthL <= 0) return null;

  let downAxis = { x: -interocularAxis.y, y: interocularAxis.x };
  if (downAxis.y < 0) downAxis = { x: -downAxis.x, y: -downAxis.y };
  const noseVector = {
    x: pts[LM.NOSE_TIP][0] - eyeMidpoint[0],
    y: pts[LM.NOSE_TIP][1] - eyeMidpoint[1],
  };

  const rollDeg = Math.atan2(interocularAxis.y, interocularAxis.x) * 180 / Math.PI;
  const metrics = {
    au4,
    ...ear,
    faceScale,
    eyeWidthR,
    eyeWidthL,
    eyeWidthRatio: eyeWidthR / eyeWidthL,
    rollDeg,
    pitchProxy: dot2d(noseVector, downAxis) / faceScale,
  };

  return Object.values(metrics).every(Number.isFinite) ? metrics : null;
}

function computeJitter(pts, faceScale) {
  const snapshot = REQUIRED_LANDMARKS.map((index) => [pts[index][0], pts[index][1]]);
  let jitter = 0;

  if (lastLandmarkSnapshot && lastLandmarkSnapshot.length === snapshot.length) {
    const movement = snapshot.reduce((sum, point, index) => {
      const previous = lastLandmarkSnapshot[index];
      return sum + Math.hypot(point[0] - previous[0], point[1] - previous[1]);
    }, 0) / snapshot.length;
    jitter = movement / faceScale;
  }

  lastLandmarkSnapshot = snapshot;
  return jitter;
}

function calibrationReference() {
  if (qualityBaseline) return qualityBaseline;
  if (calibration.samples.length < 5) return null;

  const fields = ["faceScale", "eyeWidthRatio", "pitchProxy", "rEAR", "lEAR"];
  return fields.reduce((reference, field) => {
    reference[field] = median(calibration.samples.map((sample) => sample[field]));
    return reference;
  }, {});
}

export function evaluateFaceQuality(pts, metrics, now = nowMs()) {
  const reasons = [];
  if (!hasRequiredLandmarks(pts) || !metrics) {
    return {
      ok: false,
      reasons: ["invalid-landmarks"],
      faceScaleRatio: null,
      eyeWidthRatio: null,
      rollDeg: null,
      pitchProxyDelta: null,
      jitter: null,
      blinkLike: false,
    };
  }

  const reference = calibrationReference();
  const jitter = computeJitter(pts, metrics.faceScale);
  const faceScaleRatio = reference?.faceScale ? metrics.faceScale / reference.faceScale : 1;
  const pitchProxyDelta = reference?.pitchProxy === undefined
    ? 0
    : Math.abs(metrics.pitchProxy - reference.pitchProxy);
  const eyeWidthRatioChange = reference?.eyeWidthRatio
    ? Math.abs(metrics.eyeWidthRatio - reference.eyeWidthRatio)
    : 0;
  const rEARRatio = reference?.rEAR ? metrics.rEAR / reference.rEAR : null;
  const lEARRatio = reference?.lEAR ? metrics.lEAR / reference.lEAR : null;
  const blinkCandidate = rEARRatio !== null
    && lEARRatio !== null
    && rEARRatio < BLINK_CLOSING_RATIO
    && lEARRatio < BLINK_CLOSING_RATIO;
  const eyesTooClosed = rEARRatio !== null && lEARRatio !== null
    ? rEARRatio < EYES_TOO_CLOSED_RATIO && lEARRatio < EYES_TOO_CLOSED_RATIO
    : metrics.rEAR < MIN_OPEN_EAR && metrics.lEAR < MIN_OPEN_EAR;

  if (metrics.faceScale < MIN_FACE_SCALE_PX || metrics.faceScale > MAX_FACE_SCALE_PX) {
    reasons.push("face-scale-absolute");
  }
  if (reference?.faceScale && (faceScaleRatio < MIN_FACE_SCALE_RATIO || faceScaleRatio > MAX_FACE_SCALE_RATIO)) {
    reasons.push("face-scale");
  }
  if (metrics.eyeWidthRatio < MIN_EYE_WIDTH_RATIO || metrics.eyeWidthRatio > MAX_EYE_WIDTH_RATIO) {
    reasons.push("eye-width-asymmetry");
  }
  if (reference?.eyeWidthRatio && eyeWidthRatioChange > MAX_EYE_WIDTH_RATIO_CHANGE) {
    reasons.push("yaw-proxy");
  }
  if (Math.abs(metrics.rollDeg) > MAX_ROLL_DEG) reasons.push("head-roll");
  if (reference?.pitchProxy !== undefined && pitchProxyDelta > MAX_PITCH_PROXY_DELTA) {
    reasons.push("pitch-proxy");
  }
  if (jitter > MAX_JITTER_RATIO) reasons.push("landmark-jitter");
  if (eyesTooClosed) reasons.push("eyes-too-closed");

  return {
    ok: reasons.length === 0,
    reasons,
    faceScaleRatio,
    eyeWidthRatio: metrics.eyeWidthRatio,
    rollDeg: metrics.rollDeg,
    pitchProxyDelta,
    jitter,
    rEARRatio,
    lEARRatio,
    blinkLike: blinkCandidate,
    blinkCandidate,
    eyesTooClosed,
    evaluatedAt: now,
  };
}

//Calibration
function createCalibrationState() {
  return {
    state: "idle",
    startedAt: null,
    settledAt: null,
    lastSampleAttemptAt: null,
    samples: [],
    attempted: 0,
    discarded: 0,
    lastDiscardReason: null,
    failureReason: null,
  };
}

function createBlinkState() {
  return {
    phase: "open", // open / closing / confirmed / recovering
    startedAt: 0,
    minRatio: 1,
    suppressionUntil: 0,
  };
}

function createDebug() {
  return {
    warmedUp: false, 
    ready: false,
    calibrationState: "idle",
    calibration: {
      attempted: 0,
      accepted: 0,
      discarded: 0,
      lastDiscardReason: null,
      failureReason: null,
    },
    baselines: null,
    robustBaseline: null,
    au4Delta: null,
    au7Delta: null,
    rawDeltas: null,
    au4Raw: null,
    quality: null,
    blink: {
      state: "open",
      candidate: false,
      confirmed: false,
      suppressionRemainingMs: 0,
    },
    au7Candidate: false,
    au7BlockedBy: ["not-ready"],
    rDrop: null,
    lDrop: null,
    meanDrop: null,
    bilateralOk: false,
    asymmetry: null,
    invalidForMs: 0,
    timers: { au4SupportMs: 0, au4ActiveMs: 0, au7ActiveMs: 0, neutralMs: 0 },
    baselineAdapted: false,
    baselineAdaptable: false,
    baselineAdaptBlockedBy: "not-ready",
    neutralForMs: 0,
  };
}

function getRobustStats(samples, field) {
  const values = samples.map((sample) => sample[field]);
  const center = median(values);
  return { median: center, mad: mad(values, center) };
}

function isRobustInlier(sample, stats) {
  return Object.entries(stats).every(([field, { median: center, mad: spread }]) => {
    const floor = ROBUST_FLOORS[field] ?? 0.001;
    const tolerance = ROBUST_Z_LIMIT * Math.max(spread ?? 0, floor);
    return Math.abs(sample[field] - center) <= tolerance;
  });
}

function robustlyFinalizeCalibration() {
  const samples = calibration.samples;
  if (samples.length < CALIBRATION_MIN_SAMPLES) {
    calibration.state = "failed";
    calibration.failureReason = "not-enough-valid-samples";
    refreshDebug();
    return false;
  }

  const fields = [
    "au4", "rEAR", "lEAR", "faceScale", "eyeWidthR", "eyeWidthL",
    "eyeWidthRatio", "rollDeg", "pitchProxy", "jitter",
  ];
  const stats = fields.reduce((result, field) => {
    result[field] = getRobustStats(samples, field);
    return result;
  }, {});
  const inliers = samples.filter((sample) => isRobustInlier(sample, stats));

  if (inliers.length < CALIBRATION_MIN_SAMPLES) {
    calibration.state = "failed";
    calibration.failureReason = "not-enough-robust-samples";
    refreshDebug({ robustStats: stats, robustSamples: inliers.length });
    return false;
  }

  const baseline = fields.reduce((result, field) => {
    result[field] = trimmedMean(inliers.map((sample) => sample[field]));
    return result;
  }, {});

  baselineAU4 = baseline.au4;
  baselineREAR = baseline.rEAR;
  baselineLEAR = baseline.lEAR;
  baselineEAR = (baselineREAR + baselineLEAR) / 2;
  qualityBaseline = {
    faceScale: baseline.faceScale,
    eyeWidthR: baseline.eyeWidthR,
    eyeWidthL: baseline.eyeWidthL,
    eyeWidthRatio: baseline.eyeWidthRatio,
    rollDeg: baseline.rollDeg,
    pitchProxy: baseline.pitchProxy,
    jitter: baseline.jitter,
    rEAR: baselineREAR,
    lEAR: baselineLEAR,
  };
  robustBaseline = {
    medians: Object.fromEntries(fields.map((field) => [field, stats[field].median])),
    mads: Object.fromEntries(fields.map((field) => [field, stats[field].mad])),
    trimmedMeans: baseline,
    inputSamples: samples.length,
    inlierSamples: inliers.length,
  };

  smoothAU4Delta = 0;
  smoothAU7Delta = 0;
  smoothREARDelta = 0;
  smoothLEARDelta = 0;
  resetDetectionTimers();
  calibration.state = "ready";
  calibration.failureReason = null;
  refreshDebug();
  return true;
}

function maybeAdvanceCalibration(now) {
  if (calibration.state === "settling" && now >= calibration.settledAt) {
    calibration.state = "sampling";
  }

  if (
    (calibration.state === "settling" || calibration.state === "sampling")
    && now - calibration.startedAt >= CALIBRATION_MAX_MS
  ) {
    robustlyFinalizeCalibration();
  }
}

function recordCalibrationSample(metrics, quality, now) {
  if (calibration.state !== "sampling") return;
  if (calibration.lastSampleAttemptAt !== null
    && now - calibration.lastSampleAttemptAt < CALIBRATION_SAMPLE_INTERVAL_MS) return;

  calibration.lastSampleAttemptAt = now;
  calibration.attempted += 1;
  const calibrationBlink = quality.blinkCandidate
    || (metrics && metrics.rEAR < CALIBRATION_MIN_EAR && metrics.lEAR < CALIBRATION_MIN_EAR);
  if (!quality.ok || calibrationBlink) {
    calibration.discarded += 1;
    const reasons = calibrationBlink
      ? [...quality.reasons, "calibration-blink-candidate"]
      : quality.reasons;
    calibration.lastDiscardReason = reasons.join(", ") || "quality-gate";
    refreshDebug({ quality: { ...quality, reasons } });
    return;
  }

  calibration.samples.push({ ...metrics, jitter: quality.jitter });
  calibration.lastDiscardReason = null;
  if (calibration.samples.length >= CALIBRATION_TARGET_SAMPLES) {
    robustlyFinalizeCalibration();
  } else {
    refreshDebug({ quality });
  }
}

export function startCalibrationSampling(at = nowMs()) {
  const now = Number.isFinite(at) ? at : nowMs();
  clearBaselines();
  calibration = createCalibrationState();
  calibration.state = "settling";
  calibration.startedAt = now;
  calibration.settledAt = now + CALIBRATION_SETTLE_MS;
  refreshDebug();
}

export function stopCalibrationSampling() {
  if (calibration.state === "settling" || calibration.state === "sampling") {
    return robustlyFinalizeCalibration();
  }
  return calibration.state === "ready";
}

function clearBaselines() {
  baselineAU4 = null;
  baselineREAR = null;
  baselineLEAR = null;
  baselineEAR = null;
  qualityBaseline = null;
  robustBaseline = null;
  smoothAU4Delta = null;
  smoothAU7Delta = null;
  smoothREARDelta = null;
  smoothLEARDelta = null;
  lastUpdateAt = null;
  lastLandmarkSnapshot = null;
  invalidSince = null;
  blinkState = createBlinkState();
  lastReliableResult = null;
  resetDetectionTimers();
}

function resetDetectionTimers() {
  au4SupportForMs = 0;
  au4ActiveForMs = 0;
  au7ActiveForMs = 0;
  neutralForMs = 0;
}

//Runtime detection
function getBlinkSnapshot(now) {
  const suppressionRemainingMs = Math.max(0, blinkState.suppressionUntil - now);
  const candidateAgeMs = blinkState.phase === "closing"
    ? Math.max(0, now - blinkState.startedAt)
    : 0;
  return {
    phase: blinkState.phase,
    candidate: blinkState.phase === "closing",
    confirmed: suppressionRemainingMs > 0,
    blocksAU7: blinkState.phase === "confirmed" || suppressionRemainingMs > 0,
    suppressionRemainingMs,
    candidateAgeMs,
    minRatio: blinkState.minRatio,
  };
}

function updateBlinkState(metrics, now) {
  const rRatio = baselineREAR > 0 ? metrics.rEAR / baselineREAR : 1;
  const lRatio = baselineLEAR > 0 ? metrics.lEAR / baselineLEAR : 1;
  const minRatio = Math.min(rRatio, lRatio);
  const bothClosing = rRatio < BLINK_CLOSING_RATIO && lRatio < BLINK_CLOSING_RATIO;
  const bothRecovered = rRatio >= BLINK_RECOVERY_RATIO && lRatio >= BLINK_RECOVERY_RATIO;
  let event = null;

  if (blinkState.phase === "open" && bothClosing) {
    blinkState.phase = "closing";
    blinkState.startedAt = now;
    blinkState.minRatio = minRatio;
    event = "start";
  } else if (blinkState.phase === "closing") {
    blinkState.minRatio = Math.min(blinkState.minRatio, minRatio);
    const closingForMs = now - blinkState.startedAt;

    if (bothRecovered) {
      if (closingForMs >= BLINK_MIN_CLOSE_MS && closingForMs <= BLINK_MAX_CLOSE_MS) {
        blinkState.phase = "confirmed";
        blinkState.suppressionUntil = now + BLINK_SUPPRESSION_MS;
        event = "confirmed";
      } else {
        blinkState.phase = "open";
        blinkState.startedAt = 0;
        blinkState.minRatio = 1;
        event = "end";
      }
    } else if (closingForMs >= BLINK_SUSTAINED_MS) {
      blinkState.phase = "recovering";
      event = "sustained";
    }
  } else if (blinkState.phase === "confirmed") {
    blinkState.phase = "recovering";
  } else if (blinkState.phase === "recovering" && bothRecovered) {
    blinkState.phase = "open";
    blinkState.startedAt = 0;
    blinkState.minRatio = 1;
    event = "end";
  }

  return { ...getBlinkSnapshot(now), rRatio, lRatio, event };
}

function inactiveResult({ quality = null, metrics = null, rawDeltas = null, blink = getBlinkSnapshot(nowMs()), au7BlockedBy = [] } = {}) {
  const ready = calibration.state === "ready" && baselineAU4 !== null;
  return {
    ready,
    quality,
    facialEvidence: "none",
    au4Active: false,
    au4Supportive: false,
    au7Active: false,
    au4Raw: metrics?.au4 ?? null,
    au4Delta: smoothAU4Delta,
    au7Delta: smoothAU7Delta,
    rEAR: metrics?.rEAR ?? null,
    lEAR: metrics?.lEAR ?? null,
    meanEAR: metrics?.meanEAR ?? null,
    rEARDelta: rawDeltas?.rEAR ?? smoothREARDelta,
    lEARDelta: rawDeltas?.lEAR ?? smoothLEARDelta,
    meanEARDelta: rawDeltas?.au7 ?? smoothAU7Delta,
    blinkCandidate: blink.candidate,
    blinkConfirmed: blink.confirmed,
    blinkState: blink.phase,
    blinkBlocksAU7: blink.blocksAU7,
    blinkCandidateAgeMs: blink.candidateAgeMs,
    au7Candidate: false,
    au7BlockedBy,
    au7TimerMs: au7ActiveForMs,
    qualitySuspended: false,
    invalidForMs: 0,
    baselineAdapted: false,
    neutralForMs,
    confusedStrong: false,
    confusedLikely: false,
    confused: false,
  };
}

function resetExpressionTimersForInvalidFrame() {
  au4SupportForMs = 0;
  au4ActiveForMs = 0;
  au7ActiveForMs = 0;
  neutralForMs = 0;
}

function pauseOrResetForInvalidQuality(now) {
  if (invalidSince === null) invalidSince = now;
  const invalidForMs = Math.max(0, now - invalidSince);
  const withinGrace = invalidForMs <= INVALID_GRACE_MS;
  if (!withinGrace) {
    resetExpressionTimersForInvalidFrame();
    lastReliableResult = null;
  }
  return { invalidForMs, withinGrace };
}

function clearInvalidQualityGrace() {
  const wasSuspended = invalidSince !== null;
  invalidSince = null;
  return wasSuspended;
}

function qualitySuspendedResult({ quality, metrics, rawDeltas, blink, invalid, au7BlockedBy }) {
  const previous = lastReliableResult ?? inactiveResult({ quality, metrics, rawDeltas, blink, au7BlockedBy });
  return {
    ...previous,
    ready: calibration.state === "ready" && baselineAU4 !== null,
    quality,
    au4Raw: metrics?.au4 ?? previous.au4Raw ?? null,
    rEAR: metrics?.rEAR ?? null,
    lEAR: metrics?.lEAR ?? null,
    meanEAR: metrics?.meanEAR ?? null,
    rEARDelta: rawDeltas?.rEAR ?? previous.rEARDelta ?? null,
    lEARDelta: rawDeltas?.lEAR ?? previous.lEARDelta ?? null,
    meanEARDelta: rawDeltas?.au7 ?? previous.meanEARDelta ?? null,
    blinkCandidate: blink.candidate,
    blinkConfirmed: blink.confirmed,
    blinkState: blink.phase,
    blinkBlocksAU7: blink.blocksAU7,
    blinkCandidateAgeMs: blink.candidateAgeMs,
    au7BlockedBy,
    au7TimerMs: au7ActiveForMs,
    qualitySuspended: true,
    invalidForMs: invalid.invalidForMs,
    baselineAdapted: false,
  };
}

function updateDuration(current, condition, dt) {
  return condition ? current + dt : 0;
}

function updateNeutralDrift(metrics, dt, conditions) {
  const blockedBy = [];
  if (!ENABLE_NEUTRAL_DRIFT) blockedBy.push("disabled");
  if (!conditions.qualityOk) blockedBy.push("quality");
  if (!conditions.neutral) blockedBy.push("expression-candidate");
  if (conditions.blinkLike) blockedBy.push("blink");
  if (neutralForMs < NEUTRAL_HOLD_MS) blockedBy.push("neutral-hold");

  const adaptable = blockedBy.length === 0;
  if (!adaptable) return { adapted: false, adaptable, blockedBy };

  const alpha = timeAlpha(dt, DRIFT_TAU_MS);
  baselineAU4 = alpha * metrics.au4 + (1 - alpha) * baselineAU4;
  baselineREAR = alpha * metrics.rEAR + (1 - alpha) * baselineREAR;
  baselineLEAR = alpha * metrics.lEAR + (1 - alpha) * baselineLEAR;
  baselineEAR = (baselineREAR + baselineLEAR) / 2;
  return { adapted: true, adaptable, blockedBy };
}

export function updateFaceAnalysis(wg, at = nowMs()) {
  const now = Number.isFinite(at) ? at : nowMs();
  const dt = clampTimeDelta(now);
  maybeAdvanceCalibration(now);

  const tracker = wg?.getTracker?.();
  const pts = tracker?.getPositions?.();
  if (!hasRequiredLandmarks(pts)) {
    const quality = evaluateFaceQuality(pts, null, now);
    if (calibration.state === "sampling") {
      recordCalibrationSample(null, quality, now);
    }
    const invalid = calibration.state === "ready" ? pauseOrResetForInvalidQuality(now) : null;
    const blink = getBlinkSnapshot(now);
    const au7BlockedBy = ["invalid-landmarks", ...(invalid?.withinGrace ? ["invalid-grace"] : [])];
    refreshDebug({ quality, rawDeltas: null, blink, invalid, au7BlockedBy });
    if (invalid?.withinGrace && !blink.blocksAU7) {
      return qualitySuspendedResult({ quality, blink, invalid, au7BlockedBy });
    }
    return inactiveResult({ quality, blink, au7BlockedBy });
  }

  const metrics = computeMetrics(pts);
  const quality = evaluateFaceQuality(pts, metrics, now);

  if (calibration.state === "settling" || calibration.state === "sampling") {
    recordCalibrationSample(metrics, quality, now);
    maybeAdvanceCalibration(now);
    refreshDebug({ quality, metrics });
    return inactiveResult({ quality, metrics });
  }

  if (calibration.state !== "ready" || baselineAU4 === null) {
    refreshDebug({ quality, metrics });
    return inactiveResult({ quality, metrics });
  }

  const blink = updateBlinkState(metrics, now);

  const rawAU4Delta = baselineAU4 - metrics.au4;
  const rawREARDelta = baselineREAR - metrics.rEAR;
  const rawLEARDelta = baselineLEAR - metrics.lEAR;
  const rawAU7Delta = baselineEAR - metrics.meanEAR;
  const au4RawPlausible = Math.abs(rawAU4Delta) <= AU4_MAX_VALID_DELTA;
  const rawDeltas = { au4: rawAU4Delta, au7: rawAU7Delta, rEAR: rawREARDelta, lEAR: rawLEARDelta };

  if (!quality.ok || !au4RawPlausible) {
    const gatedQuality = au4RawPlausible
      ? quality
      : { ...quality, ok: false, reasons: [...quality.reasons, "au4-outlier"] };
    const invalid = pauseOrResetForInvalidQuality(now);
    if (blink.blocksAU7) au7ActiveForMs = 0;
    const au7BlockedBy = [
      ...gatedQuality.reasons,
      ...(blink.blocksAU7 ? ["blink-confirmed"] : []),
      ...(invalid.withinGrace ? ["invalid-grace"] : ["invalid-too-long"]),
    ];
    refreshDebug({
      quality: gatedQuality,
      metrics,
      rawDeltas,
      blink,
      invalid,
      au7BlockedBy,
    });
    if (invalid.withinGrace && !blink.blocksAU7) {
      return qualitySuspendedResult({ quality: gatedQuality, metrics, rawDeltas, blink, invalid, au7BlockedBy });
    }
    return inactiveResult({ quality: gatedQuality, metrics, rawDeltas, blink, au7BlockedBy });
  }

  const resumedAfterQualitySuspension = clearInvalidQualityGrace();

  smoothAU4Delta = timeEma(smoothAU4Delta, rawAU4Delta, dt, DELTA_SMOOTH_TAU_MS);
  smoothAU7Delta = timeEma(smoothAU7Delta, rawAU7Delta, dt, DELTA_SMOOTH_TAU_MS);
  smoothREARDelta = timeEma(smoothREARDelta, rawREARDelta, dt, DELTA_SMOOTH_TAU_MS);
  smoothLEARDelta = timeEma(smoothLEARDelta, rawLEARDelta, dt, DELTA_SMOOTH_TAU_MS);

  const au4Reliable = Math.abs(smoothAU4Delta) <= AU4_MAX_VALID_DELTA;
  const au4SupportHit = au4Reliable && smoothAU4Delta > AU4_SUPPORT_DELTA;
  const au4ActiveHit = au4Reliable && smoothAU4Delta > AU4_ACTIVE_DELTA;
  const asymmetry = Math.abs(smoothREARDelta - smoothLEARDelta);
  const bilateralOk = (
    smoothREARDelta > AU7_DELTA * AU7_EYE_MIN_FRACTION
    && smoothLEARDelta > AU7_DELTA * AU7_EYE_MIN_FRACTION
    && asymmetry <= AU7_MAX_ASYMMETRY
  );
  const au7Candidate = bilateralOk && smoothAU7Delta > AU7_DELTA;
  const au7BlockedBy = [];
  if (!bilateralOk) au7BlockedBy.push("bilateral-check");
  if (smoothAU7Delta <= AU7_DELTA) au7BlockedBy.push("mean-drop");
  if (blink.blocksAU7) au7BlockedBy.push("blink-confirmed");

  const timerDt = resumedAfterQualitySuspension ? 0 : dt;
  au4SupportForMs = updateDuration(au4SupportForMs, au4SupportHit, timerDt);
  au4ActiveForMs = updateDuration(au4ActiveForMs, au4ActiveHit, timerDt);
  if (blink.blocksAU7) {
    au7ActiveForMs = 0;
  } else { 
    au7ActiveForMs = updateDuration(au7ActiveForMs, au7Candidate, timerDt);
  }

  const au4Supportive = au4SupportForMs >= AU4_SUPPORT_MS;
  const au4Active = au4ActiveForMs >= AU4_ACTIVE_MS;
  const au7Active = au7ActiveForMs >= AU7_ACTIVE_MS && !blink.blocksAU7;
  const neutral = au4Reliable && !au4SupportHit && !au7Candidate && !blink.blocksAU7;
  neutralForMs = updateDuration(neutralForMs, neutral, timerDt);
  const drift = updateNeutralDrift(metrics, dt, {
    qualityOk: quality.ok,
    neutral,
    blinkLike: blink.blocksAU7 || quality.eyesTooClosed,
  });

  const confusedStrong = au4Active && au7Active;
  const confusedLikely = au7Active && au4Supportive;
  const facialEvidence = confusedStrong ? "strong" : confusedLikely ? "weak" : "none";

  refreshDebug({
    quality,
    metrics,
    rawDeltas,
    blink,
    drift,
    au7Candidate,
    au7BlockedBy,
    bilateralOk,
    asymmetry,
  });

  const result = {
    ready: true,
    quality,
    facialEvidence,
    au4Active,
    au4Supportive,
    au7Active,
    au4Delta: smoothAU4Delta,
    au7Delta: smoothAU7Delta,
    rEAR: metrics.rEAR,
    lEAR: metrics.lEAR,
    meanEAR: metrics.meanEAR,
    rEARDelta: smoothREARDelta,
    lEARDelta: smoothLEARDelta,
    meanEARDelta: smoothAU7Delta,
    blinkCandidate: blink.candidate,
    blinkConfirmed: blink.confirmed,
    blinkState: blink.phase,
    blinkBlocksAU7: blink.blocksAU7,
    blinkCandidateAgeMs: blink.candidateAgeMs,
    au7Candidate,
    au7BlockedBy,
    au7TimerMs: au7ActiveForMs,
    qualitySuspended: false,
    invalidForMs: 0,
    baselineAdapted: drift.adapted,
    neutralForMs,
    confusedStrong,
    confusedLikely,
    confused: confusedStrong || confusedLikely,
  };
  lastReliableResult = result;
  return result;
}

function refreshDebug({
  quality = _debug.quality,
  metrics = null,
  rawDeltas = null,
  blink = getBlinkSnapshot(nowMs()),
  invalid = null,
  drift = null,
  robustStats = null,
  robustSamples = null,
  au7Candidate = false,
  au7BlockedBy = [],
  bilateralOk = false,
  asymmetry = null,
} = {}) {
  const ready = calibration.state === "ready" && baselineAU4 !== null;
  _debug = {
    warmedUp: ready,
    ready,
    calibrationState: calibration.state,
    calibration: {
      startedAt: calibration.startedAt,
      attempted: calibration.attempted,
      accepted: calibration.samples.length,
      discarded: calibration.discarded,
      targetSamples: CALIBRATION_TARGET_SAMPLES,
      minimumSamples: CALIBRATION_MIN_SAMPLES,
      lastDiscardReason: calibration.lastDiscardReason,
      failureReason: calibration.failureReason,
      settlingRemainingMs: calibration.state === "settling"
        ? Math.max(0, calibration.settledAt - nowMs())
        : 0,
    },
    baselines: ready ? {
      au4: baselineAU4,
      rEAR: baselineREAR,
      lEAR: baselineLEAR,
      ear: baselineEAR,
      quality: qualityBaseline,
    } : null,
    robustBaseline: robustBaseline ?? (robustStats ? {
      medians: Object.fromEntries(Object.entries(robustStats).map(([field, value]) => [field, value.median])),
      mads: Object.fromEntries(Object.entries(robustStats).map(([field, value]) => [field, value.mad])),
      inlierSamples: robustSamples,
    } : null),
    au4Delta: rounded(smoothAU4Delta, 3),
    au7Delta: rounded(smoothAU7Delta, 3),
    au4Raw: rounded(metrics?.au4, 4),
    rEAR: rounded(metrics?.rEAR, 3),
    lEAR: rounded(metrics?.lEAR, 3),
    meanEAR: rounded(metrics?.meanEAR, 3),
    rEARDelta: rounded(rawDeltas?.rEAR ?? smoothREARDelta, 3),
    lEARDelta: rounded(rawDeltas?.lEAR ?? smoothLEARDelta, 3),
    meanEARDelta: rounded(rawDeltas?.au7 ?? smoothAU7Delta, 3),
    rawDeltas: rawDeltas && Object.fromEntries(
      Object.entries(rawDeltas).map(([key, value]) => [key, rounded(value, 4)])
    ),
    smoothDeltas: {
      au4: rounded(smoothAU4Delta, 4),
      au7: rounded(smoothAU7Delta, 4),
      rEAR: rounded(smoothREARDelta, 4),
      lEAR: rounded(smoothLEARDelta, 4),
    },
    quality,
    blink: {
      state: blink.phase,
      candidate: blink.candidate,
      confirmed: blink.confirmed,
      minRatio: rounded(blink.minRatio, 3),
      suppressionRemainingMs: Math.round(blink.suppressionRemainingMs),
      event: blink.event ?? null,
    },
    blinkState: blink.phase,
    blinkCandidate: blink.candidate,
    blinkConfirmed: blink.confirmed,
    blinkBlocksAU7: blink.blocksAU7,
    blinkCandidateAgeMs: Math.round(blink.candidateAgeMs),
    blinkSuppressionRemainingMs: Math.round(blink.suppressionRemainingMs),
    au7Candidate,
    au7BlockedBy,
    rDrop: rounded(rawDeltas?.rEAR, 4),
    lDrop: rounded(rawDeltas?.lEAR, 4),
    meanDrop: rounded(rawDeltas?.au7, 4),
    bilateralOk,
    asymmetry: rounded(asymmetry, 4),
    invalidForMs: Math.round(invalid?.invalidForMs ?? 0),
    invalidWithinGrace: Boolean(invalid?.withinGrace),
    qualitySuspended: Boolean(invalid?.withinGrace && !blink.blocksAU7),
    timers: {
      au4SupportMs: Math.round(au4SupportForMs),
      au4ActiveMs: Math.round(au4ActiveForMs),
      au7ActiveMs: Math.round(au7ActiveForMs),
      neutralMs: Math.round(neutralForMs),
    },
    au7TimerMs: Math.round(au7ActiveForMs),
    baselineAdapted: drift?.adapted ?? false,
    baselineAdaptable: drift?.adaptable ?? false,
    baselineAdaptBlockedBy: drift?.blockedBy?.join(", ") ?? (ready ? "disabled" : "not-ready"),
    neutralForMs: Math.round(neutralForMs),
  };
}

export function resetFaceAnalysis() {
  clearBaselines();
  calibration = createCalibrationState();
  _debug = createDebug();
}
