/*
 LLM integration with a flag-driven backend switch.
 default: OpenRouter
 optional local fallback: local vllm-mlx backend
 */

const MODELS = [
  "deepseek/deepseek-v4-flash:free",
  "nvidia/nemotron-3-super-120b-a12b:free",
  "google/gemma-4-26b-a4b-it:free",
  "qwen/qwen3-next-80b-a3b-instruct:free",
];
const OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions";
const LOCAL_API_URL = "/api/local-llm";

const TARGET_PARAGRAPH_WORDS = 70;
const MAX_PARAGRAPH_WORDS = 110;
const EXPAND_EXISTING_PARAGRAPH_WORDS = 135;
const SPLIT_SEGMENT_TARGET_WORDS = 28;
const SPLIT_SEGMENT_MAX_WORDS = 42;
const OPENROUTER_TIMEOUT_MS = 30000;
const LOCAL_TIMEOUT_MS = 60000;
const LOCAL_WARMUP_TIMEOUT_MS = 90000;
const LOCAL_EXPLANATION_MAX_TOKENS = 192;
const REMOTE_EXPLANATION_MAX_TOKENS = 240;
const SPLIT_BREAKPOINT_MAX_TOKENS = 48;
const EXPLANATION_SENTENCE_LIMIT = 2;

let localWarmupPromise = null;

function envFlag(value) {
  return ["1", "true", "yes", "on"].includes(String(value ?? "").trim().toLowerCase());
}

function useLocalLlm() {
  return envFlag(import.meta.env.VITE_USE_LOCAL_LLM);
}

export function isLocalLlmEnabled() {
  return useLocalLlm();
}

function apiKey() {
  const k = import.meta.env.VITE_OPENROUTER_API_KEY ?? "";
  if (!k || k.startsWith("sk-or-your")) {
    throw new Error("Imposta VITE_OPENROUTER_API_KEY nel file .env e riavvia il server.");
  }
  return k;
}

function timeoutErrorFor(error) {
  return error?.name === "AbortError" || error === "timeout"
    ? new Error(`OpenRouter timeout dopo ${Math.round(OPENROUTER_TIMEOUT_MS / 1000)}s.`)
    : error;
}

async function readOpenRouterStream(res, onChunk) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let fullText = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line.startsWith("data:")) continue;

      const payload = line.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;

      let json;
      try {
        json = JSON.parse(payload);
      } catch {
        continue;
      }

      const delta = json.choices?.[0]?.delta?.content;
      if (delta) {
        fullText += delta;
        onChunk(fullText);
      }
    }
  }

  return fullText.trim();
}

async function callOpenRouter(messages, maxTokens, onChunk) {
  let lastError;

  for (const model of MODELS) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort("timeout"), OPENROUTER_TIMEOUT_MS);

    let res;
    try {
      res = await fetch(OPENROUTER_API_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey()}`,
          "Content-Type": "application/json",
          "HTTP-Referer": window.location.href,
        },
        body: JSON.stringify({ model, max_tokens: maxTokens, messages, stream: Boolean(onChunk) }),
        signal: controller.signal,
      });
    } catch (error) {
      window.clearTimeout(timeoutId);
      throw timeoutErrorFor(error);
    }

    if (res.ok) {
      try {
        return onChunk
          ? await readOpenRouterStream(res, onChunk)
          : (await res.json()).choices?.[0]?.message?.content?.trim() ?? "";
      } catch (error) {
        throw timeoutErrorFor(error);
      } finally {
        window.clearTimeout(timeoutId);
      }
    }

    window.clearTimeout(timeoutId);
    const body = await res.text().catch(() => "");
    lastError = new Error(`OpenRouter ${res.status} (${model}): ${body.slice(0, 120)}`);
    if (res.status !== 429 && res.status < 500) throw lastError;
  }

  throw lastError;
}

async function callLocalEndpoint(payload, timeoutMs = LOCAL_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort("timeout"), timeoutMs);

  const res = await fetch(LOCAL_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal: controller.signal,
  }).catch((error) => {
    if (error?.name === "AbortError" || error === "timeout") {
      throw new Error(
        `Il modello locale non ha risposto entro ${Math.round(timeoutMs / 1000)}s.`
      );
    }
    throw error;
  }).finally(() => {
    window.clearTimeout(timeoutId);
  });

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new Error(data.error || `Local LLM ${res.status}`);
  }

  return (data.content ?? "").trim();
}

async function callLocalModel(messages, maxTokens) {
  return callLocalEndpoint({
    requestType: "generic",
    messages,
    maxTokens,
  });
}

async function callModel(messages, maxTokens, onChunk) {
  if (useLocalLlm()) {
    return callLocalModel(messages, maxTokens);
  }

  return callOpenRouter(messages, maxTokens, onChunk);
}

export function prewarmLocalLlm() {
  if (!useLocalLlm()) {
    return Promise.resolve(false);
  }

  if (!localWarmupPromise) {
    localWarmupPromise = callLocalEndpoint(
      { warmup: true, requestType: "warmup" },
      LOCAL_WARMUP_TIMEOUT_MS
    ).catch((error) => {
      localWarmupPromise = null;
      throw error;
    });
  }

  return localWarmupPromise;
}

export function cleanExplanationText(text) {
  const normalized = String(text || "")
    .replace(/\s+/g, " ")
    .trim()
    .replace(
      /^(?:ecco\s+(?:una\s+)?spiegazione(?:\s+del\s+paragrafo)?(?:\s+in\s+[^:]+)?\s*:?\s*)/i,
      ""
    )
    .replace(/^[.!?,;:]+\s*/, "")
    .trim();

  if (!normalized) {
    return "";
  }

  const sentences = normalized.match(/[^.!?]+[.!?]+["')\]]?\s*/g) ?? [];
  if (sentences.length >= EXPLANATION_SENTENCE_LIMIT) {
    return sentences.slice(0, EXPLANATION_SENTENCE_LIMIT).join(" ").trim();
  }

  if (!/[.!?]["')\]]?$/.test(normalized)) {
    const lastPunctuation = Math.max(
      normalized.lastIndexOf("."),
      normalized.lastIndexOf("!"),
      normalized.lastIndexOf("?")
    );

    if (lastPunctuation >= 0) {
      return normalized.slice(0, lastPunctuation + 1).trim();
    }
  }

  return normalized;
}

//Paragraph splitting
export function wordCount(text) {
  const words = text.trim().match(/\S+/g);
  return words ? words.length : 0;
}

export function normalizeText(text) {
  return text.replace(/\s+/g, " ").trim();
}

export function stripParagraphLabel(text) {
  return text.replace(/^\s*(?:paragraph|paragrafo)\s*\d+\s*[:.)-]?\s*/i, "").trim();
}

function splitIntoCandidateSegments(text) {
  return splitLongParagraph(text, SPLIT_SEGMENT_TARGET_WORDS, SPLIT_SEGMENT_MAX_WORDS);
}

function buildSplitPrompt(segments) {
  const numberedSegments = segments
    .map((segment, index) => `${index + 1}. ${segment}`)
    .join("\n");

  return (
    "You are segmenting a text into logical paragraphs.\n" +
    "Each numbered item is a consecutive segment of the original text.\n" +
    "Return ONLY a comma-separated list of segment numbers after which a paragraph break should be inserted.\n" +
    "Rules:\n" +
    "- Use ascending numbers.\n" +
    "- Do not include the last segment number.\n" +
    "- If no paragraph break is needed, return only: none\n\n" +
    numberedSegments
  );
}

export function parseBreakpoints(reply, segmentCount) {
  const cleanedReply = stripParagraphLabel(reply).trim().toLowerCase();
  if (!cleanedReply || cleanedReply === "none") {
    return [];
  }

  const numbers = cleanedReply.match(/\d+/g)?.map(Number) ?? [];
  if (!numbers.length) {
    return null;
  }

  const uniqueAscending = [...new Set(numbers)];
  const valid = uniqueAscending.every((value, index) => (
    Number.isInteger(value) &&
    value > 0 &&
    value < segmentCount &&
    (index === 0 || value > uniqueAscending[index - 1])
  ));

  return valid ? uniqueAscending : null;
}

export function applyBreakpoints(segments, breakpoints) {
  if (!breakpoints?.length) {
    return [segments.join(" ").trim()].filter(Boolean);
  }

  const paragraphs = [];
  let start = 0;

  breakpoints.forEach((breakAfter) => {
    const end = breakAfter;
    const paragraph = segments.slice(start, end).join(" ").trim();
    if (paragraph) {
      paragraphs.push(paragraph);
    }
    start = end;
  });

  const tail = segments.slice(start).join(" ").trim();
  if (tail) {
    paragraphs.push(tail);
  }

  return paragraphs;
}

export function splitBySentences(text, perParagraph = 4) {
  const sentences = text.match(/[^.!?]+[.!?]+["']?\s*/g) ?? [];
  if (sentences.length < 2) return [text.trim()];

  const out = [];
  for (let i = 0; i < sentences.length; i += perParagraph) {
    out.push(sentences.slice(i, i + perParagraph).join("").trim());
  }

  return out.filter(Boolean);
}

export function splitLongParagraph(text, targetWords = TARGET_PARAGRAPH_WORDS, maxWords = MAX_PARAGRAPH_WORDS) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  if (wordCount(trimmed) <= maxWords) return [trimmed];

  const tokens = trimmed.match(/\S+\s*/g) ?? [trimmed];
  const paragraphs = [];

  let chunkStart = 0;
  let wordsSinceBreak = 0;
  let lastPreferredBreak = -1;

  function isHardBreak(token) {
    return /[.!?;:]["')\]]?$/.test(token.trim());
  }

  function isSoftBreak(token) {
    return /,["')\]]?$/.test(token.trim());
  }

  function pushChunk(endExclusive) {
    const paragraph = tokens.slice(chunkStart, endExclusive).join("").trim();
    if (paragraph) {
      paragraphs.push(paragraph);
    }

    chunkStart = endExclusive;
    wordsSinceBreak = 0;
    lastPreferredBreak = -1;
  }

  tokens.forEach((token, index) => {
    wordsSinceBreak += 1;

    if (
      isHardBreak(token) ||
      (wordsSinceBreak >= targetWords && isSoftBreak(token))
    ) {
      lastPreferredBreak = index + 1;
    }

    const reachedHardTarget = wordsSinceBreak >= maxWords;
    const reachedSoftTarget = wordsSinceBreak >= targetWords && isHardBreak(token);

    if (reachedSoftTarget || reachedHardTarget) {
      const breakAt = lastPreferredBreak > chunkStart ? lastPreferredBreak : index + 1;
      pushChunk(breakAt);
    }
  });

  if (chunkStart < tokens.length) {
    pushChunk(tokens.length);
  }

  return paragraphs.length ? paragraphs : [trimmed];
}

export function enforceParagraphStructure(paragraphs) {
  const expanded = paragraphs.flatMap((paragraph) => {
    const limit = paragraphs.length > 1
      ? EXPAND_EXISTING_PARAGRAPH_WORDS
      : MAX_PARAGRAPH_WORDS;

    return splitLongParagraph(paragraph, TARGET_PARAGRAPH_WORDS, limit);
  });

  return expanded.filter(Boolean);
}

export function heuristicSplit(rawText) {
  const bySentences = splitBySentences(rawText, 4);

  if (bySentences.length > 1) {
    return enforceParagraphStructure(bySentences);
  }

  return enforceParagraphStructure(
    splitLongParagraph(rawText, TARGET_PARAGRAPH_WORDS, MAX_PARAGRAPH_WORDS)
  );
}

export async function splitIntoSections(rawText) {
  const byBlankLine = rawText.split(/\n{2,}/).map((p) => p.trim()).filter(Boolean);
  if (byBlankLine.length > 1) {
    return enforceParagraphStructure(byBlankLine);
  }

  const normalizedText = rawText.trim();
  const splitSegments = splitIntoCandidateSegments(normalizedText);
  if (splitSegments.length < 2) {
    return enforceParagraphStructure([normalizedText]);
  }

  let backendAvailable = false;

  if (useLocalLlm()) {
    await prewarmLocalLlm();
    backendAvailable = true;
  } else {
    try {
      apiKey();
      backendAvailable = true;
    } catch {
      backendAvailable = false;
    }
  }

  if (backendAvailable) {
    try {
      const splitMessages = [{
        role: "user",
        content: buildSplitPrompt(splitSegments),
      }];
      const reply = useLocalLlm()
        ? await callLocalEndpoint({
            requestType: "split",
            messages: splitMessages,
            maxTokens: SPLIT_BREAKPOINT_MAX_TOKENS,
          })
        : await callModel(splitMessages, SPLIT_BREAKPOINT_MAX_TOKENS);

      const breakpoints = parseBreakpoints(reply, splitSegments.length);
      if (breakpoints !== null) {
        return enforceParagraphStructure(applyBreakpoints(splitSegments, breakpoints));
      }
    } catch (error) {
      if (useLocalLlm()) {
        throw new Error(
          `Lo split del testo richiede il modello locale pronto. ${error.message}`
        );
      }
    }
  }

  return heuristicSplit(rawText);
}

// Paragraph explanation
const cache = new Map();

export async function explainParagraph(paragraphText, onChunk) {
  const key = paragraphText.trim();
  if (cache.has(key)) return cache.get(key);

  const usingLocal = useLocalLlm();
  if (usingLocal) {
    await prewarmLocalLlm();
  }

  const explanationMessages = [{
    role: "user",
    content:
      usingLocal
        ? "Spiega il seguente paragrafo nella stessa lingua, in 2 frasi brevi e informative. " +
          "Rispondi solo con la spiegazione, senza introduzioni, senza titoli e senza formule utilizzando un linguaggio semplice come " +
          "\"Ecco una spiegazione\". " +
          "Nella prima frase riassumi il senso generale. " +
          "Nella seconda frase chiarisci il dettaglio più importante o il punto più difficile in modo concreto, " +
          "senza ripetere il testo parola per parola. " +
          "Non superare 2 frasi.\n\n" +
          key
        : "You are a reading comprehension assistant. " +
          "Explain this paragraph in simple language. " +
          "Use 2 short informative sentences: summarize the main idea, then clarify the most important or difficult concept concretely. " +
          "Do not exceed 2 sentences. " +
          "Do not quote the paragraph verbatim unless necessary. " +
          "Respond in the same language as the paragraph.\n\n" +
          `Paragraph: "${key}"`,
  }];

  const explanation = usingLocal
    ? await callLocalEndpoint({
        requestType: "explain",
        messages: explanationMessages,
        maxTokens: LOCAL_EXPLANATION_MAX_TOKENS,
      })
    : await callModel(explanationMessages, REMOTE_EXPLANATION_MAX_TOKENS, onChunk);

  const cleanedExplanation = cleanExplanationText(explanation);
  cache.set(key, cleanedExplanation);
  return cleanedExplanation;
}

export function clearExplanationCache() {
  cache.clear();
}
