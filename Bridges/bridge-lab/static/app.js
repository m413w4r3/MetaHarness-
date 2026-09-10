const $ = (id) => document.getElementById(id);

const surfaces = {
  bridge: [["chat", "Chat Completions"], ["responses", "Responses"], ["native", "Bridge natif /runs"]],
  webai: [["chat", "Chat Completions"], ["stateless", "Stateless Chat"]],
};

const surfaceHints = {
  bridge: {
    chat: "OpenAI-compatible Chat Completions. Le modèle est un label API / traçage.",
    responses: "Responses adapté par le Bridge. Le modèle est un label API / traçage.",
    native: "Contrat natif du Bridge. UI model peut demander un changement vérifié du sélecteur ChatGPT.",
  },
  webai: {
    chat: "Chat Completions transmis à Gemini WebAPI. Le modèle est un identifiant provider éditable.",
    stateless: "Chat stateless transmis à Gemini WebAPI, sans conversation persistante.",
  },
};

const defaultModels = { bridge: "chatgpt-web", webai: "gemini-3-flash" };
const terminalStatuses = new Set(["completed", "failed", "needs_review"]);
const POLL_INTERVAL_MS = 1000;
const POLL_TIMEOUT_MS = 120000;
let pollController = null;

function pretty(value) { return typeof value === "string" ? value : JSON.stringify(value, null, 2); }
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));
}
function currentProvider() { return $("provider").value; }
function currentSurface() { return $("surface").value; }
function currentModel() { return $("model").value.trim() || defaultModels[currentProvider()]; }
function currentUiModel() { return $("uiModel").value.trim() || null; }

function updateSurfaceControls() {
  const provider = currentProvider();
  const surface = currentSurface();
  $("surfaceHint").textContent = surfaceHints[provider][surface] || "";
  $("modelLabel").firstChild.textContent = provider === "webai" ? "Provider model" : "API model label";
  $("modelNote").textContent = provider === "webai"
    ? "Valeur initiale: gemini-3-flash. Les suggestions sont reported by /v1/models et ne remplacent pas ce champ."
    : "Label API / traçage; il ne sélectionne pas le modèle dans l’UI ChatGPT. UI model est séparé pour le Bridge natif.";
  $("uiModelField").hidden = !(provider === "bridge" && surface === "native");
  for (const button of document.querySelectorAll("[data-bridge-only]")) button.hidden = provider !== "bridge";
}

function updateSurfaces() {
  const provider = currentProvider();
  $("surface").innerHTML = "";
  for (const [value, label] of surfaces[provider]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    $("surface").appendChild(option);
  }
  $("model").value = defaultModels[provider];
  $("uiModel").value = "";
  updateSurfaceControls();
  loadModels();
  applyPreset("text");
}

async function loadModels() {
  const provider = currentProvider();
  $("modelSuggestions").innerHTML = "";
  try {
    const res = await fetch(`/api/models/${provider}`);
    const data = await res.json();
    const list = data?.body?.data;
    if (!Array.isArray(list)) return;
    if (currentProvider() !== provider) return;
    for (const item of list) {
      if (!item?.id) continue;
      const option = document.createElement("option");
      option.value = item.id;
      option.label = "reported by /v1/models";
      option.title = "reported by /v1/models (suggestion only)";
      $("modelSuggestions").appendChild(option);
    }
  } catch (_) {
    // The configured value remains usable when the informational catalogue is down.
  }
}

function payloadFor(preset) {
  const provider = currentProvider();
  const model = currentModel();
  if (provider === "bridge" && preset === "native") {
    $("surface").value = "native";
    updateSurfaceControls();
    return { requested_model: model, ui_model: currentUiModel(), input: "Réponds exactement avec le texte BRIDGE_NATIVE_OK", web_search: false, background: false };
  }
  if (provider === "bridge" && preset === "background") {
    $("surface").value = "responses";
    updateSurfaceControls();
    return { model, input: "Réponds exactement avec le texte BRIDGE_BACKGROUND_OK", background: true };
  }
  if (preset === "stream") {
    if (["responses", "native"].includes(currentSurface())) {
      $("surface").value = "chat";
      updateSurfaceControls();
    }
    return { model, messages: [{ role: "user", content: "Réponds exactement avec STREAM_OK" }], stream: true };
  }
  const prompt = "Réponds exactement avec le texte BRIDGE_LAB_OK";
  if (currentSurface() === "responses") return { model, input: prompt, background: false };
  if (currentSurface() === "native") return { requested_model: model, ui_model: currentUiModel(), input: prompt, web_search: false, background: false };
  return { model, messages: [{ role: "user", content: prompt }], stream: false };
}

function applyPreset(name) {
  if (name === "native" && currentProvider() !== "bridge") name = "text";
  if (name === "background" && currentProvider() !== "bridge") name = "text";
  $("payload").value = JSON.stringify(payloadFor(name), null, 2);
  updateSurfaceControls();
}

function compactStatus(data) {
  const checks = data?.checks || {};
  const lines = [];
  if (data?.summary) {
    const summary = data.summary;
    lines.push(`Global: ${summary.label}`);
    if (data.provider === "webai") {
      lines.push(`Process             ${summary.process}`);
      lines.push(`Gemini WebAPI auth  ${summary.gemini_webapi_auth}`);
      lines.push(`Browser readiness   ${summary.browser_readiness}`);
    }
    lines.push("");
  }
  for (const [path, result] of Object.entries(checks)) {
    const browserReady = data.provider === "webai" && path === "/ready";
    const state = browserReady ? "INFO" : result.ok ? "OK  " : "ERR ";
    lines.push(`${state} ${String(result.http_status ?? "-").padStart(3)} ${String(result.latency_ms ?? "-").padStart(5)}ms  ${path}`);
    if (!result.ok && result.body && !browserReady) lines.push(`    ${pretty(result.body).slice(0, 400)}`);
    if (result.error && !browserReady) lines.push(`    ${result.error}`);
  }
  return lines.join("\n");
}

async function refreshStatus() {
  for (const provider of ["bridge", "webai"]) {
    const badge = $(`${provider}Badge`);
    const output = $(`${provider}Status`);
    badge.textContent = "…";
    badge.className = "badge";
    try {
      const res = await fetch(`/api/status/${provider}`);
      const data = await res.json();
      output.textContent = compactStatus(data);
      const summary = data.summary || {};
      badge.textContent = summary.label || (summary.ok ? "joignable" : "hors ligne");
      badge.className = `badge ${summary.ok || summary.reachable ? "ok" : "bad"}`;
    } catch (err) {
      output.textContent = String(err);
      badge.textContent = "erreur";
      badge.className = "badge bad";
    }
  }
}

function renderDiagnostic(result, polling = null) {
  const raw = result.raw ?? result.body;
  const rows = [
    ["HTTP", result.http_status ?? "—", result.http_status >= 200 && result.http_status < 300],
    ["Durée", result.latency_ms == null ? "—" : `${result.latency_ms} ms`, true],
    ["Texte assistant", result.extracted_text ? `${result.extracted_text.length} caractères` : "non extrait", !!result.extracted_text],
  ];
  if (result.stream) rows.push(["Transport", "SSE buffered", true]);
  const responseStatus = raw && typeof raw === "object" ? raw.status : null;
  if (responseStatus) rows.push(["Statut provider", responseStatus, !["failed", "needs_review"].includes(responseStatus)]);
  if (polling) {
    rows.push(["Background id", polling.id, true]);
    rows.push(["Polls", polling.polls, true]);
    rows.push(["Statut courant", polling.status, terminalStatuses.has(polling.status)]);
  }
  $("diagnostic").className = "diagnostic";
  let html = rows.map(([name, value, ok]) => `<div class="metric"><span>${escapeHtml(name)}</span><strong class="${ok ? "pass" : "fail"}">${escapeHtml(value)}</strong></div>`).join("");
  if (result.response_parse_error) html += `<pre class="fail">${escapeHtml(result.response_parse_error)}</pre>`;
  if (result.provider_metadata && Object.keys(result.provider_metadata).length) {
    html += `<h3>Métadonnées provider</h3><pre>${escapeHtml(pretty(result.provider_metadata))}</pre>`;
    if (result.provider_metadata.error) html += `<h3>Erreur terminale</h3><pre class="fail">${escapeHtml(pretty(result.provider_metadata.error))}</pre>`;
  }
  if (result.sse) html += `<h3>SSE buffered</h3><pre>${escapeHtml(pretty(result.sse))}</pre>`;
  $("diagnostic").innerHTML = html;
}

function renderResult(result, polling = null) {
  $("raw").textContent = pretty(result.raw ?? result.body ?? result);
  $("extracted").textContent = result.extracted_text ?? "—";
  renderDiagnostic(result, polling);
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => { clearTimeout(timer); reject(new DOMException("Polling arrêté", "AbortError")); }, { once: true });
  });
}

function stopPolling() { if (pollController) pollController.abort(); }

async function pollBackground(responseId, initial) {
  stopPolling();
  const controller = new AbortController();
  pollController = controller;
  $("stopPolling").disabled = false;
  const started = performance.now();
  let polls = 0;
  let status = initial?.status || "queued";
  try {
    while (!terminalStatuses.has(status)) {
      if (performance.now() - started > POLL_TIMEOUT_MS) throw new Error("Timeout global du polling background.");
      await sleep(POLL_INTERVAL_MS, controller.signal);
      const res = await fetch(`/api/poll/responses/${encodeURIComponent(responseId)}`, { signal: controller.signal });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(pretty(data));
      polls += 1;
      const body = data.body || {};
      status = body.status || "unknown";
      renderResult(data, { id: responseId, polls, status });
      $("requestState").textContent = `polling • ${status} • ${polls} poll(s)`;
    }
    $("requestState").textContent = `terminé • ${status} • ${polls} poll(s)`;
  } catch (err) {
    if (err.name === "AbortError") $("requestState").textContent = `polling arrêté • ${polls} poll(s)`;
    else {
      $("requestState").textContent = "Erreur de polling";
      $("diagnostic").innerHTML += `<pre class="fail">${escapeHtml(String(err))}</pre>`;
    }
  } finally {
    if (pollController === controller) pollController = null;
    $("stopPolling").disabled = true;
  }
}

async function send() {
  stopPolling();
  $("requestState").textContent = "Requête en cours…";
  $("send").disabled = true;
  try {
    const payload = JSON.parse($("payload").value);
    const body = { provider: currentProvider(), surface: currentSurface(), payload, idempotency_key: $("idem").value.trim() || null };
    const res = await fetch("/api/send", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) throw new Error(pretty(data));
    renderResult(data);
    const initial = data.raw && typeof data.raw === "object" ? data.raw : null;
    if (currentProvider() === "bridge" && currentSurface() === "responses" && payload.background && initial?.id) await pollBackground(initial.id, initial);
    else $("requestState").textContent = `${data.http_status} • ${data.latency_ms} ms`;
  } catch (err) {
    if (err.name !== "AbortError") {
      $("requestState").textContent = "Erreur";
      $("raw").textContent = String(err);
      $("extracted").textContent = "—";
      $("diagnostic").className = "diagnostic";
      $("diagnostic").innerHTML = `<pre class="fail">${escapeHtml(String(err))}</pre>`;
    }
  } finally { $("send").disabled = false; }
}

$("provider").addEventListener("change", updateSurfaces);
$("surface").addEventListener("change", () => { updateSurfaceControls(); applyPreset("text"); });
$("model").addEventListener("change", () => applyPreset("text"));
$("uiModel").addEventListener("change", () => applyPreset("text"));
$("refreshStatus").addEventListener("click", refreshStatus);
$("send").addEventListener("click", send);
$("stopPolling").addEventListener("click", stopPolling);
for (const button of document.querySelectorAll("[data-preset]")) button.addEventListener("click", () => applyPreset(button.dataset.preset));

updateSurfaces();
refreshStatus();
