
const $ = (id) => document.getElementById(id);

const surfaces = {
  bridge: [
    ["chat", "Chat Completions"],
    ["responses", "Responses"],
    ["native", "Bridge /runs"],
  ],
  webai: [
    ["chat", "Chat Completions"],
    ["stateless", "Stateless Chat"],
  ],
};

const defaultModels = { bridge: "chatgpt-web", webai: "gemini-3-flash" };

function pretty(value) {
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function currentProvider() { return $("provider").value; }
function currentSurface() { return $("surface").value; }
function currentModel() { return $("model").value || defaultModels[currentProvider()]; }

function updateSurfaces() {
  const provider = currentProvider();
  $("surface").innerHTML = "";
  for (const [value, label] of surfaces[provider]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    $("surface").appendChild(option);
  }
  loadModels();
  applyPreset("text");
}

async function loadModels() {
  const provider = currentProvider();
  $("model").innerHTML = `<option value="${defaultModels[provider]}">${defaultModels[provider]}</option>`;
  try {
    const res = await fetch(`/api/models/${provider}`);
    const data = await res.json();
    const list = data?.body?.data;
    if (Array.isArray(list) && list.length) {
      $("model").innerHTML = "";
      for (const item of list) {
        if (!item?.id) continue;
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = item.label ? `${item.id} — ${item.label}` : item.id;
        $("model").appendChild(option);
      }
    }
  } catch (_) {}
}

function payloadFor(preset) {
  const provider = currentProvider();
  const model = currentModel();

  if (provider === "bridge" && preset === "native") {
    $("surface").value = "native";
    return {
      requested_model: model,
      input: "Réponds exactement avec le texte BRIDGE_NATIVE_OK",
      web_search: false,
      background: false,
    };
  }

  if (provider === "bridge" && preset === "background") {
    $("surface").value = "responses";
    return {
      model,
      input: "Réponds exactement avec le texte BRIDGE_BACKGROUND_OK",
      background: true,
    };
  }

  if (preset === "json") {
    const prompt = 'Réponds UNIQUEMENT avec ce JSON valide, sans Markdown ni commentaire: {"ok":true,"source":"lab"}';
    if (currentSurface() === "responses") return { model, input: prompt, background: false };
    return { model, messages: [{ role: "user", content: prompt }], stream: false };
  }

  if (preset === "nested") {
    const prompt = 'Réponds UNIQUEMENT avec un JSON strict de cette forme: {"items":[{"name":"alpha","score":1}],"summary":"ok"}. Aucun ``` et aucun texte autour.';
    if (currentSurface() === "responses") return { model, input: prompt, background: false };
    return { model, messages: [{ role: "user", content: prompt }], stream: false };
  }

  if (preset === "stream") {
    if (currentSurface() === "responses") $("surface").value = "chat";
    return {
      model,
      messages: [{ role: "user", content: "Réponds exactement avec STREAM_OK" }],
      stream: true,
    };
  }

  const prompt = "Réponds exactement avec le texte BRIDGE_LAB_OK";
  if (currentSurface() === "responses") return { model, input: prompt, background: false };
  if (currentSurface() === "native") {
    return { requested_model: model, input: prompt, web_search: false, background: false };
  }
  return { model, messages: [{ role: "user", content: prompt }], stream: false };
}

function applyPreset(name) {
  if (name === "native" && currentProvider() !== "bridge") name = "text";
  if (name === "background" && currentProvider() !== "bridge") name = "text";
  const payload = payloadFor(name);
  $("payload").value = JSON.stringify(payload, null, 2);

  if (name === "json") {
    $("schema").value = JSON.stringify({
      type: "object",
      additionalProperties: false,
      required: ["ok", "source"],
      properties: {
        ok: { type: "boolean", const: true },
        source: { type: "string" }
      }
    }, null, 2);
  } else if (name === "nested") {
    $("schema").value = JSON.stringify({
      type: "object",
      additionalProperties: false,
      required: ["items", "summary"],
      properties: {
        items: {
          type: "array",
          minItems: 1,
          items: {
            type: "object",
            additionalProperties: false,
            required: ["name", "score"],
            properties: {
              name: { type: "string" },
              score: { type: "integer" }
            }
          }
        },
        summary: { type: "string" }
      }
    }, null, 2);
  } else {
    $("schema").value = "";
  }
}

function compactStatus(data) {
  const checks = data?.checks || {};
  const lines = [];
  for (const [path, result] of Object.entries(checks)) {
    const state = result.ok ? "OK " : "ERR";
    lines.push(`${state} ${String(result.http_status ?? "-").padStart(3)} ${String(result.latency_ms ?? "-").padStart(5)}ms  ${path}`);
    if (!result.ok && result.body) lines.push(`    ${pretty(result.body).slice(0, 400)}`);
    if (result.error) lines.push(`    ${result.error}`);
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
      const checks = Object.values(data.checks || {});
      const healthy = checks.some(x => x.ok);
      badge.textContent = healthy ? "joignable" : "hors ligne";
      badge.className = `badge ${healthy ? "ok" : "bad"}`;
    } catch (err) {
      output.textContent = String(err);
      badge.textContent = "erreur";
      badge.className = "badge bad";
    }
  }
}

function renderDiagnostic(result) {
  const p = result.parse || {};
  const rows = [
    ["HTTP", result.http_status, result.http_status >= 200 && result.http_status < 300],
    ["Latence", `${result.latency_ms} ms`, true],
    ["Texte extrait", p.has_text ? "oui" : "non", !!p.has_text],
    ["JSON strict", p.strict_json ? "PASS" : "FAIL", !!p.strict_json],
    ["Code fence ```", p.fenced ? "détecté" : "non", !p.fenced],
  ];
  if (p.schema_valid !== null && p.schema_valid !== undefined) {
    rows.push(["JSON Schema", p.schema_valid ? "PASS" : "FAIL", !!p.schema_valid]);
  }
  $("diagnostic").className = "diagnostic";
  $("diagnostic").innerHTML = rows.map(([name, value, ok]) =>
    `<div class="metric"><span>${name}</span><strong class="${ok ? "pass" : "fail"}">${value}</strong></div>`
  ).join("") + (p.error ? `<pre class="fail">${p.error}</pre>` : "") +
    (p.schema_errors ? `<pre>${pretty(p.schema_errors)}</pre>` : "");
}

async function send() {
  $("requestState").textContent = "Requête en cours…";
  $("send").disabled = true;
  try {
    const payload = JSON.parse($("payload").value);
    const schemaText = $("schema").value.trim();
    const schema = schemaText ? JSON.parse(schemaText) : null;
    const body = {
      provider: currentProvider(),
      surface: currentSurface(),
      payload,
      idempotency_key: $("idem").value.trim() || null,
      schema,
    };
    const res = await fetch("/api/send", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(pretty(data));
    $("raw").textContent = pretty(data.raw);
    $("extracted").textContent = data.extracted_text ?? "—";
    renderDiagnostic(data);
    $("requestState").textContent = `${data.http_status} • ${data.latency_ms} ms`;
  } catch (err) {
    $("requestState").textContent = "Erreur";
    $("raw").textContent = String(err);
    $("extracted").textContent = "—";
    $("diagnostic").className = "diagnostic";
    $("diagnostic").innerHTML = `<pre class="fail">${String(err)}</pre>`;
  } finally {
    $("send").disabled = false;
  }
}

$("provider").addEventListener("change", updateSurfaces);
$("surface").addEventListener("change", () => applyPreset("text"));
$("model").addEventListener("change", () => applyPreset("text"));
$("refreshStatus").addEventListener("click", refreshStatus);
$("send").addEventListener("click", send);
for (const button of document.querySelectorAll("[data-preset]")) {
  button.addEventListener("click", () => applyPreset(button.dataset.preset));
}

updateSurfaces();
refreshStatus();
