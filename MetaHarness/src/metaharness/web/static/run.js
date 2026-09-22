// MetaHarness run page: targeted live status without full-page reloads.
//
// Static file, never templated.  It polls GET /api/runs/<id>/live every two
// seconds only while the run is running, and updates the server-rendered
// header, pipeline, tokens and recent events exclusively through
// textContent, classList and hidden.  Plan, diff, artifacts and every
// <details> element are never touched, so open sections, scroll position
// and text selection are preserved.  Polling stops as soon as the run is
// terminal or waiting for a human; the "Actualiser les détails" button then
// offers the complete server-rendered page on demand.
(function () {
  "use strict";

  var POLL_MS = 2000;
  var STOP_STATUSES = {
    committed: true, published: true, failed: true, blocked: true,
    plan_rejected: true, interrupted: true, awaiting_plan_approval: true
  };
  var SYMBOLS = {complete: "✓", running: "▶", failed: "✗", waiting: "·", resumable: "↻", skipped: "–"};
  var STATES = ["complete", "running", "failed", "waiting", "resumable", "skipped"];

  var root = document.getElementById("run");
  if (!root) {
    return;
  }
  var runId = root.getAttribute("data-run-id") || "";
  var initialStatus = root.getAttribute("data-status") || "";
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(runId) || STOP_STATUSES[initialStatus]) {
    return;
  }
  var endpoint = "/api/runs/" + encodeURIComponent(runId) + "/live";
  var timer = null;
  var busy = false;

  function byId(id) {
    return document.getElementById(id);
  }

  function setText(id, value) {
    var node = byId(id);
    if (node && typeof value === "string" && node.textContent !== value) {
      node.textContent = value;
    }
  }

  function tokens(usage) {
    if (!usage || typeof usage !== "object") {
      return "0 input / 0 output";
    }
    return String(usage.input_tokens || 0) + " input / " + String(usage.output_tokens || 0) + " output";
  }

  function updatePipeline(items) {
    if (!Array.isArray(items)) {
      return;
    }
    items.forEach(function (item) {
      if (!item || typeof item.key !== "string") {
        return;
      }
      var row = document.querySelector('[data-pipeline-key="' + item.key.replace(/[^A-Za-z0-9_-]/g, "") + '"]');
      if (!row) {
        return;
      }
      var state = STATES.indexOf(item.state) >= 0 ? item.state : "waiting";
      STATES.forEach(function (name) {
        row.classList.toggle("state-" + name, name === state);
      });
      var symbol = row.querySelector(".symbol");
      if (symbol) {
        symbol.textContent = SYMBOLS[state];
      }
    });
  }

  function updateEvents(events) {
    var list = byId("live-events");
    if (!list || !Array.isArray(events)) {
      return;
    }
    var slots = list.querySelectorAll("li[data-slot]");
    for (var index = 0; index < slots.length; index += 1) {
      var value = index < events.length ? events[index] : null;
      slots[index].textContent = typeof value === "string" ? value : "";
      slots[index].hidden = typeof value !== "string";
    }
    var empty = byId("live-events-empty");
    if (empty) {
      empty.hidden = events.length > 0;
    }
  }

  function apply(payload) {
    if (!payload || typeof payload !== "object") {
      return;
    }
    var status = typeof payload.status === "string" ? payload.status : "";
    setText("live-status", status.toUpperCase());
    setText("live-updated", typeof payload.updated_at === "string" ? payload.updated_at : "");
    setText("live-current", typeof payload.current_label === "string" ? payload.current_label : "—");
    setText("live-next", typeof payload.next_label === "string" ? payload.next_label : "—");
    var totals = payload.token_totals || {};
    ["planner", "implementer", "check_repair", "semantic_reviser", "final_reviewer"].forEach(function (name) {
      setText("live-tokens-" + name, tokens(totals[name]));
    });
    var badge = byId("live-status");
    if (badge) {
      badge.classList.toggle("failed", status === "failed" || status === "interrupted");
      badge.classList.toggle("success", status === "published" || status === "committed");
    }
    var failure = byId("live-failure");
    if (failure) {
      var reason = payload.failure && typeof payload.failure.reason === "string" ? payload.failure.reason : "";
      failure.textContent = reason;
      failure.hidden = reason === "";
    }
    updatePipeline(payload.pipeline);
    updateEvents(payload.progress_events);
    if (status !== initialStatus) {
      var refresh = byId("refresh-details");
      if (refresh) {
        refresh.hidden = false;
      }
    }
    if (STOP_STATUSES[status] || payload.running === false) {
      stop();
    }
  }

  function stop() {
    if (timer !== null) {
      window.clearInterval(timer);
      timer = null;
    }
    var refresh = byId("refresh-details");
    if (refresh) {
      refresh.hidden = false;
    }
  }

  function poll() {
    if (busy) {
      return;
    }
    busy = true;
    window.fetch(endpoint, {credentials: "same-origin", cache: "no-store", headers: {Accept: "application/json"}})
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(apply)
      .catch(function () {
        // A transient read failure keeps the last known state on screen.
      })
      .then(function () {
        busy = false;
      });
  }

  var refreshButton = byId("refresh-details");
  if (refreshButton) {
    refreshButton.addEventListener("click", function () {
      window.location.reload();
    });
  }
  timer = window.setInterval(poll, POLL_MS);
  poll();
}());
