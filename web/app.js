"use strict";

const state = {
  csrfToken: sessionStorage.getItem("fileAgentCsrf") || "",
  currentRunId: null,
  eventSource: null,
  lastEventId: 0,
  filePath: null,
  nextStartLine: null,
  eventCount: 0,
};

const byId = (id) => document.getElementById(id);

function setText(id, value) {
  byId(id).textContent = value == null ? "" : String(value);
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.method && options.method !== "GET" && state.csrfToken) {
    headers.set("X-CSRF-Token", state.csrfToken);
  }
  if (options.body) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload.error && payload.error.message ? payload.error.message : "Request failed.";
    throw new Error(message);
  }
  return payload;
}

function showAuthenticated(authenticated) {
  byId("login-view").hidden = authenticated;
  byId("app-view").hidden = !authenticated;
}

async function restoreSession() {
  try {
    const payload = await api("/api/session");
    state.csrfToken = payload.csrf_token;
    sessionStorage.setItem("fileAgentCsrf", state.csrfToken);
    showAuthenticated(true);
    await loadTree();
    if (payload.active_run_id) {
      state.currentRunId = payload.active_run_id;
      connectEvents(payload.active_run_id);
    }
  } catch (_error) {
    state.csrfToken = "";
    sessionStorage.removeItem("fileAgentCsrf");
    showAuthenticated(false);
  }
}

byId("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  setText("login-error", "");
  const input = byId("access-code");
  try {
    const payload = await api("/api/auth", {
      method: "POST",
      body: JSON.stringify({ access_code: input.value }),
    });
    input.value = "";
    state.csrfToken = payload.csrf_token;
    sessionStorage.setItem("fileAgentCsrf", state.csrfToken);
    showAuthenticated(true);
    await loadTree();
  } catch (error) {
    input.value = "";
    setText("login-error", error.message);
  }
});

async function loadTree() {
  setText("tree-status", "Loading…");
  const payload = await api("/api/workspace/tree");
  const container = byId("file-tree");
  container.replaceChildren();
  payload.entries.forEach((entry) => {
    const row = document.createElement(entry.type === "file" ? "button" : "div");
    row.className = `tree-entry ${entry.type}`;
    row.textContent = `${entry.type === "directory" ? "▸" : "·"} ${entry.path}`;
    if (entry.type === "file") {
      row.type = "button";
      row.addEventListener("click", () => openFile(entry.path, 1));
    }
    container.appendChild(row);
  });
  setText("tree-status", `${payload.count} entries`);
}

async function openFile(path, startLine) {
  try {
    const query = new URLSearchParams({ path, start_line: String(startLine), max_lines: "300" });
    const payload = await api(`/api/workspace/file?${query.toString()}`);
    state.filePath = payload.path;
    state.nextStartLine = payload.next_start_line;
    setText("viewer-path", `${payload.path} · lines ${payload.start_line}–${payload.end_line} of ${payload.total_lines}`);
    setText("file-content", payload.content);
    byId("next-page").disabled = payload.next_start_line == null;
  } catch (error) {
    setText("viewer-path", "Unable to read file.");
    setText("file-content", error.message);
  }
}

byId("next-page").addEventListener("click", () => {
  if (state.filePath && state.nextStartLine) {
    openFile(state.filePath, state.nextStartLine);
  }
});

function clearRunDisplay() {
  byId("trace-list").replaceChildren();
  state.eventCount = 0;
  state.lastEventId = 0;
  setText("event-count", "0 events");
  setText("final-answer", "Running…");
  ["model-calls", "tool-calls", "input-tokens", "output-tokens", "total-tokens", "changed", "failed"].forEach((name) => setText(`stat-${name}`, "0"));
  setText("stat-elapsed", "0 ms");
}

function addTraceEvent(type, payload) {
  state.eventCount += 1;
  setText("event-count", `${state.eventCount} events`);
  const item = document.createElement("li");
  item.className = "trace-event";
  const title = document.createElement("strong");
  title.textContent = type === "tool_completed" ? `${payload.step}. ${payload.tool}` : type;
  item.appendChild(title);
  if (type === "tool_completed") {
    const args = document.createElement("pre");
    args.textContent = JSON.stringify(payload.args, null, 2);
    item.appendChild(args);
    const summary = document.createElement("p");
    summary.textContent = payload.result_summary || "";
    item.appendChild(summary);
  } else if (type === "model_completed") {
    const summary = document.createElement("p");
    summary.textContent = `Model call ${payload.model_calls}; ${payload.tool_call_count} tool calls proposed.`;
    item.appendChild(summary);
  }
  byId("trace-list").appendChild(item);
}

function displayResult(payload) {
  setText("agent-status", payload.status);
  setText("final-answer", payload.answer || payload.reason || "Run finished without an answer.");
  setText("stat-model-calls", payload.model_calls || 0);
  setText("stat-tool-calls", payload.tool_calls || 0);
  const usage = payload.usage || {};
  setText("stat-input-tokens", usage.input_tokens || 0);
  setText("stat-output-tokens", usage.output_tokens || 0);
  setText("stat-total-tokens", usage.total_tokens || 0);
  setText("stat-elapsed", `${Math.round(payload.elapsed_ms || 0)} ms`);
  setText("stat-changed", payload.changed_mutations || 0);
  setText("stat-failed", payload.failed_mutations || 0);
  byId("run-button").disabled = false;
  byId("cancel-button").disabled = true;
}

function connectEvents(runId) {
  if (state.eventSource) {
    state.eventSource.close();
  }
  const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
  state.eventSource = source;
  setText("connection-status", "Streaming");
  ["run_started", "model_completed", "tool_completed"].forEach((type) => {
    source.addEventListener(type, (event) => {
      const eventId = Number(event.lastEventId || 0);
      if (eventId && eventId <= state.lastEventId) {
        return;
      }
      state.lastEventId = Math.max(state.lastEventId, eventId);
      addTraceEvent(type, JSON.parse(event.data));
    });
  });
  source.addEventListener("run_finished", async (event) => {
    const payload = JSON.parse(event.data);
    displayResult(payload);
    source.close();
    state.eventSource = null;
    state.currentRunId = null;
    setText("connection-status", "Connected");
    await loadTree();
  });
  source.onerror = () => setText("connection-status", "Reconnecting");
}

byId("run-button").addEventListener("click", async () => {
  setText("task-error", "");
  clearRunDisplay();
  try {
    const payload = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({ task: byId("task-input").value }),
    });
    state.currentRunId = payload.run_id;
    byId("run-button").disabled = true;
    byId("cancel-button").disabled = false;
    setText("agent-status", "running");
    connectEvents(payload.run_id);
  } catch (error) {
    setText("task-error", error.message);
    byId("run-button").disabled = false;
  }
});

byId("cancel-button").addEventListener("click", async () => {
  if (!state.currentRunId) {
    return;
  }
  try {
    await api(`/api/runs/${encodeURIComponent(state.currentRunId)}/cancel`, { method: "POST" });
    setText("agent-status", "cancelling");
  } catch (error) {
    setText("task-error", error.message);
  }
});

byId("reset-button").addEventListener("click", async () => {
  try {
    await api("/api/workspace/reset", { method: "POST" });
    await loadTree();
    setText("viewer-path", "Workspace reset.");
    setText("file-content", "Select a file.");
  } catch (error) {
    setText("task-error", error.message);
  }
});

byId("logout-button").addEventListener("click", async () => {
  try {
    await api("/api/logout", { method: "POST" });
  } finally {
    if (state.eventSource) {
      state.eventSource.close();
    }
    state.csrfToken = "";
    sessionStorage.removeItem("fileAgentCsrf");
    showAuthenticated(false);
  }
});

byId("refresh-tree").addEventListener("click", () => loadTree());
restoreSession();
