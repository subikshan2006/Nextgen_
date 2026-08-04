// Shared helpers: auth token, API calls, markdown renderer.
const TOKEN_KEY = "nextgen_token";

function getToken() { return localStorage.getItem(TOKEN_KEY); }
function setToken(t) { localStorage.setItem(TOKEN_KEY, t); }
function clearToken() { localStorage.removeItem(TOKEN_KEY); }

function getCurrentUser() {
  try { return JSON.parse(localStorage.getItem("nextgen_user")); }
  catch { return null; }
}

function setCurrentUser(u) { localStorage.setItem("nextgen_user", JSON.stringify(u)); }

function authHeaders() {
  return { "Content-Type": "application/json", Authorization: "Bearer " + getToken() };
}

async function api(path, opts = {}) {
  const res = await fetch("/api" + path, {
    headers: opts.headers || authHeaders(),
    ...opts,
  });
  if (res.status === 401) {
    clearToken();
    window.location.href = "/login";
    throw new Error("Not authenticated");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : (data.message || res.statusText);
    throw new Error(detail || "Request failed");
  }
  return data;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Minimal-but-solid Markdown renderer (headings, bold, italic, code, code blocks,
// fenced blocks, links, lists, blockquote, tables, hr).
function escapeCode(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderInline(text) {
  let s = escapeHtml(text);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}

function md2html(md) {
  if (!md) return "";
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  let html = "";
  let listType = null;
  let inCode = false;
  let codeBuf = [];
  let codeLang = "";
  let inTable = false;
  let tableRows = [];

  const flushCode = () => {
    html += "<pre><code>" + escapeCode(codeBuf.join("\n")) + "</code></pre>\n";
    codeBuf = [];
    inCode = false;
  };
  const flushList = () => { if (listType) { html += "</" + listType + ">\n"; listType = null; } };
  const flushTable = () => {
    if (!inTable) return;
    html += "<table><thead><tr>" + tableRows[0].map(c => "<th>" + renderInline(c) + "</th>").join("") + "</tr></thead><tbody>";
    for (let i = 2; i < tableRows.length; i++) {
      if (tableRows[i].some(c => /^:?-{3,}:?$/.test(c.trim()))) continue;
      html += "<tr>" + tableRows[i].map(c => "<td>" + renderInline(c) + "</td>").join("") + "</tr>";
    }
    html += "</tbody></table>\n";
    inTable = false; tableRows = [];
  };

  for (const raw of lines) {
    const line = raw.trimEnd();

    if (inCode) {
      if (/^\s*```/.test(line)) { flushCode(); continue; }
      codeBuf.push(line);
      continue;
    }
    if (/^\s*```/.test(line)) {
      flushList(); flushTable();
      codeLang = line.replace(/^\s*```\s*/, "").trim();
      inCode = true;
      continue;
    }
    if (inTable) {
      if (line.trim() === "") { flushTable(); continue; }
      if (!line.includes("|")) { flushTable(); }
      else { tableRows.push(line.split("|").slice(1, -1).map(s => s.trim())); continue; }
    }
    if (line.includes("|") && lines.indexOf(raw) + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[lines.indexOf(raw) + 1])) {
      flushList();
      tableRows = [line.split("|").slice(1, -1).map(s => s.trim())];
      inTable = true;
      continue;
    }
    if (line.trim() === "") { flushList(); flushTable(); continue; }
    flushTable();

    if (/^\s*#{1,6}\s/.test(line)) {
      flushList();
      const lvl = line.match(/^(\s*#{1,6})\s/)[1].trim().length;
      html += "<h" + lvl + ">" + renderInline(line.replace(/^\s*#{1,6}\s*/, "")) + "</h" + lvl + ">\n";
    } else if (/^\s*[-*+]\s/.test(line)) {
      if (listType !== "ul") { flushList(); html += "<ul>\n"; listType = "ul"; }
      html += "<li>" + renderInline(line.replace(/^\s*[-*+]\s*/, "")) + "</li>\n";
    } else if (/^\s*\d+\.\s/.test(line)) {
      if (listType !== "ol") { flushList(); html += "<ol>\n"; listType = "ol"; }
      html += "<li>" + renderInline(line.replace(/^\s*\d+\.\s*/, "")) + "</li>\n";
    } else if (/^\s*&gt;\s/.test(line) || /^\s*>\s/.test(line)) {
      flushList();
      html += "<blockquote>" + renderInline(line.replace(/^\s*&gt;\s*/, "").replace(/^\s*>\s*/, "")) + "</blockquote>\n";
    } else if (/^\s*(-{3,}|\*{3,})$/.test(line)) {
      flushList();
      html += "<hr>\n";
    } else {
      flushList();
      html += "<p>" + renderInline(line) + "</p>\n";
    }
  }
  if (inCode) flushCode();
  flushList();
  flushTable();
  return html;
}
