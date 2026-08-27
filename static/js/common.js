// Shared helpers: auth token, API calls, markdown renderer, code-block actions.
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

// Download an authenticated URL (e.g. the project zip endpoint) as a file.
async function downloadUrl(path, name) {
  const res = await fetch("/api" + path, { headers: authHeaders() });
  if (!res.ok) throw new Error("Download failed (" + res.status + ")");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name || "download";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

function escapeHtml(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function escapeAttr(s) {
  return escapeHtml(s).replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function escapeCode(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderInline(text) {
  let s = escapeHtml(text);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  s = s.replace(/~~([^~]+)~~/g, "<del>$1</del>");
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}

// Markdown -> HTML. Code fences get a header bar with language, filename
// (if the fence is tagged filename="..."), a Copy and a Download button.
function md2html(md) {
  if (!md) return "";
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  let html = "";
  let listType = null;
  let inCode = false;
  let codeBuf = [];
  let codeMeta = null;
  let inTable = false;
  let tableRows = [];

  const flushCode = () => {
    const lang = (codeMeta && codeMeta.lang) || "text";
    const file = (codeMeta && codeMeta.file) || "";
    const cls = lang.replace(/[^\w+#.+-]/g, "");
    html += '<div class="codeblock">';
    html += '<div class="codehead"><span class="codelang">' + escapeHtml(lang) + "</span>";
    if (file) html += '<span class="codefile">' + escapeHtml(file) + "</span>";
    html += '<span class="codebtns"><button type="button" class="code-copy">Copy</button>' +
            '<button type="button" class="code-dl">Download</button></span></div>';
    html += '<pre><code class="language-' + cls + '">' + escapeCode(codeBuf.join("\n")) + "</code></pre></div>\n";
    codeBuf = [];
    inCode = false;
    codeMeta = null;
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
      const fm = line.match(/^\s*```\s*([^\s]+)?(?:\s+filename\s*=\s*["']([^"']+)["'])?/);
      codeMeta = { lang: (fm && fm[1]) || "", file: (fm && fm[2]) || "" };
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

    // Standalone image (data URL or http link)
    const imgM = line.match(/^!\[([^\]]*)\]\(([^)]+)\)\s*$/);
    if (imgM) {
      const src = imgM[2];
      if (/^(data:image\/|https?:\/\/)/.test(src)) {
        flushList();
        const isData = /^data:/.test(src);
        const name = imgM[1] || (isData ? "image.png" : "image");
        html += '<div class="chat-imgbox"><img class="chat-img" src="' + escapeAttr(src) + '" alt="' + escapeAttr(name) + '">';
        if (isData) {
          html += '<a class="img-dl" href="' + escapeAttr(src) + '" download="image.png" title="Save image">⬇ Save</a>';
        }
        html += "</div>\n";
        continue;
      }
    }

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

// Highlight <pre><code> blocks with highlight.js if loaded.
function highlightBlocks(root) {
  if (!root) return;
  root.querySelectorAll("pre code:not([data-hl])").forEach(el => {
    try {
      if (window.hljs && el.className) hljs.highlightElement(el);
      else el.classList.add("plain");
    } catch (e) { el.classList.add("plain"); }
    el.setAttribute("data-hl", "1");
  });
}

function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text);
  }
  return new Promise((resolve) => {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) {}
    ta.remove();
    resolve();
  });
}

// Delegated handlers for code-block Copy / Download buttons.
document.addEventListener("click", (e) => {
  const copyBtn = e.target.closest(".code-copy");
  if (copyBtn) {
    e.preventDefault();
    const code = copyBtn.closest(".codeblock").querySelector("pre code");
    copyText(code.textContent).then(() => {
      const old = copyBtn.textContent;
      copyBtn.textContent = "Copied";
      setTimeout(() => { copyBtn.textContent = old; }, 1500);
    });
    return;
  }
  const dlBtn = e.target.closest(".code-dl");
  if (dlBtn) {
    e.preventDefault();
    const block = dlBtn.closest(".codeblock");
    const code = block.querySelector("pre code").textContent;
    const fileEl = block.querySelector(".codefile");
    const langEl = block.querySelector(".codelang");
    const lang = langEl ? langEl.textContent.trim() : "txt";
    let name = fileEl ? fileEl.textContent.trim() : "code." + (lang || "txt");
    if (!/\.\w+$/.test(name)) name = "code." + (lang || "txt");
    const blob = new Blob([code], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  }
});
