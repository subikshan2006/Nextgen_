"""Free, no-API-key web search for the chat.

Engine chain, in order:
  1. Bing HTML (general web results — direct LinkedIn/Indeed/etc. links)
  2. Google News RSS (reliable from serverless IPs, good for news queries)
  3. Wikipedia API (encyclopedic fallback)

Dependency-free: stdlib urllib + xml + regex only. Every function fails soft
and returns [] / False on any error.
"""
import base64
import html as _html
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as _ET
from typing import Optional

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_BING_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Phrases that hint the user wants live web info / links.
_SEARCH_PHRASES = (
    "search", "google", "web search", "look up", "look it up", "find online",
    "search the web", "search online", "current", "latest", "news", "today",
    "recent", "live", "breaking", "up-to-date", "who won", "weather", "price",
    "stock", "score", "standings", "facts", "update on", "what is the",
    "how much", "as of", "give link", "give links", "give me the link",
    "give me links", "give me link", "give me the links", "show me the link",
    "show me links", "job application", "job applications", "jobs", "vacancies",
    "openings", "hiring", "careers", "in online", "online jobs",
)

# Short words matched on word boundaries (substring matching would misfire on
# words like "linkedin" or "jobber").
_SEARCH_WORDS = ("job", "link", "links", "source", "site:")

# Leading conversational phrases stripped before searching ("search for X" -> "X").
_STRIP_LEAD = re.compile(
    r"^\s*(?:(?:please\s+|can you\s+|could you\s+|kindly\s+|"
    r"i want you to\s+|i need you to\s+|i'd like you to\s+|would you\s+)*"
    r"(?:search\s+for|search\s+the\s+web|search\s+online|search\s+google|"
    r"search|google\s+it|google|look\s+it\s+up|look\s+up|look|find\s+me|"
    r"find\s+out|find\s+online|find|web\s+search|browse|research|show\s+me|"
    r"give\s+me|tell\s+me(?:\s+about)?|display|list)\s*)"
    r"(?:for\s+|the\s+|on\s+|about\s+|up\s+)?"
    r"[:,\-]?\s*",
    re.I,
)
_STRIP_TRAIL = re.compile(
    r"(?:\s*(?:and\s+give\s+(?:me\s+)?(?:the\s+)?(?:links?|sources)\s*|"
    r"with\s+(?:links?|sources)\s*|give\s+me\s+(?:the\s+)?(?:links?|sources)\s*|"
    r"(?:links?|sources)\s+please|and\s+link\s*|in\s+online\s*)\s*[.?!]*\s*)$",
    re.I,
)
_TRACKING_RE = re.compile(r"[&?]((utm|fbclid|gclid|ref|source|sr_)=[^&]+)")

# Job-query cleanup: turn "show me job applications for the role software
# engineering in online and give link" into "software engineering jobs".
_JOB_LIST_WORDS = {
    "applications", "application", "vacancies", "vacancy", "openings",
    "hiring", "postings", "careers", "positions", "roles",
}
_JOB_STOPWORDS = {
    "me", "for", "the", "role", "and", "give", "link", "links", "in",
    "online", "of", "a", "an", "please", "show", "find", "list", "some",
    "your", "with", "sources", "source", "need", "want", "to", "about",
    "is", "are", "all", "any", "best", "top",
}


def _looks_job(q: str) -> bool:
    low = (q or "").lower()
    return any(
        w in low for w in ("job", "jobs", "vacanc", "application", "opening",
                           "hiring", "postings", "careers", "positions", "roles")
    ) or " in online" in low


# Field -> role title normalization so job queries match how job boards word
# titles ("software engineering jobs" -> "software engineer jobs", which Bing
# ranks far better).
_JOB_FIELD_TO_ROLE = {
    "engineering": "engineer",
    "development": "developer",
    "programming": "developer",
    "designing": "designer",
    "design": "designer",
    "analysis": "analyst",
    "analytics": "analyst",
    "management": "manager",
    "administration": "administrator",
    "consulting": "consultant",
    "marketing": "marketer",
    "accounting": "accountant",
}


def _clean_job_query(q: str) -> str:
    toks = re.findall(r"[a-z0-9+#.\-]+", q.lower())
    kept = [t for t in toks if t not in _JOB_STOPWORDS and len(t) > 1]
    seen, uniq = set(), []
    for t in kept:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    has_list = any(t in _JOB_LIST_WORDS for t in uniq)
    has_job = "job" in uniq or "jobs" in uniq
    if has_list:
        uniq = [t for t in uniq if t not in _JOB_LIST_WORDS and t not in ("job", "jobs")]
        uniq.append("jobs")
    elif not has_job:
        uniq.append("jobs")
    uniq = [_JOB_FIELD_TO_ROLE.get(t, t) for t in uniq]
    return " ".join(uniq) if uniq else q


def clean_query(text: str) -> str:
    """Strip conversational intent prefixes so the engine gets a clean query."""
    q = (text or "").strip()
    q = q.split("\n", 1)[0]
    q = _STRIP_TRAIL.sub("", q)
    q = _STRIP_LEAD.sub("", q)
    q = q.strip()
    if not q:
        return (text or "").strip()
    if _looks_job(q):
        q = _clean_job_query(q)
    return q


def strip_tracking(url: str) -> str:
    return _TRACKING_RE.sub("", url).rstrip("&?")


# Job-site keywords: when the query looks like a job search, tell the model
# that some boards (LinkedIn) require sign-in.
_JOB_TERMS = ("jobs", "job ", "vacanc", "opening", "hiring", "careers", "employment", "work at")


def _is_job_query(q: str) -> bool:
    low = (q or "").lower()
    return any(t in low for t in _JOB_TERMS)


def canonicalize_job_link(url: str) -> str:
    """Rewrite job-board search slugs into search-result URLs that actually
    open a results page in a browser (instead of the site home page)."""
    low = (url or "").lower()
    try:
        if "indeed.com" in low and ("/q-" in low or "/in-" in low):
            m = re.search(r"indeed\.com/(?:q-|in-)([^?#]+)", low)
            if m:
                body = m.group(1).split(".html", 1)[0]
                parts = body.split("-l-")
                query = parts[0].replace("-jobs", "").replace("-", " ").strip()
                if query:
                    out = "https://www.indeed.com/jobs?q=" + urllib.parse.quote(query)
                    if len(parts) > 1:
                        loc = parts[1].replace("-jobs", "").replace("-", " ").strip()
                        if loc:
                            out += "&l=" + urllib.parse.quote(loc)
                    return out
        elif "linkedin.com" in low and "/jobs/" in low:
            m = re.search(r"linkedin\.com/jobs/([^?#]+)", low)
            if m:
                slug = m.group(1).strip("/")
                mq = re.search(r"(.+?)-jobs(?:-|$)", slug)
                query = (mq.group(1).replace("-", " ") if mq else slug.replace("-", " ")).strip().rstrip(".")
                if query:
                    out = "https://www.linkedin.com/jobs/search?keywords=" + urllib.parse.quote(query)
                    if mq and mq.group(0).rstrip("-") != mq.group(1) and "-jobs-" in slug:
                        loc = slug.split("-jobs-", 1)[1].replace("-", " ").strip()
                        if loc:
                            out += "&location=" + urllib.parse.quote(loc)
                    return out
        elif "glassdoor.com" in low and "/job/" in low:
            m = re.search(r"glassdoor\.com/job/([^?#]+)-jobs-srch", low)
            if m:
                kw = m.group(1).replace("-", " ").strip()
                if kw:
                    return "https://www.glassdoor.com/Job/jobs.htm?sc.keyword=" + urllib.parse.quote(kw)
    except Exception:
        pass
    return url


def _clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", _html.unescape(s)).strip()


def _open(
    url: str,
    data: Optional[bytes] = None,
    timeout: int = 12,
    headers: Optional[dict] = None,
) -> str:
    hdrs = {"User-Agent": _USER_AGENT}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _unwrap_bing_url(href: str) -> str:
    """Turn a bing.com/ck/a redirect into its real destination when possible."""
    href = _html.unescape(href)
    if "bing.com/ck/a" not in href:
        return strip_tracking(href)
    m = re.search(r"[?&]u=([^&]+)", href)
    if not m:
        return href  # p= format is encrypted; link still works in a browser
    try:
        b64 = urllib.parse.unquote(m.group(1))
        if b64.startswith("a1"):
            b64 = b64[2:]
        b64 += "=" * (-len(b64) % 4)
        decoded = base64.urlsafe_b64decode(b64).decode("utf-8", "replace")
        return strip_tracking(decoded.split("&p=")[0])
    except Exception:
        return href


def _bing(q: str, timeout: int = 12) -> list:
    url = (
        "https://www.bing.com/search?q=" + urllib.parse.quote(q)
        + "&count=10&setlang=en"
    )
    text = _open(url, timeout=timeout, headers=_BING_HEADERS)
    out = []
    for block in re.split(r'<li class="b_algo"', text)[1:]:
        m = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>([\s\S]*?)</a>', block)
        if not m:
            continue
        title = _clean(m.group(2))
        url = _unwrap_bing_url(m.group(1))
        if not title or not url:
            continue
        snippet = ""
        pm = re.search(r"<p[^>]*>([\s\S]*?)</p>", block)
        if pm:
            snippet = _clean(pm.group(1))
        out.append({"title": title, "url": url, "snippet": snippet})
        if len(out) >= 10:
            break
    return out


def _google_news_rss(q: str, timeout: int = 12) -> list:
    """Google News RSS search — no key, real titles + working links."""
    url = (
        "https://news.google.com/rss/search?q=%s&hl=en-US&gl=US&ceid=US:en"
        % urllib.parse.quote(q)
    )
    text = _open(url, timeout=timeout)
    if "<rss" not in text[:500]:
        return []
    root = _ET.fromstring(text)
    out = []
    for item in root.iter("item"):
        title, link = "", ""
        for ch in item:
            tag = ch.tag.split("}")[-1]
            if tag == "title":
                title = _clean(ch.text)
            elif tag == "link":
                link = (ch.text or "").strip()
        if not title or not link:
            continue
        out.append({"title": title, "url": link, "snippet": ""})
        if len(out) >= 10:
            break
    return out


def _wikipedia(q: str, timeout: int = 12) -> list:
    api = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        "&list=search&srlimit=6&srsearch=" + urllib.parse.quote(q)
    )
    data = json.loads(_open(api, timeout=timeout))
    out = []
    for it in data.get("query", {}).get("search", []):
        title = it.get("title", "")
        if not title:
            continue
        out.append({
            "title": title,
            "url": "https://en.wikipedia.org/wiki/"
                   + urllib.parse.quote(title.replace(" ", "_")),
            "snippet": _clean(it.get("snippet", "")),
        })
    return out


def search_web(query: str, max_results: int = 6, timeout: int = 12) -> list:
    """Return up to max_results [{title, url, snippet}] or [] on failure."""
    q = clean_query(query)
    if not q:
        return []
    results = []
    try:
        results = _bing(q, timeout)
    except Exception:
        results = []
    if len(results) < 3:
        try:
            results = _google_news_rss(q, timeout)
        except Exception:
            pass
    if len(results) < 3:
        try:
            results = _wikipedia(q, timeout)
        except Exception:
            pass
    seen, out = set(), []
    for r in results:
        url = canonicalize_job_link(r.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({"title": r.get("title") or "", "url": url, "snippet": r.get("snippet") or ""})
        if len(out) >= max_results:
            break
    return out


def should_search(text: str) -> bool:
    low = (text or "").lower()
    if any(p in low for p in _SEARCH_PHRASES):
        return True
    return any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in _SEARCH_WORDS)


def format_search_context(query: str, results: list) -> str:
    """Build the block injected into the worker prompt when web results exist."""
    if not results:
        return ""
    lines = ['[WEB SEARCH RESULTS for "%s"]' % query]
    for i, r in enumerate(results, 1):
        lines.append("%d. [%s](%s)" % (i, r.get("title") or r.get("url"), r.get("url")))
        if r.get("snippet"):
            lines.append("   %s" % r["snippet"])
    lines.append(
        "\nYou were given optional web search results. Use them only if they are "
        "relevant to the user's question. If they are relevant, answer from them "
        "and cite each source as a markdown link with its full URL "
        "([title](url)) next to the claim it supports. If they are not relevant "
        "to the question, answer normally without forcing them in. Never invent "
        "facts or URLs."
    )
    if _is_job_query(query):
        lines.append(
            "For job searches, list the relevant job-board links clearly. "
            "Note: LinkedIn requires you to be signed in to see its job search "
            "results, while Indeed and Glassdoor open without signing in."
        )
    return "\n".join(lines)
