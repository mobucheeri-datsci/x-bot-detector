(function () {
  "use strict";

  console.info("[bot-detector] thread.js loaded on", location.pathname);

  const scored = new Map();
  const pending = new Map();
  let batchTimer = null;
  let panelEl = null;
  let lastClusters = [];
  let totals = { human: 0, uncertain: 0, bot: 0 };

  function isThreadPage() {
    return /^\/[A-Za-z0-9_]+\/status\/\d+/.test(location.pathname);
  }

  let lastSeenUrl = location.href;

  function init() {
    if (!isThreadPage()) {
      console.info("[bot-detector] not a thread page, idling");
      return;
    }
    console.info("[bot-detector] init on thread page");
    ensurePanel();
    scanNow();
    const observer = new MutationObserver(() => {
      if (location.href !== lastSeenUrl) {
        lastSeenUrl = location.href;
        if (!isThreadPage()) {
          console.info("[bot-detector] navigated away from thread, cleaning up");
          cleanupPanel();
          return;
        }
        console.info("[bot-detector] navigated to a different thread, resetting state");
        resetState();
      }
      scanSoon();
    });
    observer.observe(document.body, { childList: true, subtree: true });
    startPollingFallback();
  }

  function cleanupPanel() {
    if (panelEl) {
      panelEl.remove();
      panelEl = null;
    }
    scored.clear();
    pending.clear();
    lastClusters = [];
    totals = { human: 0, uncertain: 0, bot: 0 };
    window._botdetectScanLogged = false;
  }

  function resetState() {
    scored.clear();
    pending.clear();
    lastClusters = [];
    totals = { human: 0, uncertain: 0, bot: 0 };
    window._botdetectScanLogged = false;
    document.querySelectorAll("[data-botdetect-seen]").forEach((el) => {
      delete el.dataset.botdetectSeen;
      delete el.dataset.botdetectHandle;
    });
    document.querySelectorAll(".botdetect-dot-wrap").forEach((el) => el.remove());
    if (panelEl) {
      updatePanel();
      const container = document.getElementById("botdetect-cib-container");
      if (container) container.innerHTML = "";
    }
    scanNow();
  }

  let pollAttempts = 0;
  function startPollingFallback() {
    const pollId = setInterval(() => {
      pollAttempts++;
      if (!isThreadPage() || pollAttempts >= 30) {
        clearInterval(pollId);
        console.info(`[bot-detector] polling stopped after ${pollAttempts} attempts`);
        return;
      }
      if (scored.size > 0) {
        clearInterval(pollId);
        console.info(`[bot-detector] polling stopped, ${scored.size} replies scored`);
        return;
      }
      console.info(`[bot-detector] poll attempt ${pollAttempts}, scanning again`);
      scanNow();
    }, 2000);
  }

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "GET_THREAD_STATS") {
      const replies = Array.from(scored.values()).map((r) => ({
        username: r.username,
        label: r.label,
        bot_score: r.bot_score,
        override_applied: r.override_applied,
      }));
      sendResponse({
        success: true,
        data: {
          on_thread_page: isThreadPage(),
          totals,
          replies,
          clusters: lastClusters,
        },
      });
      return false;
    }
  });

  let scanTimer = null;
  function scanSoon() {
    if (scanTimer) return;
    scanTimer = setTimeout(() => {
      scanTimer = null;
      if (!isThreadPage()) return;
      scanNow();
    }, 500);
  }

  function scanNow() {
    let cells = document.querySelectorAll('[data-testid="cellInnerDiv"]');
    let selectorUsed = "cellInnerDiv";
    if (cells.length === 0) {
      cells = document.querySelectorAll('article[data-testid="tweet"]');
      selectorUsed = "article[tweet]";
    }
    if (cells.length === 0) {
      cells = document.querySelectorAll('[data-testid="User-Name"]');
      selectorUsed = "User-Name (fallback)";
    }
    updatePanelStatus(`scanning... ${cells.length} cells (${selectorUsed})`);
    if (cells.length > 0 && !window._botdetectScanLogged) {
      console.info(`[bot-detector] scanning ${cells.length} cells via ${selectorUsed}`);
      window._botdetectScanLogged = true;
    }
    cells.forEach((cell) => {
      const userBlockHost = cell.querySelector('[data-testid="User-Name"]') ? cell : cell.closest('[data-testid="cellInnerDiv"], article') || cell;
      extractReply(userBlockHost);
    });
    queueBatch();
  }

  function updatePanelStatus(text) {
    const s = document.getElementById("bd-status");
    if (s) s.textContent = text;
  }

  function extractReply(cell) {
    if (cell.dataset.botdetectSeen) return;
    const userBlock = cell.querySelector('[data-testid="User-Name"]');
    if (!userBlock) return;

    const links = userBlock.querySelectorAll('a[role="link"]');
    let username = null;
    for (const a of links) {
      const href = a.getAttribute("href") || "";
      const m = href.match(/^\/([A-Za-z0-9_]+)$/);
      if (m) { username = m[1]; break; }
    }
    if (!username) return;

    const displayName = (userBlock.querySelector("span")?.textContent || "").trim();
    const isVerified = !!userBlock.querySelector('svg[data-testid="icon-verified"]');

    cell.dataset.botdetectSeen = "1";
    cell.dataset.botdetectHandle = username;

    if (scored.has(username)) {
      injectBadge(cell, scored.get(username));
      return;
    }
    if (pending.has(username)) return;
    pending.set(username, { displayName, isVerified });
  }

  function queueBatch() {
    if (pending.size === 0) return;
    if (batchTimer) clearTimeout(batchTimer);
    if (pending.size >= 20) {
      sendBatch();
      return;
    }
    batchTimer = setTimeout(sendBatch, 2000);
  }

  function sendBatch() {
    if (batchTimer) { clearTimeout(batchTimer); batchTimer = null; }
    if (pending.size === 0) return;

    const items = Array.from(pending.entries()).slice(0, 50);
    const profiles = items.map(([username, info]) => ({
      username,
      display_name: info.displayName,
      bio: "",
      followers_count: 0,
      following_count: 0,
      tweet_count: 0,
      account_age_days: 365,
      is_verified: info.isVerified,
      has_default_avatar: false,
    }));

    console.info(`[bot-detector] sending batch of ${profiles.length} profiles`);
    chrome.runtime.sendMessage(
      { type: "PREDICT_BATCH", profiles },
      (response) => {
        items.forEach(([username]) => pending.delete(username));
        if (!response || !response.success || !response.data) {
          console.warn("[bot-detector] batch failed", response);
          updatePanel();
          if (pending.size > 0) queueBatch();
          return;
        }
        const results = response.data.results || [];
        console.info(`[bot-detector] batch returned ${results.length} results`);
        results.forEach((r) => {
          scored.set(r.username, r);
          totals[r.label] = (totals[r.label] || 0) + 1;
          const cell = document.querySelector(`[data-botdetect-handle="${r.username}"]`);
          if (cell) injectBadge(cell, r);
        });
        updatePanel();
        runCIB();
        if (pending.size > 0) queueBatch();
      }
    );
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  const labelText = { human: "HUMAN", uncertain: "UNCERTAIN", bot: "BOT" };

  function buildHoverCard(result) {
    const score = result.bot_score;
    const label = result.label;
    const threshold = typeof result.threshold === "number" ? result.threshold : 0.58;
    const margin = typeof result.margin === "number" ? result.margin : 0.1;
    const humanEnd = Math.max(0, Math.min(100, (threshold - margin) * 100));
    const botStart = Math.max(0, Math.min(100, (threshold + margin) * 100));
    const markerLeft = Math.max(0, Math.min(100, score));

    const contribs = result.contributions || { toward_bot: [], toward_human: [] };
    const bot = (contribs.toward_bot || []).slice(0, 3);
    const hum = (contribs.toward_human || []).slice(0, 3);
    const max = Math.max(...[...bot, ...hum].map((e) => Math.abs(e.contribution)), 0.5);

    function rowsHtml(entries, direction) {
      if (!entries.length) return `<div class="botdetect-hover-empty">no strong signals</div>`;
      return entries.map((e) => {
        const pct = Math.round((Math.abs(e.contribution) / max) * 100);
        const sign = e.contribution >= 0 ? "+" : "";
        return `
          <div class="botdetect-hover-row">
            <div class="botdetect-hover-rowtext">
              <span class="botdetect-hover-desc">${escapeHtml(e.description || e.feature)}</span>
              <span class="botdetect-hover-mag">${sign}${e.contribution.toFixed(2)}</span>
            </div>
            <div class="botdetect-hover-track"><div class="botdetect-hover-fill botdetect-hover-fill-${direction}" style="width:${pct}%"></div></div>
          </div>
        `;
      }).join("");
    }

    const card = document.createElement("div");
    card.className = "botdetect-hover-card";
    card.innerHTML = `
      <div class="botdetect-hover-header">
        <div class="botdetect-hover-score botdetect-hover-score-${label}">${score}<span class="botdetect-hover-of">/100</span></div>
        <div class="botdetect-hover-badge botdetect-hover-badge-${label}">${labelText[label] || "--"}</div>
      </div>
      <div class="botdetect-hover-legend">
        <div class="botdetect-hover-legend-zone botdetect-hover-legend-human" style="width:${humanEnd}%"></div>
        <div class="botdetect-hover-legend-zone botdetect-hover-legend-uncertain" style="width:${botStart - humanEnd}%"></div>
        <div class="botdetect-hover-legend-zone botdetect-hover-legend-bot" style="width:${100 - botStart}%"></div>
        <div class="botdetect-hover-legend-marker" style="left:${markerLeft}%"></div>
      </div>
      <div class="botdetect-hover-coltitle botdetect-hover-coltitle-bot">toward bot</div>
      ${rowsHtml(bot, "bot")}
      <div class="botdetect-hover-coltitle botdetect-hover-coltitle-human">toward human</div>
      ${rowsHtml(hum, "human")}
    `;
    return card;
  }

  function injectBadge(cell, result) {
    const userBlock = cell.querySelector('[data-testid="User-Name"]');
    if (!userBlock) return;
    if (userBlock.querySelector(".botdetect-dot-wrap")) return;

    const wrap = document.createElement("span");
    wrap.className = "botdetect-dot-wrap";

    const dot = document.createElement("span");
    dot.className = `botdetect-dot botdetect-dot-${result.label}`;
    wrap.appendChild(dot);

    const card = buildHoverCard(result);
    wrap.appendChild(card);

    userBlock.appendChild(wrap);
  }

  function ensurePanel() {
    if (panelEl) return;
    panelEl = document.createElement("div");
    panelEl.id = "botdetect-thread-panel";
    panelEl.innerHTML = `
      <div class="botdetect-panel-title">Bot Detector
        <span id="bd-status" class="bd-status">scanning...</span>
      </div>
      <div class="botdetect-panel-stats">
        <span class="bd-stat bd-human"><b id="bd-count-human">0</b> human</span>
        <span class="bd-stat bd-uncertain"><b id="bd-count-uncertain">0</b> unclear</span>
        <span class="bd-stat bd-bot"><b id="bd-count-bot">0</b> bot</span>
      </div>
      <div class="botdetect-panel-note">replies are scored from visible metadata only. click any profile for full scoring.</div>
      <div id="botdetect-cib-container"></div>
    `;
    document.body.appendChild(panelEl);
    console.info("[bot-detector] sticky panel injected");
  }

  function updatePanel() {
    const h = document.getElementById("bd-count-human");
    const u = document.getElementById("bd-count-uncertain");
    const b = document.getElementById("bd-count-bot");
    if (h) h.textContent = totals.human;
    if (u) u.textContent = totals.uncertain;
    if (b) b.textContent = totals.bot;
    const total = totals.human + totals.uncertain + totals.bot;
    if (total > 0) updatePanelStatus(`${total} scored`);
  }

  function tokenize(s) {
    return new Set((s || "").toLowerCase().split(/\W+/).filter(t => t.length >= 3));
  }

  function jaccard(a, b) {
    if (a.size === 0 && b.size === 0) return 0;
    let inter = 0;
    a.forEach(t => { if (b.has(t)) inter++; });
    return inter / (a.size + b.size - inter);
  }

  function runCIB() {
    if (scored.size < 4) return;
    const entries = Array.from(scored.values());
    const features = entries.map(e => {
      const handle = e.username || "";
      const digits = (handle.match(/\d/g) || []).length;
      return {
        username: handle,
        label: e.label,
        score: e.bot_score,
        digit_ratio: digits / Math.max(handle.length, 1),
        ends_in_digits: /\d{3,}$/.test(handle) ? 1 : 0,
        handle_len: handle.length,
        tokens: tokenize(handle),
      };
    });

    const used = new Set();
    const clusters = [];
    for (let i = 0; i < features.length; i++) {
      if (used.has(i)) continue;
      const cluster = [features[i]];
      used.add(i);
      for (let j = i + 1; j < features.length; j++) {
        if (used.has(j)) continue;
        const a = features[i];
        const b = features[j];
        const sim =
          (Math.abs(a.handle_len - b.handle_len) <= 2 ? 0.25 : 0) +
          (a.ends_in_digits === b.ends_in_digits && a.ends_in_digits === 1 ? 0.30 : 0) +
          (Math.abs(a.digit_ratio - b.digit_ratio) < 0.15 ? 0.20 : 0) +
          jaccard(a.tokens, b.tokens) * 0.30;
        if (sim >= 0.55) {
          cluster.push(b);
          used.add(j);
        }
      }
      if (cluster.length >= 3) clusters.push(cluster);
    }

    const suspicious = clusters.filter(cluster => {
      const allEndDigits = cluster.every(c => c.ends_in_digits === 1);
      const labelOK = cluster.every(c => c.label === "bot" || c.label === "uncertain");
      const sharedDigitRatio = cluster.every(c => c.digit_ratio > 0.2);
      let signals = 0;
      if (allEndDigits) signals++;
      if (labelOK) signals++;
      if (sharedDigitRatio) signals++;
      return signals >= 2;
    });

    lastClusters = suspicious.map((c) => ({
      size: c.length,
      handles: c.map((m) => m.username),
      median_score: Math.round(c.reduce((s, m) => s + m.score, 0) / c.length),
    }));
    renderCIB(suspicious);
  }

  function renderCIB(clusters) {
    const container = document.getElementById("botdetect-cib-container");
    if (!container) return;
    container.innerHTML = "";
    if (clusters.length === 0) return;

    const header = document.createElement("div");
    header.className = "botdetect-cib-header";
    header.textContent = `${clusters.length} suspicious cluster${clusters.length > 1 ? "s" : ""} detected`;
    container.appendChild(header);

    clusters.slice(0, 3).forEach(cluster => {
      const banner = document.createElement("div");
      banner.className = "botdetect-cib-banner";
      const handles = cluster.slice(0, 4).map(c => `@${c.username}`).join(", ");
      const more = cluster.length > 4 ? ` +${cluster.length - 4} more` : "";
      const reasons = [];
      if (cluster.every(c => c.ends_in_digits === 1)) reasons.push("handles end in digits");
      if (cluster.every(c => c.label === "bot" || c.label === "uncertain")) reasons.push("none scored as human");
      banner.innerHTML = `
        <div class="botdetect-cib-title">cluster of ${cluster.length}</div>
        <div class="botdetect-cib-handles">${handles}${more}</div>
        <div class="botdetect-cib-reason">${reasons.join(" \u00b7 ")}</div>
      `;
      container.appendChild(banner);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
