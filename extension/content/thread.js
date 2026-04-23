(function () {
  "use strict";
  console.info("[bot-detector] thread.js loaded on", location.pathname);
  const scored = new Map();
  const pending = new Map();
  let batchTimer = null;
  let panelEl = null;
  let totals = { typical: 0, possibly_suspicious: 0, suspicious: 0 };

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
    window.addEventListener("scroll", () => scanSoon(), { passive: true });
    startPollingFallback();
  }

  function cleanupPanel() {
    if (panelEl) {
      panelEl.remove();
      panelEl = null;
    }
    scored.clear();
    pending.clear();
    totals = { typical: 0, possibly_suspicious: 0, suspicious: 0 };
    window._botdetectScanLogged = false;
  }

  function resetState() {
    scored.clear();
    pending.clear();
    totals = { typical: 0, possibly_suspicious: 0, suspicious: 0 };
    window._botdetectScanLogged = false;
    document.querySelectorAll("[data-botdetect-seen]").forEach((el) => {
      delete el.dataset.botdetectSeen;
      delete el.dataset.botdetectHandle;
    });
    document.querySelectorAll(".botdetect-dot-wrap").forEach((el) => el.remove());
    document.querySelectorAll(".botdetect-highlight-suspicious, .botdetect-highlight-possibly_suspicious").forEach((el) => {
      el.classList.remove("botdetect-highlight-suspicious");
      el.classList.remove("botdetect-highlight-possibly_suspicious");
    });
    if (panelEl) updatePanel();
    scanNow();
  }

  function startPollingFallback() {
    setInterval(() => {
      if (!isThreadPage()) return;
      scanNow();
    }, 3000);
  }

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "GET_THREAD_STATS") {
      const replies = Array.from(scored.values()).map((r) => ({
        username: r.username,
        flag: r.flag,
        reasons: r.reasons,
      }));
      sendResponse({
        success: true,
        data: {
          on_thread_page: isThreadPage(),
          totals,
          replies,
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
    updatePanelStatus(`${scored.size} scanned`);
    if (cells.length > 0 && !window._botdetectScanLogged) {
      console.info(`[bot-detector] scanning ${cells.length} cells via ${selectorUsed}`);
      window._botdetectScanLogged = true;
    }
    cells.forEach((cell) => {
      const userBlockHost = cell.querySelector('[data-testid="User-Name"]') ? cell : cell.closest('[data-testid="cellInnerDiv"], article') || cell;
      extractReply(userBlockHost);
    });
    queueBatch();
    reapplyBadges();
  }

  function updatePanelStatus(text) {
    const s = document.getElementById("bd-status");
    if (s) s.textContent = text;
  }

  function extractReply(cell) {
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
      if (!userBlock.querySelector(".botdetect-dot-wrap")) {
        injectBadge(cell, scored.get(username));
      }
      return;
    }
    if (pending.has(username)) return;
    pending.set(username, { displayName, isVerified });
  }

  function queueBatch() {
    if (pending.size === 0) return;
    if (pending.size >= 20) {
      if (batchTimer) { clearTimeout(batchTimer); batchTimer = null; }
      sendBatch();
      return;
    }
    if (!batchTimer) {
      batchTimer = setTimeout(sendBatch, 2000);
    }
  }

  function sendBatch() {
    if (batchTimer) { clearTimeout(batchTimer); batchTimer = null; }
    if (pending.size === 0) return;
    const items = Array.from(pending.entries()).slice(0, 50);
    const replies = items.map(([username, info]) => ({
      username,
      display_name: info.displayName,
      is_verified: info.isVerified,
    }));

    console.info(`[bot-detector] sending batch of ${replies.length} replies`);
    chrome.runtime.sendMessage(
      { type: "PREDICT_THREAD_BATCH", replies },
      (response) => {
        items.forEach(([username]) => pending.delete(username));
        if (!response || !response.success || !response.data) {
          console.warn("[bot-detector] batch failed", response);
          updatePanel();
          if (pending.size > 0) queueBatch();
          return;
        }
        const results = response.data.results || [];
        results.forEach((r) => {
          scored.set(r.username, r);
          totals[r.flag] = (totals[r.flag] || 0) + 1;
        });
        reapplyBadges();
        updatePanel();
        if (pending.size > 0) queueBatch();
      }
    );
  }

  function reapplyBadges() {
    const userBlocks = document.querySelectorAll('[data-testid="User-Name"]');
    userBlocks.forEach((userBlock) => {
      const links = userBlock.querySelectorAll('a[role="link"]');
      let username = null;
      for (const a of links) {
        const href = a.getAttribute("href") || "";
        const m = href.match(/^\/([A-Za-z0-9_]+)$/);
        if (m) { username = m[1]; break; }
      }
      if (!username) return;
      const outerCell = userBlock.closest('[data-testid="cellInnerDiv"]') || userBlock.closest("article") || userBlock;
      outerCell.dataset.botdetectHandle = username;
      if (scored.has(username)) {
        const result = scored.get(username);
        if (!userBlock.querySelector(".botdetect-dot-wrap")) {
          const wrap = document.createElement("span");
          wrap.className = "botdetect-dot-wrap";
          const dot = document.createElement("span");
          dot.className = `botdetect-dot botdetect-dot-${result.flag}`;
          wrap.appendChild(dot);
          const card = buildHoverCard(result);
          wrap.appendChild(card);
          userBlock.appendChild(wrap);
        }
        if (result.flag !== "typical") {
          outerCell.classList.add(`botdetect-highlight-${result.flag}`);
        }
        return;
      }

      if (!pending.has(username)) {
        const displayName = (userBlock.querySelector("span")?.textContent || "").trim();
        const isVerified = !!userBlock.querySelector('svg[data-testid="icon-verified"]');
        pending.set(username, { displayName, isVerified });
      }
    });

    if (pending.size > 0) queueBatch();
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  const flagText = {
    typical: "TYPICAL",
    possibly_suspicious: "POSSIBLY SUSPICIOUS",
    suspicious: "SUSPICIOUS",
  };

  function buildHoverCard(result) {
    const flag = result.flag;
    const reasons = result.reasons || [];
    const card = document.createElement("div");
    card.className = "botdetect-hover-card";
    const reasonList = reasons.map(r => `<li>${escapeHtml(r)}</li>`).join("");
    card.innerHTML = `
      <div class="botdetect-hover-header">
        <div class="botdetect-hover-badge botdetect-hover-badge-${flag}">${flagText[flag] || "--"}</div>
      </div>
      <div class="botdetect-hover-reasons">
        <ul>${reasonList}</ul>
      </div>
      <div class="botdetect-hover-footer">
        quick heuristic scan. click profile for a full score.
      </div>
    `;
    return card;
  }

  function injectBadge(cell, result) {
    const userBlock = cell.querySelector('[data-testid="User-Name"]') || cell;
    if (!userBlock) return;
    if (userBlock.querySelector(".botdetect-dot-wrap")) return;
    const wrap = document.createElement("span");
    wrap.className = "botdetect-dot-wrap";
    const dot = document.createElement("span");
    dot.className = `botdetect-dot botdetect-dot-${result.flag}`;
    wrap.appendChild(dot);
    const card = buildHoverCard(result);
    wrap.appendChild(card);
    userBlock.appendChild(wrap);

    if (result.flag !== "typical") {
      const outerCell = cell.closest('[data-testid="cellInnerDiv"]') || cell.closest("article") || cell;
      outerCell.classList.add(`botdetect-highlight-${result.flag}`);
    }
  }

  function ensurePanel() {
    if (panelEl) return;
    panelEl = document.createElement("div");
    panelEl.id = "botdetect-thread-panel";
    panelEl.innerHTML = `
      <div class="botdetect-panel-title">Thread Scan
        <span id="bd-status" class="bd-status">scanning...</span>
      </div>
      <div class="botdetect-panel-stats">
        <button class="bd-stat bd-typical" data-flag="typical">
          <b id="bd-count-typical">0</b>
          <span id="bd-label-typical">typical</span>
        </button>
        <button class="bd-stat bd-possibly" data-flag="possibly_suspicious">
          <b id="bd-count-possibly">0</b>
          <span id="bd-label-possibly">possibly suspicious</span>
        </button>
        <button class="bd-stat bd-suspicious" data-flag="suspicious">
          <b id="bd-count-suspicious">0</b>
          <span id="bd-label-suspicious">suspicious</span>
        </button>
      </div>
      <button class="bd-report-btn" id="bd-report-btn">View full report</button>
      <div class="botdetect-panel-note">click a count to jump through flagged replies. hover any dot for details.</div>
    `;
    document.body.appendChild(panelEl);

    panelEl.querySelectorAll(".bd-stat").forEach((btn) => {
      btn.addEventListener("click", () => cycleFlag(btn.dataset.flag));
    });

    document.getElementById("bd-report-btn").addEventListener("click", openReport);

    console.info("[bot-detector] sticky panel injected");
  }

  const cycleIndex = { typical: -1, possibly_suspicious: -1, suspicious: -1 };

  function cycleFlag(flag) {
    const cells = Array.from(document.querySelectorAll(`[data-botdetect-handle]`)).filter((cell) => {
      const handle = cell.dataset.botdetectHandle;
      const result = scored.get(handle);
      return result && result.flag === flag;
    });

    if (cells.length === 0) {
      const totalFlagged = totals[flag] || 0;
      if (totalFlagged === 0) {
        console.info(`[bot-detector] no ${flag} cells found`);
        return;
      }
      updatePanelStatus(`scroll to load more replies`);
      window.scrollBy({ top: window.innerHeight * 0.8, behavior: "smooth" });
      setTimeout(() => {
        const total = totals.typical + totals.possibly_suspicious + totals.suspicious;
        updatePanelStatus(`${total} scanned`);
      }, 1500);
      return;
    }

    cycleIndex[flag] = (cycleIndex[flag] + 1) % cells.length;
    const idx = cycleIndex[flag];
    const target = cells[idx];
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    target.classList.add("botdetect-flash");
    setTimeout(() => target.classList.remove("botdetect-flash"), 1200);
    updateCycleIndicator(flag, idx + 1, cells.length);
  }

  function updateCycleIndicator(flag, current, total) {
    const btnMap = {
      typical: "bd-count-typical",
      possibly_suspicious: "bd-count-possibly",
      suspicious: "bd-count-suspicious",
    };
    const labelMap = {
      typical: "bd-label-typical",
      possibly_suspicious: "bd-label-possibly",
      suspicious: "bd-label-suspicious",
    };
    const countEl = document.getElementById(btnMap[flag]);
    const labelEl = document.getElementById(labelMap[flag]);
    if (!countEl || !labelEl) return;
    countEl.textContent = current;
    labelEl.textContent = `of ${total}`;
    setTimeout(() => {
      if (totals[flag] === total) {
        countEl.textContent = total;
        labelEl.textContent = flag.replace("_", " ");
      }
    }, 3000);
  }

  function openReport() {
    const data = {
      threadUrl: location.href,
      timestamp: Date.now(),
      totals,
      replies: Array.from(scored.values()),
    };
    const url = chrome.runtime.getURL("report/report.html") + "?data=" + encodeURIComponent(JSON.stringify(data));
    window.open(url, "_blank");
  }

  function updatePanel() {
    const t = document.getElementById("bd-count-typical");
    const p = document.getElementById("bd-count-possibly");
    const s = document.getElementById("bd-count-suspicious");
    if (t) t.textContent = totals.typical;
    if (p) p.textContent = totals.possibly_suspicious;
    if (s) s.textContent = totals.suspicious;
    const total = totals.typical + totals.possibly_suspicious + totals.suspicious;
    if (total > 0) updatePanelStatus(`${total} scanned`);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();