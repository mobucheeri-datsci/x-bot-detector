document.addEventListener("DOMContentLoaded", () => {
  const notTwitter = document.getElementById("not-twitter");
  const loading = document.getElementById("loading");
  const result = document.getElementById("result");
  const threadMode = document.getElementById("thread-mode");
  const error = document.getElementById("error");

  const isFullView = new URLSearchParams(location.search).get("fullview") === "1";
  if (isFullView) {
    document.body.classList.add("fullview");
    chrome.storage.local.get(["last_analysis"], (stored) => {
      if (stored.last_analysis) {
        notTwitter.style.display = "none";
        handleResponse({ success: true, data: stored.last_analysis });
      } else {
        notTwitter.style.display = "block";
        document.querySelector("#not-twitter .muted").textContent = "no analysis cached. open the popup on a profile first.";
      }
    });
    return;
  }

  document.getElementById("open-fullview").addEventListener("click", () => {
    chrome.tabs.create({ url: chrome.runtime.getURL("popup/popup.html?fullview=1") });
  });

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const url = tabs[0]?.url || "";
    const tabId = tabs[0]?.id;
    const threadMatch = url.match(/https?:\/\/(?:x\.com|twitter\.com)\/[A-Za-z0-9_]+\/status\/\d+/);
    const profileMatch = url.match(/https?:\/\/(x\.com|twitter\.com)\/([A-Za-z0-9_]+)\/?$/);

    if (threadMatch) {
      notTwitter.style.display = "none";
      loading.style.display = "block";
      chrome.tabs.sendMessage(tabId, { type: "GET_THREAD_STATS" }, (response) => {
        loading.style.display = "none";
        if (chrome.runtime.lastError) {
          showThreadNotLoaded(chrome.runtime.lastError.message);
          return;
        }
        if (!response || !response.success) {
          showThreadNotLoaded("thread.js did not respond");
          return;
        }
        showThreadMode(response.data);
      });
      return;
    }

    if (!profileMatch) {
      notTwitter.style.display = "block";
      return;
    }

    const username = profileMatch[2];
    const reserved = [
      "home", "explore", "search", "notifications", "messages",
      "settings", "i", "compose", "hashtag",
    ];
    if (reserved.includes(username.toLowerCase())) {
      notTwitter.style.display = "block";
      return;
    }

    notTwitter.style.display = "none";
    loading.style.display = "block";

    chrome.tabs.sendMessage(tabId, { type: "GET_RESULT" }, (response) => {
      loading.style.display = "none";

      if (chrome.runtime.lastError || !response) {
        chrome.runtime.sendMessage(
          { type: "PREDICT", profile: { username, display_name: username } },
          handleResponse
        );
        return;
      }

      handleResponse(response);
    });
  });

  function showThreadNotLoaded(reason) {
    threadMode.style.display = "block";
    document.getElementById("thread-counts").innerHTML = `
      <div class="thread-error">
        <div class="thread-error-title">Thread analyser not loaded</div>
        <div class="thread-error-body">
          ${reason ? "(" + reason + ")" : ""}
          <br>
          1. Open <code>chrome://extensions</code><br>
          2. Click reload on Twitter Bot Detector<br>
          3. Refresh this X tab<br>
          4. Check the page console for [bot-detector] logs
        </div>
      </div>
    `;
    document.getElementById("thread-clusters").innerHTML = "";
  }

  function showThreadEmpty() {
    threadMode.style.display = "block";
    document.getElementById("thread-counts").innerHTML = `
      <div class="thread-empty">Thread analyser is loaded but no replies have been scored yet. Scroll the thread to load replies, wait a few seconds, then reopen this popup.</div>
    `;
  }

  function showThreadMode(data) {
    threadMode.style.display = "block";
    const total = (data.totals?.human || 0) + (data.totals?.uncertain || 0) + (data.totals?.bot || 0);
    if (total === 0) {
      showThreadEmpty();
      return;
    }
    const h = data.totals.human || 0;
    const u = data.totals.uncertain || 0;
    const b = data.totals.bot || 0;
    const hPct = Math.round((h / total) * 100);
    const uPct = Math.round((u / total) * 100);
    const bPct = Math.round((b / total) * 100);
    document.getElementById("thread-counts").innerHTML = `
      <div class="thread-total">${total} replies analysed</div>
      <div class="thread-bar">
        <div class="thread-bar-human" style="width:${hPct}%"></div>
        <div class="thread-bar-uncertain" style="width:${uPct}%"></div>
        <div class="thread-bar-bot" style="width:${bPct}%"></div>
      </div>
      <div class="thread-legend">
        <span class="thread-stat thread-stat-human"><b>${h}</b> human</span>
        <span class="thread-stat thread-stat-uncertain"><b>${u}</b> unclear</span>
        <span class="thread-stat thread-stat-bot"><b>${b}</b> bot</span>
      </div>
    `;
    const clusters = data.clusters || [];
    const clusterEl = document.getElementById("thread-clusters");
    if (clusters.length === 0) {
      clusterEl.innerHTML = `<div class="thread-empty">No suspicious clusters detected.</div>`;
    } else {
      clusterEl.innerHTML = clusters.slice(0, 3).map((c) => {
        const handles = c.handles.slice(0, 4).map((h) => "@" + h).join(", ");
        const more = c.handles.length > 4 ? " +" + (c.handles.length - 4) : "";
        return `
          <div class="thread-cluster">
            <div class="thread-cluster-title">cluster of ${c.size}, median score ${c.median_score}</div>
            <div class="thread-cluster-handles">${handles}${more}</div>
          </div>
        `;
      }).join("");
    }
  }

  function handleResponse(response) {
    loading.style.display = "none";

    if (!response || !response.success) {
      error.style.display = "block";
      document.getElementById("error-text").textContent =
        response?.error || "Could not reach the bot detection API.";
      return;
    }

    if (!isFullView) {
      chrome.storage.local.set({ last_analysis: response.data });
    }
    showResult(response.data);
  }

  function showResult(data) {
    result.style.display = "block";

    const score = data.bot_score;
    const label = data.label;
    const colors = { human: "#34C759", uncertain: "#FF9500", bot: "#FF3B30" };
    const labelText = { human: "HUMAN", uncertain: "UNCERTAIN", bot: "BOT" };
    const color = colors[label] || "#8E8E93";

    const scoreEl = document.getElementById("score-number");
    scoreEl.textContent = score;
    scoreEl.style.color = color;

    const badge = document.getElementById("label-badge");
    badge.textContent = labelText[label] || "--";
    badge.className = `badge ${label}`;

    const bar = document.getElementById("score-bar");
    bar.style.backgroundColor = color;
    setTimeout(() => { bar.style.width = `${score}%`; }, 50);

    document.getElementById("username-display").textContent = `@${data.username}`;
    document.getElementById("confidence-text").textContent = `Confidence: ${data.confidence}`;

    renderLegend(score, data.threshold, data.margin);
    renderOverride(data.override_applied);

    const grid = document.getElementById("contrib-grid");
    const signalsList = document.getElementById("signals-list");

    if (data.contributions) {
      grid.style.display = "grid";
      signalsList.style.display = "none";
      renderContribCol("contrib-bot-list", data.contributions.toward_bot || [], "bot");
      renderContribCol("contrib-human-list", data.contributions.toward_human || [], "human");
    } else {
      grid.style.display = "none";
      signalsList.style.display = "flex";
      signalsList.innerHTML = "";
      for (const signal of data.signals || []) {
        const chip = document.createElement("div");
        chip.className = "signal-chip";
        chip.textContent = signal;
        signalsList.appendChild(chip);
      }
    }
  }

  function renderLegend(score, threshold, margin) {
    const el = document.getElementById("legend");
    if (!el) return;
    const t = typeof threshold === "number" ? threshold : 0.58;
    const m = typeof margin === "number" ? margin : 0.1;
    const humanEnd = Math.max(0, Math.min(100, (t - m) * 100));
    const botStart = Math.max(0, Math.min(100, (t + m) * 100));
    const markerLeft = Math.max(0, Math.min(100, score));
    el.innerHTML = `
      <div class="legend-track">
        <div class="legend-zone legend-human" style="width:${humanEnd}%"></div>
        <div class="legend-zone legend-uncertain" style="width:${botStart - humanEnd}%"></div>
        <div class="legend-zone legend-bot" style="width:${100 - botStart}%"></div>
        <div class="legend-marker" style="left:${markerLeft}%"></div>
      </div>
      <div class="legend-labels">
        <span>0</span>
        <span>${Math.round(t * 100)} threshold</span>
        <span>100</span>
      </div>
    `;
  }

  function renderContribCol(elementId, entries, direction) {
    const el = document.getElementById(elementId);
    el.innerHTML = "";
    if (!entries.length) {
      const empty = document.createElement("div");
      empty.className = "contrib-empty";
      empty.textContent = "no strong signals";
      el.appendChild(empty);
      return;
    }
    const max = Math.max(...entries.map((e) => Math.abs(e.contribution)), 0.5);
    for (const entry of entries) {
      const row = document.createElement("div");
      row.className = "contrib-row";
      const pct = Math.round((Math.abs(entry.contribution) / max) * 100);
      const sign = entry.contribution >= 0 ? "+" : "";
      row.innerHTML = `
        <div class="contrib-text">
          <span class="contrib-desc">${escapeHtml(entry.description || entry.feature)}</span>
          <span class="contrib-mag">${sign}${entry.contribution.toFixed(2)}</span>
        </div>
        <div class="contrib-bar-track">
          <div class="contrib-bar-fill contrib-bar-${direction}" style="width:${pct}%"></div>
        </div>
        <div class="contrib-value">${escapeHtml(entry.value)}</div>
      `;
      el.appendChild(row);
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function renderOverride(kind) {
    const el = document.getElementById("override-banner");
    if (!kind) {
      el.style.display = "none";
      return;
    }
    el.style.display = "block";
    if (kind === "news_org") {
      el.innerHTML = '<div class="override-title">News organisation override</div>Verified, established, large-following news/org account. Score capped to keep this human.';
    } else {
      el.innerHTML = `<div class="override-title">Override applied</div>${kind}`;
    }
  }
});
