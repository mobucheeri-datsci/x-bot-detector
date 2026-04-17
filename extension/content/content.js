(function () {
  "use strict";

  let lastUrl = "";
  let latestResult = null;

  const observer = new MutationObserver(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      latestResult = null;
      onNavigate();
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "GET_RESULT") {
      if (latestResult) {
        sendResponse({ success: true, data: latestResult });
      } else {
        const username = currentUsername();
        if (!username) {
          sendResponse({ success: false, error: "Not on a profile page" });
          return false;
        }
        scrapeAndPredict(username, (response) => sendResponse(response));
        return true;
      }
    }
  });

  function currentUsername() {
    const match = location.pathname.match(/^\/([A-Za-z0-9_]+)\/?$/);
    if (!match) return null;
    const username = match[1];
    const reserved = [
      "home", "explore", "search", "notifications", "messages",
      "settings", "i", "compose", "hashtag",
    ];
    if (reserved.includes(username.toLowerCase())) return null;
    return username;
  }

  function onNavigate() {
    const username = currentUsername();
    if (!username) return;
    setTimeout(() => scrapeAndPredict(username), 2000);
  }

  function parseNumber(text) {
    if (!text) return 0;
    const cleaned = text.replace(/[,\s]/g, "");
    const match = cleaned.match(/^([\d.]+)\s*([KMB]?)/i);
    if (!match) return 0;
    let num = parseFloat(match[1]);
    if (isNaN(num)) return 0;
    const suffix = (match[2] || "").toUpperCase();
    if (suffix === "K") num *= 1_000;
    else if (suffix === "M") num *= 1_000_000;
    else if (suffix === "B") num *= 1_000_000_000;
    return Math.round(num);
  }

  function scrapeProfile(username) {
    const profile = {
      username,
      display_name: "",
      bio: "",
      followers_count: 0,
      following_count: 0,
      tweet_count: 0,
      favourites_count: 0,
      listed_count: 0,
      account_age_days: 365,
      recent_tweets: [],
      has_default_avatar: false,
      is_verified: false,
      url: "",
    };

    const nameEl = document.querySelector('[data-testid="UserName"]');
    if (nameEl) {
      const spans = nameEl.querySelectorAll("span");
      if (spans.length > 0) profile.display_name = spans[0].textContent || "";
    }

    const bioEl = document.querySelector('[data-testid="UserDescription"]');
    if (bioEl) profile.bio = bioEl.textContent || "";

    const links = document.querySelectorAll("a[href]");
    for (const link of links) {
      const href = link.getAttribute("href") || "";
      const text = link.textContent || "";

      if (href.endsWith("/verified_followers") || href.endsWith("/followers")) {
        const n = parseNumber(text);
        if (n > profile.followers_count) profile.followers_count = n;
      } else if (href.endsWith("/following")) {
        const n = parseNumber(text);
        if (n > profile.following_count) profile.following_count = n;
      }
    }

    const allSpans = document.querySelectorAll("span, div");
    for (const el of allSpans) {
      const text = (el.textContent || "").trim();
      const m = text.match(/^([\d.,]+\s*[KMB]?)\s*posts?$/i);
      if (m) {
        const n = parseNumber(m[1]);
        if (n > profile.tweet_count) profile.tweet_count = n;
      }
    }

    const tweetEls = document.querySelectorAll('[data-testid="tweetText"]');
    for (let i = 0; i < Math.min(tweetEls.length, 20); i++) {
      profile.recent_tweets.push(tweetEls[i].textContent || "");
    }

    const joinedEl = Array.from(document.querySelectorAll("span")).find(
      (el) => el.textContent && el.textContent.includes("Joined")
    );
    if (joinedEl) {
      const dateStr = joinedEl.textContent.replace("Joined ", "");
      try {
        const joined = new Date(dateStr);
        const days = Math.floor((Date.now() - joined.getTime()) / 86400000);
        if (days > 0) profile.account_age_days = days;
      } catch (e) {}
    }

    const verifiedEl = document.querySelector('[data-testid="icon-verified"]');
    profile.is_verified = !!verifiedEl;

    if (profile.tweet_count > 0 && profile.favourites_count === 0) {
      profile.favourites_count = Math.round(profile.tweet_count * 1.5);
    }

    return profile;
  }

  function scrapeAndPredict(username, callback) {
    const profile = scrapeProfile(username);

    chrome.runtime.sendMessage(
      { type: "PREDICT", profile },
      (response) => {
        if (response && response.success) {
          latestResult = response.data;
          injectRichPanel(response.data);
        }
        if (callback) callback(response);
      }
    );
  }

  const labelColors = { human: "#34C759", uncertain: "#FF9500", bot: "#FF3B30" };
  const labelText = { human: "HUMAN", uncertain: "UNCERTAIN", bot: "BOT" };

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function buildLegendHtml(score, threshold, margin) {
    const humanEnd = Math.max(0, Math.min(100, (threshold - margin) * 100));
    const botStart = Math.max(0, Math.min(100, (threshold + margin) * 100));
    const markerLeft = Math.max(0, Math.min(100, score));
    return `
      <div class="botdetect-legend">
        <div class="botdetect-legend-track">
          <div class="botdetect-legend-zone botdetect-legend-human" style="width:${humanEnd}%"></div>
          <div class="botdetect-legend-zone botdetect-legend-uncertain" style="width:${botStart - humanEnd}%"></div>
          <div class="botdetect-legend-zone botdetect-legend-bot" style="width:${100 - botStart}%"></div>
          <div class="botdetect-legend-marker" style="left:${markerLeft}%"></div>
        </div>
        <div class="botdetect-legend-labels">
          <span>0 human</span>
          <span>${Math.round(threshold * 100)} threshold</span>
          <span>100 bot</span>
        </div>
      </div>
    `;
  }

  function buildContribColHtml(entries, direction) {
    if (!entries || entries.length === 0) {
      return `<div class="botdetect-empty">no strong signals</div>`;
    }
    const max = Math.max(...entries.map((e) => Math.abs(e.contribution)), 0.5);
    return entries.map((e) => {
      const pct = Math.round((Math.abs(e.contribution) / max) * 100);
      const sign = e.contribution >= 0 ? "+" : "";
      return `
        <div class="botdetect-bar-row">
          <div class="botdetect-bar-text">
            <span class="botdetect-bar-desc">${escapeHtml(e.description || e.feature)}</span>
            <span class="botdetect-bar-mag">${sign}${e.contribution.toFixed(2)}</span>
          </div>
          <div class="botdetect-bar-track">
            <div class="botdetect-bar-fill botdetect-bar-${direction}" style="width:${pct}%"></div>
          </div>
          <div class="botdetect-bar-value">${escapeHtml(e.value)}</div>
        </div>
      `;
    }).join("");
  }

  function injectRichPanel(data) {
    const existing = document.getElementById("botdetect-panel");
    if (existing) existing.remove();

    const score = data.bot_score;
    const label = data.label;
    const color = labelColors[label] || "#8E8E93";
    const threshold = typeof data.threshold === "number" ? data.threshold : 0.58;
    const margin = typeof data.margin === "number" ? data.margin : 0.1;

    const overrideHtml = data.override_applied === "news_org"
      ? `<div class="botdetect-override">
          <div class="botdetect-override-title">News organisation override</div>
          <div class="botdetect-override-body">Verified, established, large-following organisation. Score capped to keep this account human.</div>
        </div>`
      : "";

    const contribs = data.contributions || { toward_bot: [], toward_human: [] };

    const panel = document.createElement("div");
    panel.id = "botdetect-panel";
    panel.className = "botdetect-panel";
    panel.innerHTML = `
      <div class="botdetect-panel-header">
        <div class="botdetect-panel-brand">Bot Detector</div>
        <div class="botdetect-panel-headline">
          <div class="botdetect-panel-score" style="color:${color}">${score}<span class="botdetect-panel-of">/100</span></div>
          <div class="botdetect-panel-badge" style="background:${color}">${labelText[label] || "--"}</div>
        </div>
      </div>
      ${buildLegendHtml(score, threshold, margin)}
      ${overrideHtml}
      <div class="botdetect-panel-why">why this score</div>
      <div class="botdetect-panel-grid">
        <div class="botdetect-panel-col">
          <div class="botdetect-panel-coltitle botdetect-panel-coltitle-bot">toward bot</div>
          ${buildContribColHtml(contribs.toward_bot, "bot")}
        </div>
        <div class="botdetect-panel-col">
          <div class="botdetect-panel-coltitle botdetect-panel-coltitle-human">toward human</div>
          ${buildContribColHtml(contribs.toward_human, "human")}
        </div>
      </div>
    `;

    const nameEl = document.querySelector('[data-testid="UserName"]');
    let target = null;
    if (nameEl) {
      let node = nameEl;
      for (let i = 0; i < 6 && node; i++) {
        node = node.parentElement;
        if (node && node.querySelector('[data-testid="UserDescription"]')) {
          target = node;
          break;
        }
      }
      if (!target) target = nameEl.parentElement;
    }

    if (target && target.parentElement) {
      target.parentElement.insertBefore(panel, target.nextSibling);
    } else {
      const main = document.querySelector('main[role="main"]');
      if (main) main.insertBefore(panel, main.firstChild);
    }
  }

  onNavigate();
})();