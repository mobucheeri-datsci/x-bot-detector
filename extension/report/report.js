(function () {
  "use strict";

  function getQueryParam(name) {
    const url = new URL(location.href);
    return url.searchParams.get(name);
  }

  function render() {
    const rawData = getQueryParam("data");
    if (!rawData) {
      document.body.innerHTML = '<div class="report-empty">No data passed to this report. Open it from the Thread Scan panel.</div>';
      return;
    }

    let data;
    try {
      data = JSON.parse(decodeURIComponent(rawData));
    } catch (e) {
      document.body.innerHTML = '<div class="report-empty">Failed to load report data.</div>';
      return;
    }

    const { threadUrl, timestamp, replies, totals } = data;

    if (threadUrl) {
      const srcEl = document.getElementById("report-source");
      srcEl.innerHTML = `Source: <a href="${threadUrl}" target="_blank" rel="noopener noreferrer">${threadUrl}</a>`;
    }

    if (timestamp) {
      const date = new Date(timestamp);
      document.getElementById("report-timestamp").textContent = `Scanned ${date.toLocaleString()}`;
    }

    document.getElementById("stat-typical").textContent = totals.typical || 0;
    document.getElementById("stat-possibly").textContent = totals.possibly_suspicious || 0;
    document.getElementById("stat-suspicious").textContent = totals.suspicious || 0;
    document.getElementById("stat-total").textContent = (totals.typical || 0) + (totals.possibly_suspicious || 0) + (totals.suspicious || 0);

    renderChart(totals);
    renderReasons(replies);
    renderTable(replies);
    attachFilters(replies);
  }

  function renderChart(totals) {
    const total = (totals.typical || 0) + (totals.possibly_suspicious || 0) + (totals.suspicious || 0);
    if (total === 0) return;

    const chart = document.getElementById("chart");
    chart.innerHTML = "";

    const segments = [
      { flag: "typical", count: totals.typical || 0, cls: "report-chart-typical", label: "typical" },
      { flag: "possibly_suspicious", count: totals.possibly_suspicious || 0, cls: "report-chart-possibly", label: "possibly" },
      { flag: "suspicious", count: totals.suspicious || 0, cls: "report-chart-suspicious", label: "suspicious" },
    ];

    segments.forEach((seg) => {
      if (seg.count === 0) return;
      const pct = (seg.count / total) * 100;
      const el = document.createElement("div");
      el.className = `report-chart-segment ${seg.cls}`;
      el.style.width = `${pct}%`;
      el.textContent = pct >= 8 ? `${seg.count} ${seg.label}` : `${seg.count}`;
      el.title = `${seg.count} ${seg.label} (${pct.toFixed(1)}%)`;
      chart.appendChild(el);
    });
  }

  function renderReasons(replies) {
    const counts = new Map();
    replies.forEach((r) => {
      if (r.flag === "typical") return;
      (r.reasons || []).forEach((reason) => {
        counts.set(reason, (counts.get(reason) || 0) + 1);
      });
    });

    const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 5);
    const container = document.getElementById("reasons");

    if (sorted.length === 0) {
      container.innerHTML = '<div class="report-empty">No flagged replies, so no reasons to show.</div>';
      return;
    }

    container.innerHTML = sorted.map(([reason, count]) => `
      <div class="report-reason-row">
        <div class="report-reason-text">${escapeHtml(reason)}</div>
        <div class="report-reason-count">${count} account${count > 1 ? "s" : ""}</div>
      </div>
    `).join("");
  }

  function renderTable(replies, filter = "all") {
    const tbody = document.getElementById("report-tbody");
    const flagOrder = { suspicious: 0, possibly_suspicious: 1, typical: 2 };

    let rows = replies.filter((r) => {
      if (filter === "all") return r.flag !== "typical";
      return r.flag === filter;
    });

    rows.sort((a, b) => (flagOrder[a.flag] || 99) - (flagOrder[b.flag] || 99));

    if (rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="report-empty">No replies match this filter.</td></tr>';
      return;
    }

    tbody.innerHTML = rows.map((r) => `
      <tr>
        <td>
          <a class="report-handle" href="https://x.com/${escapeHtml(r.username)}" target="_blank" rel="noopener noreferrer">@${escapeHtml(r.username)}</a>
        </td>
        <td>
          <span class="report-flag-badge report-flag-badge-${r.flag}">${flagLabel(r.flag)}</span>
        </td>
        <td class="report-reasons-cell">${(r.reasons || []).map(escapeHtml).join("<br>")}</td>
        <td>
          <div class="report-actions">
            <button class="report-action-btn" data-action="copy" data-username="${escapeHtml(r.username)}">Copy @</button>
          </div>
        </td>
      </tr>
    `).join("");

    tbody.querySelectorAll('[data-action="copy"]').forEach((btn) => {
      btn.addEventListener("click", () => {
        navigator.clipboard.writeText(`@${btn.dataset.username}`);
        btn.textContent = "Copied";
        setTimeout(() => { btn.textContent = "Copy @"; }, 1500);
      });
    });
  }

  function attachFilters(replies) {
    document.querySelectorAll(".report-filter-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".report-filter-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        renderTable(replies, btn.dataset.filter);
      });
    });
  }

  function flagLabel(flag) {
    return ({
      suspicious: "SUSPICIOUS",
      possibly_suspicious: "POSSIBLY SUSPICIOUS",
      typical: "TYPICAL",
    })[flag] || flag.toUpperCase();
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  render();
})();