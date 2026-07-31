const apiBase = "https://mobucheeri-twitter-bot-detector.hf.space";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "PREDICT") {
    apiPost("/predict", message.profile).then(sendResponse);
    return true;
  }
  if (message.type === "PREDICT_BATCH") {
    apiPost("/predict_batch", { profiles: message.profiles }).then(sendResponse);
    return true;
  }
  if (message.type === "PREDICT_THREAD_BATCH") {
    apiPost("/predict_thread_batch", { replies: message.replies }).then(sendResponse);
    return true;
  }
});

async function getApiKey() {
  const stored = await chrome.storage.local.get("apiKey");
  if (stored.apiKey) return stored.apiKey;
  return registerApiKey();
}

async function registerApiKey() {
  const response = await fetch(`${apiBase}/register`, { method: "POST" });
  if (!response.ok) throw new Error(`registration failed (${response.status})`);
  const data = await response.json();
  await chrome.storage.local.set({ apiKey: data.api_key });
  return data.api_key;
}

async function apiPost(path, body) {
  try {
    // Registration failure is non-fatal: the server only rejects keyless
    // requests once REQUIRE_API_KEY is switched on.
    let key = "";
    try {
      key = await getApiKey();
    } catch (err) {
      console.warn("X Bot Detector: could not obtain API key:", err.message);
    }

    const doFetch = (apiKey) =>
      fetch(`${apiBase}${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiKey ? { "X-API-Key": apiKey } : {}),
        },
        body: JSON.stringify(body),
      });

    let response = await doFetch(key);

    if (response.status === 401) {
      await chrome.storage.local.remove("apiKey");
      key = await registerApiKey();
      response = await doFetch(key);
    }

    if (response.status === 429) {
      return {
        success: false,
        error: "Scan limit reached. Please wait a minute and try again.",
      };
    }

    if (!response.ok) {
      const error = await response.text();
      return { success: false, error: `API error ${response.status}: ${error}` };
    }

    const data = await response.json();
    return { success: true, data };
  } catch (err) {
    return { success: false, error: `Connection failed: ${err.message}` };
  }
}
