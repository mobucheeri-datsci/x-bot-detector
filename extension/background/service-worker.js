const apiBase = "http://localhost:8000";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "PREDICT") {
    handlePredict(message.profile).then(sendResponse);
    return true;
  }
  if (message.type === "PREDICT_BATCH") {
    handlePredictBatch(message.profiles).then(sendResponse);
    return true;
  }
});

async function handlePredict(profile) {
  try {
    const response = await fetch(`${apiBase}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile),
    });

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

async function handlePredictBatch(profiles) {
  try {
    const response = await fetch(`${apiBase}/predict_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profiles }),
    });

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
