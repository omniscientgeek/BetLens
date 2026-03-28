// API configuration — reads from build-time env vars (REACT_APP_*)
// Fallback to "/api" (relative) so the request goes through the same origin,
// which works behind a reverse-proxy in production and with CRA's proxy in dev.
export const API_BASE = process.env.REACT_APP_API_BASE || "/api";

export const SOCKET_URL =
  process.env.REACT_APP_SOCKET_URL || window.location.origin;

// Retry-enabled fetch: retries on network errors / 5xx with exponential backoff
export async function fetchWithRetry(url, options = {}, maxRetries = 3) {
  let lastError;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetch(url, options);
      if (res.ok || res.status < 500) return res; // success or client error (no retry)
      lastError = new Error(`HTTP ${res.status}`);
    } catch (err) {
      lastError = err;
    }
    if (attempt < maxRetries) {
      await new Promise((r) => setTimeout(r, 1000 * Math.pow(2, attempt))); // 1s, 2s, 4s
    }
  }
  throw lastError;
}
