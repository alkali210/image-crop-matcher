"use strict";

const STATUS_POLL_DELAY_MS = 750;
const STATUS_RETRY_MESSAGE = "暂时无法读取索引，正在重试...";
const MATCH_ERROR_MESSAGE = "匹配失败，请重试";

const state = {
  selectedFile: null,
  queryUrl: null,
  statusTimer: null,
  statusPending: false,
  galleryPending: false,
  gallerySubmitting: false,
  hasActiveGallery: false,
};

const dropZone = document.querySelector("#drop-zone");
const fileInput = document.querySelector("#file-input");
const indexStatus = document.querySelector("#index-status");
const statusText = document.querySelector("#status-text");
const loadingStatus = document.querySelector("#loading-status");
const errorMessage = document.querySelector("#error-message");
const uploadView = document.querySelector("#upload-view");
const resultsView = document.querySelector("#results-view");
const resultsHeading = document.querySelector("#results-heading");
const querySummary = document.querySelector("#query-summary");
const queryPreview = document.querySelector("#query-preview");
const queryName = document.querySelector("#query-name");
const queryMeta = document.querySelector("#query-meta");
const resultList = document.querySelector("#result-list");
const reuploadButton = document.querySelector("#reupload-button");
const currentGalleryPath = document.querySelector("#current-gallery-path");
const gallerySwitchForm = document.querySelector("#gallery-switch-form");
const galleryPath = document.querySelector("#gallery-path");
const switchGalleryButton = document.querySelector("#switch-gallery-button");
const gallerySwitchStatus = document.querySelector("#gallery-switch-status");

function setUploadEnabled(enabled) {
  const canUpload = enabled && loadingStatus.hidden;
  dropZone.setAttribute("aria-disabled", String(!canUpload));
  fileInput.disabled = !canUpload;
}

function renderStatus(status) {
  const count = Number.isFinite(status.indexed_images) ? status.indexed_images : 0;
  state.hasActiveGallery = status.state === "ready";
  indexStatus.dataset.state = status.state;

  if (status.state === "ready") {
    statusText.textContent = `索引已就绪，共 ${count} 张原图`;
    setUploadEnabled(true);
    return;
  }

  setUploadEnabled(false);
  if (status.state === "building") {
    statusText.textContent = "正在建立图片索引...";
    return;
  }

  const detail = typeof status.error === "string" && status.error ? `：${status.error}` : "";
  statusText.textContent = `图片索引不可用${detail}`;
}

function setGalleryPending(pending) {
  state.galleryPending = pending;
  switchGalleryButton.disabled = pending;
}

function setGallerySwitchMessage(message, statusState = "") {
  gallerySwitchStatus.textContent = message;
  if (statusState) {
    gallerySwitchStatus.dataset.state = statusState;
  } else {
    delete gallerySwitchStatus.dataset.state;
  }
}

function renderGalleryStatus(status) {
  const activePath = typeof status.gallery_dir === "string" ? status.gallery_dir : "";
  const pendingPath =
    typeof status.pending_gallery_dir === "string" ? status.pending_gallery_dir : "";
  const wasPending = state.galleryPending;
  currentGalleryPath.textContent = activePath || "尚无可用图库";

  if (status.reindexing && pendingPath) {
    setGalleryPending(true);
    setGallerySwitchMessage(`正在建立新图库索引：${pendingPath}`, "pending");
    return;
  }

  if (!state.gallerySubmitting) {
    setGalleryPending(false);
  }
  if (typeof status.switch_error === "string" && status.switch_error) {
    setGallerySwitchMessage(`图库切换失败：${status.switch_error}。当前图库仍可使用。`, "error");
  } else if (wasPending) {
    setGallerySwitchMessage("图库已切换", "");
  }
}

function clearStatusTimer() {
  if (state.statusTimer !== null) {
    window.clearTimeout(state.statusTimer);
    state.statusTimer = null;
  }
}

function scheduleStatusPoll() {
  clearStatusTimer();
  state.statusTimer = window.setTimeout(() => {
    state.statusTimer = null;
    pollStatus();
  }, STATUS_POLL_DELAY_MS);
}

function renderStatusUnavailable() {
  indexStatus.dataset.state = "error";
  statusText.textContent = STATUS_RETRY_MESSAGE;
  setUploadEnabled(state.hasActiveGallery);
}

async function pollStatus() {
  if (state.statusPending) {
    return;
  }

  clearStatusTimer();
  state.statusPending = true;

  try {
    const response = await fetch("/api/status");
    if (!response.ok) {
      throw new Error("无法读取索引状态");
    }
    const status = await response.json();
    renderStatus(status);
    renderGalleryStatus(status);
    if (status.state === "building" || status.reindexing) {
      scheduleStatusPoll();
    }
  } catch (_error) {
    renderStatusUnavailable();
    scheduleStatusPoll();
  } finally {
    state.statusPending = false;
  }
}

function setBusy(busy) {
  loadingStatus.hidden = !busy;
  reuploadButton.disabled = busy;
  setUploadEnabled(!busy && state.hasActiveGallery);
}

function clearError() {
  errorMessage.hidden = true;
  errorMessage.textContent = "";
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function revokeQueryUrl() {
  if (state.queryUrl !== null) {
    URL.revokeObjectURL(state.queryUrl);
    state.queryUrl = null;
  }
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KiB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function renderQuery(file, query, elapsedMs) {
  revokeQueryUrl();
  state.queryUrl = URL.createObjectURL(file);
  queryPreview.src = state.queryUrl;
  queryName.textContent = file.name || "未命名图片";

  const dimensions = `${query.width} × ${query.height} px`;
  queryMeta.textContent = `${dimensions} · ${formatFileSize(file.size)} · ${elapsedMs} ms`;
  querySummary.hidden = false;
}

function safeImageUrl(value) {
  try {
    const url = new URL(value, window.location.origin);
    if (url.origin === window.location.origin && url.pathname.startsWith("/api/images/")) {
      return `${url.pathname}${url.search}`;
    }
  } catch (_error) {
    return "";
  }
  return "";
}

function isValidMatchResponse(body) {
  if (body === null || typeof body !== "object" || Array.isArray(body)) {
    return false;
  }
  const query = body.query;
  if (
    query === null ||
    typeof query !== "object" ||
    !Number.isInteger(query.width) ||
    query.width <= 0 ||
    !Number.isInteger(query.height) ||
    query.height <= 0 ||
    !Number.isInteger(body.elapsed_ms) ||
    body.elapsed_ms < 0 ||
    !Array.isArray(body.matches) ||
    body.matches.length === 0
  ) {
    return false;
  }
  return body.matches.every(
    (match) =>
      match !== null &&
      typeof match === "object" &&
      typeof match.image_id === "string" &&
      match.image_id.length > 0 &&
      typeof match.parent_name === "string" &&
      typeof match.filename === "string" &&
      match.filename.length > 0 &&
      Number.isInteger(match.width) &&
      match.width > 0 &&
      Number.isInteger(match.height) &&
      match.height > 0 &&
      typeof match.similarity === "number" &&
      Number.isFinite(match.similarity) &&
      match.similarity >= 0 &&
      match.similarity <= 100 &&
      typeof match.image_url === "string" &&
      safeImageUrl(match.image_url) !== "",
  );
}

function textElement(tagName, className, text) {
  const element = document.createElement(tagName);
  element.className = className;
  element.textContent = text;
  return element;
}

function createMatchRow(match, rank) {
  const row = document.createElement("li");
  row.className = "match-row";

  const imageUrl = safeImageUrl(match.image_url);
  const thumbnailLink = document.createElement("a");
  thumbnailLink.className = "thumbnail-link";
  thumbnailLink.href = imageUrl || "#";
  thumbnailLink.target = "_blank";
  thumbnailLink.rel = "noopener";
  thumbnailLink.setAttribute("aria-label", `打开原图：${match.filename}`);

  const thumbnail = document.createElement("img");
  thumbnail.className = "match-thumbnail";
  thumbnail.src = imageUrl;
  thumbnail.alt = match.filename;
  thumbnail.loading = "lazy";
  thumbnailLink.append(thumbnail);

  const information = document.createElement("div");
  information.className = "match-information";
  information.append(
    textElement("p", "match-rank", `匹配 #${rank}`),
    textElement("p", "match-parent", match.parent_name),
    textElement("h2", "match-filename", match.filename),
    textElement("p", "match-dimensions", `${match.width} × ${match.height} px`),
  );

  const score = document.createElement("div");
  score.className = "match-score";
  score.append(textElement("p", "score-label", "相似度"));

  const scoreValue = document.createElement("p");
  scoreValue.className = "score-value";
  const numericSimilarity = Number(match.similarity);
  scoreValue.append(
    document.createTextNode(Number.isFinite(numericSimilarity) ? numericSimilarity.toFixed(1) : "—"),
    textElement("span", "", "%"),
  );
  score.append(scoreValue);

  const originalLink = document.createElement("a");
  originalLink.className = "original-link";
  originalLink.href = imageUrl || "#";
  originalLink.target = "_blank";
  originalLink.rel = "noopener";
  originalLink.textContent = "查看原图";
  score.append(originalLink);

  if (!imageUrl) {
    thumbnailLink.removeAttribute("href");
    originalLink.removeAttribute("href");
    thumbnail.removeAttribute("src");
  }

  row.append(thumbnailLink, information, score);
  return row;
}

function renderMatches(matches) {
  resultList.replaceChildren(
    ...matches.map((match, index) => createMatchRow(match, index + 1)),
  );
}

function showResults() {
  uploadView.hidden = true;
  resultsView.hidden = false;
  resultsHeading.focus();
}

async function parseJsonResponse(response) {
  try {
    return await response.json();
  } catch (_error) {
    return null;
  }
}

async function submitGallery(event) {
  event.preventDefault();
  const path = galleryPath.value.trim();
  if (!path) {
    setGallerySwitchMessage("请输入图库的绝对路径", "error");
    galleryPath.focus();
    return;
  }

  state.gallerySubmitting = true;
  setGalleryPending(true);
  setGallerySwitchMessage(`正在请求切换：${path}`, "pending");
  let remainsPending = false;

  try {
    const response = await fetch("/api/gallery", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const status = await parseJsonResponse(response);
    if (!response.ok) {
      const apiMessage = status?.error?.message;
      throw new Error(
        typeof apiMessage === "string" && apiMessage.trim() ? apiMessage : "无法切换图库",
      );
    }
    renderStatus(status);
    renderGalleryStatus(status);
    remainsPending = status.reindexing === true;
    if (remainsPending) {
      scheduleStatusPoll();
    } else {
      setGallerySwitchMessage("当前图库已就绪", "");
    }
  } catch (error) {
    const detail = error instanceof Error ? error.message : "无法切换图库";
    setGallerySwitchMessage(`图库切换失败：${detail}。当前图库仍可使用。`, "error");
  } finally {
    state.gallerySubmitting = false;
    setGalleryPending(remainsPending);
  }
}

async function submitFile(file) {
  if (!file || dropZone.getAttribute("aria-disabled") === "true") {
    return;
  }

  state.selectedFile = file;
  setBusy(true);
  clearError();
  const form = new FormData();
  form.append("file", file);
  let failureMessage = MATCH_ERROR_MESSAGE;

  try {
    const response = await fetch("/api/match", { method: "POST", body: form });
    const body = await parseJsonResponse(response);
    if (!response.ok) {
      const apiMessage = body?.error?.message;
      if (typeof apiMessage === "string" && apiMessage.trim()) {
        failureMessage = apiMessage;
      }
      throw new Error(MATCH_ERROR_MESSAGE);
    }
    if (!isValidMatchResponse(body)) {
      throw new Error(MATCH_ERROR_MESSAGE);
    }
    renderQuery(file, body.query, body.elapsed_ms);
    renderMatches(body.matches);
    showResults();
  } catch (_error) {
    showError(failureMessage);
  } finally {
    setBusy(false);
    fileInput.value = "";
  }
}

function openFilePicker() {
  if (dropZone.getAttribute("aria-disabled") !== "true") {
    fileInput.click();
  }
}

function resetUpload() {
  revokeQueryUrl();
  state.selectedFile = null;
  queryPreview.removeAttribute("src");
  queryName.textContent = "";
  queryMeta.textContent = "";
  querySummary.hidden = true;
  resultList.replaceChildren();
  resultsView.hidden = true;
  uploadView.hidden = false;
  clearError();
  fileInput.value = "";
  dropZone.focus();
}

function restorePage() {
  if (state.queryUrl !== null && !resultsView.hidden) {
    queryPreview.src = state.queryUrl;
  }

  setUploadEnabled(state.hasActiveGallery);
  if (state.statusTimer === null && !state.statusPending) {
    pollStatus();
  }
}

dropZone.addEventListener("click", openFilePicker);
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openFilePicker();
  }
});

dropZone.addEventListener("dragenter", (event) => {
  event.preventDefault();
  if (dropZone.getAttribute("aria-disabled") !== "true") {
    dropZone.classList.add("is-dragging");
  }
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
});

dropZone.addEventListener("dragleave", (event) => {
  if (!dropZone.contains(event.relatedTarget)) {
    dropZone.classList.remove("is-dragging");
  }
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  submitFile(event.dataTransfer?.files[0]);
});

fileInput.addEventListener("change", () => {
  submitFile(fileInput.files[0]);
});

reuploadButton.addEventListener("click", resetUpload);
gallerySwitchForm.addEventListener("submit", submitGallery);
window.addEventListener("pagehide", (event) => {
  if (event.persisted) {
    return;
  }
  clearStatusTimer();
  revokeQueryUrl();
});
window.addEventListener("pageshow", restorePage);
