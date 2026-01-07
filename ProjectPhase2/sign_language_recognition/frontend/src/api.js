import axios from "axios";

const API_BASE = "/api";

export const captureFrame = () =>
  axios.get(`${API_BASE}/capture-frame`);

export const bufferFrame = (frame) =>
  axios.post(`${API_BASE}/buffer-frame`, frame);

export const getBufferStatus = () =>
  axios.get(`${API_BASE}/buffer-status`);

export const saveLabel = (label) =>
  axios.post(`${API_BASE}/save-label?label=${encodeURIComponent(label)}`);

export const trainModel = (k = 3) =>
  axios.post(`${API_BASE}/train-model?k=${k}`);

export const listLabels = () =>
  axios.get(`${API_BASE}/list-labels`);

export const predictLive = (frames) =>
  axios.post(`${API_BASE}/predict-live`, { frames });

export const getStatus = () =>
  axios.get(`${API_BASE}/status`);
