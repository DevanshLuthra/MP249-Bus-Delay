'use strict';

const API_BASE = window.location.origin;
const form = document.getElementById('prediction-form');
const statusEl = document.getElementById('api-status');
const statusBox = statusEl.closest('.live-indicator');
const dateInput = document.getElementById('date');
const hourInput = document.getElementById('hour');
const expInput = document.getElementById('driver_exp_yr');
const expValue = document.getElementById('experience-value');
const submitButton = document.getElementById('predict-button');
const buttonLabel = document.getElementById('button-label');
const messageEl = document.getElementById('form-message');
const resultPanel = document.getElementById('result-panel');
const resultNumber = document.getElementById('result-number');
const resultStatus = document.getElementById('result-status');
const resultEyebrow = document.getElementById('result-eyebrow');
const resultDescription = document.getElementById('result-description');

for (let h = 0; h < 24; h++) {
  const option = document.createElement('option');
  option.value = String(h);
  option.textContent = `${String(h).padStart(2, '0')}:00 — ${String((h + 1) % 24).padStart(2, '0')}:00`;
  hourInput.append(option);
}

const now = new Date();
const istNow = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
const todayIST = [istNow.getFullYear(), String(istNow.getMonth() + 1).padStart(2, '0'), String(istNow.getDate()).padStart(2, '0')].join('-');
dateInput.value = todayIST;
hourInput.value = String(istNow.getHours());
document.getElementById('local-time').textContent = 'SURAT · IST';

function updateExperience() { expValue.textContent = `${String(expInput.value).padStart(2, '0')} YRS`; }
expInput.addEventListener('input', updateExperience);
updateExperience();

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error('API unavailable');
    const data = await response.json();
    const healthy = data.status === 'healthy';
    statusEl.textContent = healthy ? 'Prediction engine online' : 'Trained model unavailable';
    statusBox.classList.toggle('offline', !healthy);
  } catch {
    statusEl.textContent = 'API offline · start server';
    statusBox.classList.add('offline');
  }
}
checkHealth();

function resetResult() {
  resultPanel.classList.remove('has-result');
  resultNumber.textContent = '--';
  resultStatus.innerHTML = '<span></span> STANDBY';
  resultEyebrow.textContent = 'YOUR TIME, DECODED';
  resultDescription.textContent = 'Your prediction will appear here. Fill in your route details to get started.';
  messageEl.textContent = '';
}

document.getElementById('reset-button').addEventListener('click', () => {
  form.reset();
  dateInput.value = todayIST;
  hourInput.value = String(istNow.getHours());
  updateExperience();
  resetResult();
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  messageEl.textContent = '';
  if (!form.reportValidity()) return;

  const payload = {
    date: dateInput.value,
    route_km: Number(document.getElementById('route_km').value),
    stops: Number(document.getElementById('stops').value),
    passengers: Number(document.getElementById('passengers').value),
    hour: Number(hourInput.value),
    temp_c: Number(document.getElementById('temp_c').value),
    driver_exp_yr: Number(expInput.value),
    weather: document.getElementById('weather').value,
    depot_zone: document.getElementById('depot_zone').value
  };

  submitButton.disabled = true;
  buttonLabel.textContent = 'CALCULATING...';
  resultStatus.innerHTML = '<span></span> ANALYZING';

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(20000)
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = typeof data.detail === 'string' ? data.detail : 'Please review the inputs and try again.';
      throw new Error(detail);
    }
    const delay = Number(data.predicted_delay_minutes);
    if (!Number.isFinite(delay)) throw new Error('Unexpected prediction response from server.');
    resultNumber.textContent = delay.toFixed(2);
    resultStatus.innerHTML = '<span></span> COMPLETE';
    resultEyebrow.textContent = 'ESTIMATED BUS DELAY';
    resultDescription.textContent = `Your bus is estimated to be delayed by ${delay.toFixed(2)} minutes. This is a model estimate, not a live vehicle tracking result.`;
    resultPanel.classList.remove('has-result');
    void resultPanel.offsetWidth;
    resultPanel.classList.add('has-result');
    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (error) {
    resultStatus.innerHTML = '<span></span> UNAVAILABLE';
    messageEl.textContent = `${error.message} Make sure FastAPI is running on port 8000.`;
  } finally {
    submitButton.disabled = false;
    buttonLabel.textContent = 'PREDICT MY DELAY';
  }
});
