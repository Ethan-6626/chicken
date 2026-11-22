let selectedFiles = [];
const threshold = 0.575848937;
// GitHub Pages 通过这个地址调用服务器上的真实模型；部署到其他环境时可覆盖 window.API_BASE_URL。
const API_BASE_URL = window.API_BASE_URL || 'https://gx10-aeef.tail1f6601.ts.net';
let backendOnline = false;
const $ = (id) => document.getElementById(id);

function showSection(section) {
  document.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.section === section));
  document.querySelectorAll('.view-section').forEach((view) => view.classList.toggle('active', view.id === section));
  const title = section === 'overview' ? '分析总览' : section === 'predict' ? '图片预测' : '实验记录';
  document.querySelector('.breadcrumb strong').textContent = title;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function checkBackend() {
  const status = $('backendStatus');
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`, { cache: 'no-store' });
    if (!response.ok) throw new Error('health check failed');
    const data = await response.json();
    backendOnline = data.status === 'ok';
    status.textContent = backendOnline ? '后端模型在线' : '后端连接异常';
    status.classList.toggle('green', backendOnline);
    status.classList.toggle('yellow', !backendOnline);
  } catch (_) {
    backendOnline = false;
    status.textContent = '后端暂不可用';
    status.classList.remove('green');
    status.classList.add('yellow');
  }
}

document.querySelectorAll('.nav-item').forEach((item) => item.addEventListener('click', () => showSection(item.dataset.section)));
document.querySelectorAll('.jump-predict').forEach((button) => button.addEventListener('click', () => showSection('predict')));
$('overviewUpload').addEventListener('click', () => {
  showSection('predict');
  $('fileInput').click();
});

function selectFiles(files) {
  if ($('fileInput').disabled || !files.length) return;
  const supported = ['image/jpeg', 'image/png', 'image/webp'];
  selectedFiles = Array.from(files).filter(file =>
    supported.includes(file.type) && file.size > 0 && file.size <= 10 * 1024 * 1024
  );
  const skipped = files.length - selectedFiles.length;
  $('fileName').textContent = `已选择 ${selectedFiles.length} 张${skipped ? `，跳过 ${skipped} 张格式或大小不符合要求的图片` : ''}`;
  $('analyzeBtn').disabled = !selectedFiles.length;
  $('predictionRows').replaceChildren();
  $('resultSub').textContent = selectedFiles.length ? '等待开始预测' : '请选择有效图片';
}

$('fileInput').addEventListener('change', event => selectFiles(event.target.files));
$('dropzone').addEventListener('dragover', event => event.preventDefault());
$('dropzone').addEventListener('drop', event => {
  event.preventDefault();
  if ($('fileInput').disabled || !event.dataTransfer.files.length) return;
  $('fileInput').value = '';
  selectFiles(event.dataTransfer.files);
});

function formatDuration(ms) {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

async function analyze() {
  const input = $('fileInput');
  const button = $('analyzeBtn');
  if (input.disabled || !selectedFiles.length) return;
  input.disabled = button.disabled = true;
  const rows = $('predictionRows');
  rows.replaceChildren();
  const batchStart = performance.now();
  let succeeded = 0;
  for (const [index, file] of selectedFiles.entries()) {
    $('resultSub').textContent = `正在预测 ${index + 1} / ${selectedFiles.length}`;
    const started = performance.now();
    let label = '预测失败';
    let score = '—';
    try {
      const form = new FormData();
      form.append('file', file);
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST', body: form, signal: AbortSignal.timeout(30000),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const probability = data.diseased_probability;
      if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
        throw new Error('无效概率');
      }
      label = probability >= (data.threshold ?? threshold) ? '患病状态' : '正常状态';
      score = `${(probability * 100).toFixed(2)}%`;
      succeeded++;
    } catch (error) {
      label = `预测失败（${error.message}）`;
    }
    const row = rows.insertRow();
    for (const value of [file.name, label, score, formatDuration(performance.now() - started)]) {
      row.insertCell().textContent = value;
    }
  }
  $('resultSub').textContent = `成功 ${succeeded} / ${selectedFiles.length} 张 · 批次总耗时 ${formatDuration(performance.now() - batchStart)}`;
  input.disabled = button.disabled = false;
}

$('analyzeBtn').addEventListener('click', analyze);
checkBackend();
let toastTimer;
function showToast(message) { const toast = $('toast'); toast.textContent = message; toast.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.classList.remove('show'), 3600); }
