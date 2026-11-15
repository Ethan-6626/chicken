const state = { file: null, previewUrl: null };
const threshold = 0.575848937;
const $ = (id) => document.getElementById(id);

function showSection(section) {
  document.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.section === section));
  document.querySelectorAll('.view-section').forEach((view) => view.classList.toggle('active', view.id === section));
  const title = section === 'overview' ? '分析总览' : section === 'predict' ? '图片预测' : '实验记录';
  document.querySelector('.breadcrumb strong').textContent = title;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

document.querySelectorAll('.nav-item').forEach((item) => item.addEventListener('click', () => showSection(item.dataset.section)));
document.querySelectorAll('.jump-predict').forEach((button) => button.addEventListener('click', () => showSection('predict')));

function setFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showToast('请选择 JPG、PNG 或 WebP 图片');
    return;
  }
  state.file = file;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  $('fileName').textContent = file.name;
  $('previewImage').src = state.previewUrl;
  $('previewName').textContent = file.name;
  $('previewMeta').textContent = `${(file.size / 1024).toFixed(1)} KB · ${file.type.split('/')[1].toUpperCase()}`;
  $('dropzone').classList.add('hidden');
  $('previewWrap').classList.remove('hidden');
  $('analyzeBtn').disabled = false;
  $('resultEmpty').classList.remove('hidden');
  $('resultContent').classList.add('hidden');
  $('resultSub').textContent = '图片已就绪，可以开始分析';
  $('resultTag').textContent = 'READY';
}

$('fileInput').addEventListener('change', (event) => setFile(event.target.files[0]));
['dragenter', 'dragover'].forEach((eventName) => $('dropzone').addEventListener(eventName, (event) => { event.preventDefault(); $('dropzone').classList.add('dragover'); }));
['dragleave', 'drop'].forEach((eventName) => $('dropzone').addEventListener(eventName, (event) => { event.preventDefault(); $('dropzone').classList.remove('dragover'); }));
$('dropzone').addEventListener('drop', (event) => setFile(event.dataTransfer.files[0]));
$('removeFile').addEventListener('click', () => {
  state.file = null;
  $('fileInput').value = '';
  $('dropzone').classList.remove('hidden');
  $('previewWrap').classList.add('hidden');
  $('analyzeBtn').disabled = true;
  $('resultEmpty').classList.remove('hidden');
  $('resultContent').classList.add('hidden');
  $('fileName').textContent = '未选择文件';
  $('resultSub').textContent = '等待上传图片';
  $('resultTag').textContent = 'IDLE';
});

function demoProbability(file) {
  const name = file.name.toLowerCase();
  if (/(normal|阴性|假阳性)/.test(name)) return 0.5928;
  if (/(diseased|阳性|强阳性|弱阳性)/.test(name)) return 0.92;
  let hash = 0;
  for (const char of `${file.name}${file.size}`) hash = (hash * 31 + char.charCodeAt(0)) % 1000;
  return 0.62 + (hash / 1000) * 0.33;
}

async function requestPrediction(file) {
  try {
    const response = await fetch('/api/predict', { method: 'POST', body: (() => { const form = new FormData(); form.append('file', file); return form; })() });
    if (!response.ok) throw new Error('API unavailable');
    const data = await response.json();
    return { probability: Number(data.diseased_probability ?? data.probability), mode: '后端模型' };
  } catch (_) {
    return { probability: demoProbability(file), mode: '演示推理' };
  }
}

async function analyze() {
  if (!state.file) return;
  const button = $('analyzeBtn');
  button.disabled = true;
  button.innerHTML = '分析中… <span>↻</span>';
  $('resultTag').textContent = 'RUNNING';
  $('resultSub').textContent = '正在计算患病概率';
  const result = await requestPrediction(state.file);
  const probability = Math.max(0, Math.min(1, result.probability));
  const diseased = probability >= threshold;
  $('resultEmpty').classList.add('hidden');
  $('resultContent').classList.remove('hidden');
  $('resultTag').textContent = 'DONE';
  $('resultSub').textContent = '已完成一次图片推理';
  $('resultBadge').classList.toggle('diseased', diseased);
  $('resultIcon').textContent = diseased ? '!' : '✓';
  $('resultLabel').textContent = diseased ? '患病状态' : '正常状态';
  $('probabilityValue').textContent = `${(probability * 100).toFixed(2)}%`;
  $('probabilityBar').style.width = `${probability * 100}%`;
  $('inferenceMode').textContent = result.mode;
  button.disabled = false;
  button.innerHTML = '重新分析 <span>↻</span>';
  showToast(result.mode === '演示推理' ? '已显示演示预测结果，可接入后端模型替换' : '后端模型分析完成');
}

$('analyzeBtn').addEventListener('click', analyze);
let toastTimer;
function showToast(message) { const toast = $('toast'); toast.textContent = message; toast.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.classList.remove('show'), 3600); }
