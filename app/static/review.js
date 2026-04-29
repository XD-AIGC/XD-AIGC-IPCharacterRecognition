const uploadForm = document.getElementById('uploadForm');
const uploadInput = document.getElementById('uploadInput');
const uploadStatus = document.getElementById('uploadStatus');
const reviewSection = document.getElementById('reviewSection');
const faceCards = document.getElementById('faceCards');
const heroImage = document.getElementById('heroImage');
const faceOverlay = document.getElementById('faceOverlay');
const imageCandidates = document.getElementById('imageCandidates');
const imageMeta = document.getElementById('imageMeta');
const reviewBadge = document.getElementById('reviewBadge');
const fillBestBtn = document.getElementById('fillBestBtn');
const saveBtn = document.getElementById('saveBtn');
const body = document.body;

let currentImageId = body.dataset.imageId ? Number(body.dataset.imageId) : null;
let reviewData = null;

function setStatus(text) {
  uploadStatus.textContent = text || '';
}

function chip(text, className = 'chip') {
  const span = document.createElement('span');
  span.className = className;
  span.textContent = text;
  return span;
}

function normalizeFaceBoxes(face) {
  // 新版优先：raw_bbox / expanded_bbox
  // 旧版兼容：bbox
  const rawBox = face.raw_bbox || face.bbox || null;
  const expandedBox = face.expanded_bbox || face.bbox || null;
  return { rawBox, expandedBox };
}

async function loadReview(imageId) {
  const res = await fetch(`/api/images/${imageId}/review-data`);
  if (!res.ok) {
    throw new Error('加载 review 数据失败');
  }
  reviewData = await res.json();
  currentImageId = imageId;
  renderReview();
  history.replaceState({}, '', `/review/${imageId}`);
}


function createOverlayBox(box, labelText, color, scaleX, scaleY) {
  const [x1, y1, x2, y2] = box;
  const el = document.createElement('div');
  el.className = 'face-box-tag';
  el.style.left = `${x1 * scaleX}px`;
  el.style.top = `${y1 * scaleY}px`;
  el.style.width = `${(x2 - x1) * scaleX}px`;
  el.style.height = `${(y2 - y1) * scaleY}px`;
  el.style.borderColor = color;
  el.style.boxShadow = `0 0 0 1px ${color} inset`;

  // 只在有类名时显示标签
  if (labelText && labelText.trim()) {
    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = labelText;
    label.style.background = color;
    label.style.color = color === 'yellow' ? '#111' : '#fff';
    el.appendChild(label);
  }

  return el;
}

function renderOverlay() {
  faceOverlay.innerHTML = '';
  if (!reviewData) return;

  const displayedWidth = heroImage.clientWidth;
  const displayedHeight = heroImage.clientHeight;
  const scaleX = displayedWidth / reviewData.width;
  const scaleY = displayedHeight / reviewData.height;

  reviewData.faces.forEach((face) => {
    const expandedBox = face.expanded_bbox || face.bbox || null;
    if (!expandedBox) return;

    // 只显示类名，不显示 F1 / Embedding框 / 原始脸框
    const className =
      face.suggested_clusters?.[0]?.cluster_name ||
      '';

    const expEl = createOverlayBox(
      expandedBox,
      className,
      'yellow',
      scaleX,
      scaleY
    );
    faceOverlay.appendChild(expEl);
  });
}



// function createOverlayBox(box, labelText, color, scaleX, scaleY) {
//   const [x1, y1, x2, y2] = box;
//   const el = document.createElement('div');
//   el.className = 'face-box-tag';
//   el.style.left = `${x1 * scaleX}px`;
//   el.style.top = `${y1 * scaleY}px`;
//   el.style.width = `${(x2 - x1) * scaleX}px`;
//   el.style.height = `${(y2 - y1) * scaleY}px`;
//   el.style.borderColor = color;
//   el.style.boxShadow = `0 0 0 1px ${color} inset`;

//   const label = document.createElement('div');
//   label.className = 'label';
//   label.textContent = labelText;
//   label.style.background = color;
//   label.style.color = color === 'yellow' ? '#111' : '#fff';

//   el.appendChild(label);
//   return el;
// }

// function renderOverlay() {
//   faceOverlay.innerHTML = '';
//   if (!reviewData) return;

//   const displayedWidth = heroImage.clientWidth;
//   const displayedHeight = heroImage.clientHeight;
//   const scaleX = displayedWidth / reviewData.width;
//   const scaleY = displayedHeight / reviewData.height;

//   reviewData.faces.forEach((face, idx) => {
//     const { rawBox, expandedBox } = normalizeFaceBoxes(face);

//     if (rawBox) {
//       const rawEl = createOverlayBox(
//         rawBox,
//         `F${face.face_index + 1} · 原始脸框`,
//         'red',
//         scaleX,
//         scaleY
//       );
//       faceOverlay.appendChild(rawEl);
//     }

//     if (expandedBox) {
//       const suggested = face.suggested_clusters?.[0]?.cluster_name || '未识别';
//       const expEl = createOverlayBox(
//         expandedBox,
//         `F${face.face_index + 1} · Embedding框 · ${suggested}`,
//         'yellow',
//         scaleX,
//         scaleY
//       );
//       faceOverlay.appendChild(expEl);
//     }
//   });
// }

function renderImageCandidates() {
  imageCandidates.innerHTML = '';
  const items = reviewData.image_candidates || [];
  if (!items.length) {
    imageCandidates.appendChild(chip('当前没有图片级候选类', 'chip warn'));
    return;
  }
  items.forEach(item => {
    imageCandidates.appendChild(chip(`${item.cluster_name} · ${item.score}`, 'chip good'));
  });
}

// function createSuggestionChip(face, candidate) {
//   const scoreText =
//     candidate.raw_best_score != null
//       ? Number(candidate.raw_best_score).toFixed(3)
//       : candidate.score != null
//         ? Number(candidate.score).toFixed(3)
//         : '-';

//   const el = chip(`${candidate.cluster_name} · ${scoreText}`, 'chip clickable');
//   el.addEventListener('click', () => {
//     const input = document.querySelector(`input[data-face-id="${face.face_id}"]`);
//     input.value = candidate.cluster_name;
//   });
//   return el;
// }

function createSuggestionCard(face, candidate) {
  const wrap = document.createElement('div');
  wrap.className = 'suggestion-card';

  const img = document.createElement('img');
  img.className = 'suggestion-thumb';
  img.src = candidate.sample_crop_url || '';
  img.alt = candidate.cluster_name;

  const meta = document.createElement('div');
  meta.className = 'suggestion-meta';
  meta.innerHTML = `
    <div class="suggestion-name">${candidate.cluster_name}</div>
    <div class="suggestion-score">
      best=${Number(candidate.raw_best_score).toFixed(3)}
      · top3=${Number(candidate.avg_top3_score).toFixed(3)}
      · hits=${candidate.hit_count}
    </div>
  `;

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'suggestion-pick-btn';
  btn.textContent = '选这个类';
  btn.addEventListener('click', () => {
    const input = document.querySelector(`input[data-face-id="${face.face_id}"]`);
    input.value = candidate.cluster_name;
  });

  wrap.appendChild(img);
  wrap.appendChild(meta);
  wrap.appendChild(btn);
  return wrap;
}

function formatBox(box) {
  if (!box || box.length !== 4) return '无';
  return `(${box.join(', ')})`;
}

function renderFaceCards() {
  faceCards.innerHTML = '';
  const tpl = document.getElementById('faceCardTemplate');

  reviewData.faces.forEach(face => {
    const { rawBox, expandedBox } = normalizeFaceBoxes(face);

    const node = tpl.content.cloneNode(true);
    const card = node.querySelector('.face-card');
    const thumb = node.querySelector('.face-thumb');
    const title = node.querySelector('.face-title');
    const score = node.querySelector('.face-score');
    const box = node.querySelector('.face-box');
    const suggestions = node.querySelector('.face-suggestions');
    const input = node.querySelector('.class-input');
    const fillBtn = node.querySelector('.fill-btn');
    const skipInput = node.querySelector('.skip-input');

    thumb.src = face.crop_url;
    title.textContent = `人脸 F${face.face_index + 1}`;

    const detectorScoreText =
      face.detector_score != null ? Number(face.detector_score).toFixed(3) : '-';
    const suggestedScoreText =
      face.suggested_score != null ? Number(face.suggested_score).toFixed(3) : '无';

    score.textContent = `检测分数：${detectorScoreText} · 建议分数：${suggestedScoreText}`;

    box.innerHTML = `
      <div>红框（原始脸框）：${formatBox(rawBox)}</div>
      <div>黄框（Embedding框）：${formatBox(expandedBox)}</div>
    `;

    input.dataset.faceId = String(face.face_id);

    const bestName = face.suggested_clusters?.[0]?.cluster_name || '';
    input.value = bestName;

    if (face.suggested_clusters?.length) {
      face.suggested_clusters.forEach(candidate => {
        suggestions.appendChild(createSuggestionCard(face, candidate));
      });
      // face.suggested_clusters.forEach(candidate => {
      //   suggestions.appendChild(createSuggestionChip(face, candidate));
      // });
    } else {
      suggestions.appendChild(chip('没有命中已有类，可直接输入新类名', 'chip warn'));
    }

    fillBtn.addEventListener('click', () => {
      input.value = bestName;
    });

    card.dataset.faceId = String(face.face_id);
    card.dataset.faceIndex = String(face.face_index);
    faceCards.appendChild(node);
  });
}

function renderReview() {
  reviewSection.classList.remove('hidden');
  heroImage.src = reviewData.image_url;
  imageMeta.textContent = `图片 ID：${reviewData.image_id} · ${reviewData.width} × ${reviewData.height} · 检测到 ${reviewData.faces.length} 张脸`;
  reviewBadge.textContent = reviewData.review_status || 'pending';

  renderImageCandidates();
  renderFaceCards();

  heroImage.onload = () => renderOverlay();
  if (heroImage.complete) renderOverlay();
}

fillBestBtn?.addEventListener('click', () => {
  document.querySelectorAll('.class-input').forEach(input => {
    if (!input.value.trim()) {
      const card = input.closest('.face-card');
      const idx = Number(card.dataset.faceIndex);
      const best =
        reviewData.faces.find(x => x.face_index === idx)?.suggested_clusters?.[0]?.cluster_name || '';
      input.value = best;
    }
  });
});

saveBtn?.addEventListener('click', async () => {
  if (!currentImageId || !reviewData) return;

  const items = reviewData.faces.map(face => {
    const input = document.querySelector(`input[data-face-id="${face.face_id}"]`);
    const card = input.closest('.face-card');
    const skip = card.querySelector('.skip-input').checked;
    return {
      face_id: face.face_id,
      class_name: input.value.trim() || null,
      cluster_id: null,
      skip,
    };
  });

  const res = await fetch(`/api/images/${currentImageId}/labels`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items }),
  });

  if (!res.ok) {
    const detail = await res.text();
    alert(`保存失败：${detail}`);
    return;
  }

  alert('保存成功，当前图片的人脸标签已写入后端。');
  await loadReview(currentImageId);
});

uploadForm?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = uploadInput.files?.[0];
  if (!file) {
    setStatus('请先选择一张图片。');
    return;
  }

  const fd = new FormData();
  fd.append('file', file);
  setStatus('正在上传并分析，请稍候…');

  const res = await fetch('/api/images/upload', {
    method: 'POST',
    body: fd,
  });

  if (!res.ok) {
    const detail = await res.text();
    setStatus(`上传失败：${detail}`);
    return;
  }

  const data = await res.json();
  setStatus(`上传完成，检测到 ${data.face_count ?? 0} 张脸。`);
  await loadReview(data.image_id);
});

if (currentImageId) {
  loadReview(currentImageId).catch(err => {
    console.error(err);
    setStatus('当前图片加载失败，请检查后端日志。');
  });
}