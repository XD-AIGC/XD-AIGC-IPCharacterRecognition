async function loadClasses() {
  const res = await fetch('/api/clusters');
  if (!res.ok) {
    throw new Error('加载类列表失败');
  }
  return await res.json();
}

function renderClassCard(item) {
  const card = document.createElement('a');
  card.className = 'class-card';
  card.href = `/classes/${item.cluster_id}`;

  const title = document.createElement('h3');
  title.textContent = item.name;

  const meta = document.createElement('div');
  meta.className = 'class-meta';
  meta.textContent = `类ID：${item.cluster_id} · 样本数：${item.face_count}`;

  const sampleGrid = document.createElement('div');
  sampleGrid.className = 'sample-grid';

  (item.samples || []).forEach(s => {
    const img = document.createElement('img');
    img.src = s.crop_url;
    img.loading = 'lazy';
    sampleGrid.appendChild(img);
  });

  card.appendChild(title);
  card.appendChild(meta);
  card.appendChild(sampleGrid);
  return card;
}

(async function init() {
  const grid = document.getElementById('classGrid');
  try {
    const data = await loadClasses();
    (data.items || []).forEach(item => grid.appendChild(renderClassCard(item)));
  } catch (err) {
    grid.innerHTML = `<div>加载失败：${err.message}</div>`;
  }
})();