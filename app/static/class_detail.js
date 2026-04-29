const body = document.body;
const clusterId = Number(body.dataset.clusterId);

async function loadClusterDetail(id) {
  const res = await fetch(`/api/clusters/${id}`);
  if (!res.ok) {
    throw new Error('加载类详情失败');
  }
  return await res.json();
}

function renderFaceItem(face) {
  const item = document.createElement('div');
  item.className = 'face-item';

  const img = document.createElement('img');
  img.src = face.crop_url;
  img.loading = 'lazy';

  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.innerHTML = `
    <div>face_id：${face.face_id}</div>
    <div>image_id：${face.image_id}</div>
    <div>检测分数：${face.detector_score ?? '-'}</div>
    <div>相似度：${face.similarity_score ?? '-'}</div>
  `;

  item.appendChild(img);
  item.appendChild(meta);
  return item;
}

(async function init() {
  const title = document.getElementById('clusterTitle');
  const meta = document.getElementById('clusterMeta');
  const gallery = document.getElementById('faceGallery');

  try {
    const data = await loadClusterDetail(clusterId);
    title.textContent = data.cluster.name;
    meta.textContent = `类ID：${data.cluster.cluster_id} · 样本数：${data.cluster.face_count}`;
    (data.faces || []).forEach(face => gallery.appendChild(renderFaceItem(face)));
  } catch (err) {
    gallery.innerHTML = `<div>加载失败：${err.message}</div>`;
  }
})();