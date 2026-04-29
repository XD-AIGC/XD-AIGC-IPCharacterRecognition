# album_backend_mvp_face

这是基于你之前 `album_backend_mvp` 思路升级的一版：

- **embedding 模型切换为本地 Hugging Face CLIP 目录**
  - 默认路径：`/AIGC_Group/models/hy-motion/clip-vit-large-patch1`
- **不再对整图做 embedding**，而是先检测动漫人脸，再对每一张脸单独做 embedding
- **多脸图像**支持：一张图可关联多个类，每张脸分别给出候选类
- **阈值**：默认余弦相似度阈值 `0.888`
- **标签页面**：新增 `/review/{image_id}`，可视化原图、人脸框、人脸裁剪图，并为每张脸填写/确认类名

## 目录

```text
app/
  main.py
  config.py
  db.py
  models.py
  routers/
    api.py
    ui.py
  services/
    face_detector.py
    face_embedder.py
    face_index.py
    face_matcher.py
    ingest.py
  templates/
    review.html
  static/
    review.css
    review.js
```

## 安装

```bash
conda activate ip_char
pip install -r requirements.txt
```

## 启动

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload
```

## 页面

- Swagger: `http://127.0.0.1:8010/docs`
- 打标签页：上传后返回 `review_url`，或者直接访问 `http://127.0.0.1:8010/review/<image_id>`

## 关键说明

### 1. 本地 CLIP 模型

代码默认用 `transformers.AutoProcessor + AutoModel.from_pretrained(local_path, local_files_only=True)`
去加载本地模型目录。

要求你的本地目录至少是 Hugging Face 可加载的模型目录，通常应包含：

- `config.json`
- `preprocessor_config.json` 或等效处理器配置
- `model.safetensors` 或 `pytorch_model.bin`

如果这个目录不是标准 CLIP/兼容视觉模型目录，需要把 `face_embedder.py` 中的加载方式改成你本地模型对应的类。

### 2. imgutils 的动漫人脸检测

代码默认使用：

```python
from imgutils.detect.face import detect_faces
```

如果你的机器**不能联网**，而 `imgutils` 首次运行时需要拉取检测模型，请提前把它的检测权重缓存好；
或者你自己提供可用的检测模型名/路径，并在配置中设置 `ANIME_FACE_MODEL_NAME`。

### 3. 多脸逻辑

- 一张图可以检测出多张脸
- 每张脸单独搜索相似已标注脸
- 每张脸返回多组候选类
- 整张图会聚合出多个图片级候选类
- **最终确认以人脸为单位**，不是整图只能有一个类

### 4. 数据流

#### 上传时

1. 保存原图
2. 检测所有动漫人脸
3. 裁剪人脸并提 embedding
4. 用已确认的人脸库做 Faiss 检索
5. 给每张脸生成候选类
6. 聚合成整图候选类
7. 返回 `review_url`

#### 打标签时

1. 在页面里为每张脸填写/确认类名
2. 后端创建或复用对应 cluster
3. 将该人脸加入 cluster
4. 把该人脸 embedding 加入 Faiss 索引
5. 后续再来相似脸时就能命中

## 旧项目替换建议

如果你要把这版直接并到原来的 `album_backend_mvp`，最简单的方式是：

- 用这里的 `app/models.py` 替换旧模型定义
- 新增 `app/services/face_*.py` 和 `app/services/ingest.py`
- 用这里的 `app/routers/api.py` 替换原上传逻辑
- 新增 `app/routers/ui.py`、`app/templates/review.html`、`app/static/*`
- 用这里的 `app/main.py` 替换原入口

## 建议的后续增强

- 对同一张图里的人脸和全身图做联合重排
- 给标签页增加“同一图多个脸一键继承现有类名”的快捷操作
- 给 cluster 单独做浏览页
- 对低置信度候选做人审队列
