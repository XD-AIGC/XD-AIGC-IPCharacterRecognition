# album_backend_mvp_face

## 项目文档使用说明在飞书：
https://xd.feishu.cn/wiki/UNoKwDOq2iicxmkLnkFcZhj2nPd

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


