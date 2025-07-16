## README

### 1. 项目背景

随着人工智能技术的快速发展，计算机视觉成为了现代 AI 应用中的重要领域之一。在图像分类任务中，基于深度学习的卷积神经网络（CNN）已经成为主流的解决方案，广泛应用于物体检测、面部识别等多个领域。猫狗识别作为一个经典的计算机视觉任务，旨在通过计算机自动识别图像中的猫狗，是入门深度学习模型训练的热门选择。

本项目以 **猫狗图像分类** 为任务，采用 **ResNet-18** 网络进行训练。利用深度学习框架 **PyTorch** 构建和训练分类模型，并通过 **Flask** 框架实现了一个简单的 **Web 应用**，用户可以在前端页面上传猫狗图片，系统会返回分类结果（猫或狗）及预测置信度。该系统的实现展示了深度学习与 Web 开发技术的结合，能够为用户提供便捷的在线图像分类服务。

![image-20250703142551693](D:\文化课资料\Python程序设计\大作业\识别结果-总)

### 2. 图像分类模块

#### 2.1 模块架构

图像分类模块主要由以下几个部分构成：

- **数据加载与预处理 (data.py)** ：从数据源 `PetImage/` 文件夹加载图像数据，按照给定的 `train_ratio, val_ratio, test_ratio` 分类构建数据集，并进行必要的图像缩放、裁剪、标准化等预处理；
- **模型定义 (model.py)**：从 torchvision 库导入分类模型 ResNet-18 并设置输出为分类为猫/狗可能性的二维向量，设置学习率、批次等超参数；
- **模型训练  (train.py)**：使用验证集对模型进行评估，优化模型参数以提高分类性能；
- **模型测试 (test.py)**：训练完成后使用测试集进行最终评估，计算模型的准确率和混淆矩阵；
- **分类器封装 (classifier.py)**：将训练好的模型封装成一个易于调用的分类器，供 Web 应用进行调用；
- **调试入口 (main.py)**：允许通过 `python main.py --mode [train/test]` 进行训练和测试；
- **全局设置 (settings.py)**：定义 `device, model_path` 等全局参数。

```
+---------------------+   +---------------------+   +---------------------+
|    数据加载与预处理    |-->|     模型定义与训练    |-->|     评估与测试模块    |
|   (Image Loading)   |   |  (Model Definition) |   | (Evaluation & Test) |
+---------------------+   +---------------------+   +---------------------+
                                                               |
                        +---------------------+     +---------------------+
                        |   Web应用（Flask）   | <-- |    分类器封装模块      |
                        |    (User Interface) |     |  (Classifier Wrap)  |
                        +---------------------+     +---------------------+
```

#### 2.2 图像预处理

- `split_and_copy`：从数据源 `PetImage/` 文件夹下读取图片，检查并丢弃不可用图片，按照 `train_ratio, val_ratio, test_ratio` 分类构建 `data/train, data/val, data/test`（使用 `symlink_to` 创建软链接，避免大量复制)；
- 设置 `train_transforms, val_test_transforms`，训练集数据随机水平反转，增强模型识别能力；
- 定义 `dataset, dataloader`，设置变换、批量和乱序参数。 

#### 2.3 模型训练/测试

- 采用 CrossEntropy 损失函数，Adam 优化器；
- $\text{epoch}=20, \text{learning\_rate}=0.001, \text{batch\_size}=128, \text{val\_loss}\space 3$ 次高于最小值触发早停。

![新建 BMP 图像](D:\python-class\大作业\新建 BMP 图像.bmp)

#### 2.4 分类器封装

为方便网页 APP 调用分类器获取分类标签和预测结果，封装 `classifier.py` 分类器，对外提供 `ImageClassifier.predict(image)` 方法，将模型输出 (logits) 取 softmax 得到概率分布张量 `probability`，取 max 返回分类标签和置信度。

### 3. 网页前端模块

#### 3.1 模块结构

```
webapp/
├── app.py          # Flask 应用的入口，处理路由和请求
├── templates/
│   └── index.html  # 前端页面，允许用户上传图片并显示结果
├── static/
│   └──  uploads/   # 存储用户上传的图片等临时文件
```

#### 3.2 Flask 框架

Webapp 部分使用了 **Flask** 框架，Flask 是一个轻量级的 Python Web 框架，适合快速构建简单的 Web 应用，方便将不同的功能模块映射到不同的 URL 路径。

- **`app.py`**：Flask 应用的主要入口。此文件包含了所有的路由配置和请求处理逻辑。它处理了用户的上传请求，将图片裁剪为相同尺寸 (224*224)，保存到 `static/uploads/`  并传给分类器，获取分类结果，最后将结果打包渲染到网页上。每次启动前清空 `static/uploads/` 缓存；
- **`index.html`**：前端页面，使用 HTML 表单来上传图片，并显示分类结果和置信度。

```py
# index() 方法打包返回 result 
results.append({
    'filename': filename,
    'url': url_for('static', filename='uploads/' + filename), # 指定图片位置，用于渲染原图
    'label': label,
    'confidence': f"{confidence * 100:.2f}%" # 按百分比形式显示置信度
})
```

#### 3.3 前端网页设计

前端部分主要由 **HTML** 和 **CSS** 组成，通过一个简单的表单允许用户上传图片，并展示分类结果。网页设计保持简洁，主要关注用户体验。用户上传图片后，页面会显示原图、分类结果和模型的置信度。

```html
htmlCopyEdit<form method="POST" enctype="multipart/form-data">
    <input type="file" name="file" required>
    <input type="submit" value="上传并分类">
</form>

{% if label %}
    <div>预测结果：{{ label }}</div>
    <div>置信度：{{ confidence * 100 }}%</div>
    <img src="{{ file_path }}" alt="上传图片" />
{% endif %}
```

在这部分的设计中，我们通过 **Flask** 的模板引擎 `Jinja2` 实现了动态内容的插入。分类结果和置信度在服务器端处理后通过 `render_template` 函数传递给前端进行显示。

