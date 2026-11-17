# 抗原凝集智能分析平台前端

这是项目的静态前端演示界面，包含分析总览、图片上传预测和实验记录三个页面。

## 本地运行

在项目根目录执行：

```bash
python3 -m http.server 8080 --directory web
```

然后打开 <http://localhost:8080>。

## 接入真实模型

前端默认调用服务器上的 `POST https://gx10-aeef.tail1f6601.ts.net/api/predict`，字段名为 `file`。接口返回 JSON 时支持以下字段：

```json
{"diseased_probability": 0.83}
```

如果后端接口不可用，页面会自动使用演示结果，并在界面上标注“演示推理”，不会把演示结果误认为真实模型输出。当前最终阈值为 `0.575848937`。

如果部署到其他后端地址，可在加载 `app.js` 前设置：

```html
<script>window.API_BASE_URL = 'https://你的后端地址';</script>
```
