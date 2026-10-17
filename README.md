# 颜值打分系统 v2 / Facial Beauty Score App

一个面向 Windows 的本地人脸颜值评分应用。输入照片后，程序在本地完成人脸检测、姿态与质量检查、关键点几何特征提取和神经网络评分，不上传用户图片。

> 本项目仅供学习、研究与娱乐，评分不能代表一个人的真实价值，也不应被用于招聘、婚恋、信贷、医疗等现实决策。

## 功能

- Tkinter 图形界面，支持选择本地图片并显示原图、人脸裁剪图和评分结果。
- 支持 `SCUT`、`MEBeauty`、`众包` 三套评分权重，以及三模型平均。
- 支持通用、男性专用、女性专用三种 SCUT 路由，界面默认使用通用模型。
- 无人脸、侧脸过大、俯仰角过大、人脸过小、图像过糊时拒绝评分或降低置信度。
- 显示原始分、校准分、置信度、人脸姿态和图像质量提示。
- 支持命令行批量推理和 JSON 输出。
- 使用 CPU 版 PyTorch，打包后的 onedir 目录约 1.25 GB。

## 下载 Windows 应用

请到本仓库的 **Releases** 页面下载：

1. `beauty-score-app-v2.1.0-windows-x64.zip`：完整 Windows 应用。
2. `weights_v2-v2.1.0.zip`：源码运行所需的模型权重。

下载后**必须解压整个压缩包**，然后运行：

```text
颜值打分系统_v2\颜值打分系统_v2.exe
```

这是 PyInstaller onedir 构建，不能只拷贝 exe；`_internal` 目录必须和 exe 放在一起。应用未做代码签名，Windows SmartScreen 可能提示未知发布者，可选择“更多信息 -> 仍要运行”。

## 命令行用法

```powershell
.\颜值打分系统_v2.exe --image "C:\path\face.jpg" --track scut --gender generic --tta --out-json result.json
```

参数：

- `--track scut|meb|crowd|ensemble`
- `--gender generic|male|female`
- `--tta`：开启水平翻转测试时增强
- `--json`：直接输出 JSON
- `--out-json result.json`：写入结果文件

## 从源码运行

建议使用 Python 3.11。

```powershell
conda create -n face-score python=3.11 -y
conda activate face-score
pip install -r requirements.txt
```

把 Release 中的 `weights_v2-v2.1.0.zip` 解压到项目根目录，使目录结构如下：

```text
face-score-app/
  app/
  app_test.py
  weights_v2/
    scut/best.pth
    scut_male/best.pth
    scut_female/best.pth
    meb/best.pth
    crowd/best.pth
```

启动 GUI：

```powershell
python app_test.py
```

源码命令行：

```powershell
python app_test.py --image "C:\path\face.jpg" --track scut --gender generic --tta
```

## 构建 Windows 应用

安装构建依赖：

```powershell
pip install -r requirements-build.txt
```

确认 `weights_v2` 已放在项目根目录，然后运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
```

输出位于 `dist\颜值打分系统_v2\`。

## 模型与真实指标

SCUT-FBP5500 官方 fold1 测试集共 1100 张，标签从 1–5 分乘 2 映射到 1–10 分。评估开启水平翻转 TTA。

| 模型 | 样本数 | MAE | Pearson | hit@0.5 |
|---|---:|---:|---:|---:|
| 通用 SCUT | 1100 | 0.4310 | 0.9114 | 67.18% |
| 男性专用 | 525 | 0.3962 | 0.9102 | 71.81% |
| 女性专用 | 575 | 0.4455 | 0.9158 | 64.70% |
| 男女合并部署 | 1100 | 0.4220 | 0.9140 | 68.09% |

目标 `|误差| <= 0.5` 达到 80%，**实际没有达到**，真实结果为 68.09%。男性专用模型有效，女性专用模型在女性子集上低于通用模型，因此 GUI 默认使用通用模型。

MEBeauty 跨域测试的 `hit@0.5` 为 31.62%，众包标签测试约 27%。众包标签自身随机分半的 `±0.5` 一致率仅约 33.4%，说明该测试集的标注噪声上限很低。完整数据见 `docs/gender_model_report.md`。

## 数据来源与引用

### SCUT-FBP5500

本项目的训练与评估使用了华南理工大学 HCIILAB 发布的 SCUT-FBP5500 数据集。数据集图片、标签和原始文件**没有**包含在本仓库中，请从原项目获取：

- <https://github.com/HCIILAB/SCUT-FBP5500-Database-Release>

引用：

```bibtex
@article{liang2017SCUT,
  title   = {SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction},
  author  = {Liang, Lingyu and Lin, Luojun and Jin, Lianwen and Xie, Duorui and Li, Mengru},
  journal = {ICPR},
  year    = {2018}
}
```

SCUT-FBP5500 仅允许用于非商业研究用途。使用本项目权重或复现实验时，请同时遵守原数据集的使用条款。

### MEBeauty

跨文化评估使用了 MEBeauty 数据集。数据集图片和标签**没有**包含在本仓库中，请从原项目获取：

- <https://github.com/fbplab/MEBeauty-database>

引用：

```bibtex
@article{lebedeva2021mebeauty,
  title={MEBeauty: a multi-ethnic facial beauty dataset in-the-wild},
  author={Lebedeva, Irina and Guo, Yi and Ying, Fangli},
  journal={Neural Computing and Applications},
  pages={1--15},
  year={2021},
  publisher={Springer}
}
```

## 隐私

- 推理在本地完成，应用不会上传图片。
- 本仓库不包含爬虫原图、用户照片、众包标注明细、登录凭据或内网穿透配置。
- 分发应用时请勿把他人照片或未授权的网络图片一起打包。

## 分发方式

推荐使用 GitHub Releases：

1. 在本地生成新的应用压缩包。
2. 打开仓库的 `Releases -> Draft a new release`。
3. 选择新 tag，例如 `v2.1.0`。
4. 上传完整应用压缩包和权重压缩包。
5. 发布后分享 `https://github.com/Ressian/face-score-app/releases/latest`。

不要把 1 GB 以上的 `dist` 目录直接提交进 Git 仓库。

## 许可证

本仓库暂未声明应用代码的开源许可证。第三方库、数据集和模型权重分别受其原作者条款约束，详见 `NOTICE.md`。如果你需要使用本仓库代码，请先联系仓库作者。

