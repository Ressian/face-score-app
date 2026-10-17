# 颜值打分系统 v2 / Facial Beauty Score App

一个面向 Windows 的本地人脸颜值评分应用。输入照片后，程序在本地完成人脸检测、姿态与质量检查、关键点几何特征提取和神经网络评分，不上传用户图片。

> 本项目仅供学习、研究与娱乐，评分不能代表一个人的真实价值，也不应被用于招聘、婚恋、信贷、医疗等现实决策。

## 致谢

非常非常非常感谢华南理工大学团队发布 SCUT-FBP5500 数据集并提供源码参考。没有你们的数据集和代码参考，这项研究完全是空中楼阁。

- 源项目地址：<https://github.com/HCIILAB/SCUT-FBP5500-Database-Release>

同时也感谢帮助完成数据标注的伙伴们，是你们让模型的性能更进一步。

## 功能

- Tkinter 图形界面，支持选择本地图片并显示原图、人脸裁剪图和评分结果。
- 支持 `SCUT`、`MEBeauty`、`众包` 三套评分权重，以及三模型平均。
- 支持通用、男性专用、女性专用三种 SCUT 路由，界面默认使用通用模型。
- 无人脸、侧脸过大、俯仰角过大、人脸过小、图像过糊时拒绝评分或降低置信度。
- 显示原始分、校准分、置信度、人脸姿态和图像质量提示。
- 支持命令行批量推理和 JSON 输出。
- 使用 CPU 版 PyTorch，打包后的 onedir 目录约 1.25 GB。

## 技术栈

| 层面 | 技术 |
|---|---|
| 语言与运行环境 | Python 3.11、Windows x64 |
| 深度学习 | PyTorch 2.9.1+cpu、torchvision 0.24.1+cpu；训练阶段使用 CUDA 环境，推理阶段使用 CPU |
| 人脸检测与关键点 | MediaPipe Face Detection、Face Mesh（468 点） |
| 图像处理 | OpenCV、NumPy、Pillow |
| 模型结构 | ResNet18 主干 + 14 维几何特征 MLP + 特征融合头 |
| 训练策略 | 多来源数据清单、加权回归损失、数据增强、验证集选点、水平翻转 TTA、分性别微调 |
| 评估指标 | MAE、RMSE、Pearson、Spearman、`hit@0.2/0.5/1.0` |
| 桌面界面 | Tkinter |
| 打包发布 | PyInstaller 6.22.3、onedir、CPU 推理 |

模型输入为对齐后的人脸图像和 14 维几何特征。几何特征包括面部比例、对称性、眼裂宽度、鼻宽、口宽、眉眼间距、脸型长宽比等。网络输出评分和辅助置信度；应用实际展示的置信度还会结合人脸姿态、清晰度和磨皮程度进行质量校准。

## 实现路线

### 1. 数据与标签统一

- SCUT-FBP5500 原始标签为 1–5 分，按 ×2 映射到 1–10 分。
- MEBeauty 原生使用 1–10 分，不重复乘 2。
- 建立 SCUT、MEBeauty、众包三套训练清单，严格区分训练集、验证集和测试集。
- 众包数据经过“不可评一票否决”、标注员权重和聚合分数清洗后参与训练。

### 2. 人脸预处理与质量控制

- 使用 MediaPipe 检测最大人脸并提取 468 个关键点。
- 根据眼睛、鼻尖和轮廓点估计 yaw、pitch、roll。
- 对侧脸、俯仰角过大、人脸过小、图像过糊、磨皮过度的输入进行拒绝或降低置信度。
- 统一裁剪、对齐并缩放到 224×224，再送入模型。

### 3. 模型训练

- 以 ResNet18 为视觉主干，提取人脸外观特征。
- 将 14 维几何特征通过 MLP 编码后与图像特征拼接融合。
- 训练时使用随机裁剪、旋转、模糊、降采样、亮度对比度变化和水平翻转等增强。
- 以验证集 `hit@0.5` 为主要选点依据，测试集只做最终评估。
- 在通用模型基础上分别微调男性、女性专用权重。

### 4. 评估与校准

- 使用 MAE、RMSE、Pearson、Spearman 和 `hit@0.2/0.5/1.0` 评估。
- 开启水平翻转 TTA，降低单次推理波动。
- 对质量不达标的图片做分数收缩和置信度修正。
- 试过 288/320 高分辨率、通用+性别融合、仿射/Isotonic 校准、4/5 模型集成和纯 L1 精调；这些方案在验证集或女性测试集上没有稳定收益，因此未部署。

### 5. 应用与发布

- `app/preprocess.py` 负责人脸检测、对齐、关键点和质量门控。
- `app/features.py` 计算 14 维几何特征。
- `app/model.py` 定义与权重完全一致的 ResNetFusion 结构。
- `app/engine.py` 负责权重路由、TTA、校准、质量修正和结果组装。
- `app/ui.py` 提供 Tkinter 界面；`app_test.py` 同时支持 GUI 和命令行。
- 使用 PyInstaller 打包为 Windows onedir 应用，并通过 GitHub Releases 分发。

## 项目历程

### 阶段一：原应用整理与可运行化

- 从 conda `py311` 环境开始，解决了启动联网等待、MediaPipe 日志、中文路径、窗口关闭后残留进程等问题。
- 关闭控制台、加入图标，并将启动依赖固定为 CPU 推理。

### 阶段二：v2 架构重构

- 将原单文件程序拆分为预处理、特征、模型、推理引擎、界面和配置模块。
- 迁移到 Tkinter 桌面界面，保留命令行批量测试入口。
- 重新整理权重目录和校准文件，避免不同路线之间混用校准参数。

### 阶段三：数据扩充与众包标注

- 引入 MEBeauty 跨文化数据，并补充网络采集样本。
- 搭建人工标注页面，按标注员可靠性加权；带“不可评”标记的样本直接剔除。
- 将众包数据用于训练和独立测试，同时保留其标注噪声上限分析。

### 阶段四：模型迭代与分性别权重

- 完成 SCUT、MEBeauty、众包三套权重。
- 在通用 SCUT 权重上分别微调男性、女性专用模型。
- 完成 SCUT、MEBeauty 和众包测试集评估，记录失败方案和真实指标。
- 最终 SCUT 部署组合的 `hit@0.5` 为 68.09%，未达到 80% 目标。

### 阶段五：发布与分发

- 完成 Windows 应用打包，完整 onedir 目录约 1.25 GB，压缩包约 603 MiB。
- 建立 GitHub 仓库和 `v2.1.0` Release，分离代码、权重和可执行程序。
- 数据集、爬虫原图、众包标注明细和本地凭据不进入公开仓库。

## 使用建议（实测）

目前测试下来，推荐以下两种组合：

- **通用 + SCUT 主模型**：给分相对慷慨，范围大致为 **1–9 分**，适合日常正脸照片。
- **通用 + 三权重平均模型**：给分相对谨慎，范围大致为 **1–7.5 分**，适合希望分数更保守的场景。

上传时建议：

- 优先上传**正脸、无遮挡**（如证件照）照片，眼睛必须可见，不要佩戴眼罩或口罩。
- 下方“水平翻转”和“主显示校验分”一般**保持叉选**即可（就是关闭），通常不需要打开。
- **不建议**上传过度美颜、重度磨皮、AIP图或侧脸照片；这些输入会显著降低评分可信度，并且给分也不准确。

## 给分小贴士

### 通用 + SCUT 主模型

- **1–2 分**：极丑或程序 bug（大概率是后者）。
- **2–3 分**：丑。
- **3–4.5 分**：一般普通人。
- **4.5–5 分**：还行普通人。
- **5–6.5 分**：小美 / 小帅 / 得体。
- **6.5–7.5 分**：美 / 帅。
- **7.5 分以上**：几乎很少出现，属于顶美 / 顶帅。

### 通用 + 三权重平均模型

三权重平均模型给分相对谨慎（约 1–7.5 分），上述判据会进一步收紧：

- **5 分是分水岭**。
- 5 分以上区分度不是很高。
- 5 分以下基本参照上文 SCUT 主模型的分段理解，看看即可。

## 免责声明

**本模型仅仅仅仅仅仅仅供技术实践参考和娱乐，千万千万千万千万千万千万千千万万不要当真！！！！！！！**

如果对评分体系有意见，**那就是您对**。

美丑不由分数定义。这个模型仅为一个维度，阳光下的人也有阴影，不要因为所谓的分数置气。

## 已知问题

- **MEBeauty（MyBeauty）权重暂不推荐使用**：实测给分主要集中在 1–5 分，整体偏保守，且偶尔出现固定 **1.00 分**的异常结果。
- MEBeauty 数据集本身质量很好，推测问题出在标签读取或分数映射环节，后续会继续排查修复。

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

