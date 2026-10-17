# -*- mode: python ; coding: utf-8 -*-
"""颜值打分 v2（Tkinter 界面 + PyTorch CPU 推理）PyInstaller 配置。

构建环境：conda env py311，torch 2.9.1+cpu / torchvision 0.24.1+cpu。
构建命令（项目根目录）：
  D:\anaconda\envs\py311\python.exe -m PyInstaller --noconfirm --clean 颜值打分系统_v2.spec
产物：dist\颜值打分系统_v2\颜值打分系统_v2.exe（onedir，需整目录分发）
"""
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

PROJECT_DIR = os.path.abspath(os.getcwd())
ENV_LIB_BIN = os.path.join(sys.prefix, "Library", "bin")
ICON_PATH = os.path.join(PROJECT_DIR, "app.ico")

binaries = []
datas = []
hiddenimports = [
    "app",
    "app.config",
    "app.features",
    "app.model",
    "app.preprocess",
    "app.engine",
    "app.ui",
    "tkinter",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "_tkinter",
    "torchvision.transforms.v2",
    "torchvision.transforms.v2.functional",
    "matplotlib",
    "matplotlib.backends.backend_agg",
    "google.protobuf",
    "google.protobuf.internal",
    "scipy.special._ufuncs",
    "scipy.special._ufuncs_cxx",
    "scipy.special._special_ufuncs",
    "scipy._cyutility",
]

datas.append((os.path.join(PROJECT_DIR, "weights_v2"), "weights_v2"))
datas.append((os.path.join(PROJECT_DIR, "app.ico"), "."))

binaries += collect_dynamic_libs("scipy")

BLAS_DLLS = [
    "liblapack.dll", "libblas.dll", "libcblas.dll",
    "mkl_rt.2.dll", "mkl_core.2.dll", "mkl_intel_thread.2.dll",
    "mkl_sequential.2.dll", "mkl_def.2.dll", "mkl_avx2.2.dll",
    "mkl_avx512.2.dll", "mkl_mc3.2.dll",
]
for _dll in BLAS_DLLS:
    _p = os.path.join(ENV_LIB_BIN, _dll)
    if os.path.isfile(_p):
        binaries.append((_p, "."))

mp_datas, mp_binaries, mp_hiddenimports = collect_all("mediapipe")
datas += mp_datas
binaries += mp_binaries
hiddenimports += mp_hiddenimports

EXCLUDES = [
    "flet", "flet_desktop", "flet_core",
    "New_Nets", "retinaface", "albumentations",
    "streamlit", "chromadb", "langchain", "langchain_core", "langchain_community",
    "langchain_openai", "langchain_text_splitters", "langchain_classic", "langgraph",
    "langsmith", "openai", "ollama", "dashscope", "anthropic",
    "fastapi", "starlette", "uvicorn", "kubernetes", "sqlalchemy",
    "pandas", "pyarrow", "altair", "pydeck", "openpyxl", "PyMySQL", "aiomysql",
    "sklearn", "scikit-learn", "transformers", "tokenizers", "huggingface_hub",
    "tensorflow", "keras", "jax", "onnx", "onnxruntime", "sentence_transformers",
    "gdown", "cookiecutter", "build", "IPython", "pytest", "notebook",
    "seaborn", "plotly", "bokeh",
]

a = Analysis(
    ["app_test.py"],
    pathex=[PROJECT_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="颜值打分系统_v2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=ICON_PATH,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="颜值打分系统_v2",
)

