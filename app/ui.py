# -*- coding: utf-8 -*-
import os
import threading
import traceback

from . import config


def run():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from PIL import Image, ImageOps, ImageTk

    from .engine import ScoreEngine, format_result

    root = tk.Tk()
    root.title("%s %s" % (config.APP_NAME, config.APP_VERSION))
    root.geometry("1060x760")
    root.minsize(900, 640)
    root.configure(bg="#f4f6fa")

    try:
        root.iconbitmap(os.path.join(config.resource_root(), "app.ico"))
    except Exception:
        pass

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TButton", font=("Microsoft YaHei UI", 10))
    style.configure("TCombobox", font=("Microsoft YaHei UI", 10))
    style.configure("TCheckbutton", font=("Microsoft YaHei UI", 9), background="#f4f6fa")

    engine_holder = {"engine": None, "busy": False, "photo": None, "photo2": None}
    tracks = config.available_tracks()
    track_names = [config.TRACK_LABELS.get(t, t) for t in tracks]
    genders = config.available_genders() or ["generic"]
    gender_names = [config.GENDER_LABELS.get(g, g) for g in genders]
    label_to_gender = dict(zip(gender_names, genders))
    default_gender = "generic" if "generic" in genders else genders[0]
    if len(tracks) >= 2:
        track_names.append("三模型平均")
    label_to_track = dict(zip(track_names, tracks))
    if len(tracks) >= 2:
        label_to_track["三模型平均"] = "ensemble"

    header = tk.Frame(root, bg="#1f2a44", height=64)
    header.pack(fill="x")
    header.pack_propagate(False)
    tk.Label(header, text="颜值打分 v2", bg="#1f2a44", fg="white",
             font=("Microsoft YaHei UI", 18, "bold")).pack(side="left", padx=22)
    tk.Label(header, text="224px · 1-10 分 · SCUT / MEBeauty / 众包",
             bg="#1f2a44", fg="#aab6d4", font=("Microsoft YaHei UI", 10)).pack(side="left")

    body = tk.Frame(root, bg="#f4f6fa")
    body.pack(fill="both", expand=True, padx=16, pady=14)

    left = tk.Frame(body, bg="white", bd=0, highlightthickness=1,
                    highlightbackground="#d9deea")
    left.pack(side="left", fill="both", expand=False, padx=(0, 12))
    right = tk.Frame(body, bg="white", bd=0, highlightthickness=1,
                     highlightbackground="#d9deea")
    right.pack(side="left", fill="both", expand=True)

    img_label = tk.Label(left, text="尚未选择图片\n\n点击下方按钮选择一张人脸照片",
                         bg="#eef1f7", fg="#6b7690", width=46, height=17,
                         font=("Microsoft YaHei UI", 10), justify="center")
    img_label.pack(padx=14, pady=(14, 8))
    crop_label = tk.Label(left, text="对齐后的人脸裁切会显示在这里",
                          bg="#eef1f7", fg="#6b7690", width=46, height=8,
                          font=("Microsoft YaHei UI", 9), justify="center")
    crop_label.pack(padx=14, pady=(0, 10))

    controls = tk.Frame(left, bg="white")
    controls.pack(fill="x", padx=14, pady=(0, 14))
    btn = ttk.Button(controls, text="选择图片并打分")
    btn.pack(fill="x", pady=(0, 8))

    row = tk.Frame(controls, bg="white")
    row.pack(fill="x", pady=(0, 6))
    tk.Label(row, text="评分路线", bg="white", font=("Microsoft YaHei UI", 9)).pack(side="left")
    track_var = tk.StringVar(value=track_names[0])
    combo = ttk.Combobox(row, textvariable=track_var, values=track_names,
                         state="readonly", width=18)
    combo.pack(side="right")

    grow = tk.Frame(controls, bg="white")
    grow.pack(fill="x", pady=(0, 6))
    tk.Label(grow, text="性别模型", bg="white", font=("Microsoft YaHei UI", 9)).pack(side="left")
    gender_var = tk.StringVar(value=config.GENDER_LABELS.get(default_gender, default_gender))
    gcombo = ttk.Combobox(grow, textvariable=gender_var, values=gender_names,
                          state="readonly", width=18)
    gcombo.pack(side="right")

    tta_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(controls, text="水平翻转 TTA（更稳但更慢）",
                    variable=tta_var).pack(anchor="w")
    calib_main_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(controls, text="主显示校准分（分布更接近 1-10）",
                    variable=calib_main_var).pack(anchor="w")

    score_var = tk.StringVar(value="—")
    score_label = tk.Label(right, textvariable=score_var, bg="white", fg="#1f2a44",
                           font=("Microsoft YaHei UI", 54, "bold"))
    score_label.pack(anchor="w", padx=26, pady=(20, 0))
    sub_var = tk.StringVar(value="选择图片后开始")
    tk.Label(right, textvariable=sub_var, bg="white", fg="#6b7690",
             font=("Microsoft YaHei UI", 10)).pack(anchor="w", padx=28)

    divider = tk.Frame(right, bg="#e6e9f1", height=1)
    divider.pack(fill="x", padx=26, pady=12)

    result_text = tk.Text(right, bg="white", fg="#26314d", bd=0, wrap="word",
                          font=("Microsoft YaHei UI", 10), padx=26, pady=2,
                          spacing1=2, spacing3=2)
    result_text.pack(fill="both", expand=True, padx=0, pady=(0, 8))
    result_text.insert("1.0", "使用说明\n\n"
                       "1. 选择一张人脸照片，程序会自动检测最大人脸、做滚转对齐并裁切。\n"
                       "2. 综合分是 SCUT 主模型的原始输出；分布校准分用于修正 1-10 分分布，"
                       "两者排序一致，但逐点误差不同，验收以综合分为准。\n"
                       "3. 无人脸、偏头超过 30°、俯仰超过 32° 会直接拒绝打分。\n"
                       "4. 远景、模糊、磨皮、仰俯视会给出可靠性提示，并在“质量修正参考分”里"
                       "做透明的启发式补偿。\n")
    result_text.insert("end", "\n5. 可切换女性/男性专用模型；SCUT 主分会使用对应权重，避免男女分数分布偏差。\n")
    result_text.configure(state="disabled")

    status_var = tk.StringVar(value="就绪")
    status = tk.Label(root, textvariable=status_var, bg="#e8ecf5", fg="#4d5875",
                      anchor="w", font=("Microsoft YaHei UI", 9), padx=16, pady=6)
    status.pack(fill="x", side="bottom")

    def set_result_text(text):
        result_text.configure(state="normal")
        result_text.delete("1.0", "end")
        result_text.insert("1.0", text)
        result_text.configure(state="disabled")

    def show_image(path, pack):
        try:
            im = Image.open(path).convert("RGB")
            im = ImageOps.exif_transpose(im)
            im.thumbnail((432, 330), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(im)
            engine_holder["photo"] = photo
            img_label.configure(image=photo, text="", width=432, height=330)
        except Exception:
            img_label.configure(image="", text="图片预览失败", width=46, height=17)
        try:
            crop = pack.get("crop")
            if crop is not None:
                crop = crop.copy()
                crop.thumbnail((432, 150), Image.Resampling.LANCZOS)
                photo2 = ImageTk.PhotoImage(crop)
                engine_holder["photo2"] = photo2
                crop_label.configure(image=photo2, text="", width=432, height=150)
            else:
                crop_label.configure(image="", text="未生成对齐裁切", width=46, height=8)
        except Exception:
            pass

    def ensure_engine():
        if engine_holder["engine"] is None:
            engine_holder["engine"] = ScoreEngine(tracks=tracks, tta=False)
        engine_holder["engine"].tta = bool(tta_var.get())
        return engine_holder["engine"]

    def worker(path, track, gender):
        try:
            eng = ensure_engine()
            result = eng.predict(path, track=track, gender=gender)
            root.after(0, lambda: finish(path, result))
        except Exception as exc:
            tb = traceback.format_exc()
            root.after(0, lambda: fail(exc, tb))

    def finish(path, result):
        engine_holder["busy"] = False
        btn.configure(state="normal")
        show_image(path, result)
        if not result.get("ok"):
            score_var.set("--.-")
            sub_var.set("无法打分")
            set_result_text("无法打分：%s" % result.get("reason", "未知错误"))
            status_var.set("未通过质量门控")
            return
        if calib_main_var.get():
            score_var.set("%.2f" % result["score_calibrated"])
            sub_var.set("校准分 / 10　·　原始 %.2f　·　置信度 %.0f%%（%s）　·　%.2fs" % (
                result["score_raw"], result["confidence"] * 100,
                result["reliability"], result["elapsed"]))
        else:
            score_var.set("%.2f" % result["score_raw"])
            sub_var.set("原始分 / 10　·　校准 %.2f　·　置信度 %.0f%%（%s）　·　%.2fs" % (
                result["score_calibrated"], result["confidence"] * 100,
                result["reliability"], result["elapsed"]))
        set_result_text(format_result(result))
        note = result.get("model_label", "")
        if note:
            sub_var.set(sub_var.get() + " | " + note)
        status_var.set("完成：%s" % os.path.basename(path))

    def fail(exc, tb):
        engine_holder["busy"] = False
        btn.configure(state="normal")
        score_var.set("--.-")
        sub_var.set("处理失败")
        set_result_text("处理失败：%s\n\n%s" % (exc, tb))
        status_var.set("处理失败")

    def choose_file():
        if engine_holder["busy"]:
            return
        path = filedialog.askopenfilename(
            title="选择人脸图片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                       ("所有文件", "*.*")])
        if not path:
            return
        engine_holder["busy"] = True
        btn.configure(state="disabled")
        score_var.set("…")
        sub_var.set("正在检测人脸并推理")
        set_result_text("正在处理：%s\n\n首次运行需要加载模型，通常 3-8 秒。" % path)
        status_var.set("正在处理…")
        track = label_to_track.get(track_var.get(), tracks[0])
        gender = label_to_gender.get(gender_var.get(), "generic")
        threading.Thread(target=worker, args=(path, track, gender), daemon=True).start()

    btn.configure(command=choose_file)

    def on_close():
        root.destroy()
        os._exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

