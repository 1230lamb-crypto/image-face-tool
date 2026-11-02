"""
AI 輔助之互動式影像處理與人臉分析工具
功能：1.影像載入與中繼資料  2.色彩空間(RGB/YUV)與同步直方圖
      3.自訂卷積核濾波 + PSNR  4.人臉偵測(Haar Cascade + 信心分數)
執行：streamlit run app.py
"""
import io
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="影像處理與人臉分析工具", layout="wide")

# ============================== 共用工具 ==============================
@st.cache_data(show_spinner=False)
def decode_image(data: bytes):
    """回傳 (BGR ndarray, 中繼資料 dict)。支援 PNG/JPG/BMP。"""
    pil = Image.open(io.BytesIO(data))
    fmt, mode, bands = pil.format, pil.mode, len(pil.getbands())
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    meta = {
        "檔案格式": fmt,
        "影像解析度": f"{w} × {h} px",
        "原始色彩模式": f"{mode}（{bands} 個通道）",
        "處理後色彩維度": "3 通道 (RGB) × 8 bit",
        "檔案大小": f"{len(data) / 1024:.1f} KB",
        "總像素數": f"{w * h:,}",
    }
    return bgr, meta


def histogram_figure(channels):
    """channels: {label: (uint8 2D array, color)}；X=強度 0–255，Y=像素數量"""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for label, (arr, color) in channels.items():
        counts = np.bincount(arr.ravel(), minlength=256)
        ax.fill_between(range(256), counts, alpha=0.35, color=color)
        ax.plot(range(256), counts, color=color, lw=1, label=label)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel intensity (0-255)")
    ax.set_ylabel("Pixel count")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def show_fig(fig):
    w = max(IMG_W, 300) / 100
    fig.set_size_inches(w, w * 0.55)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ============================== 側欄：載入 ==============================
st.title("🖼️ AI 輔助之互動式影像處理與人臉分析工具")
with st.sidebar:
    st.header("影像載入")
    up = st.file_uploader("支援 PNG / JPG / JPEG / BMP", type=["png", "jpg", "jpeg", "bmp"])
    st.header("顯示設定")
    IMG_W = st.slider("圖片顯示寬度 (px)", 150, 900, 450, 50,
                      help="調小可讓整頁內容縮進同一個螢幕，方便截圖")

if up is None:
    st.info("請從左側上傳圖片開始。")
    st.stop()

try:
    bgr, meta = decode_image(up.getvalue())
except Exception as e:
    st.error(f"無法讀取此檔案：{e}")
    st.stop()

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
tab1, tab2, tab3, tab4 = st.tabs(
    ["① 載入與中繼資料", "② 色彩空間與直方圖", "③ 卷積濾波與 PSNR", "④ 人臉偵測"])

# ============================== 功能 1 ==============================
with tab1:
    c1, c2 = st.columns([3, 2])
    c1.image(rgb, caption=up.name, width=IMG_W)
    c2.subheader("影像中繼資料")
    c2.table(pd.DataFrame({"項目": meta.keys(), "內容": meta.values()}).set_index("項目"))

# ============================== 功能 2 ==============================
with tab2:
    mode = st.radio("檢視模式", ["Original", "R", "G", "B", "Y", "U", "V"], horizontal=True)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    colors = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4"}

    left, right = st.columns(2)
    if mode == "Original":
        left.image(rgb, caption="Original", width=IMG_W)
        hist = {"R": (r, colors["R"]), "G": (g, colors["G"]), "B": (b, colors["B"])}
    elif mode in "RGB":
        ch = {"R": r, "G": g, "B": b}[mode]
        tint = st.checkbox("以對應顏色顯示", value=True)
        if tint:
            img = np.zeros_like(rgb)
            img[:, :, "RGB".index(mode)] = ch
        else:
            img = ch
        left.image(img, caption=f"{mode} channel", width=IMG_W, clamp=True)
        hist = {mode: (ch, colors[mode])}
    else:
        idx = "YUV".index(mode)
        ch = yuv[:, :, idx]
        left.image(ch, caption=f"{mode} component (OpenCV BT.601, U/V 以 128 為零點)",
                   width=IMG_W, clamp=True)
        hist = {mode: (ch, {"Y": "#444444", "U": "#1f77b4", "V": "#d62728"}[mode])}
    with right:
        show_fig(histogram_figure(hist))
        sel = hist[next(iter(hist))][0]
        st.caption(f"平均強度 {sel.mean():.1f}　標準差 {sel.std():.1f}")

# ============================== 功能 3 ==============================
PRESETS = {
    "邊緣銳化 3×3": "0 -1 0\n-1 5 -1\n0 -1 0",
    "高斯平滑 3×3": "1 2 1\n2 4 2\n1 2 1",
    "高斯平滑 5×5": "1 4 6 4 1\n4 16 24 16 4\n6 24 36 24 6\n4 16 24 16 4\n1 4 6 4 1",
    "浮雕 3×3": "-2 -1 0\n-1 1 1\n0 1 2",
    "均值模糊 3×3": "1 1 1\n1 1 1\n1 1 1",
    "Laplacian 邊緣": "0 1 0\n1 -4 1\n0 1 0",
    "自訂（請直接編輯）": "0 0 0\n0 1 0\n0 0 0",
}


def parse_kernel(text):
    rows = [x for x in text.strip().splitlines() if x.strip()]
    if not rows:
        raise ValueError("矩陣為空，請輸入數值。")
    try:
        m = [[float(v) for v in re.split(r"[,\s;]+", row.strip()) if v] for row in rows]
    except ValueError:
        raise ValueError("含有非數值字元，請只輸入數字（以空白或逗號分隔）。")
    if len({len(row) for row in m}) != 1:
        raise ValueError("每一列的元素數量必須相同。")
    k = np.array(m, dtype=np.float64)
    if k.shape[0] != k.shape[1]:
        raise ValueError(f"目前為 {k.shape[0]}×{k.shape[1]} 非方陣，請輸入 N×N 矩陣。")
    if k.shape[0] % 2 == 0:
        raise ValueError("卷積核邊長必須為奇數（3、5、7…），才有中心點。")
    if k.shape[0] > 15:
        raise ValueError("卷積核過大（上限 15×15）。")
    if not np.all(np.isfinite(k)):
        raise ValueError("矩陣含有 NaN 或無限大。")
    return k


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return mse, (float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))


with tab3:
    st.latex(r"\mathrm{MSE}=\frac{1}{MN}\sum_{i,j}\big[I(i,j)-K(i,j)\big]^2,\quad "
             r"\mathrm{PSNR}=10\log_{10}\!\frac{255^2}{\mathrm{MSE}}\ (\mathrm{dB})")
    cA, cB = st.columns([1, 2])
    with cA:
        preset = st.selectbox("預設核", list(PRESETS))
        text = st.text_area("卷積核矩陣（每列一行，數值以空白/逗號分隔）",
                            PRESETS[preset], height=170, key=f"k_{preset}")
        norm = st.radio("正規化", ["不正規化", "除以權重總和"], horizontal=True)
        bias = st.slider("輸出偏移量 (bias)", 0, 255, 128 if "浮雕" in preset else 0,
                         help="浮雕等含負值的核通常加上 128 以顯示灰階起伏")
        gray_mode = st.checkbox("轉灰階後濾波")
    try:
        k = parse_kernel(text)
        if norm == "除以權重總和":
            s = k.sum()
            if abs(s) < 1e-8:
                st.warning("權重總和為 0（如 Laplacian、浮雕），無法正規化，已略過以避免除以零。")
            else:
                k = k / s
        src = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if gray_mode else bgr
        # cv2.filter2D 實為相關(correlation)；翻轉核 180° 才是數學上的卷積
        out = cv2.filter2D(src, -1, cv2.flip(k, -1), delta=float(bias),
                           borderType=cv2.BORDER_REFLECT)
        mse, val = psnr(src, out)
        disp = (lambda x: x) if gray_mode else (lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
        with cB:
            i1, i2 = st.columns(2)
            i1.image(disp(src), caption="濾波前", width=IMG_W, clamp=True)
            i2.image(disp(out), caption="濾波後", width=IMG_W, clamp=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("PSNR", "∞ dB（完全相同）" if val == float("inf") else f"{val:.2f} dB")
            m2.metric("MSE", f"{mse:.2f}")
            m3.metric("核權重總和", f"{k.sum():.3f}")
            if st.checkbox("顯示差異圖（|原圖 − 濾波後|，放大 3 倍）"):
                diff = cv2.convertScaleAbs(cv2.absdiff(src, out), alpha=3)
                st.image(disp(diff), width=IMG_W, clamp=True)
    except ValueError as e:
        cB.error(f"⚠️ {e}")

# ============================== 功能 4 ==============================
@st.cache_resource
def face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_faces(img, scale_factor, neighbors, min_size):
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    res = face_cascade().detectMultiScale3(
        gray, scaleFactor=scale_factor, minNeighbors=neighbors,
        minSize=(min_size, min_size), outputRejectLevels=True)
    rects, _, weights = res
    if len(rects) == 0:
        return []
    # levelWeights 為 Haar 最後一級的強分類器分數，非機率；
    # 以 1-exp(-w/3) 壓縮至 (0,1) 作為「相對信心度」
    return [(tuple(int(v) for v in r), float(w), 1 - np.exp(-max(float(w), 0) / 3))
            for r, w in zip(rects, np.ravel(weights))]


with tab4:
    p1, p2, p3 = st.columns(3)
    sf = p1.slider("scaleFactor", 1.05, 1.5, 1.1, 0.01)
    nb = p2.slider("minNeighbors", 1, 10, 5)
    ms = p3.slider("最小人臉尺寸 (px)", 20, 300, 40)
    faces = detect_faces(bgr, sf, nb, ms)
    canvas = bgr.copy()
    ih, iw = bgr.shape[:2]
    t = max(1, iw // 400)                       # 線條粗細
    fs = max(0.35, min(iw, ih) / 900)           # 字體隨圖片尺寸縮放
    for i, ((x, y, w, h), raw, conf) in enumerate(faces, 1):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 200, 0), t)
        label = f"#{i} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        ty = y - 4 if y - th - base - 4 >= 0 else y + th + 4   # 上方放不下就放框內
        cv2.rectangle(canvas, (x, ty - th - base // 2), (x + tw, ty + base // 2), (0, 200, 0), -1)
        cv2.putText(canvas, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t)
    st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), width=IMG_W)
    if faces:
        st.success(f"偵測到 {len(faces)} 張人臉")
        st.dataframe(pd.DataFrame([{
            "編號": i, "x": x, "y": y, "寬": w, "高": h,
            "levelWeight": round(raw, 3), "相對信心度": round(conf, 3)}
            for i, ((x, y, w, h), raw, conf) in enumerate(faces, 1)]),
            hide_index=True, use_container_width=True)
    else:
        st.warning("未偵測到人臉。可降低 minNeighbors 或縮小最小人臉尺寸。")
    st.caption("方法：Haar Cascade（Viola–Jones）。信心度由 levelWeights 正規化而得，僅供相對比較。")
