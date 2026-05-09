"""
AccidentShield AI — Accident Detection from CCTV Footage
CNN Deep Learning | Image & Video Detection
"""
import streamlit as st
import numpy as np
import cv2
import pickle
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AccidentShield AI", page_icon="🚨",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif!important;}
.stApp{background:#060810!important;}
.main .block-container{background:#060810!important;padding:1.5rem 2rem;max-width:1400px;}
body,p,span,div,label,h1,h2,h3{color:#e2e8f0!important;}
#MainMenu,footer,header{visibility:hidden;}
.hero{background:linear-gradient(135deg,#1a0a0a 0%,#2d1515 50%,#1a0a0a 100%);
  border:1px solid #da3633;border-radius:20px;padding:36px 44px;margin-bottom:28px;
  box-shadow:0 0 40px rgba(218,54,51,0.15);}
.hero-title{font-size:2.3rem;font-weight:800;color:#fff!important;margin:0;}
.hero-sub{font-size:0.92rem;color:#8b949e!important;margin:8px 0 18px;}
.badge{display:inline-block;background:rgba(218,54,51,0.15);border:1px solid rgba(218,54,51,0.4);
  color:#f85149!important;font-size:0.72rem;font-weight:600;padding:4px 12px;border-radius:99px;margin:0 5px 5px 0;}
.card{background:#0d1117;border:1px solid #21262d;border-radius:14px;padding:22px 24px;margin-bottom:18px;}
.card-title{font-size:0.7rem!important;font-weight:700!important;text-transform:uppercase;
  letter-spacing:1.8px;color:#f85149!important;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid #21262d;}
.res-accident{background:linear-gradient(135deg,#1a0a0a,#3d1515);border:1px solid #da3633;
  border-radius:14px;padding:26px;text-align:center;box-shadow:0 0 40px rgba(218,54,51,0.25);margin-bottom:16px;}
.res-safe{background:linear-gradient(135deg,#051a10,#0a2d1c);border:1px solid #238636;
  border-radius:14px;padding:26px;text-align:center;box-shadow:0 0 40px rgba(35,134,54,0.2);margin-bottom:16px;}
.res-title{font-size:1.5rem!important;font-weight:800!important;color:#fff!important;margin:8px 0 4px;}
.res-prob{font-size:1rem!important;color:rgba(255,255,255,0.85)!important;}
.res-note{font-size:0.78rem!important;color:rgba(255,255,255,0.55)!important;font-style:italic;margin-top:6px;}
.sbox{background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #30363d;
  border-radius:10px;padding:14px;text-align:center;margin-bottom:8px;}
.sbox-val{font-size:1.5rem!important;font-weight:800!important;color:#f85149!important;}
.sbox-lbl{font-size:0.65rem!important;color:#8b949e!important;text-transform:uppercase;letter-spacing:0.8px;}
.rrow{display:flex;align-items:center;gap:10px;background:#0d1117;border:1px solid #21262d;
  border-radius:8px;padding:9px 13px;margin-bottom:6px;}
.rdot{width:9px;height:9px;border-radius:50%;flex-shrink:0;}
.dr{background:#da3633;box-shadow:0 0 8px rgba(218,54,51,0.6);}
.dg{background:#238636;box-shadow:0 0 8px rgba(35,134,54,0.6);}
.rlbl{font-size:0.8rem!important;color:#c9d1d9!important;}
.pill-ok{display:inline-block;background:rgba(35,134,54,0.15);border:1px solid #238636;
  color:#3fb950!important;padding:6px 16px;border-radius:99px;font-size:0.78rem;font-weight:700;}
.pill-err{display:inline-block;background:rgba(218,54,51,0.15);border:1px solid #da3633;
  color:#f85149!important;padding:6px 16px;border-radius:99px;font-size:0.78rem;font-weight:700;}
.mrow{display:flex;gap:10px;margin:12px 0;}
.mt{flex:1;background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:12px;text-align:center;}
.mv{font-size:1.35rem!important;font-weight:800!important;color:#f85149!important;}
.ml{font-size:0.63rem!important;color:#8b949e!important;text-transform:uppercase;letter-spacing:0.8px;}
.atable{width:100%;border-collapse:collapse;}
.atable th{background:#161b22;color:#f85149!important;font-size:0.7rem;font-weight:700;
  text-transform:uppercase;letter-spacing:0.8px;padding:10px 14px;text-align:left;border-bottom:1px solid #30363d;}
.atable td{padding:10px 14px;color:#c9d1d9!important;font-size:0.83rem;border-bottom:1px solid #161b22;}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:#0d1117;border-radius:10px;padding:4px;border:1px solid #21262d;}
.stTabs [data-baseweb="tab"]{border-radius:8px!important;color:#8b949e!important;font-weight:600!important;
  font-size:0.86rem!important;padding:8px 20px!important;background:transparent!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#da3633,#991b1b)!important;color:#fff!important;}
.stButton>button{background:linear-gradient(135deg,#da3633,#991b1b)!important;color:white!important;
  border:none!important;border-radius:10px!important;padding:13px 28px!important;font-size:0.9rem!important;
  font-weight:700!important;width:100%!important;box-shadow:0 4px 20px rgba(218,54,51,0.35)!important;text-transform:uppercase!important;}
[data-testid="stSidebar"]{background:#060810!important;border-right:1px solid #21262d!important;}
[data-testid="stSidebar"] *{color:#e2e8f0!important;}
[data-testid="stFileUploader"]{background:#0d1117!important;border:2px dashed #30363d!important;border-radius:12px!important;}
.stProgress>div>div{background:linear-gradient(90deg,#da3633,#f85149)!important;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import tensorflow as tf
        for p in ['accident_detection_model_new.h5', 'accident_detection_model.h5', 'fine_tuned_best.keras',
                  'transfer_learning_best.keras', 'cnn_scratch_best.keras']:
            if os.path.exists(p):
                return tf.keras.models.load_model(p), True, f"Loaded: {p}"
        files = str(os.listdir('.'))
        return None, False, f"No model found. Files in directory: {files}"
    except Exception as e:
        return None, False, str(e)


@st.cache_resource(show_spinner=False)
def load_classes():
    if os.path.exists('class_names.pkl'):
        with open('class_names.pkl', 'rb') as f:
            return pickle.load(f)
    return ['Accident', 'Non Accident']


def preprocess_image(img_array):
    img = cv2.resize(img_array, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def predict_image(model, img_array, class_names):
    prob = float(model.predict(preprocess_image(img_array), verbose=0)[0][0])
    accident_prob = prob if class_names[0] == 'Accident' else 1 - prob
    label = 'Accident' if accident_prob >= 0.5 else 'Non Accident'
    return label, accident_prob


model, loaded, load_msg = load_model()
class_names = load_classes()

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:18px 0 12px;">
      <div style="font-size:2.8rem;">🚨</div>
      <div style="font-size:1.1rem;font-weight:800;color:#f85149!important;">AccidentShield AI</div>
      <div style="font-size:0.65rem;color:#8b949e!important;text-transform:uppercase;letter-spacing:1px;">CNN Deep Learning</div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,#30363d,transparent);margin:10px 0;"></div>
    <p style="font-size:0.65rem!important;color:#8b949e!important;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:8px;">📊 Model Info</p>
    <div class="sbox"><div class="sbox-val">CNN</div><div class="sbox-lbl">Architecture</div></div>
    <div class="sbox"><div class="sbox-val">224×224</div><div class="sbox-lbl">Input Size</div></div>
    <div class="sbox"><div class="sbox-val">2</div><div class="sbox-lbl">Classes</div></div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,#30363d,transparent);margin:14px 0;"></div>
    <p style="font-size:0.65rem!important;color:#8b949e!important;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:8px;">🎯 Classes</p>
    <div class="rrow"><div class="rdot dr"></div><span class="rlbl"><b>Accident</b> — Collision detected</span></div>
    <div class="rrow"><div class="rdot dg"></div><span class="rlbl"><b>Non Accident</b> — Normal traffic</span></div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,#30363d,transparent);margin:14px 0;"></div>
    <p style="font-size:0.65rem!important;color:#8b949e!important;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:8px;">⚙️ Model Status</p>
    """, unsafe_allow_html=True)
    if loaded:
        st.markdown('<div class="pill-ok">✅ &nbsp; Model Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="pill-err">❌ &nbsp; Model Not Found</div>', unsafe_allow_html=True)
        st.caption(load_msg)

st.markdown("""
<div class="hero">
  <div class="hero-title">🚨 AccidentShield AI</div>
  <div class="hero-sub">Accident Detection from CCTV Footage &nbsp;·&nbsp; CNN Deep Learning + Transfer Learning</div>
  <span class="badge">🧠 CNN from Scratch</span>
  <span class="badge">🔁 MobileNetV2</span>
  <span class="badge">📸 Image Detection</span>
  <span class="badge">🎬 Video Detection</span>
  <span class="badge">⚡ Real-time Analysis</span>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📸  Image Detection", "🎬  Video Detection", "ℹ️  About"])

with tab1:
    st.markdown('<div class="card"><div class="card-title">📸 Upload Image for Accident Detection</div><p style="color:#8b949e;font-size:0.86rem;margin-bottom:0;">Upload a CCTV image (JPG/PNG)</p></div>', unsafe_allow_html=True)
    uploaded_img = st.file_uploader("Upload CCTV Image", type=["jpg","jpeg","png"])
    if uploaded_img:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<div class="card"><div class="card-title">🖼️ Uploaded Image</div>', unsafe_allow_html=True)
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card"><div class="card-title">🎯 Detection Result</div>', unsafe_allow_html=True)
            if not loaded:
                st.error(f"❌ Model not loaded.\n\n{load_msg}")
            else:
                with st.spinner("Analysing..."):
                    label, acc_prob = predict_image(model, img_rgb, class_names)
                if label == 'Accident':
                    st.markdown(f"""<div class="res-accident">
                      <div style="font-size:3rem;">🚨</div>
                      <p class="res-title">ACCIDENT DETECTED</p>
                      <p class="res-prob">Confidence: <b>{acc_prob*100:.1f}%</b></p>
                      <p class="res-note">Immediate emergency response recommended</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="res-safe">
                      <div style="font-size:3rem;">✅</div>
                      <p class="res-title">NO ACCIDENT</p>
                      <p class="res-prob">Confidence: <b>{(1-acc_prob)*100:.1f}%</b></p>
                      <p class="res-note">Normal traffic conditions</p>
                    </div>""", unsafe_allow_html=True)
                a_pct = int(acc_prob*100)
                st.markdown(f"""
                <div style="margin:12px 0 8px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:#f85149;font-size:0.82rem;font-weight:600;">🚨 Accident</span>
                    <span style="color:#f85149;font-size:0.82rem;font-weight:700;">{a_pct}%</span>
                  </div>
                  <div style="height:8px;background:#161b22;border-radius:4px;">
                    <div style="width:{a_pct}%;height:100%;border-radius:4px;background:linear-gradient(90deg,#da3633,#f85149);"></div>
                  </div>
                </div>
                <div>
                  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:#3fb950;font-size:0.82rem;font-weight:600;">✅ Non Accident</span>
                    <span style="color:#3fb950;font-size:0.82rem;font-weight:700;">{100-a_pct}%</span>
                  </div>
                  <div style="height:8px;background:#161b22;border-radius:4px;">
                    <div style="width:{100-a_pct}%;height:100%;border-radius:4px;background:linear-gradient(90deg,#238636,#3fb950);"></div>
                  </div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card"><div class="card-title">🎬 Upload Video for Accident Detection</div><p style="color:#8b949e;font-size:0.86rem;margin-bottom:0;">Upload a CCTV video (MP4/AVI)</p></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a: frame_skip = st.slider("⚡ Analyse every N frames", 5, 60, 15)
    with col_b: threshold = st.slider("🎯 Accident threshold (%)", 30, 90, 50)
    uploaded_vid = st.file_uploader("Upload CCTV Video", type=["mp4","avi","mov"])
    if uploaded_vid:
        if not loaded:
            st.error(f"❌ Model not loaded.\n\n{load_msg}")
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_vid.read()); tfile.close()
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            st.markdown(f"""<div class="mrow">
              <div class="mt"><div class="mv">{total_frames}</div><div class="ml">Frames</div></div>
              <div class="mt"><div class="mv">{fps:.0f}</div><div class="ml">FPS</div></div>
              <div class="mt"><div class="mv">{total_frames/fps:.1f}s</div><div class="ml">Duration</div></div>
            </div>""", unsafe_allow_html=True)
            if st.button("🔍  ANALYSE VIDEO"):
                progress = st.progress(0); status = st.empty()
                accident_frames = []; analysed = 0; frame_idx = 0
                c1, c2 = st.columns(2)
                with c1: fd = st.empty()
                with c2: dd = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_idx % frame_skip == 0:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        label, acc_prob = predict_image(model, img_rgb, class_names)
                        analysed += 1; fd.image(img_rgb, use_container_width=True)
                        if acc_prob >= threshold/100:
                            accident_frames.append({'Frame':frame_idx,'Time':f'{frame_idx/fps:.1f}s','Confidence':f'{acc_prob*100:.1f}%'})
                            ann = img_rgb.copy()
                            cv2.putText(ann,f'ACCIDENT {acc_prob*100:.1f}%',(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,50,50),2)
                            dd.image(ann, use_container_width=True)
                        progress.progress(min(frame_idx/max(total_frames,1),1.0))
                        status.markdown(f'<p style="color:#8b949e;font-size:0.8rem;">Frame {frame_idx}/{total_frames} | Accidents: {len(accident_frames)}</p>', unsafe_allow_html=True)
                    frame_idx += 1
                cap.release(); os.unlink(tfile.name)
                if accident_frames:
                    st.markdown(f"""<div class="res-accident"><div style="font-size:2.5rem;">🚨</div>
                      <p class="res-title">ACCIDENTS DETECTED</p>
                      <p class="res-prob">Found <b>{len(accident_frames)}</b> accident frames</p></div>""", unsafe_allow_html=True)
                    import pandas as pd
                    df_r = pd.DataFrame(accident_frames)
                    st.dataframe(df_r, use_container_width=True)
                    st.download_button("⬇️ Download Report", df_r.to_csv(index=False).encode(), "accident_report.csv","text/csv")
                else:
                    st.markdown(f"""<div class="res-safe"><div style="font-size:2.5rem;">✅</div>
                      <p class="res-title">NO ACCIDENTS DETECTED</p>
                      <p class="res-prob">Analysed <b>{analysed}</b> frames — all clear</p></div>""", unsafe_allow_html=True)

with tab3:
    c1,c2 = st.columns(2,gap="large")
    with c1:
        st.markdown("""<div class="card"><div class="card-title">🧠 Model Info</div>
          <table class="atable"><tr><th>Component</th><th>Detail</th></tr>
          <tr><td>Model 1</td><td>CNN from Scratch (4 Conv blocks)</td></tr>
          <tr><td>Model 2</td><td>MobileNetV2 Transfer Learning</td></tr>
          <tr><td>Fine Tuning</td><td>Last 20 layers unfrozen</td></tr>
          <tr><td>Input</td><td>224 × 224 × 3</td></tr>
          <tr><td>Output</td><td>Sigmoid (binary)</td></tr>
          <tr><td>CNN Accuracy</td><td>53%</td></tr>
          <tr><td>Transfer Accuracy</td><td>88% ✅</td></tr>
          </table></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">📁 Repo Files</div>', unsafe_allow_html=True)
        st.code("""Accidcent-CNN/
├── app.py
├── requirements.txt
├── accident_detection_model.h5
├── class_names.pkl
└── render.yaml""","")
        st.markdown('</div>', unsafe_allow_html=True)
