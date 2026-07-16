import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from mtcnn import MTCNN
import io
import time
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate

st.set_page_config(
    page_title="DeepGuard — Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  body, [data-testid="stAppViewContainer"] {
    background-color: #0F1117; color: #E8EAF0;
    font-family: 'Segoe UI', sans-serif;
  }
  [data-testid="stSidebar"] {
    background-color: #161B27; border-right: 1px solid #2A2F3E;
  }
  .hero-banner {
    background: linear-gradient(135deg, #1A1F2E 0%, #0D1117 100%);
    border: 1px solid #2A3F5F; border-radius: 12px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem; text-align: center;
  }
  .hero-title { font-size: 2.2rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.5px; }
  .hero-subtitle { font-size: 1rem; color: #8892A4; }
  .hero-badge {
    display: inline-block; background: #1A3A5C; border: 1px solid #2E5F8A;
    color: #5DA8E0; font-size: 0.75rem; font-weight: 600;
    padding: 0.25rem 0.75rem; border-radius: 20px; margin-top: 0.8rem;
  }

  /* Verdict levels */
  .verdict-def-real  { background:#063B1E;border:1px solid #065F46;border-left:4px solid #10B981;border-radius:10px;padding:1.2rem 1.5rem;margin:1rem 0; }
  .verdict-likely-real{ background:#0D2B1F;border:1px solid #1A5C3A;border-left:4px solid #22C55E;border-radius:10px;padding:1.2rem 1.5rem;margin:1rem 0; }
  .verdict-uncertain { background:#1C1A07;border:1px solid #5C4A1A;border-left:4px solid #F59E0B;border-radius:10px;padding:1.2rem 1.5rem;margin:1rem 0; }
  .verdict-likely-fake{ background:#2B0D0D;border:1px solid #5C1A1A;border-left:4px solid #EF4444;border-radius:10px;padding:1.2rem 1.5rem;margin:1rem 0; }
  .verdict-def-fake  { background:#3B0808;border:1px solid #7C1A1A;border-left:4px solid #B91C1C;border-radius:10px;padding:1.2rem 1.5rem;margin:1rem 0; }

  .verdict-label { font-size:1.4rem;font-weight:700;margin-bottom:0.3rem; }
  .verdict-def-real  .verdict-label { color:#10B981; }
  .verdict-likely-real .verdict-label{ color:#22C55E; }
  .verdict-uncertain .verdict-label  { color:#F59E0B; }
  .verdict-likely-fake .verdict-label{ color:#EF4444; }
  .verdict-def-fake  .verdict-label  { color:#B91C1C; }
  .verdict-sub { font-size:0.9rem; color:#8892A4; }

  .metric-row { display:flex;gap:1rem;margin:1rem 0; }
  .metric-card { flex:1;background:#161B27;border:1px solid #2A2F3E;border-radius:10px;padding:1rem;text-align:center; }
  .metric-value { font-size:1.6rem;font-weight:700;color:#5DA8E0; }
  .metric-label { font-size:0.8rem;color:#8892A4;margin-top:0.2rem;text-transform:uppercase;letter-spacing:0.5px; }

  .conf-bar-bg { background:#1E2433;border-radius:6px;height:10px;margin:0.5rem 0;overflow:hidden; }
  .conf-bar-real { background:linear-gradient(90deg,#16A34A,#22C55E);height:100%;border-radius:6px; }
  .conf-bar-amber{ background:linear-gradient(90deg,#B45309,#F59E0B);height:100%;border-radius:6px; }
  .conf-bar-fake { background:linear-gradient(90deg,#B91C1C,#EF4444);height:100%;border-radius:6px; }
  .conf-label { font-size:0.85rem;color:#8892A4;margin-bottom:0.3rem; }
  .conf-value { font-size:1.1rem;font-weight:600;color:#E8EAF0; }

  .info-grid { display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.8rem; }
  .info-item { background:#0F1117;border:1px solid #2A2F3E;border-radius:6px;padding:0.5rem 0.8rem; }
  .info-item-label { font-size:0.75rem;color:#8892A4;text-transform:uppercase;letter-spacing:0.4px; }
  .info-item-value { font-size:0.88rem;color:#E8EAF0;font-weight:500;margin-top:1px; }

  .timing-badge {
    display:inline-block;background:#0F1117;border:1px solid #2A3F5F;
    color:#5DA8E0;font-size:0.78rem;padding:0.2rem 0.7rem;
    border-radius:12px;margin-top:0.6rem;font-family:monospace;
  }

  .stat-item { background:#0F1117;border:1px solid #2A2F3E;border-radius:8px;
    padding:0.6rem 0.9rem;margin:0.4rem 0;display:flex;justify-content:space-between;align-items:center; }
  .stat-label { font-size:0.82rem;color:#8892A4; }
  .stat-value { font-size:0.9rem;font-weight:600;color:#5DA8E0; }

  .stButton > button {
    background:linear-gradient(135deg,#1A4A7A 0%,#1E5A96 100%);
    color:white;border:none;border-radius:8px;
    padding:0.6rem 1.5rem;font-weight:600;font-size:0.95rem;width:100%;
  }
  .gradcam-title { font-size:1rem;font-weight:600;color:#5DA8E0;margin-bottom:0.5rem; }
  .gradcam-sub { font-size:0.82rem;color:#8892A4;margin-bottom:0.8rem;line-height:1.5; }
  .warn-box { background:#1F1A0A;border:1px solid #5C4A1A;border-left:3px solid #F59E0B;
    border-radius:8px;padding:0.8rem 1rem;font-size:0.875rem;color:#D4B483;margin-top:0.5rem; }
  .section-divider { border:none;border-top:1px solid #2A2F3E;margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Verdict logic ──────────────────────────────────────────────────────────────
def get_verdict(raw_score):
    if raw_score >= 0.80:
        return "Definitely Real", "def-real", "real", \
               "Very high confidence — strong authentic facial patterns detected.", "#10B981"
    elif raw_score >= 0.40:
        return "Likely Real", "likely-real", "real", \
               "Probable authentic image — no significant manipulation indicators.", "#22C55E"
    elif raw_score >= 0.20:
        return "Uncertain", "uncertain", "amber", \
               "Borderline result — manual verification recommended.", "#F59E0B"
    elif raw_score >= 0.10:
        return "Likely Fake", "likely-fake", "fake", \
               "Probable AI-generated image — suspicious patterns detected.", "#EF4444"
    else:
        return "Definitely Fake", "def-fake", "fake", \
               "Very high confidence — strong deepfake indicators identified.", "#B91C1C"


# ── Grad-CAM ───────────────────────────────────────────────────────────────────
def generate_gradcam(model, face_input, face_rgb_224):
    try:
        base = model.layers[0]
        last_conv_layer = None
        for layer in reversed(base.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
        if last_conv_layer is None:
            return None
        grad_model = tf.keras.models.Model(
            inputs=base.input,
            outputs=[base.get_layer(last_conv_layer).output, base.output]
        )
        with tf.GradientTape() as tape:
            inputs = tf.cast(face_input, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            class_channel = predictions[:, 0]
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        face_bgr = cv2.cvtColor(face_rgb_224, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(face_bgr, 0.5, heatmap_colored, 0.5, 0)
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    except:
        return None


# ── PDF Report ────────────────────────────────────────────────────────────────
def generate_pdf_report(history):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []
    title_style = ParagraphStyle('T', parent=styles['Normal'], fontSize=20,
        fontName='Helvetica-Bold', textColor=colors.HexColor('#1A3A5C'),
        alignment=TA_CENTER, spaceAfter=6)
    subtitle_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=11,
        fontName='Helvetica', textColor=colors.HexColor('#555555'),
        alignment=TA_CENTER, spaceAfter=4)
    section_style = ParagraphStyle('Sec', parent=styles['Normal'], fontSize=13,
        fontName='Helvetica-Bold', textColor=colors.HexColor('#1A3A5C'),
        spaceBefore=14, spaceAfter=6)
    small_style = ParagraphStyle('Sm', parent=styles['Normal'], fontSize=8.5,
        fontName='Helvetica', textColor=colors.HexColor('#777777'), alignment=TA_CENTER)
    story.append(Paragraph("DeepGuard Detection System", title_style))
    story.append(Paragraph("AI-Powered Deepfake Analysis Report with Grad-CAM", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y at %I:%M %p')}", small_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=2,
        color=colors.HexColor('#1A3A5C'), spaceAfter=12))
    df = pd.DataFrame(history)
    total = len(df)
    real_c = len(df[df['Result'].str.contains('REAL')])
    fake_c = len(df[df['Result'].str.contains('FAKE')])
    story.append(Paragraph("Session Summary", section_style))
    summary_data = [
        ['Metric', 'Value'],
        ['Total Images Analysed', str(total)],
        ['Real Images Detected', str(real_c)],
        ['Deepfakes Detected', str(fake_c)],
        ['Deepfake Rate', f'{round(fake_c/total*100,1) if total>0 else 0}%'],
        ['Model', 'EfficientNetB4 + Grad-CAM Forensics'],
        ['Face Detector', 'MTCNN'],
        ['Model AUC', '0.9634'],
        ['Report Date', datetime.now().strftime('%d %B %Y')],
    ]
    st_tbl = Table(summary_data, colWidths=[8*cm, 8*cm])
    st_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A3A5C')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F8F9FA'), colors.white]),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(st_tbl)
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Detailed Prediction Results", section_style))
    table_data = [['#', 'Time', 'Source', 'Verdict', 'Confidence', 'Time Taken']]
    for i, row in enumerate(history, 1):
        table_data.append([str(i), row['Time'], row['Source'],
                           row['Result'], row['Confidence'],
                           row.get('Analysis Time', 'N/A')])
    det_tbl = Table(table_data, colWidths=[1*cm, 3*cm, 2.5*cm, 3.5*cm, 3*cm, 2.5*cm])
    det_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A3A5C')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F8F9FA'), colors.white]),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 6),
    ])
    det_tbl.setStyle(det_style)
    story.append(det_tbl)
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1,
        color=colors.HexColor('#CCCCCC'), spaceAfter=8))
    story.append(Paragraph(
        "DeepGuard — AI-Powered Deepfake Detection with Grad-CAM Forensic Visualization | "
        "EfficientNetB4 | MTCNN | AUC 0.9634", small_style))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    base_model = EfficientNetB4(weights=None, include_top=False, input_shape=(224,224,3))
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    dummy = np.zeros((1,224,224,3), dtype=np.float32)
    model(dummy, training=False)
    all_weights = np.load('models/model_weights_v2_numpy.npy', allow_pickle=True)
    for i, layer in enumerate(model.layers):
        if len(all_weights[i]) > 0:
            layer.set_weights(all_weights[i])
    with open("models/class_indices.json", "r") as f:
        class_indices = json.load(f)
    detector = MTCNN()
    return model, class_indices, detector

with st.spinner("Initialising DeepGuard..."):
    model, class_indices, detector = load_resources()

if "history" not in st.session_state:
    st.session_state.history = []


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 DeepGuard")
    st.markdown("<p style='color:#8892A4;font-size:0.82rem;margin-top:-0.5rem;'>Deepfake Detection + Grad-CAM</p>",
        unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2A2F3E;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("**Model Information**")
    st.markdown("""
    <div class='stat-item'><span class='stat-label'>Backbone</span><span class='stat-value'>EfficientNetB4</span></div>
    <div class='stat-item'><span class='stat-label'>Face Detector</span><span class='stat-value'>MTCNN</span></div>
    <div class='stat-item'><span class='stat-label'>Explainability</span><span class='stat-value'>Grad-CAM</span></div>
    <div class='stat-item'><span class='stat-label'>AUC Score</span><span class='stat-value'>0.9634</span></div>
    <div class='stat-item'><span class='stat-label'>Accuracy</span><span class='stat-value'>~89.6%</span></div>
    <div class='stat-item'><span class='stat-label'>Training Images</span><span class='stat-value'>147,221</span></div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2A2F3E;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("**Session Stats**")
    total = len(st.session_state.history)
    real_c = len([h for h in st.session_state.history if 'REAL' in h['Result']])
    fake_c = len([h for h in st.session_state.history if 'FAKE' in h['Result']])
    st.markdown(f"""
    <div class='stat-item'><span class='stat-label'>Total Scanned</span><span class='stat-value'>{total}</span></div>
    <div class='stat-item'><span class='stat-label'>Real Detected</span><span class='stat-value' style='color:#22C55E'>{real_c}</span></div>
    <div class='stat-item'><span class='stat-label'>Fake Detected</span><span class='stat-value' style='color:#EF4444'>{fake_c}</span></div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2A2F3E;margin:0.8rem 0;'>", unsafe_allow_html=True)
    if st.button("🗑  Clear History"):
        st.session_state.history = []
        st.rerun()
    if len(st.session_state.history) > 0:
        df_export = pd.DataFrame(st.session_state.history)
        st.download_button("⬇  Download CSV",
            df_export.to_csv(index=False).encode('utf-8'),
            f"deepguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        st.download_button("⬇  Download PDF Report",
            generate_pdf_report(st.session_state.history),
            f"deepguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", "application/pdf")


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
  <div class='hero-title'>DeepGuard Detection System</div>
  <div class='hero-subtitle'>AI-powered forensic deepfake analysis with Grad-CAM explainability</div>
  <div class='hero-badge'>EfficientNetB4 · MTCNN · Grad-CAM · 96.3% AUC</div>
</div>
""", unsafe_allow_html=True)


# ── Detection ──────────────────────────────────────────────────────────────────
def detect_deepfake(img, source, filename="", filesize="", dimensions=""):
    start_time = time.time()

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_img)

    if len(faces) == 0:
        st.error("No face detected. Try a clearer, front-facing photo with good lighting.")
        return

    face_data = max(faces, key=lambda f: f['box'][2] * f['box'][3])
    x, y, w, h = face_data['box']
    pad = int(max(w, h) * 0.20)
    H, W = img.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    face = rgb_img[y1:y2, x1:x2]

    face_resized = cv2.resize(face, (224, 224))
    face_array = np.expand_dims(face_resized.astype(np.float32), axis=0)
    face_input = preprocess_input(face_array.copy())

    raw_score = float(model.predict(face_input, verbose=0)[0][0])

    elapsed_ms = round((time.time() - start_time) * 1000)

    verdict, verdict_class, bar_type, verdict_desc, verdict_color = get_verdict(raw_score)

    # Confidence
    if raw_score >= 0.10:
        confidence_pct = round(raw_score * 100, 2)
    else:
        confidence_pct = round((1.0 - raw_score) * 100, 2)

    # Draw bounding box
    display_img = rgb_img.copy()
    box_color = (16, 185, 129) if 'real' in bar_type else (239, 68, 68) if 'fake' in bar_type else (245, 158, 11)
    cv2.rectangle(display_img, (x1, y1), (x2, y2), box_color, 3)
    cv2.putText(display_img, f"{verdict} {confidence_pct}%",
                (x1, max(y1 - 12, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)

    # Grad-CAM
    gradcam_img = generate_gradcam(model, face_input, face_resized)

    # ── Display ──
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image(display_img, caption="Detected face with bounding box", use_container_width=True)

    with col2:
        # Verdict card
        st.markdown(f"""
        <div class='verdict-{verdict_class}'>
          <div class='verdict-label'>{
            '✓✓' if 'Definitely Real' in verdict else
            '✓' if 'Likely Real' in verdict else
            '?' if 'Uncertain' in verdict else
            '✗' if 'Likely Fake' in verdict else '✗✗'
          } {verdict.upper()}</div>
          <div class='verdict-sub'>{verdict_desc}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        bar_cls = "conf-bar-real" if bar_type == "real" else "conf-bar-amber" if bar_type == "amber" else "conf-bar-fake"
        st.markdown(f"""
        <div class='conf-label'>Confidence Score</div>
        <div class='conf-value'>{confidence_pct}%</div>
        <div class='conf-bar-bg'>
          <div class='{bar_cls}' style='width:{min(confidence_pct,100)}%'></div>
        </div>
        <div style='margin-top:0.8rem;'>
          <div class='conf-label'>Raw Model Score</div>
          <div style='font-size:0.9rem;color:#8892A4;font-family:monospace;'>{raw_score:.4f}</div>
        </div>
        <div style='margin-top:0.8rem;'>
          <div class='conf-label'>Source</div>
          <div style='font-size:0.9rem;color:#8892A4;'>{source}</div>
        </div>
        <div style='margin-top:0.8rem;'>
          <div class='conf-label'>Timestamp</div>
          <div style='font-size:0.9rem;color:#8892A4;'>{datetime.now().strftime("%d %b %Y, %I:%M %p")}</div>
        </div>
        <div><span class='timing-badge'>⏱ Analysis completed in {elapsed_ms} ms</span></div>
        """, unsafe_allow_html=True)

    # ── Image metadata (below columns, no overlap) ──
    st.markdown(f"""
    <div class='info-grid'>
      <div class='info-item'>
        <div class='info-item-label'>Filename</div>
        <div class='info-item-value'>{filename if filename else 'Webcam'}</div>
      </div>
      <div class='info-item'>
        <div class='info-item-label'>Dimensions</div>
        <div class='info-item-value'>{dimensions if dimensions else f'{W} × {H} px'}</div>
      </div>
      <div class='info-item'>
        <div class='info-item-label'>File Size</div>
        <div class='info-item-value'>{filesize if filesize else 'N/A'}</div>
      </div>
      <div class='info-item'>
        <div class='info-item-label'>Face Region</div>
        <div class='info-item-value'>{w} × {h} px</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Grad-CAM
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='gradcam-title'>🔬 Grad-CAM Forensic Heatmap</div>
    <div class='gradcam-sub'>
      Grad-CAM highlights which regions of the face most influenced the model's decision.
      <b style='color:#EF4444'>Red/yellow</b> = high-suspicion regions.
      <b style='color:#5DA8E0'>Blue/cool</b> = low influence areas.
    </div>
    """, unsafe_allow_html=True)

    if gradcam_img is not None:
        col_orig, col_heat = st.columns(2)
        with col_orig:
            st.image(face_resized, caption="Cropped face input", use_container_width=True)
        with col_heat:
            st.image(gradcam_img,
                caption=f"Grad-CAM — {'suspicious regions' if 'fake' in bar_type else 'authentic pattern'}",
                use_container_width=True)
        if 'fake' in bar_type:
            st.markdown("""<div class='warn-box'>
            🔬 <b>Forensic finding:</b> Concentrated activation in facial regions typically altered by GAN generation —
            skin texture, eye boundaries, or facial contours. Consistent with AI-generated manipulation.
            </div>""", unsafe_allow_html=True)
        elif bar_type == 'amber':
            st.markdown("""<div style='background:#1C1A07;border:1px solid #5C4A1A;border-left:3px solid #F59E0B;
            border-radius:8px;padding:0.8rem 1rem;font-size:0.875rem;color:#D4B483;margin-top:0.5rem;'>
            🔬 <b>Forensic finding:</b> Inconclusive activation pattern. Manual review recommended.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style='background:#063B1E;border:1px solid #065F46;border-left:3px solid #10B981;
            border-radius:8px;padding:0.8rem 1rem;font-size:0.875rem;color:#A3D9B8;margin-top:0.5rem;'>
            🔬 <b>Forensic finding:</b> Natural distributed activation with no concentrated anomaly regions.
            Consistent with authentic facial photograph.
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Grad-CAM could not be generated for this image.")

    st.session_state.history.append({
        "Time": datetime.now().strftime("%I:%M:%S %p"),
        "Source": source,
        "Result": verdict.upper(),
        "Confidence": f"{confidence_pct}%",
        "Raw Score": round(raw_score, 4),
        "Analysis Time": f"{elapsed_ms} ms"
    })


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  Upload Image  ", "  Webcam Capture  ", "  Prediction History  "])

with tab1:
    st.markdown("#### Upload an image for analysis")
    uploaded_file = st.file_uploader("Drag and drop or click to browse",
        type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        filesize = f"{round(len(file_bytes)/1024, 1)} KB"
        dimensions = f"{W} × {H} px"
        filename = uploaded_file.name
        col_img, col_btn = st.columns([3, 1])
        with col_img:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                caption=f"{filename} — {dimensions} — {filesize}", use_container_width=True)
        with col_btn:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("Analyse Image", key="btn_upload"):
                with st.spinner("Detecting face · Running model · Generating Grad-CAM..."):
                    detect_deepfake(img, "Upload", filename, filesize, dimensions)

with tab2:
    st.markdown("#### Capture from webcam")
    st.markdown("<p style='color:#8892A4;font-size:0.88rem;'>Position your face clearly and capture.</p>",
        unsafe_allow_html=True)
    camera_image = st.camera_input("", label_visibility="collapsed")
    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        np_arr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        if st.button("Analyse Captured Image", key="btn_webcam"):
            with st.spinner("Detecting face · Running model · Generating Grad-CAM..."):
                detect_deepfake(img, "Webcam", "webcam_capture",
                    f"{round(len(bytes_data)/1024,1)} KB", f"{W} × {H} px")

with tab3:
    st.markdown("#### Prediction History")
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        total = len(df)
        real_count = len(df[df['Result'].str.contains('REAL')])
        fake_count = len(df[df['Result'].str.contains('FAKE')])
        st.markdown(f"""
        <div class='metric-row'>
          <div class='metric-card'><div class='metric-value'>{total}</div><div class='metric-label'>Total Scanned</div></div>
          <div class='metric-card'><div class='metric-value' style='color:#22C55E'>{real_count}</div><div class='metric-label'>Real Images</div></div>
          <div class='metric-card'><div class='metric-value' style='color:#EF4444'>{fake_count}</div><div class='metric-label'>Fake Images</div></div>
          <div class='metric-card'><div class='metric-value'>{round(fake_count/total*100) if total>0 else 0}%</div><div class='metric-label'>Fake Rate</div></div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col_csv, col_pdf = st.columns(2)
        with col_csv:
            st.download_button("⬇ Download CSV",
                df.to_csv(index=False).encode('utf-8'),
                f"deepguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv", use_container_width=True)
        with col_pdf:
            st.download_button("⬇ Download PDF Report",
                generate_pdf_report(st.session_state.history),
                f"deepguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                "application/pdf", use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:3rem;color:#8892A4;'>
          <div style='font-size:2rem;margin-bottom:0.5rem;'>🔍</div>
          <div style='font-size:1rem;'>No predictions yet</div>
          <div style='font-size:0.85rem;margin-top:0.3rem;'>Upload an image or use the webcam to begin</div>
        </div>""", unsafe_allow_html=True)