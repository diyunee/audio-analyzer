import os
import json
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 페이지 설정 + 커스텀 스타일 (스튜디오 VU미터 컨셉)
# ============================================================
st.set_page_config(page_title="Audio Analyzer", page_icon="🎛️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Manrope:wght@400;600;800&display=swap');

html, body, [class*="css"]  { font-family: 'Manrope', sans-serif; }

.stApp {
    background: #FFFFFF;
    color: #1A1D24;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.3rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #14171C;
    margin-bottom: 0;
}
.hero-sub {
    font-family: 'Manrope', sans-serif;
    color: #6B7280;
    font-size: 0.95rem;
    margin-top: 4px;
}

/* VU 미터 바 (헤더 장식) */
.vu-strip { display:flex; gap:4px; margin: 14px 0 22px 0; }
.vu-bar { width: 6px; border-radius: 2px; background: #E5E7EB; }
.vu-bar.on-teal { background: linear-gradient(180deg, #1F8F7B, #146B5C); }
.vu-bar.on-amber { background: linear-gradient(180deg, #E08A2E, #B96E1C); }

/* 결과 카드 */
.metric-card {
    background: #F7F8FA;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 0.78rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: #1F8F7B;
    font-weight: 700;
}
.metric-value.amber { color: #B96E1C; }
.metric-sub { color: #8A93A3; font-size: 0.8rem; margin-top: 4px; }

.section-label {
    font-family: 'Space Mono', monospace;
    color: #B96E1C;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 26px 0 10px 0;
    border-bottom: 1px solid #E5E7EB;
    padding-bottom: 6px;
}

.genre-row {
    display:flex; justify-content: space-between;
    font-family: 'Space Mono', monospace;
    padding: 6px 0; border-bottom: 1px dashed #E5E7EB;
    font-size: 0.9rem;
    color: #1A1D24;
}

.similarity-score {
    font-family: 'Space Mono', monospace;
    font-size: 3.2rem;
    font-weight: 700;
    color: #1F8F7B;
    text-align: center;
}

section[data-testid="stFileUploader"] {
    background: #F7F8FA; border: 1px dashed #D1D5DB; border-radius: 14px; padding: 10px;
}
</style>
""", unsafe_allow_html=True)


def vu_strip(seed=0):
    import random
    random.seed(seed)
    bars = ""
    for i in range(28):
        cls = "on-teal" if random.random() > 0.35 else ("on-amber" if random.random() > 0.7 else "")
        h = random.randint(6, 26)
        bars += f'<div class="vu-bar {cls}" style="height:{h}px;"></div>'
    return f'<div class="vu-strip">{bars}</div>'


st.markdown('<div class="hero-title">🎛️ AUDIO ANALYZER</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Essentia 기반 음원 분석 · 유사도 측정</div>', unsafe_allow_html=True)
st.markdown(vu_strip(1), unsafe_allow_html=True)

# ============================================================
# 모델 다운로드 (최초 1회만, 캐시됨)
# ============================================================
MODEL_URLS = {
    "discogs-effnet-bs64-1.pb": "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
    "genre_discogs400-discogs-effnet-1.pb": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    "genre_discogs400-discogs-effnet-1.json": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
    "mood_acoustic-discogs-effnet-1.pb": "https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.pb",
    "mood_acoustic-discogs-effnet-1.json": "https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.json",
    "voice_instrumental-discogs-effnet-1.pb": "https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb",
    "voice_instrumental-discogs-effnet-1.json": "https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.json",
    "msd-musicnn-1.pb": "https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.pb",
    "emomusic-msd-musicnn-2.pb": "https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-msd-musicnn-2.pb",
    "emomusic-msd-musicnn-2.json": "https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-msd-musicnn-2.json",
}
MODEL_DIR = "models"
LIBRARY_PATH = "library_data.json"


@st.cache_resource(show_spinner="분석 모델 준비 중 (최초 1회, 1~2분 소요)...")
def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fname, url in MODEL_URLS.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
    return True


def mpath(fname):
    return os.path.join(MODEL_DIR, fname)


# ============================================================
# 장르 기반 BPM 옥타브 보정
# ============================================================
GENRE_BPM_CENTER = {
    "Ballad": 70, "Vocal": 75, "R&B": 80, "Soul": 80,
    "Chillwave": 90, "Downtempo": 85, "Ambient": 70, "Trip Hop": 85,
    "Lo-Fi": 80, "Bossa": 90,
    "Indie Pop": 115, "Synth-pop": 112, "Pop": 110, "Dance-pop": 118,
    "Tropical House": 105, "House": 124, "Deep House": 122,
    "Hip Hop": 90, "Trap": 140,
    "Techno": 130, "Trance": 136, "Dubstep": 140, "Drum": 170,
}


def genre_based_octave_correction(bpm_candidate, labels, avg_predictions, top_n=5):
    top_idx = np.argsort(avg_predictions)[::-1][:top_n]
    top_genres = [(labels[i], avg_predictions[i]) for i in top_idx]
    half_bpm, double_bpm = bpm_candidate / 2, bpm_candidate * 2

    def score(bpm_value):
        s = 0
        for genre_label, prob in top_genres:
            for keyword, center in GENRE_BPM_CENTER.items():
                if keyword.lower() in genre_label.lower():
                    distance = abs(bpm_value - center)
                    s += np.exp(-(distance ** 2) / (2 * 25 ** 2)) * prob
        return s

    scores = {"원본": (bpm_candidate, score(bpm_candidate)),
              "절반": (half_bpm, score(half_bpm)),
              "두배": (double_bpm, score(double_bpm))}
    best = max(scores, key=lambda k: scores[k][1])
    return scores[best][0]


def format_duration(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}분 {secs}초"


def interpret_bpm(v):
    if v <= 75: return "매우 느림 (발라드, 앰비언트)"
    if v <= 95: return "느긋함 (미드템포 발라드, R&B)"
    if v <= 115: return "보통 (팝, 미드템포 댄스)"
    if v <= 130: return "업비트 (댄스, 하우스)"
    return "빠름 (EDM, 트랩, 펑크)"

def interpret_danceability(v):
    if v < 0.5: return "리듬이 불규칙하거나 약함 (자유박, 앰비언트)"
    if v < 1.0: return "리듬감 보통"
    if v < 1.5: return "리듬이 뚜렷하고 규칙적 (댄스곡 다수 포함)"
    return "매우 규칙적 · 반복적인 그루브 (EDM 등)"

def interpret_loudness(v):
    if v < 0.3: return "조용하게 믹싱됨"
    if v < 0.6: return "보통 수준의 라우드니스"
    return "강하게 마스터링됨 (라우드니스 워 성향)"

def interpret_dynamic_complexity(v):
    if v < 3: return "다이내믹이 좁음 (강하게 압축된 믹스)"
    if v <= 6: return "일반적인 대중음악 수준"
    return "다이내믹 폭이 넓음 (라이브, 클래식, 어쿠스틱 성향)"

def interpret_spectral_centroid(v):
    if v < 1000: return "어둡고 저음 중심 (베이스 강조, 따뜻한 톤)"
    if v < 2000: return "중간 밝기 (보컬 중심 믹스에 흔함)"
    if v < 3500: return "밝은 편 (신스, 하이햇 강조)"
    return "매우 밝음/샤프함 (harsh하게 들릴 수 있음)"

def interpret_zcr(v):
    if v < 0.05: return "부드럽고 톤(음정)이 뚜렷한 소리 위주"
    if v < 0.1: return "타악기/노이즈 요소가 어느 정도 섞임"
    return "노이즈성·타격감이 강함 (퍼커시브, 디스토션)"

def interpret_acousticness(v):
    if v < 0.3: return "전자·신스 기반 사운드, 어쿠스틱 악기 비중 낮음"
    if v < 0.6: return "전자 요소와 어쿠스틱 요소가 섞임"
    return "어쿠스틱 악기(기타, 피아노, 현악 등) 중심 사운드"

def interpret_energy(v):
    if v < 0.3: return "차분하고 잔잔함 (발라드, 앰비언트)"
    if v < 0.6: return "보통 수준의 에너지"
    return "강렬하고 활동적 (댄스, 록, EDM)"

def interpret_instrumentalness(v):
    if v < 0.3: return "보컬이 뚜렷하게 존재"
    if v < 0.6: return "보컬과 연주 비중이 비슷함"
    return "보컬이 거의 없는 연주곡 성격"

def interpret_valence(v):
    if v < 0.3: return "어둡고 우울한 정서 (마이너 성향과 자주 연관)"
    if v < 0.6: return "중립적인 정서"
    return "밝고 긍정적인 정서 (메이저 성향과 자주 연관)"


# ============================================================
# 핵심 분석 함수
# ============================================================
@st.cache_data(show_spinner="음원 분석 중...")
def analyze_audio(file_bytes, filename):
    import essentia.standard as es
    from essentia.standard import TensorflowPredictMusiCNN

    tmp_path = f"/tmp/{filename}"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    extractor = es.MusicExtractor(
        lowlevelStats=['mean', 'stdev'],
        rhythmStats=['mean', 'stdev'],
        tonalStats=['mean', 'stdev']
    )
    feats, _ = extractor(tmp_path)

    audio_16k = es.MonoLoader(filename=tmp_path, sampleRate=16000, resampleQuality=4)()

    embedding_model = es.TensorflowPredictEffnetDiscogs(
        graphFilename=mpath("discogs-effnet-bs64-1.pb"), output="PartitionedCall:1"
    )
    embeddings = embedding_model(audio_16k)

    # 장르
    genre_model = es.TensorflowPredict2D(
        graphFilename=mpath("genre_discogs400-discogs-effnet-1.pb"),
        input="serving_default_model_Placeholder", output="PartitionedCall:0"
    )
    genre_pred = np.mean(genre_model(embeddings), axis=0)
    with open(mpath("genre_discogs400-discogs-effnet-1.json")) as f:
        genre_labels = json.load(f)['classes']

    # acousticness
    acoustic_model = es.TensorflowPredict2D(
        graphFilename=mpath("mood_acoustic-discogs-effnet-1.pb"),
        input="model/Placeholder", output="model/Softmax"
    )
    acoustic_pred = np.mean(acoustic_model(embeddings), axis=0)
    with open(mpath("mood_acoustic-discogs-effnet-1.json")) as f:
        acoustic_labels = json.load(f)['classes']
    acousticness = float(acoustic_pred[acoustic_labels.index("acoustic")])

    # instrumentalness
    voice_model = es.TensorflowPredict2D(
        graphFilename=mpath("voice_instrumental-discogs-effnet-1.pb"),
        input="model/Placeholder", output="model/Softmax"
    )
    voice_pred = np.mean(voice_model(embeddings), axis=0)
    with open(mpath("voice_instrumental-discogs-effnet-1.json")) as f:
        voice_labels = json.load(f)['classes']
    instrumentalness = float(voice_pred[voice_labels.index("instrumental")])

    # valence / energy (musicnn 임베딩 별도 사용)
    musicnn_embedding_model = TensorflowPredictMusiCNN(
        graphFilename=mpath("msd-musicnn-1.pb"), output="model/dense/BiasAdd"
    )
    musicnn_embeddings = musicnn_embedding_model(audio_16k)
    emomusic_model = es.TensorflowPredict2D(
        graphFilename=mpath("emomusic-msd-musicnn-2.pb"), output="model/Identity"
    )
    emomusic_pred = np.mean(emomusic_model(musicnn_embeddings), axis=0)
    with open(mpath("emomusic-msd-musicnn-2.json")) as f:
        emomusic_labels = [c.lower() for c in json.load(f)['classes']]
    valence = float((emomusic_pred[emomusic_labels.index("valence")] - 1) / 8)
    energy = float((emomusic_pred[emomusic_labels.index("arousal")] - 1) / 8)

    final_bpm = float(genre_based_octave_correction(feats['rhythm.bpm'], genre_labels, genre_pred))

    top5_idx = np.argsort(genre_pred)[::-1][:5]
    top_genres = [(genre_labels[i], float(genre_pred[i])) for i in top5_idx]

    return {
        "duration": float(feats['metadata.audio_properties.length']),
        "bpm": final_bpm,
        "danceability": float(feats['rhythm.danceability']),
        "loudness": float(feats['lowlevel.average_loudness']),
        "dynamic_complexity": float(feats['lowlevel.dynamic_complexity']),
        "spectral_centroid": float(feats['lowlevel.spectral_centroid.mean']),
        "zcr": float(feats['lowlevel.zerocrossingrate.mean']),
        "acousticness": acousticness,
        "energy": energy,
        "instrumentalness": instrumentalness,
        "valence": valence,
        "top_genres": top_genres,
        "embedding_vector": np.mean(embeddings, axis=0).tolist(),
        "audio_16k": audio_16k,  # 파형/스펙트로그램용 (라이브러리에는 저장 안 함)
    }


def cosine_similarity(vec_a, vec_b):
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================
# 라이브러리 저장/불러오기
# ============================================================
def make_library_entry(result, filename):
    entry = {k: v for k, v in result.items() if k != "audio_16k"}
    entry["filename"] = filename
    return entry


def load_library():
    if os.path.exists(LIBRARY_PATH):
        try:
            with open(LIBRARY_PATH) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_to_library(entry):
    lib = load_library()
    lib = [s for s in lib if s["filename"] != entry["filename"]]
    lib.append(entry)
    with open(LIBRARY_PATH, "w") as f:
        json.dump(lib, f)
    return lib


# ============================================================
# 시각화 (레이더 차트 / 파형·스펙트로그램)
# ============================================================
def plot_radar(result, title=""):
    labels = ["Acousticness", "Energy", "Instrumentalness", "Valence", "Danceability"]
    values = [
        result["acousticness"],
        result["energy"],
        result["instrumentalness"],
        result["valence"],
        min(result["danceability"] / 1.5, 1.0),
    ]
    values = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F7F8FA")
    ax.plot(angles, values, color="#1F8F7B", linewidth=2)
    ax.fill(angles, values, color="#1F8F7B", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color="#1A1D24")
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title, color="#1A1D24", fontsize=11, pad=20)
    return fig


def plot_waveform_spectrogram(audio_array, sr=16000):
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.5))
    fig.patch.set_facecolor("white")

    t = np.linspace(0, len(audio_array) / sr, len(audio_array))
    axes[0].plot(t, audio_array, color="#1F8F7B", linewidth=0.4)
    axes[0].set_title("Waveform", fontsize=10, color="#1A1D24")
    axes[0].set_facecolor("#F7F8FA")
    axes[0].set_xlabel("Time (s)", fontsize=8)

    axes[1].specgram(audio_array, Fs=sr, cmap="viridis")
    axes[1].set_title("Spectrogram", fontsize=10, color="#1A1D24")
    axes[1].set_xlabel("Time (s)", fontsize=8)
    axes[1].set_ylabel("Frequency (Hz)", fontsize=8)

    plt.tight_layout()
    return fig


def metric_card(label, value, sub="", amber=False):
    cls = "metric-value amber" if amber else "metric-value"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="{cls}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def render_report(result):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("길이", format_duration(result["duration"]))
    with c2:
        metric_card("BPM (보정됨)", f'{result["bpm"]:.1f}', sub=interpret_bpm(result["bpm"]))
    with c3:
        metric_card("댄서빌리티", f'{result["danceability"]:.2f}', sub=interpret_danceability(result["danceability"]), amber=True)
    with c4:
        metric_card("다이내믹 범위", f'{result["dynamic_complexity"]:.2f} dB', sub=interpret_dynamic_complexity(result["dynamic_complexity"]))

    st.markdown('<div class="section-label">다이내믹스 / 라우드니스</div>', unsafe_allow_html=True)
    c1, = st.columns(1)
    with c1:
        metric_card("평균 러프니스", f'{result["loudness"]:.2f}', sub=interpret_loudness(result["loudness"]))

    st.markdown('<div class="section-label">감성 / 속성 지표</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Acousticness", f'{result["acousticness"]:.2f}', sub=interpret_acousticness(result["acousticness"]))
    with c2:
        metric_card("Energy", f'{result["energy"]:.2f}', sub=interpret_energy(result["energy"]), amber=True)
    with c3:
        metric_card("Instrumentalness", f'{result["instrumentalness"]:.2f}', sub=interpret_instrumentalness(result["instrumentalness"]))
    with c4:
        metric_card("Valence", f'{result["valence"]:.2f}', sub=interpret_valence(result["valence"]), amber=True)

    st.markdown('<div class="section-label">음색 (Timbre)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        metric_card("스펙트럴 센트로이드 (밝기)", f'{result["spectral_centroid"]:.0f} Hz', sub=interpret_spectral_centroid(result["spectral_centroid"]))
    with c2:
        metric_card("제로크로싱레이트 (타격감)", f'{result["zcr"]:.4f}', sub=interpret_zcr(result["zcr"]))

    st.markdown('<div class="section-label">장르 예측 TOP 5</div>', unsafe_allow_html=True)
    for label, prob in result["top_genres"]:
        st.markdown(f"""
        <div class="genre-row"><span>{label}</span><span>{prob*100:.1f}%</span></div>
        """, unsafe_allow_html=True)


# ============================================================
# 탭 구성
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎧 단일 곡 분석", "🔗 두 곡 유사도 비교", "📖 해석 가이드", "📚 라이브러리 & 유사곡"
])

with tab1:
    uploaded = st.file_uploader("음원 파일 업로드 (mp3, wav)", type=["mp3", "wav"], key="single")
    if uploaded:
        ensure_models()
        result = analyze_audio(uploaded.getvalue(), uploaded.name)
        render_report(result)

        st.markdown('<div class="section-label">파형 / 스펙트로그램</div>', unsafe_allow_html=True)
        st.pyplot(plot_waveform_spectrogram(result["audio_16k"]))

        st.markdown('<div class="section-label">감성 프로필 (레이더)</div>', unsafe_allow_html=True)
        st.pyplot(plot_radar(result, title=uploaded.name))

        if st.button("📚 라이브러리에 저장", key="save_single"):
            save_to_library(make_library_entry(result, uploaded.name))
            st.success(f"'{uploaded.name}'을(를) 라이브러리에 저장했어요.")

with tab2:
    colA, colB = st.columns(2)
    with colA:
        file_a = st.file_uploader("곡 A", type=["mp3", "wav"], key="a")
    with colB:
        file_b = st.file_uploader("곡 B", type=["mp3", "wav"], key="b")

    if file_a and file_b:
        ensure_models()
        result_a = analyze_audio(file_a.getvalue(), file_a.name)
        result_b = analyze_audio(file_b.getvalue(), file_b.name)
        sim = cosine_similarity(result_a["embedding_vector"], result_b["embedding_vector"])

        st.markdown('<div class="section-label">유사도 점수</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="similarity-score">{sim*100:.1f}%</div>', unsafe_allow_html=True)
        st.caption("discogs-effnet 임베딩(1280차원) 간 코사인 유사도 기준입니다.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{file_a.name}**")
            render_report(result_a)
        with c2:
            st.markdown(f"**{file_b.name}**")
            render_report(result_b)

with tab3:
    st.markdown("""
## 1. 리듬 / 템포

**BPM (rhythm.bpm → 장르 기반 옥타브 보정 적용)**
장르 예측 Top5와 각 장르별 전형적인 템포 중심값을 대조해서, 원본/절반/두배 BPM 중 가장 그럴듯한 값을 자동 채택합니다.

| 범위 | 느낌 |
|---|---|
| ~75 이하 | 매우 느림 (발라드, 앰비언트) |
| 76~95 | 느긋함 (미드템포 발라드, R&B) |
| 96~115 | 보통 (팝, 미드템포 댄스) |
| 116~130 | 업비트 (댄스, 하우스) |
| 131 이상 | 빠름 (EDM, 트랩, 펑크) |

**댄서빌리티 (rhythm.danceability)**
DFA(detrended fluctuation analysis) 기반 리듬 규칙성 지표. 1을 넘을 수 있는 상대값입니다.

| 범위 | 의미 |
|---|---|
| 0 ~ 0.5 | 리듬이 불규칙하거나 약함 (자유박, 앰비언트) |
| 0.5 ~ 1.0 | 리듬감 보통 |
| 1.0 ~ 1.5 | 리듬이 뚜렷하고 규칙적 (댄스곡 다수 포함) |
| 1.5 이상 | 매우 규칙적 · 반복적인 그루브 (EDM 등) |

---

## 2. 다이내믹스 / 라우드니스

**평균 러프니스 (lowlevel.average_loudness, 0~1 정규화)**

| 범위 | 의미 |
|---|---|
| 0.3 미만 | 조용하게 믹싱됨 |
| 0.3 ~ 0.6 | 보통 수준의 라우드니스 |
| 0.6 이상 | 강하게 마스터링됨 (라우드니스 워 성향) |

**다이내믹 복잡도 (lowlevel.dynamic_complexity, dB 단위)**

| 범위 | 의미 |
|---|---|
| 3 미만 | 다이내믹이 좁음 (강하게 압축된 믹스) |
| 3 ~ 6 | 일반적인 대중음악 수준 |
| 6 이상 | 다이내믹 폭이 넓음 (라이브, 클래식, 어쿠스틱 성향) |

---

## 3. 음색 (Timbre)

**스펙트럴 센트로이드 (lowlevel.spectral_centroid.mean, Hz)**
소리의 "밝기"를 나타내는 무게중심 주파수.

| 범위 | 의미 |
|---|---|
| 1000 Hz 미만 | 어둡고 저음 중심 (베이스 강조, 따뜻한 톤) |
| 1000 ~ 2000 Hz | 중간 밝기 (보컬 중심 믹스에 흔함) |
| 2000 ~ 3500 Hz | 밝은 편 (신스, 하이햇 강조) |
| 3500 Hz 이상 | 매우 밝음/샤프함 (harsh하게 들릴 수 있음) |

**제로크로싱레이트 (lowlevel.zerocrossingrate.mean)**

| 범위 | 의미 |
|---|---|
| 0.05 미만 | 부드럽고 톤(음정)이 뚜렷한 소리 위주 |
| 0.05 ~ 0.1 | 타악기/노이즈 요소가 어느 정도 섞임 |
| 0.1 이상 | 노이즈성·타격감이 강함 (퍼커시브, 디스토션) |

---

## 4. 감성 / 속성 지표 (0~1 정규화)

**acousticness (mood_acoustic 모델, "acoustic" 클래스 확률)**

| 범위 | 의미 |
|---|---|
| 0.3 미만 | 전자·신스 기반 사운드, 어쿠스틱 악기 비중 낮음 |
| 0.3 ~ 0.6 | 전자 요소와 어쿠스틱 요소가 섞임 |
| 0.6 이상 | 어쿠스틱 악기(기타, 피아노, 현악 등) 중심 사운드 |

**energy (emomusic 모델의 arousal 값 정규화, 각성도·강렬함)**

| 범위 | 의미 |
|---|---|
| 0.3 미만 | 차분하고 잔잔함 (발라드, 앰비언트) |
| 0.3 ~ 0.6 | 보통 수준의 에너지 |
| 0.6 이상 | 강렬하고 활동적 (댄스, 록, EDM) |

**instrumentalness (voice_instrumental 모델, "instrumental" 클래스 확률)**

| 범위 | 의미 |
|---|---|
| 0.3 미만 | 보컬이 뚜렷하게 존재 |
| 0.3 ~ 0.6 | 보컬과 연주 비중이 비슷함 |
| 0.6 이상 | 보컬이 거의 없는 연주곡 성격 |

**valence (emomusic 모델의 valence 값 정규화, 정서적 긍정성)**

| 범위 | 의미 |
|---|---|
| 0.3 미만 | 어둡고 우울한 정서 (마이너 성향과 자주 연관) |
| 0.3 ~ 0.6 | 중립적인 정서 |
| 0.6 이상 | 밝고 긍정적인 정서 (메이저 성향과 자주 연관) |
    """)

with tab4:
    st.markdown('<div class="section-label">일괄 분석 (여러 곡 한 번에)</div>', unsafe_allow_html=True)
    batch_files = st.file_uploader(
        "음원 파일 여러 개 업로드", type=["mp3", "wav"], accept_multiple_files=True, key="batch"
    )
    if batch_files:
        ensure_models()
        batch_results = []
        progress = st.progress(0)
        for i, f in enumerate(batch_files):
            r = analyze_audio(f.getvalue(), f.name)
            entry = make_library_entry(r, f.name)
            save_to_library(entry)
            batch_results.append(entry)
            progress.progress((i + 1) / len(batch_files))
        st.success(f"{len(batch_files)}곡 분석 및 라이브러리 저장 완료!")

        df = pd.DataFrame(batch_results)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("평균 BPM", f'{df["bpm"].mean():.1f}')
        with c2:
            metric_card("평균 댄서빌리티", f'{df["danceability"].mean():.2f}', amber=True)
        with c3:
            metric_card("평균 Energy", f'{df["energy"].mean():.2f}', amber=True)
        with c4:
            metric_card("평균 Valence", f'{df["valence"].mean():.2f}')

        top1_genres = [r["top_genres"][0][0] for r in batch_results]
        common = Counter(top1_genres).most_common(3)
        st.markdown('<div class="section-label">가장 흔한 장르 (Top1 기준)</div>', unsafe_allow_html=True)
        for g, cnt in common:
            st.markdown(f'<div class="genre-row"><span>{g}</span><span>{cnt}곡</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">저장된 라이브러리</div>', unsafe_allow_html=True)
    library = load_library()
    if not library:
        st.caption("아직 저장된 곡이 없어요. 단일 분석 탭에서 저장하거나 위에서 일괄 분석을 실행해보세요.")
    else:
        table_df = pd.DataFrame(library)[
            ["filename", "bpm", "danceability", "energy", "valence", "acousticness", "instrumentalness"]
        ]
        st.dataframe(table_df, use_container_width=True)

        st.markdown('<div class="section-label">유사곡 추천</div>', unsafe_allow_html=True)
        selected_name = st.selectbox("기준 곡 선택", [s["filename"] for s in library])
        selected = next(s for s in library if s["filename"] == selected_name)

        others = [s for s in library if s["filename"] != selected_name]
        if others:
            sims = [
                (s["filename"], cosine_similarity(selected["embedding_vector"], s["embedding_vector"]))
                for s in others
            ]
            sims.sort(key=lambda x: x[1], reverse=True)
            for name, score in sims[:3]:
                st.markdown(f'<div class="genre-row"><span>{name}</span><span>{score*100:.1f}%</span></div>', unsafe_allow_html=True)
        else:
            st.caption("비교할 다른 곡이 아직 없어요.")

        st.markdown('<div class="section-label">감성 프로필 (레이더)</div>', unsafe_allow_html=True)
        st.pyplot(plot_radar(selected, title=selected_name))

        if st.button("🗑️ 라이브러리 전체 삭제"):
            if os.path.exists(LIBRARY_PATH):
                os.remove(LIBRARY_PATH)
            st.rerun()
