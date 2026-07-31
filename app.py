import os
import json
import urllib.request

import numpy as np
import streamlit as st

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
    }


def cosine_similarity(vec_a, vec_b):
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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
        metric_card("BPM (보정됨)", f'{result["bpm"]:.1f}')
    with c3:
        metric_card("댄서빌리티", f'{result["danceability"]:.2f}', amber=True)
    with c4:
        metric_card("다이내믹 범위", f'{result["dynamic_complexity"]:.2f} dB')

    st.markdown('<div class="section-label">감성 / 속성 지표</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Acousticness", f'{result["acousticness"]:.2f}')
    with c2:
        metric_card("Energy", f'{result["energy"]:.2f}', amber=True)
    with c3:
        metric_card("Instrumentalness", f'{result["instrumentalness"]:.2f}')
    with c4:
        metric_card("Valence", f'{result["valence"]:.2f}', amber=True)

    st.markdown('<div class="section-label">음색 (Timbre)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        metric_card("스펙트럴 센트로이드 (밝기)", f'{result["spectral_centroid"]:.0f} Hz')
    with c2:
        metric_card("제로크로싱레이트 (타격감)", f'{result["zcr"]:.4f}')

    st.markdown('<div class="section-label">장르 예측 TOP 5</div>', unsafe_allow_html=True)
    for label, prob in result["top_genres"]:
        st.markdown(f"""
        <div class="genre-row"><span>{label}</span><span>{prob*100:.1f}%</span></div>
        """, unsafe_allow_html=True)


# ============================================================
# 탭 구성: 단일 분석 / 유사도 비교
# ============================================================
tab1, tab2, tab3 = st.tabs(["🎧 단일 곡 분석", "🔗 두 곡 유사도 비교", "📖 해석 가이드"])

with tab1:
    uploaded = st.file_uploader("음원 파일 업로드 (mp3, wav)", type=["mp3", "wav"], key="single")
    if uploaded:
        ensure_models()
        result = analyze_audio(uploaded.getvalue(), uploaded.name)
        render_report(result)

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
### 리듬 / 템포
| 항목 | 범위 | 의미 |
|---|---|---|
| BPM | 76~95 | 느긋함 (발라드, R&B) |
| BPM | 96~115 | 보통 (팝, 미드템포 댄스) |
| BPM | 116~130 | 업비트 (댄스, 하우스) |
| 댄서빌리티 | 1.0~1.5 | 리듬이 뚜렷하고 규칙적 |

### 다이내믹스
| 항목 | 범위 | 의미 |
|---|---|---|
| 다이내믹 복잡도 | 3~6 dB | 대중음악 표준 범위 |
| 다이내믹 복잡도 | 6 dB 이상 | 다이내믹 폭 넓음 (라이브·클래식 성향) |

### 감성 / 속성 지표 (0~1)
| 지표 | 0.6 이상일 때 의미 |
|---|---|
| Acousticness | 어쿠스틱 악기 중심 사운드 |
| Energy | 강렬하고 활동적 (댄스, 록, EDM) |
| Instrumentalness | 보컬이 거의 없는 연주곡 |
| Valence | 밝고 긍정적인 정서 |

### 음색
| 항목 | 범위 | 의미 |
|---|---|---|
| 스펙트럴 센트로이드 | 1000Hz 미만 | 어둡고 저음 중심 |
| 스펙트럴 센트로이드 | 2000~3500Hz | 밝은 편 (신스, 하이햇 강조) |
    """)
