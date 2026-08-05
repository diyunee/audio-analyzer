import os
import json
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
import requests
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

.main .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
div[data-testid="stVerticalBlock"] { gap: 0.3rem; }
div[data-testid="stHorizontalBlock"] { gap: 0.4rem; }

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
    border-radius: 8px;
    padding: 6px 10px;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.62rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 2px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 0.98rem;
    color: #1F8F7B;
    font-weight: 700;
    line-height: 1.15;
}
.metric-value.amber { color: #B96E1C; }
.metric-sub { color: #8A93A3; font-size: 0.62rem; margin-top: 1px; line-height: 1.15; }

.section-label {
    font-family: 'Space Mono', monospace;
    color: #B96E1C;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 12px 0 4px 0;
    border-bottom: 1px solid #E5E7EB;
    padding-bottom: 3px;
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

.guide-content { font-size: 0.82rem; line-height: 1.55; color: #374151; }
.guide-content h2 {
    font-family: 'Space Mono', monospace;
    font-size: 1.0rem; color: #1A1D24;
    margin: 22px 0 8px 0; border-bottom: 1px solid #E5E7EB; padding-bottom: 4px;
}
.guide-content h3 {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem; color: #1F8F7B; margin: 12px 0 4px 0;
}
.guide-content p { font-size: 0.78rem; color: #6B7280; margin: 2px 0 8px 0; }
.guide-content table { width: 100%; table-layout: fixed; border-collapse: collapse; font-size: 0.76rem; margin-bottom: 10px; }
.guide-content th, .guide-content td {
    border-bottom: 1px dashed #E5E7EB; padding: 4px 8px; text-align: left;
}
.guide-content th:first-child, .guide-content td:first-child { width: 32%; }
.guide-content th { color: #6B7280; font-weight: 600; }
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
# Last.fm API 설정
# ============================================================
try:
    _default_lastfm_key = st.secrets.get("LASTFM_API_KEY", "")
except Exception:
    _default_lastfm_key = ""

if not _default_lastfm_key:
    # TODO: 공개 저장소에 올릴 경우 이 줄을 지우고 .streamlit/secrets.toml의
    # LASTFM_API_KEY로 옮기는 걸 권장합니다.
    _default_lastfm_key = "763df6909f59e8cd9ab54e8ac100f4da"

with st.sidebar:
    st.markdown("### ⚙️ 설정")
    _lastfm_key_input = st.text_input(
        "Last.fm API Key",
        value=_default_lastfm_key,
        help="secrets.toml에 LASTFM_API_KEY를 등록해두면 자동으로 채워집니다. 발급받은 키를 그대로 붙여넣으세요.",
        placeholder="Last.fm API Key를 붙여넣으세요",
    )
    LASTFM_API_KEY = _lastfm_key_input.strip()
    if LASTFM_API_KEY:
        st.caption(f"✅ 키 등록됨 ({LASTFM_API_KEY[:4]}...{LASTFM_API_KEY[-4:]})")
    else:
        st.caption("⚠️ 아직 Last.fm API Key가 입력되지 않았어요.")
    st.caption("💡 파일명을 '제목_아티스트.mp3'로 올리면 제목/아티스트가 자동으로 채워져요.")

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
AUDIO_DIR = "audio_files"


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


METRIC_LABELS = {
    "duration": "Duration(길이)",
    "bpm": "BPM(템포)",
    "danceability": "Danceability(댄서빌리티)",
    "dynamic_complexity": "Dynamic Complexity(다이내믹)",
    "loudness": "Loudness(러프니스)",
    "acousticness": "Acousticness(어쿠스틱함)",
    "energy": "Energy(에너지)",
    "instrumentalness": "Instrumentalness(보컬없음)",
    "valence": "Valence(긍정정서)",
    "spectral_centroid": "Spectral Centroid(음색밝기)",
    "zcr": "Zero Crossing Rate(타격감)",
}


# ============================================================
# 핵심 분석 함수
# ============================================================
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    import essentia.standard as es
    return es.TensorflowPredictEffnetDiscogs(
        graphFilename=mpath("discogs-effnet-bs64-1.pb"), output="PartitionedCall:1"
    )


@st.cache_resource(show_spinner=False)
def get_genre_model():
    import essentia.standard as es
    return es.TensorflowPredict2D(
        graphFilename=mpath("genre_discogs400-discogs-effnet-1.pb"),
        input="serving_default_model_Placeholder", output="PartitionedCall:0"
    )


@st.cache_resource(show_spinner=False)
def get_acoustic_model():
    import essentia.standard as es
    return es.TensorflowPredict2D(
        graphFilename=mpath("mood_acoustic-discogs-effnet-1.pb"),
        input="model/Placeholder", output="model/Softmax"
    )


@st.cache_resource(show_spinner=False)
def get_voice_model():
    import essentia.standard as es
    return es.TensorflowPredict2D(
        graphFilename=mpath("voice_instrumental-discogs-effnet-1.pb"),
        input="model/Placeholder", output="model/Softmax"
    )


@st.cache_resource(show_spinner=False)
def get_musicnn_embedding_model():
    from essentia.standard import TensorflowPredictMusiCNN
    return TensorflowPredictMusiCNN(
        graphFilename=mpath("msd-musicnn-1.pb"), output="model/dense/BiasAdd"
    )


@st.cache_resource(show_spinner=False)
def get_emomusic_model():
    import essentia.standard as es
    return es.TensorflowPredict2D(
        graphFilename=mpath("emomusic-msd-musicnn-2.pb"), output="model/Identity"
    )


@st.cache_data(show_spinner=False)
def get_genre_labels():
    with open(mpath("genre_discogs400-discogs-effnet-1.json")) as f:
        return json.load(f)['classes']


@st.cache_data(show_spinner=False)
def get_acoustic_labels():
    with open(mpath("mood_acoustic-discogs-effnet-1.json")) as f:
        return json.load(f)['classes']


@st.cache_data(show_spinner=False)
def get_voice_labels():
    with open(mpath("voice_instrumental-discogs-effnet-1.json")) as f:
        return json.load(f)['classes']


@st.cache_data(show_spinner=False)
def get_emomusic_labels():
    with open(mpath("emomusic-msd-musicnn-2.json")) as f:
        return [c.lower() for c in json.load(f)['classes']]


@st.cache_data(show_spinner="음원 분석 중...")
def analyze_audio(file_bytes, filename):
    import essentia.standard as es

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

    embeddings = get_embedding_model()(audio_16k)

    # 장르
    genre_pred = np.mean(get_genre_model()(embeddings), axis=0)
    genre_labels = get_genre_labels()

    # acousticness
    acoustic_pred = np.mean(get_acoustic_model()(embeddings), axis=0)
    acoustic_labels = get_acoustic_labels()
    acousticness = float(acoustic_pred[acoustic_labels.index("acoustic")])

    # instrumentalness
    voice_pred = np.mean(get_voice_model()(embeddings), axis=0)
    voice_labels = get_voice_labels()
    instrumentalness = float(voice_pred[voice_labels.index("instrumental")])

    # valence / energy (musicnn 임베딩 별도 사용)
    musicnn_embeddings = get_musicnn_embedding_model()(audio_16k)
    emomusic_pred = np.mean(get_emomusic_model()(musicnn_embeddings), axis=0)
    emomusic_labels = get_emomusic_labels()
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


# ============================================================
# 파일명 / 태그에서 제목·아티스트 추출
# ============================================================
def parse_filename_title_artist(filename):
    """'제목_아티스트.mp3' 형식의 파일명을 파싱. 언더스코어 없으면 제목만 반환."""
    name = os.path.splitext(filename)[0]
    if "_" in name:
        title, artist = name.split("_", 1)
        return title.strip(), artist.strip()
    return name.strip(), ""


def extract_audio_tags(path):
    """오디오 파일의 ID3 등 태그에서 제목/아티스트 추출. 실패 시 빈 문자열."""
    try:
        from mutagen import File as MutagenFile
        audio = MutagenFile(path, easy=True)
        title = str(audio["title"][0]) if audio and "title" in audio and audio["title"] else ""
        artist = str(audio["artist"][0]) if audio and "artist" in audio and audio["artist"] else ""
        return title, artist
    except Exception:
        return "", ""


def resolve_title_artist(filename, path):
    """파일명 파싱을 우선으로 하고, 정보가 없는 부분만 ID3 태그로 보완."""
    name_title, name_artist = parse_filename_title_artist(filename)
    tag_title, tag_artist = extract_audio_tags(path) if path else ("", "")
    title = name_title or tag_title
    artist = name_artist or tag_artist
    return title, artist


# ============================================================
# Last.fm 유사 아티스트 / 유사곡
# ============================================================
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_similar_artists(artist_name, api_key, limit=6):
    """Last.fm artist.getSimilar 호출. [{name, match, url}, ...] 반환."""
    if not artist_name or not api_key:
        return []
    url = "https://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "artist.getsimilar",
        "artist": artist_name,
        "api_key": api_key,
        "format": "json",
        "limit": limit,
        "autocorrect": 1,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()
        if "error" in data:
            return []
        artists = data.get("similarartists", {}).get("artist", [])
        result = []
        for a in artists:
            result.append({
                "name": a.get("name", ""),
                "match": float(a.get("match", 0) or 0),
                "url": a.get("url", ""),
            })
        return result
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_similar_tracks(artist_name, track_name, api_key, limit=6):
    """Last.fm track.getSimilar 호출. [{name, artist, match, url}, ...] 반환."""
    if not artist_name or not track_name or not api_key:
        return []
    url = "https://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "track.getsimilar",
        "artist": artist_name,
        "track": track_name,
        "api_key": api_key,
        "format": "json",
        "limit": limit,
        "autocorrect": 1,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()
        if "error" in data:
            return []
        tracks = data.get("similartracks", {}).get("track", [])
        result = []
        for t in tracks:
            artist_field = t.get("artist", {})
            artist_name_out = artist_field.get("name", "") if isinstance(artist_field, dict) else str(artist_field)
            result.append({
                "name": t.get("name", ""),
                "artist": artist_name_out,
                "match": float(t.get("match", 0) or 0),
                "url": t.get("url", ""),
            })
        return result
    except Exception:
        return []


def render_similar_artists(artist_name, api_key, limit=6):
    if not api_key:
        st.caption("⚠️ 사이드바에 Last.fm API Key를 입력하면 유사 아티스트를 볼 수 있어요.")
        return
    if not artist_name:
        st.caption("아티스트 이름을 입력하면 유사 아티스트를 찾아드려요.")
        return
    similar = get_similar_artists(artist_name, api_key, limit=limit)
    if not similar:
        st.caption(f"'{artist_name}'에 대한 유사 아티스트 정보를 찾을 수 없어요.")
        return
    for a in similar:
        st.markdown(
            f'<div class="genre-row"><span><a href="{a["url"]}" target="_blank" '
            f'style="color:#1A1D24;text-decoration:none;">{a["name"]}</a></span>'
            f'<span>{a["match"]*100:.1f}%</span></div>',
            unsafe_allow_html=True,
        )


def render_similar_tracks(artist_name, track_name, api_key, limit=6):
    if not api_key:
        st.caption("⚠️ 사이드바에 Last.fm API Key를 입력하면 유사곡을 볼 수 있어요.")
        return
    if not artist_name or not track_name:
        st.caption("제목과 아티스트를 입력하면 유사곡을 찾아드려요.")
        return
    similar = get_similar_tracks(artist_name, track_name, api_key, limit=limit)
    if not similar:
        st.caption(f"'{track_name}'에 대한 유사곡 정보를 찾을 수 없어요.")
        return
    for t in similar:
        label = f'{t["name"]} — {t["artist"]}'
        st.markdown(
            f'<div class="genre-row"><span><a href="{t["url"]}" target="_blank" '
            f'style="color:#1A1D24;text-decoration:none;">{label}</a></span>'
            f'<span>{t["match"]*100:.1f}%</span></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# 라이브러리 저장/불러오기
# ============================================================
def make_library_entry(result, filename, title="", artist=""):
    entry = {k: v for k, v in result.items() if k != "audio_16k"}
    entry["filename"] = filename
    entry["title"] = title
    entry["artist"] = artist
    return entry


def save_audio_file(file_bytes, filename):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    path = os.path.join(AUDIO_DIR, filename)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


def get_audio_path(filename):
    path = os.path.join(AUDIO_DIR, filename)
    return path if os.path.exists(path) else None


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


def update_library_meta(filename, title, artist):
    lib = load_library()
    for s in lib:
        if s["filename"] == filename:
            s["title"] = title
            s["artist"] = artist
    with open(LIBRARY_PATH, "w") as f:
        json.dump(lib, f)
    return lib


def delete_from_library(filename):
    lib = load_library()
    lib = [s for s in lib if s["filename"] != filename]
    with open(LIBRARY_PATH, "w") as f:
        json.dump(lib, f)
    audio_path = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return lib


# ============================================================
# 시각화 (레이더 차트)
# ============================================================
def plot_radar(result, title="", figsize=(2.8, 2.8)):
    labels = [
        METRIC_LABELS["acousticness"].split("(")[0],
        METRIC_LABELS["energy"].split("(")[0],
        METRIC_LABELS["danceability"].split("(")[0],
        METRIC_LABELS["valence"].split("(")[0],
        METRIC_LABELS["dynamic_complexity"].split("(")[0],
    ]
    values = [
        result["acousticness"],
        result["energy"],
        min(result["danceability"] / 1.5, 1.0),
        result["valence"],
        min(result["dynamic_complexity"] / 8, 1.0),
    ]
    values = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F7F8FA")
    ax.plot(angles, values, color="#1F8F7B", linewidth=1.5)
    ax.fill(angles, values, color="#1F8F7B", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=5.0, color="#1A1D24")
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.2)
    ax.tick_params(pad=1)
    if title:
        ax.set_title(title, color="#1A1D24", fontsize=7, pad=10)
    fig.subplots_adjust(left=0.24, right=0.76, top=0.80, bottom=0.20)
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
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card(METRIC_LABELS["duration"], format_duration(result["duration"]))
    with c2:
        metric_card(METRIC_LABELS["bpm"], f'{result["bpm"]:.1f}', sub=interpret_bpm(result["bpm"]))
    with c3:
        metric_card(METRIC_LABELS["danceability"], f'{result["danceability"]:.2f}', sub=interpret_danceability(result["danceability"]), amber=True)
    with c4:
        metric_card(METRIC_LABELS["dynamic_complexity"], f'{result["dynamic_complexity"]:.2f} dB', sub=interpret_dynamic_complexity(result["dynamic_complexity"]))
    with c5:
        metric_card(METRIC_LABELS["loudness"], f'{result["loudness"]:.2f}', sub=interpret_loudness(result["loudness"]))

    st.markdown('<div class="section-label">감성 / 속성 지표</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(METRIC_LABELS["acousticness"], f'{result["acousticness"]:.2f}', sub=interpret_acousticness(result["acousticness"]))
    with c2:
        metric_card(METRIC_LABELS["energy"], f'{result["energy"]:.2f}', sub=interpret_energy(result["energy"]), amber=True)
    with c3:
        metric_card(METRIC_LABELS["instrumentalness"], f'{result["instrumentalness"]:.2f}', sub=interpret_instrumentalness(result["instrumentalness"]))
    with c4:
        metric_card(METRIC_LABELS["valence"], f'{result["valence"]:.2f}', sub=interpret_valence(result["valence"]), amber=True)

    st.markdown('<div class="section-label">음색 (Timbre)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        metric_card(METRIC_LABELS["spectral_centroid"], f'{result["spectral_centroid"]:.0f} Hz', sub=interpret_spectral_centroid(result["spectral_centroid"]))
    with c2:
        metric_card(METRIC_LABELS["zcr"], f'{result["zcr"]:.4f}', sub=interpret_zcr(result["zcr"]))

    st.markdown('<div class="section-label">장르 예측 TOP 5</div>', unsafe_allow_html=True)
    for label, prob in result["top_genres"]:
        st.markdown(f"""
        <div class="genre-row"><span>{label}</span><span>{prob*100:.1f}%</span></div>
        """, unsafe_allow_html=True)


def render_lastfm_block(key_prefix, default_title, default_artist, api_key):
    """제목/아티스트 입력 UI + 유사 아티스트/유사곡 결과를 함께 렌더링."""
    c1, c2 = st.columns(2)
    with c1:
        title_value = st.text_input(
            "곡 제목", value=default_title, key=f"title_{key_prefix}",
            placeholder="예: 좋은날"
        )
    with c2:
        artist_value = st.text_input(
            "아티스트 이름", value=default_artist, key=f"artist_{key_prefix}",
            placeholder="예: IU, 아이유"
        )

    st.markdown('<div class="section-label">유사 아티스트 (Last.fm)</div>', unsafe_allow_html=True)
    render_similar_artists(artist_value, api_key)

    st.markdown('<div class="section-label">유사곡 (Last.fm)</div>', unsafe_allow_html=True)
    render_similar_tracks(artist_value, title_value, api_key)

    return title_value, artist_value


# ============================================================
# 탭 구성
# ============================================================
tab1, tab4, tab3 = st.tabs([
    "🎧 단일 곡 분석", "📚 라이브러리 & 유사곡", "📖 해석 가이드"
])

with tab1:
    uploaded = st.file_uploader("음원 파일 업로드 (mp3, wav)", type=["mp3", "wav"], key="single")
    if uploaded:
        ensure_models()
        result = analyze_audio(uploaded.getvalue(), uploaded.name)

        left_col, right_col = st.columns([4, 1])

        with left_col:
            st.audio(uploaded.getvalue())
            render_report(result)

            tmp_tag_path = f"/tmp/tag_{uploaded.name}"
            with open(tmp_tag_path, "wb") as f:
                f.write(uploaded.getvalue())
            default_title, default_artist = resolve_title_artist(uploaded.name, tmp_tag_path)
            title_value, artist_value = render_lastfm_block(
                "single", default_title, default_artist, LASTFM_API_KEY
            )

        with right_col:
            st.markdown('<div class="section-label">감성 프로필</div>', unsafe_allow_html=True)
            st.pyplot(plot_radar(result, figsize=(2.0, 2.0)), use_container_width=False)

        if st.button("📚 라이브러리에 저장", key="save_single"):
            save_to_library(make_library_entry(result, uploaded.name, title=title_value, artist=artist_value))
            save_audio_file(uploaded.getvalue(), uploaded.name)
            st.success(f"'{uploaded.name}'을(를) 라이브러리에 저장했어요.")

with tab3:
    st.markdown("""
    <div class="guide-content">

    <h2>1. 리듬 / 템포</h2>

    <h3>BPM(템포)</h3>
    <p>장르 예측 Top5와 각 장르별 전형적인 템포 중심값을 대조해서 원본/절반/두배 BPM 중 가장 그럴듯한 값을 자동으로 채택하는 방식입니다.</p>
    <table>
    <tr><th>범위</th><th>느낌</th></tr>
    <tr><td>~75 이하</td><td>매우 느림 (발라드, 앰비언트)</td></tr>
    <tr><td>76~95</td><td>느긋함 (미드템포 발라드, R&amp;B)</td></tr>
    <tr><td>96~115</td><td>보통 (팝, 미드템포 댄스)</td></tr>
    <tr><td>116~130</td><td>업비트 (댄스, 하우스)</td></tr>
    <tr><td>131 이상</td><td>빠름 (EDM, 트랩, 펑크)</td></tr>
    </table>

    <h3>Danceability(댄서빌리티)</h3>
    <p>DFA(detrended fluctuation analysis) 기반 리듬 규칙성 지표이며, 1을 넘을 수 있는 상대값입니다.</p>
    <table>
    <tr><th>범위</th><th>의미</th></tr>
    <tr><td>0 ~ 0.5</td><td>리듬이 불규칙하거나 약함 (자유박, 앰비언트)</td></tr>
    <tr><td>0.5 ~ 1.0</td><td>리듬감 보통</td></tr>
    <tr><td>1.0 ~ 1.5</td><td>리듬이 뚜렷하고 규칙적 (댄스곡 다수 포함)</td></tr>
    <tr><td>1.5 이상</td><td>매우 규칙적 · 반복적인 그루브 (EDM 등)</td></tr>
    </table>

    <h2>2. 다이내믹스 / 라우드니스</h2>

    <h3>Loudness(러프니스)</h3>
    <p>0~1 사이로 정규화한 값입니다.</p>
    <table>
    <tr><th>범위</th><th>의미</th></tr>
    <tr><td>0.3 미만</td><td>조용하게 믹싱됨</td></tr>
    <tr><td>0.3 ~ 0.6</td><td>보통 수준의 라우드니스</td></tr>
    <tr><td>0.6 이상</td><td>강하게 마스터링됨 (라우드니스 워 성향)</td></tr>
    </table>

    <h3>Dynamic Complexity(다이내믹)</h3>
    <p>dB 단위로 측정한 값입니다.</p>
    <table>
    <tr><th>범위</th><th>의미</th></tr>
    <tr><td>3 미만</td><td>다이내믹이 좁음 (강하게 압축된 믹스)</td></tr>
    <tr><td>3 ~ 6</td><td>일반적인 대중음악 수준</td></tr>
    <tr><td>6 이상</td><td>다이내믹 폭이 넓음 (라이브, 클래식, 어쿠스틱 성향)</td></tr>
    </table>

    <h2>3. 음색 (Timbre)</h2>

    <h3>Spectral Centroid(음색밝기)</h3>
    <p>소리의 "밝기"를 나타내는 무게중심 주파수(Hz)입니다.</p>
    <table>
    <tr><th>범위</th><th>의미</th></tr>
    <tr><td>1000 Hz 미만</td><td>어둡고 저음 중심 (베이스 강조, 따뜻한 톤)</td></tr>
    <tr><td>1000 ~ 2000 Hz</td><td>중간 밝기 (보컬 중심 믹스에 흔함)</td></tr>
    <tr><td>2000 ~ 3500 Hz</td><td>밝은 편 (신스, 하이햇 강조)</td></tr>
    <tr><td>3500 Hz 이상</td><td>매우 밝음/샤프함 (harsh하게 들릴 수 있음)</td></tr>
    </table>

    <h3>Zero Crossing Rate(타격감)</h3>
    <p>신호가 0을 지나는 빈도를 나타내는 값입니다.</p>
    <table>
    <tr><th>범위</th><th>의미</th></tr>
    <tr><td>0.05 미만</td><td>부드럽고 톤(음정)이 뚜렷한 소리 위주</td></tr>
    <tr><td>0.05 ~ 0.1</td><td>타악기/노이즈 요소가 어느 정도 섞임</td></tr>
    <tr><td>0.1 이상</td><td>노이즈성·타격감이 강함 (퍼커시브, 디스토션)</td></tr>
    </table>

    <h2>4. 감성 / 속성 지표 (0~1 정규화)</h2>

    <h3>Acousticness(어쿠스틱함)</h3>
    <p>mood_acoustic 모델이 예측한 "acoustic" 클래스 확률입니다.</p>
    <table>
    <tr><th>범위</th><th>의미</th></tr>
    <tr><td>0.3 미만</td><td>전자·신스 기반 사운드, 어쿠스틱 악기 비중 낮음</td></tr>
    <tr><td>0.3 ~ 0.6</td><td>전자 요소와 어쿠스틱 요소가 섞임</td></tr>
    <tr><td>0.6 이상</td><td>어쿠스틱 악기(기타, 피아노, 현악 등) 중심 사운드</td></tr>
    </table>

    <h3>Energy(에너지)</h3>
    <p>emomusic 모델의 arousal 값을 정규화한 각성도·강렬함 지표입니다.</p>
    <table>
    <tr><th>범위</th><th>의미</th></tr>
    <tr><td>0.3 미만</td><td>차분하고 잔잔함 (발라드, 앰비언트)</td></tr>
    <tr><td>0.3 ~ 0.6</td><td>보통 수준의 에너지</td></tr>
    <tr><td>0.6 이상</td><td>강렬하고 활동적 (댄스, 록, EDM)</td></tr>
    </table>

    <h3>Instrumentalness(보컬없음)</h3>
    <p>voice_instrumental 모델이 예측한 "instrumental" 클래스 확률입니다.</p>
    <table>
    <tr><th>범위</th><th>의미</th></tr>
    <tr><td>0.3 미만</td><td>보컬이 뚜렷하게 존재</td></tr>
    <tr><td>0.3 ~ 0.6</td><td>보컬과 연주 비중이 비슷함</td></tr>
    <tr><td>0.6 이상</td><td>보컬이 거의 없는 연주곡 성격</td></tr>
    </table>

    <h3>Valence(긍정정서)</h3>
    <p>emomusic 모델의 valence 값을 정규화한 정서적 긍정성 지표입니다.</p>
    <table>
    <tr><th>범위</th><th>의미</th></tr>
    <tr><td>0.3 미만</td><td>어둡고 우울한 정서 (마이너 성향과 자주 연관)</td></tr>
    <tr><td>0.3 ~ 0.6</td><td>중립적인 정서</td></tr>
    <tr><td>0.6 이상</td><td>밝고 긍정적인 정서 (메이저 성향과 자주 연관)</td></tr>
    </table>

    </div>
    """, unsafe_allow_html=True)

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
            tmp_tag_path = f"/tmp/tag_{f.name}"
            with open(tmp_tag_path, "wb") as tf:
                tf.write(f.getvalue())
            default_title, default_artist = resolve_title_artist(f.name, tmp_tag_path)
            entry = make_library_entry(r, f.name, title=default_title, artist=default_artist)
            save_to_library(entry)
            save_audio_file(f.getvalue(), f.name)
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
        for entry in library:
            with st.expander(f"🎵 {entry['filename']}"):
                left_col, right_col = st.columns([4, 1])

                with left_col:
                    audio_path = get_audio_path(entry["filename"])
                    if audio_path:
                        st.audio(audio_path)
                    else:
                        st.caption("⚠️ 재생 파일 없음 (재배포로 초기화됐을 수 있어요)")

                    render_report(entry)

                    fallback_title, fallback_artist = resolve_title_artist(entry["filename"], get_audio_path(entry["filename"]))
                    default_title = entry.get("title") or fallback_title
                    default_artist = entry.get("artist") or fallback_artist

                    title_value, artist_value = render_lastfm_block(
                        entry["filename"], default_title, default_artist, LASTFM_API_KEY
                    )
                    if title_value != entry.get("title", "") or artist_value != entry.get("artist", ""):
                        update_library_meta(entry["filename"], title_value, artist_value)

                with right_col:
                    st.markdown('<div class="section-label">감성 프로필</div>', unsafe_allow_html=True)
                    st.pyplot(plot_radar(entry, figsize=(2.0, 2.0)), use_container_width=False)
                    if st.button("🗑️ 이 곡 삭제", key=f"delete_{entry['filename']}"):
                        delete_from_library(entry["filename"])
                        st.rerun()

        st.markdown('<div class="section-label">유사곡 추천 (내 라이브러리 기준)</div>', unsafe_allow_html=True)
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

        if st.button("🗑️ 라이브러리 전체 삭제"):
            if os.path.exists(LIBRARY_PATH):
                os.remove(LIBRARY_PATH)
            st.rerun()
