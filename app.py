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
import matplotlib.font_manager as fm
import plotly.express as px

# ============================================================
# 그래프용 한글 폰트 설정 (한글 깨짐 방지)
# ============================================================
KOREAN_FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
FONT_DIR = "fonts"
FONT_PATH = os.path.join(FONT_DIR, "NanumGothic-Regular.ttf")


def ensure_korean_font():
    os.makedirs(FONT_DIR, exist_ok=True)
    if not os.path.exists(FONT_PATH):
        try:
            urllib.request.urlretrieve(KOREAN_FONT_URL, FONT_PATH)
        except Exception:
            pass
    if os.path.exists(FONT_PATH):
        try:
            fm.fontManager.addfont(FONT_PATH)
            font_name = fm.FontProperties(fname=FONT_PATH).get_name()
            matplotlib.rcParams['font.family'] = font_name
        except Exception:
            pass
    matplotlib.rcParams['axes.unicode_minus'] = False


ensure_korean_font()

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
div[data-testid="stVerticalBlock"] { gap: 0.6rem; }
div[data-testid="stHorizontalBlock"] { gap: 0.8rem; }

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
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 12px;
    min-height: 84px;
    width: 100%;
    box-sizing: border-box;
}
.metric-card.recommend-card {
    width: 100%;
}
.metric-label {
    font-size: 0.66rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.05rem;
    color: #1F8F7B;
    font-weight: 700;
    line-height: 1.2;
}
.metric-value.amber { color: #B96E1C; }
.metric-value.recommend-score {
    font-size: 1.7rem;
    line-height: 1.3;
}
.metric-sub { color: #8A93A3; font-size: 0.68rem; margin-top: 3px; line-height: 1.35; }

.mood-card {
    background: #EFF7F5;
    border: 1px solid #1F8F7B;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 16px;
}
.mood-label {
    font-size: 0.66rem;
    color: #146B5C;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}
.mood-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.2rem;
    color: #146B5C;
    font-weight: 700;
}
.mood-desc { color: #4B7A70; font-size: 0.78rem; margin-top: 4px; }

.section-label {
    font-family: 'Space Mono', monospace;
    color: #B96E1C;
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 20px 0 10px 0;
    border-bottom: 1px solid #E5E7EB;
    padding-bottom: 5px;
}

.genre-row {
    display:flex; justify-content: space-between;
    font-family: 'Space Mono', monospace;
    padding: 8px 2px; border-bottom: 1px dashed #E5E7EB;
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
# 장르 기반 BPM 옥타브 보정 (+ 댄서빌리티·에너지 prior)
# ============================================================
GENRE_BPM_CENTER = {
    # 주의: 아래 키워드는 장르명에 '부분 문자열'로 매칭되므로, 짧고 모호한 단어는
    # 다른 의미의 복합 장르명과 충돌할 수 있다(예: "Vocal"이 "Vocal House"에도 걸림).
    # 그런 충돌이 실제로 있는 조합은 아래처럼 구체적인 복합 장르명으로 직접 등록해서,
    # genre_score()의 '가장 긴(구체적인) 키워드 우선' 규칙으로 올바른 쪽이 이기게 한다.
    "Ballad": 70, "R&B": 80, "Contemporary R&B": 78,
    "Chillwave": 90, "Downtempo": 85, "Ambient": 70, "Trip Hop": 85,
    "Lo-Fi": 80, "Bossa": 90,
    # 빠른 록 계열을 "Pop"이나 포괄적인 "Rock"보다 먼저 구체적으로 매칭한다.
    # 특히 Pop Punk/Punk Rock은 비트 추적기가 약 100 BPM으로 잡아도 실제 체감 템포가
    # 약 150 BPM인 3:2 오류가 자주 생겨, 아래의 1.5배 후보를 올바르게 평가할 수 있게 한다.
    "Skate Punk": 170, "Pop Punk": 155, "Punk Rock": 160, "Punk": 155,
    "Alternative Rock": 125, "Indie Rock": 125, "Hard Rock": 125,
    "Pop Rock": 125, "Rock": 125,
    "Indie Pop": 115, "Synth-pop": 112, "Pop": 110, "Dance-pop": 118, "Dance": 122,
    "Disco": 118, "Boogie": 116, "Funk": 108, "Freestyle": 118, "New Jack Swing": 100,
    "Tropical House": 105, "House": 124, "Deep House": 122, "Progressive House": 126,
    "Soulful House": 121, "Vocal House": 124, "Vocal Trance": 136,
    "Electro": 128, "Electropop": 118, "Eurodance": 138, "Big Room": 128, "K-Pop": 120,
    "Hip Hop": 90, "Trap": 140,
    "Techno": 130, "Trance": 136, "Dubstep": 140, "Drum": 170,
    "Hardstyle": 150, "Garage": 130, "EDM": 128,
}


def _danceability_bpm_range(danceability):
    """댄서빌리티(리듬 규칙성) 값이 그럴듯하게 나올 수 있는 BPM 대역을 추정.
    리듬이 규칙적일수록(값이 높을수록) 빠른 BPM대를, 불규칙할수록 느린 BPM대를 선호.
    어디까지나 '참고용 대역'이며, 아래 genre_based_octave_correction에서는
    이 대역을 하드 컷오프가 아니라 연속적인 점수로만 반영한다."""
    if danceability < 0.5:
        return (40, 95)
    elif danceability < 1.0:
        return (70, 115)
    elif danceability < 1.3:
        return (90, 130)
    else:
        return (100, 190)


def _energy_bpm_range(energy):
    """emomusic arousal 기반 Energy(0~1)가 그럴듯한 BPM 대역을 추정.
    danceability만으로는 리듬 규칙성 오검출(예: 강렬한 곡인데 DFA 값이 낮게 나오는 경우)에
    취약해서, 서로 다른 축인 Energy를 보조 신호로 함께 반영한다. 값이 높을수록(강렬할수록)
    빠른 BPM대를, 낮을수록 느린 BPM대를 선호. 마찬가지로 하드 컷오프가 아니라 연속 점수로만
    반영한다."""
    if energy < 0.3:
        return (40, 100)
    elif energy < 0.5:
        return (70, 120)
    elif energy < 0.7:
        return (90, 145)
    else:
        return (110, 190)


def genre_based_octave_correction(bpm_candidate, labels, avg_predictions, danceability=1.0, energy=0.5, top_n=8):
    """원본/절반/두배 BPM 후보 중 하나를 고르는 옥타브 보정.

    과거 버전은 댄서빌리티 대역에 후보가 '단 하나'만 걸리면 장르 신호를 완전히 무시하고
    그 후보를 그대로 채택했다(veto 방식). 문제는 DFA 기반 댄서빌리티 값이 EDM/댄스곡에서도
    낮게 측정되는 경우가 있어서, 이때 대역이 실제보다 느린 쪽으로 잡히고 정작 그 대역에
    "절반 BPM"(예: 실제 128인데 64) 후보 하나만 우연히 걸리면 장르가 명백히 댄스/일렉트로닉인데도
    발라드 템포로 잘못 보정되는 사례가 있었다.

    지금 버전은 하드 컷오프를 없애고, 장르 유사도 점수 · 댄서빌리티 대역 적합도 점수 ·
    Energy(emomusic arousal) 대역 적합도 점수를 모두 연속값으로 계산해 가중합으로 최종
    후보를 고른다. Energy는 danceability와 다른 축의 신호라서, danceability 하나가
    오검출되더라도 Energy가 보완해준다(예: 강렬하고 빠른 곡인데 DFA 리듬 규칙성이 낮게
    측정돼 danceability 대역만으로는 느린 BPM으로 잘못 끌려가는 경우). 장르 예측(400종
    분류)은 음향 prior 두 개보다 신뢰도가 높은 신호이므로, 상위 장르 중 하나라도 BPM
    성향이 뚜렷한 장르(장르명이 GENRE_BPM_CENTER 키워드와 매칭)와 겹치면 장르 쪽에 가장
    큰 비중을 준다."""
    top_idx = np.argsort(avg_predictions)[::-1][:top_n]
    top_genres = [(labels[i], avg_predictions[i]) for i in top_idx]
    # 절반/두배(옥타브 오류)뿐 아니라, 스윙/트리플렛 느낌의 리듬에서 흔한
    # 1.5배 · 2/3배 오검출도 후보에 포함시킨다.
    candidates = {
        "원본": bpm_candidate,
        "절반": bpm_candidate / 2,
        "두배": bpm_candidate * 2,
        "2/3배": bpm_candidate * 2 / 3,
        "1.5배": bpm_candidate * 1.5,
    }

    def genre_score(bpm_value):
        """장르별 (거리 기반 유사도)를 예측 확률로 가중평균한 값. 0~1 범위를 유지해야
        음향 prior 점수들(0~1)과 같은 스케일에서 공정하게 비교/가중합할 수 있다.
        (과거 버전은 매칭 개수로 단순 평균해서 값이 항상 작게 나왔고, 그 결과 가중치를
        줘도 사실상 음향 신호에 항상 밀리는 문제가 있었음)"""
        weighted_sum = 0.0
        weight_total = 0.0
        for genre_label, prob in top_genres:
            label_lower = genre_label.lower()
            best_match = None
            for keyword, center in GENRE_BPM_CENTER.items():
                if keyword.lower() in label_lower:
                    if best_match is None or len(keyword) > len(best_match[0]):
                        best_match = (keyword, center)
            if best_match is not None:
                distance = abs(bpm_value - best_match[1])
                similarity = np.exp(-(distance ** 2) / (2 * 25 ** 2))  # 0~1
                weighted_sum += similarity * prob
                weight_total += prob
        return float(weighted_sum / weight_total) if weight_total > 0 else 0.0

    def range_score(bpm_value, lo, hi):
        """대역 안이면 1.0, 밖이면 대역과의 거리에 따라 완만하게 감소 (하드 컷오프 아님)."""
        if lo <= bpm_value <= hi:
            return 1.0
        dist = (lo - bpm_value) if bpm_value < lo else (bpm_value - hi)
        return float(np.exp(-(dist ** 2) / (2 * 20 ** 2)))

    dance_lo, dance_hi = _danceability_bpm_range(danceability)
    energy_lo, energy_hi = _energy_bpm_range(energy)

    genre_scores = {k: genre_score(v) for k, v in candidates.items()}
    dance_scores = {k: range_score(v, dance_lo, dance_hi) for k, v in candidates.items()}
    energy_scores = {k: range_score(v, energy_lo, energy_hi) for k, v in candidates.items()}

    # 상위 장르 중 BPM 성향이 뚜렷한 장르와 매칭되는 게 하나라도 있으면(=장르 신호 존재)
    # 장르 쪽 비중을 가장 크게 두고, danceability·energy는 서로를 보완하는 보조 신호로
    # 절반씩 나눠 가진다. 매칭이 전혀 없으면 두 음향 신호에만 균등하게 의존한다.
    has_genre_signal = any(score > 0 for score in genre_scores.values())
    if has_genre_signal:
        genre_weight, dance_weight, energy_weight = 0.6, 0.2, 0.2
    else:
        genre_weight, dance_weight, energy_weight = 0.0, 0.5, 0.5

    # "원본"에 작은 가산점을 줘서, 다른 후보와 점수 차가 크지 않은 애매한 상황에서는
    # Essentia가 직접 검출한 원래 BPM을 함부로 뒤집지 않도록 한다. 장르/음향 신호가
    # 뚜렷하게 다른 후보를 가리킬 때는 이 가산점을 넘어서므로 여전히 보정이 이루어진다.
    ORIGINAL_BIAS = 0.08
    combined = {
        k: genre_weight * genre_scores[k] + dance_weight * dance_scores[k] + energy_weight * energy_scores[k]
        + (ORIGINAL_BIAS if k == "원본" else 0.0)
        for k in candidates
    }
    best_key = max(combined, key=combined.get)
    return candidates[best_key]


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
# 무드 자동 생성 (Essentia feature 기반 규칙 분류)
# ============================================================
# 각 방향마다 강도별 완성형 문구를 사용한다.
# 접두어를 기계적으로 조합하지 않아 자연스러운 한국어 무드명이 표시된다.
MOOD_ANCHORS = [
    (0,   ("은은하고 따뜻한", "따뜻하고 편안한", "밝고 포근한"),
     "따뜻하고 안정적인 긍정의 정서가 느껴져요"),
    (45,  ("가볍고 경쾌한", "밝고 신나는", "강렬하고 신나는"),
     "밝고 활기찬 에너지가 느껴져요"),
    (90,  ("긴장감이 감도는", "긴장감 있고 역동적인", "긴박하고 강렬한"),
     "팽팽한 긴장감과 움직임이 느껴져요"),
    (135, ("약간 거친", "거칠고 격렬한", "폭발적이고 격렬한"),
     "날카롭고 강한 에너지가 느껴져요"),
    (180, ("차분하고 어두운", "무겁고 어두운", "깊고 우울한"),
     "무겁고 가라앉은 정서가 느껴져요"),
    (225, ("잔잔하고 쓸쓸한", "쓸쓸하고 차분한", "깊고 애잔한"),
     "차분하면서도 쓸쓸한 여운이 남아요"),
    (270, ("편안하고 느긋한", "느긋하고 나른한", "몽환적이고 나른한"),
     "힘을 뺀 느슨하고 편안한 분위기예요"),
    (315, ("잔잔하고 편안한", "고요하고 편안한", "깊고 평온한"),
     "차분하고 안정적인 편안함이 느껴져요"),
]
MOOD_CENTER = ("담담하고 편안한", "감정의 치우침이 크지 않은 편안한 분위기예요")


def _angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def determine_mood(result):
    vx = 2 * result["valence"] - 1
    ey = 2 * result["energy"] - 1
    radius = float(np.hypot(vx, ey))

    if radius < 0.15:
        mood_label = f"{MOOD_CENTER[0]} 무드"
        base_desc = MOOD_CENTER[1]
    else:
        angle = float(np.degrees(np.arctan2(ey, vx)) % 360)
        anchor = min(MOOD_ANCHORS, key=lambda a: _angle_diff(a[0], angle))
        mood_words, base_desc = anchor[1], anchor[2]

        if radius >= 0.62:
            intensity_index = 2
        elif radius >= 0.35:
            intensity_index = 1
        else:
            intensity_index = 0
        mood_label = f"{mood_words[intensity_index]} 무드"

    return mood_label, base_desc


def determine_music_character(result):
    """BPM, ZCR, 예측 장르와 Acousticness를 사람이 읽기 쉬운 세 가지 특성으로 요약."""
    bpm = result.get("bpm", 100)
    zcr = result.get("zcr", 0.05)
    acousticness = result.get("acousticness", 0.5)

    if bpm <= 75:
        tempo_tag = "느린 템포"
    elif bpm <= 95:
        tempo_tag = "느긋한 템포"
    elif bpm <= 115:
        tempo_tag = "보통 템포"
    else:
        tempo_tag = "빠른 템포"

    if zcr >= 0.1:
        texture_tag = "거친 타격감"
    elif zcr >= 0.05:
        texture_tag = "선명한 타격감"
    else:
        texture_tag = "부드러운 음색"

    # '밴드 사운드'는 악기 직접 검출값이 아니라 상위 장르 예측을 바탕으로 한 표현이다.
    top_genres = [
        str(genre).lower()
        for genre, _score in result.get("top_genres", [])[:5]
    ]
    band_keywords = ("rock", "punk", "metal", "hardcore", "garage", "grunge")
    is_band_style = any(
        keyword in genre
        for genre in top_genres
        for keyword in band_keywords
    )

    if is_band_style:
        sound_tag = "밴드 사운드"
    elif acousticness >= 0.6:
        sound_tag = "어쿠스틱 사운드"
    elif acousticness < 0.3:
        sound_tag = "일렉트로닉 사운드"
    else:
        sound_tag = "혼합형 사운드"

    return [tempo_tag, texture_tag, sound_tag]


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

    genre_pred = np.mean(get_genre_model()(embeddings), axis=0)
    genre_labels = get_genre_labels()

    acoustic_pred = np.mean(get_acoustic_model()(embeddings), axis=0)
    acoustic_labels = get_acoustic_labels()
    acousticness = float(acoustic_pred[acoustic_labels.index("acoustic")])

    voice_pred = np.mean(get_voice_model()(embeddings), axis=0)
    voice_labels = get_voice_labels()
    instrumentalness = float(voice_pred[voice_labels.index("instrumental")])

    musicnn_embeddings = get_musicnn_embedding_model()(audio_16k)
    emomusic_pred = np.mean(get_emomusic_model()(musicnn_embeddings), axis=0)
    emomusic_labels = get_emomusic_labels()
    valence = float((emomusic_pred[emomusic_labels.index("valence")] - 1) / 8)
    energy = float((emomusic_pred[emomusic_labels.index("arousal")] - 1) / 8)

    danceability = float(feats['rhythm.danceability'])
    final_bpm = float(genre_based_octave_correction(
        feats['rhythm.bpm'], genre_labels, genre_pred,
        danceability=danceability, energy=energy
    ))

    top5_idx = np.argsort(genre_pred)[::-1][:5]
    top_genres = [(genre_labels[i], float(genre_pred[i])) for i in top5_idx]

    result = {
        "duration": float(feats['metadata.audio_properties.length']),
        "bpm": final_bpm,
        "danceability": danceability,
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
    mood_label, mood_desc = determine_mood(result)
    result["mood_label"] = mood_label
    result["mood_desc"] = mood_desc
    return result


def cosine_similarity(vec_a, vec_b):
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_recommendations(selected, others):
    """Essentia 임베딩(discogs-effnet) 코사인 유사도만으로 추천 점수를 계산.
    (Last.fm 아티스트 연관도는 더 이상 점수에 반영하지 않음 — 순수 음향 기반 추천)"""
    results = []
    for s in others:
        score = cosine_similarity(selected["embedding_vector"], s["embedding_vector"])
        results.append({
            "filename": s["filename"],
            "title": s.get("title") or s["filename"],
            "artist": (s.get("artist") or "").strip(),
            "score": score,
            "song": s,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def build_recommend_reason(selected, rec):
    """음향 임베딩 유사도 한 줄로 뭉뚱그리지 않고, 실제로 어떤 지표가 비슷한지
    장르 / 템포 / 무드 / 감성 프로필 / 음향 텍스처 순으로 구체적으로 짚어주는 추천 이유."""
    song = rec["song"]
    parts = []

    # 1) 예측 장르 겹침 (Top5 기준)
    selected_genres = [g for g, _ in selected.get("top_genres", [])[:5]]
    other_genres = [g for g, _ in song.get("top_genres", [])[:5]]
    selected_set = {g.lower() for g in selected_genres}
    overlap = [g for g in other_genres if g.lower() in selected_set]
    if overlap:
        parts.append(f"공통 예상 장르 '{overlap[0]}'" + (f" 외 {len(overlap)-1}개" if len(overlap) > 1 else ""))

    # 2) BPM 근접도
    bpm_diff = abs(selected["bpm"] - song["bpm"])
    if bpm_diff <= 4:
        parts.append(f"BPM 거의 동일 (약 {song['bpm']:.0f})")
    elif bpm_diff <= 12:
        parts.append(f"템포대 비슷함 (약 {song['bpm']:.0f} BPM)")

    # 3) 무드 라벨 일치
    if selected.get("mood_label") and selected.get("mood_label") == song.get("mood_label"):
        parts.append(f"무드 동일 ({song.get('mood_label')})")

    # 4) 감성 프로필(Valence·Energy) 근접도
    energy_diff = abs(selected["energy"] - song["energy"])
    valence_diff = abs(selected["valence"] - song["valence"])
    if energy_diff <= 0.12 and valence_diff <= 0.12:
        parts.append("에너지·긍정정서 프로필 거의 동일")
    elif energy_diff <= 0.2 and valence_diff <= 0.2:
        parts.append("감성 프로필(에너지·긍정정서) 비슷")

    # 5) 음향 텍스처(어쿠스틱함 · 댄서빌리티)
    if abs(selected["acousticness"] - song["acousticness"]) <= 0.15:
        parts.append("어쿠스틱함 정도 비슷")
    if abs(selected.get("danceability", 1.0) - song.get("danceability", 1.0)) <= 0.2:
        parts.append("리듬감(댄서빌리티) 비슷")

    if not parts:
        # 개별 지표는 크게 안 겹치지만 임베딩 전체로는 유사하다고 판단된 경우
        parts.append("개별 지표보단 전체적인 음향 임베딩(장르·리듬·음색을 종합한 벡터)이 유사")

    return " · ".join(parts[:3])


# ============================================================
# 파일명 / 태그에서 제목·아티스트 추출
# ============================================================
def parse_filename_title_artist(filename):
    name = os.path.splitext(filename)[0]
    if "_" in name:
        title, artist = name.split("_", 1)
        return title.strip(), artist.strip()
    return name.strip(), ""


def extract_audio_tags(path):
    try:
        from mutagen import File as MutagenFile
        audio = MutagenFile(path, easy=True)
        title = str(audio["title"][0]) if audio and "title" in audio and audio["title"] else ""
        artist = str(audio["artist"][0]) if audio and "artist" in audio and audio["artist"] else ""
        return title, artist
    except Exception:
        return "", ""


def resolve_title_artist(filename, path):
    name_title, name_artist = parse_filename_title_artist(filename)
    tag_title, tag_artist = extract_audio_tags(path) if path else ("", "")
    title = name_title or tag_title
    artist = name_artist or tag_artist
    return title, artist


# ============================================================
# Last.fm 유사 아티스트 / 유사 앨범
# ============================================================
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_similar_artists(artist_name, api_key, limit=6):
    """Last.fm artist.getSimilar 호출. [{name, match, url}, ...] 반환.
    - 실제 네트워크/API 오류(타임아웃, 잘못된 키, 레이트리밋 등)가 발생하면 None을 반환.
    - '해당 이름의 아티스트를 찾을 수 없음'(Last.fm 에러코드 6)처럼 정상적으로 결과가
      없는 경우에는 빈 리스트를 반환. 두 상황을 구분해야 상위 로직에서
      '오류라서 못 가져온 것'과 '그냥 결과가 없는 것'을 다르게 처리할 수 있음."""
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
            if data.get("error") == 6:  # 해당 이름의 아티스트 없음 (정상적인 빈 결과)
                return []
            return None
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
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_artist_top_albums(artist_name, api_key, limit=3):
    """Last.fm artist.getTopAlbums 호출. [{name, artist, url, playcount}, ...] 반환.
    get_similar_artists와 동일한 규칙으로 오류(None) / 정상 빈 결과([])를 구분한다.
    (앨범 단위 getSimilar API가 없어서, 유사 아티스트의 대표 앨범을 유사 앨범 후보로 사용)"""
    if not artist_name or not api_key:
        return []
    url = "https://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "artist.gettopalbums",
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
            if data.get("error") == 6:
                return []
            return None
        albums = data.get("topalbums", {}).get("album", [])
        result = []
        for alb in albums:
            name = alb.get("name", "")
            if not name:
                continue
            result.append({
                "name": name,
                "artist": artist_name,
                "url": alb.get("url", ""),
                "playcount": int(alb.get("playcount", 0) or 0),
            })
        return result
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_similar_albums(artist_name, api_key, limit=6):
    """유사 아티스트들의 대표 앨범을 모아 '유사 앨범' 후보 목록을 만듦.
    각 앨범은 원 아티스트와의 Last.fm 유사도(match)를 그대로 이어받음.
    유사 아티스트 조회 자체가 오류(None)면 그대로 None을 반환해 오류 상태를 전달한다.

    주의: 이건 곡의 오디오 특성(BPM·에너지·임베딩 등)을 기반으로 한 유사도가 아니라,
    Last.fm의 청취 기록 기반 아티스트 연관도를 그대로 앨범 단위로 끌어온 것이다.
    (곡 단위 음향 유사 앨범을 원한다면 Last.fm이 아니라 라이브러리 내 임베딩 비교가 필요함)

    반환: [{album, artist, match, url}, ...] match 내림차순, 오류 시 None."""
    if not artist_name or not api_key:
        return []
    similar_artists = get_similar_artists(artist_name, api_key, limit=limit)
    if similar_artists is None:
        return None
    if not similar_artists:
        return []

    # Last.fm이 표기가 다른 원 아티스트 자신을 유사 아티스트 목록에 섞어 보내는
    # 경우가 드물게 있어서, 그로 인해 같은 아티스트의 앨범이 '유사 앨범'으로
    # 다시 나오는 걸 막기 위해 자기 자신을 제외한다.
    original_lower = artist_name.strip().lower()
    seen_albums = set()
    results = []
    for a in similar_artists:
        if a["name"].strip().lower() == original_lower:
            continue
        top_albums = get_artist_top_albums(a["name"], api_key, limit=1)
        if top_albums is None:
            continue
        for alb in top_albums:
            dedup_key = (alb["name"].strip().lower(), alb["artist"].strip().lower())
            if dedup_key in seen_albums:
                continue
            seen_albums.add(dedup_key)
            results.append({
                "album": alb["name"],
                "artist": alb["artist"],
                "match": a["match"],
                "url": alb.get("url", a.get("url", "")),
            })
    results.sort(key=lambda x: x["match"], reverse=True)
    return results[:limit]


def render_similar_artists(artist_name, api_key, limit=6):
    if not api_key:
        st.caption("⚠️ 사이드바에 Last.fm API Key를 입력하면 유사 아티스트를 볼 수 있어요.")
        return
    if not artist_name:
        st.caption("아티스트 이름을 입력하면 유사 아티스트를 찾아드려요.")
        return
    similar = get_similar_artists(artist_name, api_key, limit=limit)
    if similar is None:
        st.caption("⚠️ Last.fm 연결에 문제가 있어 유사 아티스트를 불러오지 못했어요.")
        return
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


def render_similar_albums(artist_name, api_key, limit=6):
    """유사 아티스트의 대표 앨범 기반 '유사 앨범' 목록 렌더링.
    (track.getSimilar는 곡 표기 차이 때문에 실패가 잦아, 더 안정적인 앨범 단위 매칭으로 대체)"""
    if not api_key:
        st.caption("⚠️ 사이드바에 Last.fm API Key를 입력하면 유사 앨범을 볼 수 있어요.")
        return
    if not artist_name:
        st.caption("아티스트 이름을 입력하면 유사 앨범을 찾아드려요.")
        return
    albums = get_similar_albums(artist_name, api_key, limit=limit)
    if albums is None:
        st.caption("⚠️ Last.fm 연결에 문제가 있어 유사 앨범을 불러오지 못했어요.")
        return
    if not albums:
        st.caption(f"'{artist_name}'과(와) 유사한 아티스트의 앨범 정보를 찾을 수 없어요.")
        return
    for alb in albums:
        label = f'{alb["album"]} — {alb["artist"]}'
        st.markdown(
            f'<div class="genre-row"><span><a href="{alb["url"]}" target="_blank" '
            f'style="color:#1A1D24;text-decoration:none;">{label}</a></span>'
            f'<span>{alb["match"]*100:.1f}%</span></div>',
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
# 시각화 (레이더 차트 / 항목별 구간 그래프 / 2D 지도)
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


# 각 지표별로 해석 가이드 탭의 구간 경계값을 그대로 사용한 설정.
# breakpoints의 길이 + 1 = band_labels 개수가 되어야 함.
RANGE_METRIC_CONFIGS = {
    "bpm": {
        "label": "BPM", "breakpoints": [75, 95, 115, 130], "vmax": 200,
        "band_labels": ["매우느림", "느긋함", "보통", "업비트", "빠름"],
        "fmt": "{:.1f}", "unit": "",
    },
    "danceability": {
        "label": "Danceability", "breakpoints": [0.5, 1.0, 1.5], "vmax": 2.0,
        "band_labels": ["불규칙", "보통", "규칙적", "매우규칙"],
        "fmt": "{:.2f}", "unit": "",
    },
    "loudness": {
        "label": "Loudness", "breakpoints": [0.3, 0.6], "vmax": 1.0,
        "band_labels": ["조용함", "보통", "강하게"],
        "fmt": "{:.2f}", "unit": "",
    },
    "dynamic_complexity": {
        "label": "Dynamic Complexity", "breakpoints": [3, 6], "vmax": 10,
        "band_labels": ["좁음", "보통", "넓음"],
        "fmt": "{:.2f}", "unit": " dB",
    },
    "spectral_centroid": {
        "label": "Spectral Centroid", "breakpoints": [1000, 2000, 3500], "vmax": 6000,
        "band_labels": ["어두움", "중간", "밝은편", "매우밝음"],
        "fmt": "{:.0f}", "unit": " Hz",
    },
    "zcr": {
        "label": "Zero Crossing Rate", "breakpoints": [0.05, 0.1], "vmax": 0.2,
        "band_labels": ["부드러움", "혼합", "강함"],
        "fmt": "{:.4f}", "unit": "",
    },
    "acousticness": {
        "label": "Acousticness", "breakpoints": [0.3, 0.6], "vmax": 1.0,
        "band_labels": ["전자적", "혼합", "어쿠스틱"],
        "fmt": "{:.2f}", "unit": "",
    },
    "energy": {
        "label": "Energy", "breakpoints": [0.3, 0.6], "vmax": 1.0,
        "band_labels": ["차분함", "보통", "강렬함"],
        "fmt": "{:.2f}", "unit": "",
    },
    "valence": {
        "label": "Valence", "breakpoints": [0.3, 0.6], "vmax": 1.0,
        "band_labels": ["우울함", "중립", "긍정적"],
        "fmt": "{:.2f}", "unit": "",
    },
    "instrumentalness": {
        "label": "Instrumentalness", "breakpoints": [0.3, 0.6], "vmax": 1.0,
        "band_labels": ["보컬있음", "혼합", "연주곡"],
        "fmt": "{:.2f}", "unit": "",
    },
}


def plot_range_gauge(value, config, figsize=(1.9, 0.62)):
    """지표 하나의 값이 해석 가이드 상 어느 구간에 속하는지 보여주는 작은 구간 막대.
    각 metric-card 바로 아래에 작게 붙는 용도라서 최대한 소형으로 설계.
    구간 경계선(breakpoints)을 색 블록으로 나누고, 실제 값 위치를 세모/막대로 표시."""
    breakpoints = config["breakpoints"]
    vmax = config["vmax"]
    n_bands = len(config["band_labels"])
    edges = [0.0] + list(breakpoints) + [vmax]
    band_colors = ["#CFE8E2", "#8FCBB9", "#1F8F7B", "#146B5C", "#0B3A31"][:n_bands]

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i in range(n_bands):
        ax.barh(0, edges[i + 1] - edges[i], left=edges[i], height=0.55,
                color=band_colors[i], edgecolor="white", linewidth=0.6)

    v = min(max(value, edges[0]), edges[-1])
    ax.plot([v, v], [-0.4, 0.4], color="#1A1D24", linewidth=1.3, solid_capstyle="round")
    ax.scatter([v], [0.42], marker="v", color="#1A1D24", s=10, zorder=5)

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(-0.5, 0.6)
    ax.set_yticks([])
    ax.set_xticks(edges)
    ax.set_xticklabels([f"{e:g}" for e in edges], fontsize=4.2, color="#8A93A3")
    tick_labels = ax.get_xticklabels()
    if tick_labels:
        tick_labels[0].set_ha("left")
        tick_labels[-1].set_ha("right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=1)

    fmt = config.get("fmt", "{:.2f}")
    unit = config.get("unit", "")
    ax.set_title(f'{fmt.format(value)}{unit}', fontsize=5.8,
                 color="#1A1D24", pad=1.5)
    # tight_layout 대신 고정 마진을 써서, 라벨 글자 수가 달라도 막대(축 영역)의
    # 실제 폭이 모든 항목에서 동일하게 렌더링되도록 함
    fig.subplots_adjust(left=0.02, right=0.98, top=0.62, bottom=0.34)
    return fig


def build_library_map(library, height=620):
    """라이브러리 전체 곡을 Valence(x) x Energy(y) 평면에 흩뿌리는 인터랙티브 지도.
    무드 배너와 같은 정서 원형모델(circumplex) 좌표계를 사용.
    곡 제목은 화면에 항상 표시하지 않고 마우스를 올렸을 때(hover)만 보여줘서,
    라이브러리에 곡이 아무리 많이 쌓여도 라벨끼리 겹치는 문제가 생기지 않도록 함."""
    df = pd.DataFrame([{
        "title": s.get("title") or s["filename"],
        "artist": s.get("artist") or "미상",
        "valence": s["valence"],
        "energy": s["energy"],
        "danceability": s.get("danceability", 1.0),
        "acousticness": s.get("acousticness", 0.5),
    } for s in library])

    fig = px.scatter(
        df, x="valence", y="energy",
        # 점 크기는 동일하게 유지하고 Danceability를 색으로 표현한다.
        # 이 분석기의 Danceability는 1을 넘을 수 있으므로 0~2 범위를 사용한다.
        color="danceability",
        # 한 계열의 순차 색상: 값이 높을수록 더 짙은 주황색으로 표시한다.
        # 가장 낮은 색도 흰 배경에서 사라지지 않도록 충분한 채도를 유지한다.
        color_continuous_scale=["#F6C982", "#E8943A", "#C85F17", "#8F3508"],
        range_color=[0, 2.0],
        custom_data=["title", "artist", "danceability", "acousticness"],
    )
    fig.update_traces(
        marker=dict(size=13, line=dict(width=1.2, color="#1A1D24"), opacity=0.9),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br>"
            "Valence %{x:.2f} · Energy %{y:.2f}<br>"
            "Danceability %{customdata[2]:.2f} · Acousticness %{customdata[3]:.2f}"
            "<extra></extra>"
        ),
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#D1D5DB", line_width=1)
    fig.add_vline(x=0.5, line_dash="dash", line_color="#D1D5DB", line_width=1)
    fig.update_layout(
        plot_bgcolor="#F7F8FA",
        paper_bgcolor="white",
        font=dict(family="Manrope, sans-serif", color="#1A1D24"),
        xaxis=dict(
            title="어두운 분위기 ← Valence → 밝은 분위기", range=[-0.08, 1.08],
            color="#374151", gridcolor="#E5E7EB", zeroline=False,
        ),
        yaxis=dict(
            title="차분함 ← Energy → 강렬함", range=[-0.08, 1.08],
            color="#374151", gridcolor="#E5E7EB", zeroline=False,
        ),
        coloraxis_colorbar=dict(
            title="Danceability<br>(춤추기 좋은 정도)", orientation="h",
            y=-0.22, len=0.85, thickness=12,
            tickvals=[0, 0.5, 1.0, 1.5, 2.0],
            ticktext=["낮음", "0.5", "보통", "1.5", "높음"],
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=height,
    )
    return fig


def metric_card(label, value, sub="", amber=False, value_class=""):
    cls = "metric-value"
    if amber:
        cls += " amber"
    if value_class:
        cls += f" {value_class}"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="{cls}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def render_mood_banner(result):
    # 저장된 예전 문구 대신 현재 규칙으로 다시 계산해 기존 라이브러리에도 즉시 반영한다.
    mood_label, mood_desc = determine_mood(result)
    music_character = determine_music_character(result)
    st.markdown(f"""
    <div class="mood-card">
        <div class="mood-label">Mood(자동분류 무드)</div>
        <div class="mood-value">{mood_label}</div>
        <div class="mood-desc">{mood_desc}</div>
        <div class="mood-desc"><strong>음악적 성격:</strong> {" · ".join(music_character)}</div>
    </div>
    """, unsafe_allow_html=True)


def render_predicted_genres(result, top_n=5):
    """Discogs-EffNet 장르 모델이 예측한 상위 장르와 모델 점수를 표시."""
    top_genres = result.get("top_genres") or []
    st.markdown('<div class="section-label">예상 장르 (Top 5)</div>', unsafe_allow_html=True)

    if not top_genres:
        st.caption("이 곡에는 저장된 예상 장르 정보가 없어요. 곡을 다시 분석하면 표시됩니다.")
        return

    for genre, score in top_genres[:top_n]:
        try:
            score_text = f"{float(score) * 100:.1f}%"
        except (TypeError, ValueError):
            score_text = "-"
        st.markdown(
            f'<div class="genre-row"><span>{genre}</span><span>{score_text}</span></div>',
            unsafe_allow_html=True,
        )
    st.caption("장르 옆 수치는 Discogs-EffNet 모델의 상대적인 예측 점수이며, 확정 장르나 정확도를 뜻하지 않아요.")


def render_report(result):
    render_mood_banner(result)
    render_predicted_genres(result)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(METRIC_LABELS["bpm"], f'{result["bpm"]:.1f}', sub=interpret_bpm(result["bpm"]))
        st.pyplot(plot_range_gauge(result["bpm"], RANGE_METRIC_CONFIGS["bpm"]), use_container_width=True)
    with c2:
        metric_card(METRIC_LABELS["danceability"], f'{result["danceability"]:.2f}', sub=interpret_danceability(result["danceability"]), amber=True)
        st.pyplot(plot_range_gauge(result["danceability"], RANGE_METRIC_CONFIGS["danceability"]), use_container_width=True)
    with c3:
        metric_card(METRIC_LABELS["dynamic_complexity"], f'{result["dynamic_complexity"]:.2f} dB', sub=interpret_dynamic_complexity(result["dynamic_complexity"]))
        st.pyplot(plot_range_gauge(result["dynamic_complexity"], RANGE_METRIC_CONFIGS["dynamic_complexity"]), use_container_width=True)
    with c4:
        metric_card(METRIC_LABELS["loudness"], f'{result["loudness"]:.2f}', sub=interpret_loudness(result["loudness"]))
        st.pyplot(plot_range_gauge(result["loudness"], RANGE_METRIC_CONFIGS["loudness"]), use_container_width=True)

    st.markdown('<div class="section-label">감성 / 속성 지표</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(METRIC_LABELS["acousticness"], f'{result["acousticness"]:.2f}', sub=interpret_acousticness(result["acousticness"]))
        st.pyplot(plot_range_gauge(result["acousticness"], RANGE_METRIC_CONFIGS["acousticness"]), use_container_width=True)
    with c2:
        metric_card(METRIC_LABELS["energy"], f'{result["energy"]:.2f}', sub=interpret_energy(result["energy"]), amber=True)
        st.pyplot(plot_range_gauge(result["energy"], RANGE_METRIC_CONFIGS["energy"]), use_container_width=True)
    with c3:
        metric_card(METRIC_LABELS["instrumentalness"], f'{result["instrumentalness"]:.2f}', sub=interpret_instrumentalness(result["instrumentalness"]))
        st.pyplot(plot_range_gauge(result["instrumentalness"], RANGE_METRIC_CONFIGS["instrumentalness"]), use_container_width=True)
    with c4:
        metric_card(METRIC_LABELS["valence"], f'{result["valence"]:.2f}', sub=interpret_valence(result["valence"]), amber=True)
        st.pyplot(plot_range_gauge(result["valence"], RANGE_METRIC_CONFIGS["valence"]), use_container_width=True)

    st.markdown('<div class="section-label">음색 (Timbre)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(METRIC_LABELS["spectral_centroid"], f'{result["spectral_centroid"]:.0f} Hz', sub=interpret_spectral_centroid(result["spectral_centroid"]))
        st.pyplot(plot_range_gauge(result["spectral_centroid"], RANGE_METRIC_CONFIGS["spectral_centroid"]), use_container_width=True)
    with c2:
        metric_card(METRIC_LABELS["zcr"], f'{result["zcr"]:.4f}', sub=interpret_zcr(result["zcr"]))
        st.pyplot(plot_range_gauge(result["zcr"], RANGE_METRIC_CONFIGS["zcr"]), use_container_width=True)


def render_lastfm_block(key_prefix, default_title, default_artist, api_key):
    """제목/아티스트 입력 UI + 유사 아티스트/유사 앨범 결과를 함께 렌더링."""
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

    st.markdown('<div class="section-label">유사 앨범 (Last.fm)</div>', unsafe_allow_html=True)
    render_similar_albums(artist_value, api_key)

    return title_value, artist_value


# ============================================================
# 탭 구성
# ============================================================
tab1, tab4, tab_map, tab3 = st.tabs([
    "🎧 단일 곡 분석", "📚 라이브러리 & 유사곡", "🗺️ 라이브러리 지도", "📖 해석 가이드"
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
    <p>원본 BPM뿐 아니라 절반·두배(옥타브 오류) 및 1.5배·2/3배(스윙·트리플렛 리듬에서 흔한 오검출) 후보까지
    총 5가지를 놓고 고릅니다. 각 후보는 ① 장르 예측 Top8과 장르별 전형적인 템포를 비교한 "장르 유사도",
    ② 댄서빌리티(리듬 규칙성)로 추정한 그럴듯한 BPM 대역과의 근접도, ③ Energy(각성도)로 추정한
    그럴듯한 BPM 대역과의 근접도, 세 점수를 0~1 범위로 정규화해 가중합한 값으로 비교합니다. Energy는
    댄서빌리티와 다른 축의 신호라서, 리듬 규칙성이 낮게 측정되는 곡(강렬하지만 DFA 값이 낮게 나오는 경우 등)
    에서도 실제 템포감을 보완해줍니다. 장르 라벨 하나에 여러 키워드가 부분 문자열로 걸릴 수 있는 경우(예:
    "Vocal House"가 "Vocal"과 "House" 둘 다에 걸리는 것) 가장 구체적인 키워드 하나만 채택해서, 이름만
    비슷한 다른 장르 때문에 엉뚱한 템포로 끌려가지 않게 합니다. 또한 뚜렷한 근거가 없을 때는 Essentia가
    직접 검출한 원본 BPM에 약간의 가산점을 줘서, 애매한 상황에서 불필요하게 배수를 뒤집지 않도록 했습니다.</p>
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

    <h2>5. Mood(자동분류 무드)</h2>
    <p>Valence(정서 긍정성)와 Energy(각성도)를 정서 원형모델(circumplex model) 평면에 좌표로 놓고,
    45도 간격 8방향(따뜻한·신나는·긴장감 있는·격앙된·우울한·쓸쓸한·나른한·평온한) + 중심(담백한) 중
    가장 가까운 방향으로 기본 무드를 정합니다. 원점에서 얼마나 떨어져 있는지(감정의 강도)에 따라
    각 분위기에 맞는 자연스러운 완성형 무드명을 선택합니다. 무드 아래의 "음악적 성격"은 BPM으로 템포를,
    ZCR로 타격감과 음색 성향을, 상위 장르 예측과 Acousticness로 밴드·어쿠스틱·일렉트로닉 등의 사운드 성격을
    요약합니다. "밴드 사운드"는 악기 구성을 직접 검출한 결과가 아니라 록·펑크·메탈 계열 장르 예측을 바탕으로 한
    설명입니다. 별도의 모델 호출 없이 이미 계산된 Essentia 지표와 장르 예측값만 사용해요.</p>

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
            recs = compute_recommendations(selected, others)
            st.caption("내 라이브러리 안에서 Essentia 음향 임베딩(장르·리듬·음색 종합)이 가장 비슷한 곡을 추천해드려요.")
            for rec in recs[:3]:
                reason = build_recommend_reason(selected, rec)
                display_name = f'{rec["title"]}' + (f' — {rec["artist"]}' if rec["artist"] else "")
                st.markdown(f"""
                <div class="metric-card recommend-card">
                    <div class="metric-label">{display_name}</div>
                    <div class="metric-value recommend-score">유사도 {rec['score']*100:.1f}%</div>
                    <div class="metric-sub">💬 추천 이유: {reason}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("비교할 다른 곡이 아직 없어요.")

        if st.button("🗑️ 라이브러리 전체 삭제"):
            if os.path.exists(LIBRARY_PATH):
                os.remove(LIBRARY_PATH)
            st.rerun()

with tab_map:
    map_library = load_library()
    if not map_library:
        st.caption("아직 저장된 곡이 없어요. 단일 분석 탭에서 저장하거나 라이브러리 탭에서 일괄 분석을 실행해보세요.")
    elif len(map_library) < 2:
        st.caption("지도를 그리려면 라이브러리에 곡이 2개 이상 있어야 해요.")
    else:
        st.markdown('<div class="section-label">라이브러리 지도 (감성 분포 한눈에 보기)</div>', unsafe_allow_html=True)
        st.caption("가로: 어두운 분위기 ↔ 밝은 분위기 · 세로: 차분함 ↔ 강렬함")
        st.caption("점 색: Danceability — 연한 주황색일수록 낮고, 짙은 주황색일수록 높아요 · 마우스를 올리면 곡 정보와 Acousticness가 나타나요")
        st.plotly_chart(build_library_map(map_library), use_container_width=True)
