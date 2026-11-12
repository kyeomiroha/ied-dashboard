# 뇌파 4초 조각 IED 자동 탐지 결과 대시보드
# 실행: streamlit run app.py
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="뇌파 IED 탐지 시스템", page_icon="🧠", layout="wide")

CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
SHOW = ["Fp1", "F3", "C3", "O1"]  # 시안에 그려진 4채널
INK, BLUE, RED, PINK, LAV = "#1d1d1f", "#1f6fe0", "#e0393e", "#fde4e4", "#dcdbf6"

# 시안(손그림) 모양: 손글씨 글꼴, 검은 둥근 테두리 카드, 연보라 라벨
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Gaegu:wght@400;700&display=swap');
.stApp, .stApp * {font-family: 'Gaegu', sans-serif !important;}
.stApp [data-testid="stIconMaterial"], .stApp [data-testid="stIconMaterial"] * {font-family: 'Material Symbols Rounded' !important;}
.stApp p, .stApp label, .stApp li {font-size: 1.15rem;}
.st-key-sian {border: 2.5px solid #1d1d1f; border-radius: 30px; padding: 22px 28px 28px; background: #fff;}
.st-key-sian [data-testid="stSelectbox"] [role="group"], .st-key-sian [data-baseweb="select"] > div
{border: 2px solid #1d1d1f; border-radius: 12px; background: #fff;}
.st-key-sian [data-testid="stSelectbox"] input, .st-key-sian [data-baseweb="select"] * {font-size: 1.15rem !important;}
@media (max-width: 640px) {.sian-lab {text-align: left;}}
.sian-title {font-size: clamp(1.6rem, 2.2vw, 2.5rem); font-weight: 700; color: #1d1d1f; line-height: 1.25;
  padding: 0 6px; white-space: nowrap; background: linear-gradient(transparent 62%, #dcdbf6 62%);}
.sian-lab {font-size: 1.4rem; color: #1d1d1f; text-align: right; white-space: nowrap;}
.sian-card {border: 2px solid #1d1d1f; border-radius: 22px; padding: 14px 22px 16px; background: #fff;
  min-height: 245px; margin-bottom: 4px;}
.sian-pill {display: inline-block; background: #dcdbf6; border-radius: 999px; padding: 0 18px;
  font-size: 1.6rem; font-weight: 700; color: #1d1d1f;}
.sian-verdict {margin: 14px 8px 6px; border-radius: 18px; padding: 10px; text-align: center;
  font-size: 3.6rem; font-weight: 700;}
.sian-note {font-size: 1.25rem; color: #555; text-align: center;}
.sian-stats {display: flex; margin-top: 14px;}
.sian-stat {flex: 1; text-align: center; color: #1d1d1f;}
.sian-stat + .sian-stat {border-left: 2px solid #9a9a9a;}
.sian-stat .k {font-size: 1.45rem; line-height: 1.15; min-height: 3.4rem; display: flex;
  align-items: center; justify-content: center;}
.sian-stat .v {font-size: 2.2rem; display: flex; align-items: center; justify-content: center; gap: 10px;
  white-space: nowrap;}
</style>""", unsafe_allow_html=True)

ICONS = {  # 시안의 쌓인 판, 경고 삼각형, 과녁 아이콘
    "layers": '<path d="M20 6 L36 14 L20 22 L4 14 Z"/><path d="M4 20 L20 28 L36 20"/><path d="M4 26 L20 34 L36 26"/>',
    "warn": '<path d="M20 5 L37 34 L3 34 Z"/><path d="M20 15 V25"/><circle cx="20" cy="30" r="1.2"/>',
    "target": '<circle cx="20" cy="20" r="12"/><circle cx="20" cy="20" r="1.5"/>'
              '<path d="M20 2 V10 M20 30 V38 M2 20 H10 M30 20 H38"/>',
}


def icon(name):
    return (f'<svg width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="{INK}" stroke-width="2.4" '
            f'stroke-linejoin="round" stroke-linecap="round">{ICONS[name]}</svg>')


def wave_svg(w, box):
    """4채널 파형 SVG. box=(시작초, 끝초)면 빨간 의심 구간을 그림"""
    W, x0, x1, top, gap = 1000, 85, 975, 105, 62
    xs = x0 + np.arange(w.shape[1]) / (w.shape[1] - 1) * (x1 - x0)
    base = top + gap * (len(w) - 1) + 50
    out = [f'<svg viewBox="0 0 {W} {base + 52}" width="100%" xmlns="http://www.w3.org/2000/svg" '
           f'style="font-family:Gaegu,sans-serif">']
    if box:
        a, b = (x0 + v / 4 * (x1 - x0) for v in box)
        out.append(f'<rect x="{a:.0f}" y="{top - 38}" width="{b - a:.0f}" height="{base - top + 26}" rx="10" '
                   f'fill="{PINK}" stroke="{RED}" stroke-width="2.5"/>')
        right = b + 250 < W  # 오른쪽에 자리가 없으면 왼쪽에 씀
        ex, tx = (b - 6, b + 42) if right else (a + 6, a - 42)
        sgn = 1 if right else -1
        out.append(f'<path d="M{tx - sgn * 4:.0f} {top - 70} Q{ex + sgn * 6:.0f} {top - 70} {ex:.0f} {top - 44}" '
                   f'fill="none" stroke="{INK}" stroke-width="2"/>'
                   f'<path d="M{ex - sgn * 3:.0f} {top - 56} L{ex:.0f} {top - 44} L{ex + sgn * 10:.0f} {top - 52}" '
                   f'fill="none" stroke="{INK}" stroke-width="2"/>'
                   f'<text x="{tx:.0f}" y="{top - 64}" font-size="32" fill="{INK}" '
                   f'text-anchor="{"start" if right else "end"}">IED 의심 구간</text>')
    for k, (ch, row) in enumerate(zip(SHOW, w)):
        y = top + k * gap - np.clip(row, -4, 4) * 7
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, y))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2.4" '
                   f'stroke-linejoin="round"/>'
                   f'<text x="{x0 - 18}" y="{top + k * gap + 10}" font-size="32" fill="{INK}" '
                   f'text-anchor="end">{ch}</text>')
    out.append(f'<path d="M{x0} {top - 45} V{base} H{x1}" fill="none" stroke="{INK}" stroke-width="2"/>')
    for sec in range(5):
        x = x0 + sec / 4 * (x1 - x0)
        out.append(f'<path d="M{x:.0f} {base} V{base - 9}" stroke="{INK}" stroke-width="2"/>'
                   f'<text x="{x:.0f}" y="{base + 38}" font-size="30" fill="{INK}" text-anchor="middle">{sec} s</text>')
    out.append("</svg>")
    return "".join(out)


DATA = Path(__file__).parent / "data"


@st.cache_data
def load(name):
    return pd.read_csv(DATA / f"{name}.csv")


sweep = load("test_sweep")
patients = load("patient_sweep")
types = load("type_sweep")
models = load("models")
reduction = load("reduction")
importance = load("importance")
segments = load("segments")

# 기준값: 확률이 이 값 이상이면 IED라고 판정 (교차검증에서 F1이 가장 높았던 0.6이 기본)
with st.sidebar:
    st.header("설정")
    thr_pct = st.slider("기준값", 5, 95, 60, step=5, format="0.%02d",
                        help="모델이 낸 확률이 이 값 이상이면 IED로 판정함")
    thr = thr_pct / 100
    st.write(f"지금 기준값: **{thr:.2f}**")
    st.caption("기본값 0.6은 개발 환자 71명 교차검증에서 F1이 가장 높았던 값. 테스트 환자는 보지 않고 정함")

# ---------------- 시안 화면: 뇌파 IED 탐지 시스템 ----------------
with st.container(key="sian"):
    c_title, c_l1, c_p, c_l2, c_s = st.columns([4.0, 1.0, 2.0, 1.0, 3.4], vertical_alignment="center")
    c_title.markdown('<span class="sian-title">뇌파 IED 탐지 시스템</span>', unsafe_allow_html=True)
    c_l1.markdown('<div class="sian-lab">환자 선택</div>', unsafe_allow_html=True)
    pids = sorted(segments.patient_id.unique())
    has_ied = set(segments.patient_id[segments.label == 1])
    # 처음 열면 IED가 있는 첫 환자의 첫 IED 구간을 보여 줌
    pid = c_p.selectbox("환자 선택", pids, index=next(k for k, x in enumerate(pids) if x in has_ied),
                        label_visibility="collapsed")
    seg = segments[segments.patient_id == pid].reset_index(drop=True)
    c_l2.markdown('<div class="sian-lab">구간 선택</div>', unsafe_allow_html=True)
    i = c_s.selectbox("구간 선택", seg.index, index=int(seg.label.idxmax()), label_visibility="collapsed",
                      format_func=lambda i: f"구간 {i + 1:03d} ({seg.start_s[i]}초~)"
                                            + (" 정답 IED" if seg.label[i] == 1 else ""))
    s = seg.loc[i]
    pred = s.prob >= thr

    # 파형 복원: int8 base64 -> (19채널, 500샘플, 125Hz), 저장할 때 8을 곱해 놨음
    wave = np.frombuffer(base64.b64decode(s.wave), dtype=np.int8).reshape(19, -1) / 8
    w4 = wave[[CHANNELS.index(c) for c in SHOW]]
    box = None
    if pred:  # 모델은 4초 단위로만 판정하므로, 그 안에서 4채널이 가장 크게 튄 0.5초를 칠함
        win = 62
        a = max(range(0, w4.shape[1] - win + 1, 8), key=lambda a: np.ptp(w4[:, a:a + win], axis=1).sum())
        box = (a / 125, (a + win) / 125)
    note = ("빨간 칸은 모델이 IED라고 판정한 4초 안에서 가장 크게 튄 0.5초. 모델은 4초 단위로만 판정함"
            if pred else "모델이 IED가 아니라고 판정한 4초라 의심 구간이 없음")
    st.markdown(f'<div class="sian-card"><span class="sian-pill">뇌파 파형</span>{wave_svg(w4, box)}'
                f'<div class="sian-note">{note}</div></div>', unsafe_allow_html=True)

    if s.label == 1:
        truth = f"정답 IED 종류 {s.ied_type}, " + ("맞힘" if pred else "미탐")
    else:
        truth = "정답 정상, " + ("오탐" if pred else "맞힘")
    color, bg, word = (RED, PINK, "IED 있음") if pred else ("#2e7d32", "#e3f2e3", "IED 없음")
    left, right = st.columns([1, 1.45])
    left.markdown(f'<div class="sian-card"><span class="sian-pill">IED 판별 결과</span>'
                  f'<div class="sian-verdict" style="background:{bg};color:{color}">{word}</div>'
                  f'<div class="sian-note">모델 확률 {s.prob:.2f} · 기준값 {thr:.2f} · {truth}</div></div>',
                  unsafe_allow_html=True)

    p = patients[(patients.thr_pct == thr_pct) & (patients.patient_id == pid)].iloc[0]
    rec = "IED 없음" if pd.isna(p.recall) else f"{p.recall * 100:.1f} %"
    stats = [("분석 구간", "layers", f"{p.n_seg:,}개"), ("IED 의심 구간", "warn", f"{p.found + p.fp:,}개"),
             ("탐지 비율<br>(재현율)", "target", rec)]
    right.markdown('<div class="sian-card"><span class="sian-pill">분석 요약</span><div class="sian-stats">'
                   + "".join(f'<div class="sian-stat"><div class="k">{k}</div><div class="v">{icon(ic)}{v}</div></div>'
                             for k, ic, v in stats)
                   + f'</div><div class="sian-note">{pid} 녹화 전체, 기준값 {thr:.2f}</div></div>',
                   unsafe_allow_html=True)

st.caption("빅데이터반 × 희귀난치병 동아리 · 데이터 vEpiSet (환자 84명) · 최종 모델 랜덤포레스트 · "
           "구간은 테스트 환자마다 IED 최대 8개, 정상 7개를 골라 둔 것이고 파형은 125Hz로 줄여서 그림")

st.subheader("자세한 결과")
tab1, tab3, tab4, tab5, tab6 = st.tabs(["결과 요약", "환자별 · 종류별", "모델 비교", "축소 실험", "특징 중요도"])

# ---------------- 1. 결과 요약 ----------------
with tab1:
    st.subheader("테스트 환자 13명 결과")
    st.write("테스트 환자는 학습에 한 번도 안 쓴 환자임. 모델을 다 정한 뒤 마지막에 한 번만 평가함")
    r = sweep[sweep.thr_pct == thr_pct].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("재현율", f"{r.recall:.0%}", help="실제 IED 중에 찾은 비율")
    c2.metric("정밀도", f"{r.precision:.0%}", help="IED라고 한 것 중에 맞은 비율")
    c3.metric("F1", f"{r.f1:.2f}")
    c4.metric("특이도", f"{r.specificity:.0%}", help="IED 아닌 조각을 아니라고 맞힌 비율")
    c5.metric("녹화 1개당 오탐", f"{r.fp_per_rec:.1f}개")

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown("**판정 결과 세기**")
        table = pd.DataFrame(
            {"실제 IED": [int(r.tp), int(r.fn)], "실제 IED 아님": [int(r.fp), int(r.tn)]},
            index=["모델이 IED라고 함", "모델이 아니라고 함"])
        st.table(table)
        st.caption(f"오탐(IED 아닌데 IED라고 함) {int(r.fp)}개, 미탐(IED인데 놓침) {int(r.fn)}개")
    with right:
        st.markdown("**기준값에 따라 바뀌는 성능**")
        chart = sweep.set_index("thr")[["recall", "precision", "f1"]]
        chart.columns = ["재현율", "정밀도", "F1"]
        st.line_chart(chart, x_label="기준값", y_label="점수")
        st.caption("기준값을 낮추면 IED를 더 많이 찾지만(재현율↑) 잘못 짚는 것도 늘어남(정밀도↓)")

# ---------------- 3. 환자별 · 종류별 ----------------
with tab3:
    st.subheader("테스트 환자별 결과")
    pt = patients[patients.thr_pct == thr_pct][["patient_id", "n_seg", "n_ied", "found", "recall", "fp"]]
    pt.columns = ["환자", "조각 수", "IED 수", "찾은 IED", "재현율", "오탐"]
    st.dataframe(pt, hide_index=True, column_config={"재현율": st.column_config.NumberColumn(format="percent")})
    st.caption("IED가 없는 환자는 재현율이 비어 있음")

    st.subheader("IED 종류별 재현율")
    ty = types[types.thr_pct == thr_pct].copy()
    ty["set"] = ty["set"].map({"cv": "개발 (교차검증)", "test": "테스트"})
    st.bar_chart(ty, x="ied_type", y="recall", color="set", stack=False,
                 x_label="IED 종류", y_label="재현율")
    tt = ty[["set", "ied_type", "n_ied", "found", "recall"]]
    tt.columns = ["구분", "IED 종류", "IED 수", "찾은 IED", "재현율"]
    st.dataframe(tt, hide_index=True, column_config={"재현율": st.column_config.NumberColumn(format="percent")})
    st.caption("종류 1은 잘 찾는데 종류 4는 거의 못 찾음. 종류 번호가 어느 위치의 IED인지는 아직 확인 못 함")

# ---------------- 4. 모델 비교 ----------------
with tab4:
    st.subheader("로지스틱 회귀 · 랜덤포레스트 · SVM")
    st.write("세 모델을 같은 특징 36개로 학습함. 모델은 개발 환자 교차검증 PR-AUC로만 골랐음")
    m = models.copy()
    m["eval"] = m["eval"].map({"cv": "교차검증 (개발 71명)", "test": "테스트 (13명)"})
    st.bar_chart(m, x="model", y="pr_auc", color="eval", stack=False, x_label="모델", y_label="PR-AUC")
    m.columns = ["평가", "모델", "PR-AUC", "ROC-AUC", "재현율", "정밀도", "F1", "특이도"]
    st.dataframe(m, hide_index=True)
    st.caption("이 표의 재현율, 정밀도, F1, 특이도는 기준값 0.5 기준. 아무렇게나 찍으면 PR-AUC는 IED 비율인 0.10 정도가 나옴")

# ---------------- 5. 축소 실험 ----------------
with tab5:
    st.subheader("샘플링 주파수나 채널 수를 줄였을 때 성능")
    st.write("뇌파를 더 성기게 찍거나(샘플링 주파수↓) 전극을 덜 붙여도(채널 수↓) 비슷하게 찾는지 보려고 랜덤포레스트를 다시 학습함 (개발 환자 교차검증)")
    fs = reduction[reduction.exp == "fs"][["cond", "pr_auc", "f1_max"]]
    ch = reduction[reduction.exp == "ch"][["cond", "pr_auc", "f1_max"]]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**샘플링 주파수**")
        st.bar_chart(fs, x="cond", y="pr_auc", x_label="샘플링 주파수", y_label="PR-AUC", sort=False)
    with c2:
        st.markdown("**채널 수와 위치**")
        st.bar_chart(ch, x="cond", y="pr_auc", x_label="채널", y_label="PR-AUC", horizontal=True, sort=False)
    st.caption("귀 주변 4채널(T3 T4 T5 T6)만 쓴 게 19채널 전체보다 PR-AUC가 높았음 (0.53 대 0.49)")

# ---------------- 6. 특징 중요도 ----------------
with tab6:
    st.subheader("랜덤포레스트가 많이 본 특징 (상위 15개)")
    st.bar_chart(importance, x="feature", y="importance", horizontal=True, sort="-importance",
                 x_label="중요도", y_label="특징")
    imp = importance[["rank", "feature", "meaning", "importance"]]
    imp.columns = ["순위", "특징", "뜻", "중요도"]
    st.dataframe(imp, hide_index=True)

st.divider()
st.caption("데이터 출처: Lin, N. et al. (2025). An EEG dataset for interictal epileptiform discharge with spatial "
           "distribution information. Scientific Data. https://doi.org/10.1038/s41597-025-04572-1")
