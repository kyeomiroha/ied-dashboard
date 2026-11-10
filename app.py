# 뇌파 4초 조각 IED 자동 탐지 결과 대시보드
# 실행: streamlit run app.py
import base64
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IED 자동 탐지", page_icon="🧠", layout="wide")

CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]


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

st.title("뇌파 4초 조각 IED 자동 탐지")
st.caption("빅데이터반 × 희귀난치병 동아리 · 데이터: vEpiSet (환자 84명) · 최종 모델: 랜덤포레스트")

# 기준값: 확률이 이 값 이상이면 IED라고 판정 (교차검증에서 F1이 가장 높았던 0.6이 기본)
with st.sidebar:
    st.header("설정")
    thr_pct = st.slider("기준값", 5, 95, 60, step=5, format="0.%02d",
                        help="모델이 낸 확률이 이 값 이상이면 IED로 판정함")
    thr = thr_pct / 100
    st.write(f"지금 기준값: **{thr:.2f}**")
    st.caption("기본값 0.6은 개발 환자 71명 교차검증에서 F1이 가장 높았던 값. 테스트 환자는 보지 않고 정함")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["결과 요약", "4초 구간 보기", "환자별 · 종류별", "모델 비교", "축소 실험", "특징 중요도"])

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

# ---------------- 2. 4초 구간 보기 ----------------
with tab2:
    st.subheader("4초 구간 하나씩 보기")
    st.write("테스트 환자의 4초 조각 중 일부(환자마다 IED 최대 8개, 정상 7개)를 골라 둠. 파형은 125Hz로 줄여서 그림")

    col_a, col_b = st.columns(2)
    pid = col_a.selectbox("환자", sorted(segments.patient_id.unique()))
    seg = segments[segments.patient_id == pid].reset_index(drop=True)

    def seg_name(i):
        s = seg.loc[i]
        answer = f"IED (종류 {s.ied_type})" if s.label == 1 else "정상"
        return f"{s.start_s}초부터 | 정답: {answer} | 확률 {s.prob:.2f}"

    i = col_b.selectbox("구간", seg.index, format_func=seg_name)
    s = seg.loc[i]

    # 파형 복원: int8 base64 -> (19채널, 500샘플), 저장할 때 8을 곱해 놨음
    wave = np.frombuffer(base64.b64decode(s.wave), dtype=np.int8).reshape(19, -1) / 8
    t = np.arange(wave.shape[1]) / 125

    fig, ax = plt.subplots(figsize=(11, 6.5))
    gap = 6
    for k, ch in enumerate(CHANNELS):
        ax.plot(t, wave[k] - k * gap, color="black", linewidth=0.7)
    ax.set_yticks([-k * gap for k in range(19)])
    ax.set_yticklabels(CHANNELS)
    ax.set_xlim(0, 4)
    ax.set_xlabel("time (s)")
    ax.grid(axis="x", alpha=0.3)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    st.pyplot(fig)

    pred = s.prob >= thr
    if s.label == 1 and pred:
        result = "맞힘 (IED를 찾음)"
    elif s.label == 1:
        result = "미탐 (IED인데 놓침)"
    elif pred:
        result = "오탐 (IED 아닌데 IED라고 함)"
    else:
        result = "맞힘 (정상이라고 함)"

    c1, c2, c3 = st.columns(3)
    c1.metric("정답", f"IED 종류 {s.ied_type}" if s.label == 1 else "정상")
    c2.metric("모델 확률", f"{s.prob:.2f}")
    c3.metric(f"판정 (기준값 {thr:.2f})", result)

    p = patients[(patients.thr_pct == thr_pct) & (patients.patient_id == pid)].iloc[0]
    st.markdown(f"**{pid} 녹화 전체 (지금 기준값)**: 조각 {p.n_seg}개 중 IED {p.n_ied}개, "
                f"찾은 IED {p.found}개, 오탐 {p.fp}개")

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
    st.subheader("장비를 줄이면 어디서 성능이 떨어지나")
    st.write("샘플링 주파수를 낮추거나 채널을 줄여서 랜덤포레스트를 다시 학습함 (개발 환자 교차검증)")
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
