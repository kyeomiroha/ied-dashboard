# 뇌파 4초 조각 IED 자동 탐지 대시보드

빅데이터반 × 희귀난치병 동아리 프로젝트 결과를 Streamlit으로 만든 대시보드예요.

뇌파(EEG)를 4초씩 잘라서 조각마다 IED(발작 사이에 나오는 뾰족한 파형)가 있는지 모델이 자동으로 판정해요.
조각마다 특징 36개를 뽑고 로지스틱 회귀, 랜덤포레스트, SVM을 비교했고, 랜덤포레스트를 최종 모델로 골랐어요.

## 탭
- 결과 요약: 테스트 환자 13명에서 재현율, 정밀도, F1, 특이도 (기준값은 사이드바에서 바꿀 수 있음)
- 4초 구간 보기: 테스트 환자의 4초 조각 파형과 모델 확률
- 환자별 · 종류별: 환자마다, IED 종류마다 얼마나 찾았는지
- 모델 비교, 축소 실험, 특징 중요도

## 실행
```
pip install -r requirements.txt
streamlit run app.py
```

## 데이터
`data/` 폴더의 CSV는 모델 결과를 정리한 거예요. `segments.csv`에는 테스트 환자 4초 조각 156개의 파형이 들어 있어요(125Hz, 19채널).

데이터 출처: Lin, N. et al. (2025). An EEG dataset for interictal epileptiform discharge with spatial distribution information. *Scientific Data*. https://doi.org/10.1038/s41597-025-04572-1
