# nara_bid_stat — 나라장터 복수예가·사정율 통계 검증 도구

나라장터 적격심사(하한 근접 최저가) 입찰 분석을 **검증 가능한 통계 모델** 위에 올리기 위한 도구입니다.
기존 예측 엔진을 바로 대체하기보다 다음 네 가지를 제공합니다.

| 기능 | 하는 일 |
|---|---|
| 사정율 예측 분포 | 발주기관의 최근 복수예가(15개) x 1365조합으로 다음 공고 사정율의 **확률분포**를 계산 |
| 워크포워드 검증 | `개찰일 < 대상 개찰일` 이력만 쓰는 시간순 검증, CRPS·구간적중·BEST-k를 **무기술 기준선과 같은 사례에서 비교** |
| 통일 투찰 산식 | A값·하한율·순공사원가·원단위(기본 절상)를 한 곳에서 계산 |
| 낙찰확률 최적화 | 사정율 분포와 **경쟁사 가정 사정율 분포**로 P(낙찰 \| 가정 사정율)을 계산 |

## 설치

Python 3.11 이상, 외부 패키지는 `numpy`, `pandas`, `openpyxl` 뿐입니다(scipy 불필요).

```bash
pip install -r requirements.txt
python -m unittest discover -s tests      # 47개 테스트
```

## 사용법

```bash
# 1) 복수예가 폴더 전체 감사(생성 규칙, 4개 선택 무작위성, 이론 대비 실제, 워크포워드, 튜닝 귀무검정)
python -m nara_bid_stat audit --prebid-dir 복수예가 --out audit_out

# 2) 발주기관 사정율 예측 분포와 상위 0.1 구간
python -m nara_bid_stat forecast --prebid-dir 복수예가 --org "충청북도 청주시" --asof 2026-10-01

# 3) 가정 사정율의 투찰금액
python -m nara_bid_stat bid --base 312450000 --lower-rate 87.745 --a-value 12340000 --rate -0.13

# 4) 경쟁사 분포(개찰결과에서 역산한 가정 사정율 CSV, 컬럼 rate)를 반영한 낙찰확률
python -m nara_bid_stat winprob --prebid-dir 복수예가 --org "충청북도 청주시" --competitors comp.csv --n 120

# 5) 임계값 자동튜닝 절차가 우연으로 통과하는 비율
python -m nara_bid_stat nulltest --cases 60 --prior 0.44
```

파이썬에서 직접:

```python
from nara_bid_stat import RateDistribution, AwardRule, bid_for_assumed_rate, win_probability_curve
from nara_bid_stat.data import load_prebid_folder, P_COLS

df = load_prebid_folder("복수예가")
hist = df[df.org == "충청북도 청주시"]
dist = RateDistribution.from_prebid_history(hist[P_COLS].to_numpy(), recent=60)
print(dist.summary(), dist.bucket_table(0.1)[:10])
```

## 실데이터 검증 결과 요약 (충북권 66개 기관, 40,971건, 2020-01 ~ 2026-09)

자세한 근거와 수치는 [docs/METHODOLOGY_KO.md](docs/METHODOLOGY_KO.md)에 있습니다.

1. **15개 예가는 칸별 균등난수입니다.** 지자체·교육청(±3%)은 음수 8칸 + 양수 7칸, 조달청 등(±2%, ±2.5%)은 15등분입니다.
   국방(D2B)은 거의 고정된 격자, 한전·LH는 자체 규칙입니다.
2. **4개 선택은 사실상 추첨입니다.** 1365조합 안에서 실제 예정가격의 위치(PIT)가 균등분포입니다.
3. **그래서 사정율 분포는 생성 규칙으로 거의 확정됩니다.** ±3% 8/7 규칙의 이론 P(양수)=44.8%, 실제 43.9%.
   과거 이력 기반 부호 예측(54.5%)은 '항상 음수'(54.1%)와 같습니다.
4. **최근 흐름(최근 20건·EWMA·직전값)은 오히려 나쁩니다.** 2024년 이후 시간순 검증에서 메커니즘 분포 대비 CRPS가 유의하게 높고
   80% 구간 적중률이 72~76%로 과신합니다. 메커니즘 분포는 50%/80% 구간 적중 50.9%/80.5%로 보정이 맞습니다.
5. **무기술 기준선:** 메커니즘 분포의 상위 0.1 구간만 골라도 BEST1 6.0%, BEST5 28.9%, **BEST10 54.1%** 가 나옵니다.
   어떤 엔진이든 BEST10 적중률은 이 값과 같은 사례에서 비교해야 합니다.
6. **소표본 임계값 자동튜닝은 위험합니다.** 60건·후보 10개·정확도 54% 기준 방식은 정보가 없는 모델로도 57~60% 확률로 통과합니다.
7. **실질적 우위는 경쟁사 분포에서 나옵니다.** 낙찰은 '실제 하한 이상 중 최저가'이므로, 사정율 분포가 정해진 이상
   승률은 경쟁사들이 어디에 몰리는지(빈 구간)를 얼마나 정확히 아느냐로 결정됩니다. 개찰결과 전체 투찰금액 수집이 최우선 과제입니다.

## 구성

```
nara_bid_stat/
  mechanism.py    생성 규칙 판별, 1365조합, RateDistribution(예측 분포)
  scoring.py      CRPS, PIT, 구간적중, Wilson CI, 이항검정, Diebold-Mariano, KS
  walkforward.py  엄격한 시간순 검증, 기준선 비교, 기존 엔진 BEST-k 장부 평가(McNemar)
  bid.py          통일 투찰 산식, 개찰결과 -> 가정 사정율 역산, 낙찰확률 곡선
  audit.py        데이터 감사, 튜닝 절차 귀무 시뮬레이션
  data.py         복수예가 Excel 폴더 로더(중복 제거 포함)
tests/            unittest 47개(합성 데이터, 실데이터 미포함)
docs/             방법론과 검증 근거
```

## 주의

* 이 저장소에는 입찰 원본 데이터가 들어 있지 않습니다. 테스트는 합성 데이터로만 동작합니다.
* 투찰금액은 공고문의 A값·순공사원가·하한율 적용 기준을 반드시 확인한 뒤 사용하십시오.
  특히 순공사원가 98% 기준에 사정율을 곱하는지(`net_cost_scales_with_rate`)는 공고·예규로 확인이 필요합니다.
* 사정율은 확률변수입니다. 이 도구의 출력은 '정답'이 아니라 보정된 확률이며, 낙찰을 보장하지 않습니다.
