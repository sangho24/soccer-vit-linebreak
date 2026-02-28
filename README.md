# soccer-vit-linebreak

Metrica sample-data 기반으로 line-breaking pass 라벨을 geometry 규칙으로 자동 생성하고, raster 입력에서 baseline/CNN/ViT를 비교한 뒤 **counterfactual + attention/focus 지표로 메커니즘을 해석**하는 프로젝트입니다.

핵심 목표는 "ViT 최고 성능" 주장이 아니라, `패스 레인 구조`와 `패서 맥락`을 모델이 어떻게 사용하는지 관찰 가능한 프레임을 만드는 것입니다.

## 1) 프로젝트 핵심

- 데이터 파이프라인: event + tracking 정렬, 좌표 정규화(+x 공격 방향 통일)
- 라벨링: 전진 패스 + corridor 내 bypassed defenders 기반 line-breaking 자동 라벨
- 입력 표현: 5채널(공격/수비/볼/패서/리시버), 7채널(+`pass_line`/+`pass_corridor`)
- 모델: `baseline`, `baseline_rule_like`, `baseline_strict`, `resnet18`, `vit_base`
- 해석: attention rollout, attention distance, focus metrics, counterfactual (local geometric intervention)
- 리포트: 질문형(Q1~Q4) figure 자동 생성 + pass-centric heatmap 비교

## 2) 핵심 결과 스냅샷 (발표용)

- 데이터셋: `N=1759`, `pos_rate≈0.151`
- 7채널 성능:
  - `baseline_strict`: AUROC≈0.937, F1≈0.716, BalAcc≈0.885
  - `resnet18`: AUROC≈0.946, F1≈0.744, BalAcc≈0.841
  - `vit_base`: AUROC≈0.787, F1≈0.409, BalAcc≈0.698
- ViT 5ch→7ch:
  - AUROC: `0.579 → 0.787`
  - CF on_line>off_line: `0.453 → 0.625`
- 해석 요약:
  - 현재 데이터/세팅에선 CNN 우세
  - 하지만 ViT도 7채널에서 구조 민감도(counterfactual + focus) 신호를 보임
  - `line_only=판별력`, `corridor_only=구조민감도`, `both=균형`의 상보 패턴 확인

## 3) 설치

```bash
cd /Users/sangho/Documents/New\ project/soccer-vit-linebreak
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[vision]
```

## 4) 데이터 준비

```bash
python -m soccer_vit.metrica.download --out data/external/sample-data
```

## 5) 라벨링 규칙 (요약)

`p0=passer`, `p1=receiver`, `D=defenders`일 때:

1. `forward_m = p1.x - p0.x`
2. `forward_m < min_forward_m`(기본 5m)이면 `label=0`
3. `t in (0,1)` and `perp_dist<=corridor_w_m`(기본 8m) and 선분 진행 범위 내부인 수비수 수를 `bypassed_count`로 계산
4. `label = 1 if bypassed_count >= k_bypassed(기본 2) else 0`

> 즉, "전진 패스 + 패스 corridor 내부에서 여러 수비수를 우회"하면 line-breaking으로 라벨링합니다.

## 6) 기본 실행 플로우 (로컬)

### 6-1. 데이터셋 빌드

```bash
python -m soccer_vit.train build-dataset --config configs/vit_mid_linecorridor.yaml
```

### 6-2. 학습

```bash
python -m soccer_vit.train fit --model baseline_strict --config configs/vit_mid_linecorridor.yaml
python -m soccer_vit.train fit --model resnet18 --config configs/vit_mid_linecorridor.yaml
python -m soccer_vit.train fit --model vit_base --config configs/vit_mid_linecorridor.yaml
```

### 6-3. 평가 + 단일 리포트 산출

```bash
python -m soccer_vit.eval run --config configs/vit_mid_linecorridor.yaml
python -m soccer_vit.report make --config configs/vit_mid_linecorridor.yaml --n-samples 30
```

## 7) 질문형 리포트(Q1~Q4) 생성

여러 report 디렉터리를 묶어 발표용 비교 그림을 자동 생성합니다.

```bash
python -m soccer_vit.report questions \
  --named-reports "both=reports/vit_mid_linecorridor,no_passer=reports/vit_mid_linecorridor_no_passer,line_only=reports/vit_mid_line_only,corridor_only=reports/vit_mid_corridor_only,compare=reports/compare_linecorridor" \
  --out-dir reports/question_figures_presentation \
  --passcentric-grid-size 96 \
  --passcentric-splat-sigma 0.85 \
  --passcentric-min-count 2.0
```

주요 출력:

- `q1_no_passer_vs_both.png`
- `q2_counterfactual_label_flip.png`
- `q3_structure_channel_role_map.png`
- `q4_cnn_vs_vit_focus_compare.png`
- `passcentric_*.png`, `q1_passcentric_*.png`, `q4_passcentric_*.png`
- `questions_summary.json`

## 8) 발표용 자산 ZIP으로 바로 저장

런타임 비용 절감을 위해 결과를 즉시 ZIP으로 묶어 다운로드할 수 있습니다.

```bash
OUT=reports/question_figures_presentation
ZIP=question_figures_presentation_$(date +%s).zip
python -m soccer_vit.report questions \
  --named-reports "both=reports/vit_mid_linecorridor,no_passer=reports/vit_mid_linecorridor_no_passer,line_only=reports/vit_mid_line_only,corridor_only=reports/vit_mid_corridor_only,compare=reports/compare_linecorridor" \
  --out-dir "$OUT" \
  --passcentric-grid-size 96 \
  --passcentric-splat-sigma 0.85 \
  --passcentric-min-count 2.0
cd reports && zip -r "../$ZIP" "$(basename "$OUT")"
```

## 9) Compare / Seed Sweep

```bash
python -m soccer_vit.experiments compare-models --config configs/compare_linecorridor.yaml
python -m soccer_vit.experiments seed-sweep --config configs/seed_sweep_linecorridor.yaml --model vit_base --seeds 41,42,43
```

## 10) Explainability export 샘플 수 제어

시각화용 샘플 수와 NPZ export 샘플 수를 분리할 수 있습니다.

- `eval.n_rollout_samples`: 화면/figure에 그릴 샘플 수
- `eval.explainability_export_n` (alias: `eval.explainability_export_n_samples`): `*.npz` 저장 샘플 수

```yaml
eval:
  n_rollout_samples: 8
  explainability_export_n: 128
```

## 11) 체크포인트/산출물 트러블슈팅 (중요)

`eval` 실행 후 `metrics.json`에 `models: {}` / `counterfactual: {}`만 나오고 아래 파일이 없으면, 거의 항상 체크포인트 누락입니다.

- 누락 파일 예시:
  - `reports/.../models/vit_base.pt`
  - `reports/.../models/resnet18.pt`
- 누락 시 생성 실패 파일:
  - `vit_rollout_samples.npz`
  - `resnet18_focus_samples.npz`
  - `counterfactual_vit_base.csv`
  - `counterfactual_resnet18.csv`

검증:

```bash
ls -l reports/vit_mid_linecorridor/models
ls -l reports/compare_linecorridor/models
```

해결:

1. 해당 config로 `train fit`을 먼저 수행하거나,
2. 이미 있는 `*_colab` report의 `models/*.pt`를 현재 report 경로로 복사/링크한 뒤 `eval run` 재실행

## 12) Data Validity 메모

- split: stratified `train/val/test=70/10/20`, `splits.npz`를 고정해 모델 간 동일 분할 사용
- 누수 점검: `sample_id` 중복 0건, split 간 overlap 0건
- 라벨 점검: 규칙(`forward_m>=5` and `bypassed_count>=2`)과 저장 라벨 불일치 0건
- 재현성: seed 3개 검증에서 핵심 해석 패턴(구조 민감도 증가 + passer 편향) 동일 방향 재현

## 13) 핵심 지표 해석 가이드

- `AUROC`: 임계값 독립 판별력
- `F1`: precision/recall 균형(양성 클래스 중심)
- `BalAcc`: 클래스 불균형에서 class-wise recall 평균
- `CF on_line>off_line rate`: `P(on) > P(off)` 비율
- `CF ΔP`: `P(on) - P(off)` 평균
- `Receiver-Passer`: `receiver_mean - passer_mean` (음수면 passer 쪽 focus)
- `Corridor/Passer`, `NearestDef/Passer`: 1보다 크면 구조 영역 focus가 passer focus를 넘음

## 14) 해석 시 주의사항 / 한계

- counterfactual은 팀 전술 전체를 재구성한 실험이 아니라 **local geometric intervention**입니다.
- ResNet focus는 Grad-CAM이 아니라 `input_grad * input` patch-mean proxy입니다.
- 데이터셋 규모가 작으므로 절대 성능 우열보다 메커니즘 해석의 일관성에 초점을 둡니다.
