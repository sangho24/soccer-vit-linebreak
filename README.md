# soccer-vit-linebreak

Metrica sample-data 기반으로 기하학 규칙 라벨(라인 브레이킹 패스)을 생성하고, Raster 입력으로 baseline/CNN/ViT 분류 및 설명(Attention Rollout/Distance) 실험을 재현하는 프로젝트입니다.

## Quickstart

```bash
cd /Users/sangho/Documents/New\ project/soccer-vit-linebreak
python -m venv .venv && source .venv/bin/activate
pip install -e .
# torch/timm까지 필요하면
pip install -e .[vision]
```

## Data

Metrica sample-data를 `data/external/sample-data`에 준비합니다.

```bash
python -m soccer_vit.metrica.download --out data/external/sample-data
```

네트워크/권한이 없는 환경에서는 clone 명령을 출력하고 경로 검증만 수행합니다.

## CLI

```bash
python -m soccer_vit.train build-dataset --config configs/default.yaml
python -m soccer_vit.train fit --model baseline --config configs/default.yaml
python -m soccer_vit.train fit --model resnet18 --config configs/default.yaml
python -m soccer_vit.train fit --model vit_base --config configs/default.yaml
python -m soccer_vit.eval run --config configs/default.yaml
python -m soccer_vit.report make --config configs/default.yaml --n-samples 30
```

## Notes

- Metrica 좌표(0..1)를 meter-centered 좌표로 변환 후 공격 방향을 +x로 정규화합니다.
- 라벨은 패스 선분 corridor를 관통하는 수비수 수(`K`) 기반의 geometry line-break 규칙을 사용합니다.
- Vision dependencies가 없으면 baseline만 동작하도록 설계했습니다.
