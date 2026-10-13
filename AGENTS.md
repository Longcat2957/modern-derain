**# AGENTS.md — Modern Derain**

별도 `modern-derain` 리포의 `AGENTS.md`로 사용하는 구현 지침이다.

**## 목표와 범위**

- EfficientDeRain의 ****v3 Rain1400 KPN****을 현재 안정판 PyTorch의 표준 연산/API로 재구축한다. 공식 코드·학습 설정은 참조 기준과 테스트 oracle로 사용하고, production 모델·데이터·학습·평가·배포 코드는 독립적으로 구현한다. 공식 체크포인트의 구조·가중치·연산 의미를 보존하며, 파일 전체 Copy-and-Paste를 재이식으로 취급하지 않는다. EfficientDeRain+나 v4 RainMix로 모델을 바꾸지 않는다.

- 리포·배포 패키지·CLI는 `modern-derain`, Python import는 `modern_derain`으로 통일한다. 단일 config, 폴더 기반 데이터 입력, `weights/`·`tests/reference/`·`artifacts/`의 역할 분리를 기본 사용 방식으로 둔다.

- 프로젝트 자체 코드는 ****AGPL-3.0-only 공개****를 기준으로 설계한다. 초기화 시 `LICENSE`와 `pyproject.toml`의 license metadata에 반영한다. 외부 원저작물의 MIT 등 원래 라이선스·저작권 고지는 별도로 보존하며, 외부 체크포인트·데이터의 출처·라이선스도 프로젝트 코드와 구분해 기록한다.

- 최종 목표는 Jetson AGX Orin의 TensorRT INT8 혼합 정밀도 추론이다. 커널 예측 CNN의 W8A8과 동적 필터의 부동소수점 실행을 구분한다. 목표 해상도·FPS·P95 지연·전력 모드는 아직 미정이며 설정으로 둔다. 실기기 측정 전에는 실시간 성능을 확정하지 않는다.

- 간단한 구성과 CLI를 우선한다. 다른 복원 리포와 공통 패키지나 상호 의존성을 만들지 않는다. 아래 스캐폴드는 단계별 목표 구조이며, 필요한 모듈만 만들고 빈 구현·추상 base class·registry·DI framework를 미리 만들지 않는다.

**## 도구와 환경**

- ****Python 패키지·가상환경·실행은 반드시 [****uv****](**https://docs.astral.sh/uv/guides/projects/**)로 관리한다.**** 의존성 변경은 `uv add`/`uv remove`, 환경 재현은 `uv sync --locked`, 실행·테스트는 `uv run --locked`를 사용한다. 필요한 extra는 명시해 실행한다. `pyproject.toml`, `uv.lock`, `.python-version`을 커밋하고 lockfile은 직접 편집하지 않는다.

- PyTorch CUDA wheel과 NVIDIA 공식 Python 패키지의 인덱스·소스·플랫폼 조건은 `pyproject.toml`에 명시한다. 개별 `pip install`로 우회하지 않는다. Orin의 JetPack CUDA·TensorRT 네이티브 SDK는 시스템 설치를 사용한다. Orin Python runner가 필요한 경우에만 uv 환경에서 호환 바인딩과 CUDA 동작을 확인한다.

- 구현 시점의 ****최신 안정판 PyTorch와 NVIDIA ModelOpt****를 우선한다. nightly는 사용하지 않는다. 공식 release·패키지 metadata·wheel을 확인하고 CUDA·ModelOpt·ONNX exporter 호환성을 작은 실행 검증으로 확인한 뒤 버전을 고정한다. 다른 리포의 후보 버전을 실행 검증된 pin으로 복사하지 않는다.

- 기본 배포 경로는 ****x86 CUDA 서버/워크스테이션에서 학습·calibration·QAT → ONNX export → ONNX를 Orin으로 전달 → 대상 Orin에서 TensorRT 엔진 build → 엔진 infer/bench****다. x86의 `export`와 Orin의 `build/infer/bench`를 분리한다. Jetson PyTorch는 추가 비교가 필요한 경우에만 NVIDIA 공식 호환 패키지·컨테이너를 검토한다.

- 전달 산출물은 ONNX와 외부 가중치 파일, 입력 shape/dtype·RGB/range·resize/padding·출력 복원 설정, hash·버전 manifest다. INT8 calibration/QAT scale은 Q/DQ ONNX에 포함해 전달한다. opset·연산·정밀도는 대상 TensorRT 지원 범위에 맞추며, x86에서 빌드한 engine의 Orin 이식성을 가정하지 않는다. [TensorRT 엔진 호환성](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)

- 모델 재이식·학습·calibration·QAT·ONNX export와 공식 품질 평가는 x86에서 실행한다. 전체 학습 패키지의 Orin 설치·실행 호환성은 필수 범위가 아니다. Orin의 초기 엔진 build·레이어 분석은 시스템 `trtexec`를 우선 사용한다. 실제 지연·FPS·전력·병목 측정은 학습/QAT 이후의 별도 단계로 둔다. [TensorRT 명령행 도구](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/reference/command-line-programs.html)

- 실제 이미지 출력 비교·end-to-end 측정·제품 연동에 필요한 경우에만 최소 TensorRT runner를 추가한다. 기존 제품 runtime이 있으면 재사용한다. Python을 선택한 경우에만 `trt` extra와 aarch64 바인딩·platform marker·source를 검증한다. runner는 PyTorch·ModelOpt·Albumentations·TorchMetrics를 요구하지 않아야 한다.

- `uv.lock`은 Python 의존성을 관리한다. JetPack/L4T·CUDA·cuDNN·TensorRT·Python ABI·GPU SKU·드라이버는 별도 환경 manifest에 기록한다. `.python-version`의 학습 호스트 기본값을 Orin에 강제하지 않으며, 시스템 바인딩을 노출하는 것만으로 lock 재현이 완료됐다고 판단하지 않는다.

- 배포 시 실제 ****AGX Orin SKU와 설치 이미지****에 맞는 [공식 JetPack 구성](https://developer.nvidia.com/embedded/jetpack/downloads)·[TensorRT 지원 범위](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html)를 확인하고 SDK 조합을 기록한다. 최신 x86 TensorRT를 Orin 배포 버전으로 자동 선택하거나 특정 JetPack의 Orin 지원을 버전 이름만으로 확정하지 않는다.

- GPU INT8을 먼저 검증하고 2:4 sparsity·DLA는 효과가 확인될 때 추가한다. 장비의 sparse INT8 TOPS를 모델의 dense INT8 처리량이나 application FPS로 바꾸어 보고하지 않는다.

**## 계층화한 최소 프로젝트 스캐폴드**

`uv init --package --name modern-derain`를 출발점으로 삼으며 기존 파일은 보존한다. 아래 파일은 해당 단계의 구현에 필요할 때 만든다.

```text
.

├── AGENTS.md

├── README.md

├── LICENSE # 프로젝트 자체 코드: AGPL-3.0-only

├── THIRD_PARTY_NOTICES.md

├── licenses/ # 포함·수정한 upstream의 원래 라이선스

├── pyproject.toml

├── uv.lock

├── .python-version

├── .gitignore

├── .gitattributes # 원본 oracle의 줄바꿈·공백 보존

├── configs/default.json

├── configs/train.json # 경로와 바꿀 옵션만 담는 짧은 config

├── weights/

│ ├── README.md # 제공 checkpoint·Release URL·SHA-256·원래 고지

│ └── efficientderain-rain1400-v3.pt # 검증된 FP32 기본 가중치; 큰 파일은 Release에서 확보

├── src/modern_derain/

│ ├── __init__.py

│ ├── cli.py # Typer app·얇은 명령·main()

│ ├── config.py # stdlib JSON 설정·검증·경로 처리

│ ├── artifacts.py # stdlib 실행 metadata·hash·결과 JSON

│ ├── models/

│ │ ├── __init__.py

│ │ ├── kpn.py # 구조·가중치 호환 KPN 재구축·checkpoint helper

│ │ └── kernel.py # 네 dilation의 동적 KernelConv 재구축

│ ├── data/

│ │ ├── __init__.py

│ │ ├── images.py # NumPy/OpenCV RGB 입출력·resize·복원

│ │ ├── augment.py # Albumentations paired Compose·seed

│ │ ├── sources.py # 폴더 탐색·일반/Rain1400 매칭·자동 snapshot

│ │ └── paired.py # Dataset·transform 호출·tensor 변환

│ ├── training/

│ │ ├── __init__.py

│ │ ├── loop.py # FP32 fine-tune/QAT 공통 루프·학습 상태

│ │ ├── losses.py # 원본 L1−0.2×SSIM 수식의 현대 이식

│ │ └── quantize.py # ModelOpt calibration·정책·저장/복원

│ ├── deployment/

│ │ ├── __init__.py

│ │ ├── export.py # PyTorch → FP/QDQ ONNX

│ │ └── tensorrt.py # 필요 시 최소 runner; 초기 build는 trtexec

│ └── evaluation/

│ ├── __init__.py

│ ├── metrics.py # 공식 PSNR/SSIM·선택적 보조 지표

│ ├── quality.py # 같은 데이터·전처리의 backend 품질 평가

│ └── benchmark.py # engine/end-to-end 측정

└── tests/

└── reference/EfficientDeRain/<UPSTREAM_COMMIT>/ # Git에 포함하는 테스트 전용 원본
```

- 각 폴더는 책임과 import 경계다. 모델·데이터 계층은 CLI·학습·배포 workflow를 import하지 않는다. 평가의 backend 연결은 `quality.py`에 두고 모델을 수정하지 않는다. 공통 코드를 거대한 `utils/`에 모으지 않는다.

- `cli.py`는 Typer app·설정·명령 dispatch만 담당하고 선택한 명령의 구현을 지연 import한다. 모든 `__init__.py`는 가볍게 유지하며 모델/SDK를 재노출하지 않는다. 패키지 import, 루트와 각 하위 명령의 `--help`가 CUDA·TensorRT·선택 extra를 요구하지 않아야 한다.

- 설정·metadata·`images.py`·TensorRT 실행 경로에는 torch import를 넣지 않는다. PyTorch tensor 변환은 `paired.py` 또는 PyTorch workflow에서 수행한다. `export.py`의 torch 의존성이 TensorRT runtime으로 전파되지 않게 한다.

- Albumentations import와 학습용 Compose 구성은 `augment.py`에 둔다. 이미지 입출력·CLI help·TensorRT runtime은 증강 패키지를 요구하지 않는다. `loop.py`는 준비된 모델을 학습하며 ModelOpt를 직접 import하지 않는다. calibration·양자화 변환은 `quantize.py`에서 준비한다.

- CLI entry point는 `modern-derain = "modern_derain.cli:main"`으로 둔다. `main()`은 Typer app을 호출하며 x86의 `train/qat/export/infer/eval`을 필요한 순서대로 등록한다. Orin은 우선 네이티브 `trtexec` 명령을 README에 남기고 자체 runner가 필요할 때 `build/infer/bench`를 확장한다. 루트 callback으로 명령이 하나인 초기 단계에도 `modern-derain infer` 형태를 유지한다. 진입 명령은 `uv run --locked modern-derain --help`다.

- Typer의 type hint·`Annotated`·`Path`로 인자/옵션과 경로 검증을 선언한다. 모델·학습·평가 함수에 Typer 의존성을 전파하지 않는다. `no_args_is_help=True`, `pretty_exceptions_show_locals=False`를 사용하며 SDK import·설정 읽기·데이터 확인은 실제 명령 실행에서 수행한다. [Typer callback](https://typer.tiangolo.com/tutorial/commands/callback/)

- 설정 처리는 stdlib JSON의 작은 재귀 병합과 검증으로 구현한다. Ultralytics 스타일의 경로·split 사용 흐름을 채택하며 설정 framework, registry, YAML 의존성을 기본으로 추가하지 않는다. 아래 설정·CLI 계약을 따른다.

- 개발 도구는 uv의 `dev` dependency group으로 관리한다. `.gitignore`에는 `/.venv/`, `/data/`, `/artifacts/`, `/dist/`, cache·bytecode와 Release로 관리하는 큰 가중치의 ****정확한 경로****를 제외한다. `weights/README.md`·작은 제공 자산·`tests/reference/`는 추적한다. `data/` 패턴으로 `src/modern_derain/data/`까지 제외하지 않는다. 테스트 파일은 해당 단계의 검증에 필요한 경우에만 추가한다.

**## 설정과 CLI 계약**

- `configs/default.json`과 코드의 기본 설정을 일치시키고 `model/data/training/quantization/export/inference/evaluation/benchmark`를 구분한다. 사용자는 ********`configs/train.json`** **하나****에 필요한 경로와 바꿀 옵션만 적는다. 데이터 목록 파일을 따로 작성하지 않는다. 적용 순서는 기본값 → 사용자 JSON → 명시된 CLI 옵션이다. 설정 전체를 복제한 실험 config를 요구하지 않는다.

- `data.path`는 데이터 루트, `data.train/val/test/calibration`은 그 아래 폴더다. 기본 split 값은 각각 `train/val/test/calibration`이며 `data.path`·checkpoint·CLI 상대 경로는 CWD 기준, split 상대 경로는 `data.path` 기준이다. split 절대 경로도 허용한다. `data.layout`의 기본값은 일반 `paired`이며 검증된 파일명 규칙의 `rain1400`을 선택할 수 있다. 폴더나 파일명의 일부만 보고 layout을 추측하지 않는다. [Ultralytics 데이터 경로 설정](https://docs.ultralytics.com/datasets/detect/)

- `train/qat/eval --data <root>`는 `data.path`를 override한다. `eval --split val|test`는 기본 `val`을 바꾼다. 일반 사용자 입력은 폴더만 받는다. ****입력 manifest reader·설정 키·****`--manifest`** **옵션·호환 fallback을 구현하지 않는다.**** ONNX 전달 metadata와 환경 기록은 산출물이며 데이터 입력 파일로 사용하지 않는다.

- CLI 미지정 값은 `None`으로 두고 그 값만 병합에서 제외한다. `--no-amp` 같은 명시적 `False`와 `qat --steps 0`을 보존한다. 알 수 없는 키·잘못된 enum·타입·빈 경로·유효하지 않은 shape는 실행 초기에 오류로 처리한다. 선택한 명령이 쓰는 split만 요구하며 train 실행에 test 폴더까지 요구하지 않는다. QAT `--steps 0`은 calibration만 사용하고 supervised train 폴더를 요구하지 않는다.

- `--data`는 실존 폴더, `--config`는 실존 파일로 검증한다. checkpoint의 존재 확인은 기본 제공 weight인지 분류한 뒤 workflow에서 수행한다. 기본 weight의 최초 다운로드를 Typer의 일괄 `exists=True` 검사로 먼저 차단하지 않으며, 제공 weight 이외의 사용자 checkpoint가 없으면 오류로 처리한다.

- `--checkpoint`는 초기 가중치·fine-tune, `--resume`는 저장된 학습 상태 재개다. 두 옵션을 동시에 지정하면 오류로 처리한다. `export.shape`·목표 FPS·전력 모드 등 미정 값은 `null`이며 export/calibration 실행에는 유효한 batch 1·RGB NCHW 고정 shape가 필요하다. 학습 해상도를 배포 목표 해상도로 자동 확정하지 않는다.

`configs/train.json`의 시작 예시:

```json
{

"data": {"path": "data/Rain1400", "layout": "rain1400"},

"training": {"checkpoint": "weights/efficientderain-rain1400-v3.pt"}

}
```

```bash
uv run --locked modern-derain --help

uv run --locked --extra train modern-derain train --config configs/train.json

uv run --locked --extra train modern-derain train \

--data data/paired --checkpoint weights/efficientderain-rain1400-v3.pt

uv run --locked --extra train --extra metrics modern-derain eval \

--config configs/train.json --checkpoint weights/efficientderain-rain1400-v3.pt --split test
```

일반 `paired` 폴더의 기본 배치는 `data/paired/{train,val,test}/{rainy,clean}/`이며

같은 상대 경로·확장자로 대응한다. Rain1400의 `rainy_image/`·`ground_truth/`와

`<clean_id>_<rain_id>.jpg` → `<clean_id>.jpg` 대응은 아래 데이터 계약을 따른다.

**## 라이브러리 선정과 의존성 프로필**

아래는 라이브러리의 역할과 설치 범위다. 버전은 구현 시 공식 metadata와 실제 실행 호환성을 확인해 uv로 고정한다.

| 라이브러리 | 역할 | 설치 범위 |

|---|---|---|

| `numpy` + `opencv-python-headless` | RGB 입출력·전후처리·공식 resize | 기본 의존성 |

| `typer` + `rich` | typed CLI·help·progress·logging·결과 표 | 기본 의존성 |

| `torch` | KPN·학습·PyTorch 추론·export | PyTorch를 쓰는 extra |

| `albumentationsx[headless]` | 정렬된 rainy/clean 증강; import는 `albumentations as A` | `train`/`qat` 필수 |

| `nvidia-modelopt` | calibration·QAT·quantizer 저장/복원 | `qat` |

| `onnx` + 선택 exporter의 필수 패키지 | FP/QDQ 그래프 export·검사 | `export` |

| `scikit-image` | 공식 평가 SSIM의 현대 API | `metrics` |

| `torchmetrics` | 선택적 보조 PSNR/SSIM | 실제 보조 지표를 구현할 때만 선택 의존성 추가 |

- 선택 프로필은 `train`(torch·AlbumentationsX), `qat`(torch·ModelOpt·AlbumentationsX), `export`(torch·ONNX 및 exporter 필수 패키지), `metrics`(scikit-image)로 둔다. PyTorch `infer`에는 `train` 등 torch가 포함된 프로필을 명시한다. 예: `uv run --locked --extra train modern-derain train ...`, PyTorch 공식 평가는 `--extra train --extra metrics`다. `trt`는 Python runner를 구현할 때만 추가하며 초기에는 전체 프로젝트의 aarch64 의존성 해결을 요구하지 않는다. `pytest`·`ruff`는 `dev` group에 둔다.

- 부동소수점 ONNX export는 `--extra export`, ModelOpt 구조·quantizer 상태를 복원하는 Q/DQ export는 `--extra export --extra qat`로 실행한다. 부동소수점 export가 ModelOpt를 import하지 않게 한다. 선택 extra 누락은 필요한 프로필과 실행 명령을 안내하고 nonzero 종료한다.

- Typer와 직접 사용하는 Rich는 직접 의존성으로 선언한다. Typer의 command/testing API를 사용하고 외부 Click이나 Typer 내부 `_click`을 직접 사용하는 의존성을 추가하지 않는다. 설정·metadata 처리는 stdlib로 시작한다. TorchMetrics 때문에 PyTorch Lightning·`torchmetrics[image]`·torchvision을 자동 도입하지 않으며, 필요 없이 ModelOpt `[all]`/`[hf]`를 설치하지 않는다.

- OpenCV 배포 패키지는 `opencv-python-headless` 하나만 사용한다. 다른 `opencv-python*`/`opencv-contrib-python*`와 `cv2` namespace를 공유하므로 중복 설치하지 않는다. Orin runner에서 필요하면 aarch64 wheel의 ABI/glibc·NumPy 호환성을 확인하며 예상치 못한 source build는 허용하지 않는다. PyPI headless wheel을 CUDA OpenCV 빌드로 가정하지 않는다. [OpenCV 패키지 안내](https://pypi.org/project/opencv-python-headless/)

- 학습/QAT 증강은 AlbumentationsX를 사용한다. AGPL-3.0-only 정책과 외부 고지를 기록하며 legacy `albumentations`와 같은 namespace를 동시에 설치하지 않는다. `albumentationsx[headless]`와 직접 사용하는 OpenCV headless의 요구 버전을 uv가 하나의 배포 패키지로 해결하게 한다. [공식 설치·라이선스 안내](https://albumentations.ai/docs/1-introduction/installation/)

- paired geometry는 `A.Compose(..., additional_targets={"clean": "image"})` 한 번에 `image=rainy, clean=clean`을 전달한다. 별도 호출로 각각 난수를 적용하지 않는다. clean RGB를 mask로 매핑하지 않는다. 공통 geometry·공통 photometric·rainy 입력에만 적용하는 degradation을 구분하고, 초기에는 공식 학습 설정에서 활성화된 증강만 재현한다. 원본 확인 없이 `RandomCrop + D4`를 공식 정책으로 가정하지 않는다. calibration/평가에는 학습용 무작위 증강을 적용하지 않는다. [additional targets](https://albumentations.ai/docs/4-advanced-guides/additional-targets/)

- 학습 증강은 uint8 공통 crop에서 처리하고 마지막에 contiguous CHW float `[0,1]`로 바꾼다. RGB decode는 sample당 필요한 rainy/clean 각각 한 번만 수행한다. 다중 DataLoader worker에서는 OpenCV 내부 스레드·OpenCL을 꺼 중첩 병렬화를 피하고 worker 수를 설정으로 조절한다. 전처리의 출력·crop 좌표·난수 분포가 원본과 같은지 검증하며 CPU 처리량은 추후 측정한다. NAFNet의 합성 MotionBlur를 derain 데이터 생성으로 이관하지 않는다. [Albumentations 성능 지침](https://albumentations.ai/docs/3-basic-usage/performance-tuning/)

- `Compose(seed=...)`와 DataLoader 전용 `torch.Generator`의 seed를 설정하고 worker 수·persistent_workers·sampler·batch/drop_last·데이터 순서를 기록한다. AlbumentationsX의 worker별 seed 처리를 사용하고 매 sample/worker를 같은 seed로 재설정하지 않는다. Compose 난수는 NumPy 전역 seed만으로 제어되지 않는다. [재현성 지침](https://albumentations.ai/docs/4-advanced-guides/reproducibility/)

- BGR → RGB와 float `[0,1]` 변환을 명시하고 ImageNet mean/std 정규화를 적용하지 않는다. `ToTensorV2`의 HWC → CHW 변환은 range 정규화를 대신하지 않는다. NumPy 배열은 contiguous를 보장하며 이를 위해 torchvision을 추가하지 않는다.

- Rich는 stderr에 사람용 progress/logging을 출력하고 정상 실행 stdout에는 기계가 읽는 결과만 둔다. Typer help는 표준 help 출력으로 유지한다. `Console(stderr=True)`와 `Progress(..., redirect_stdout=False)`를 사용하며 비 TTY에서는 live progress를 비활성화한다. benchmark 측정 구간 안에서 출력·refresh·추가 GPU 동기화를 하지 않는다. 별도 TUI framework는 추가하지 않는다.

**## 데이터 계약과 실행 기록**

- ****데이터 입력은 폴더만 사용한다.**** `sources.py`에서 시작 시 한 번 탐색하고 정렬한 sample 목록을 고정한다. 준비 단계에서 이미지를 decode하지 않고 sample마다 디렉터리를 재탐색하지 않는다. 일반 `paired`는 split 아래 `rainy/`·`clean/`의 동일 상대 경로·파일명·확장자를 매칭하며 시퀀스 하위 폴더도 보존한다. 누락·중복 rainy sample·모호한 구조·읽기 실패를 오류로 처리하고 이미지 크기·RGB/range를 검증한다.

- `rain1400`은 split 아래 `rainy_image/`·`ground_truth/`를 읽고, ****검증한** **`<clean_id>_<rain_id>.jpg`** **→** **`<clean_id>.jpg`** **규칙****으로 대응한다. 여러 rainy가 하나의 clean을 공유하는 것은 정상이며 GT 파일을 rain 수만큼 복제하거나 이름을 바꾸지 않는다. sample ID에는 rainy의 상대 경로를, 원본 구분에는 clean ID를 기록한다. `source_id`는 선택한 배포본에서 검증한 원본 GT 식별자이며, clean ID가 전역적으로 유일한 경우에만 그대로 사용한다. split마다 번호를 재부여한 배포본에서 숫자 파일명 교집합만으로 같은 원본이라 판정하지 않는다. 배포 출처의 ID 대응·GT 동일성을 준비 단계에서 확인하고, 대응이 불명확한 상태를 누수 검증 통과로 보고하지 않는다. 모호한 ID·누락된 GT·중복 rain ID를 거부하고 각 split의 rainy 수·고유 clean 수를 기록한다. 임의의 정규식 설정이나 데이터셋 registry를 추가하지 않는다.

- upstream `utils.get_files`의 ****활성**** 경로는 다른 데이터셋의 `small/rain`·`small/norain`이며 Rain1400 대응 코드는 주석 처리되어 있다. 구현 전에 선택한 Rain1400 배포본의 실제 train/test 파일명·ID·확장자·쌍 수·공식 split 출처를 확인하고 그 규칙을 fixture로 고정한다. 주석이나 관행적인 쌍 수만으로 실제 test 구조를 검증했다고 보고하지 않는다. 확인되지 않은 배포본은 명확한 오류로 처리하고 추측 매칭으로 통과시키지 않는다. [고정 원본 데이터 경로 코드](https://github.com/tsingqguo/efficientderain/blob/465759da2428a4263eeee6ab464a0135fca2900c/utils.py)

- Rain1400의 선택한 공식 train/test를 보존한다. train 내부의 calibration/val은 ****clean 원본 ID 단위로 분리해 폴더로 준비****하며 같은 clean에서 파생된 rainy 이미지가 train과 held-out split에 섞이지 않게 한다. 준비한 split과 저장된 학습 snapshot의 source ID 교집합을 검사하며, 이를 위해 미사용 split까지 존재하도록 요구하지 않는다. loader가 임의 자동 split을 만들거나 최종 test를 calibration·튜닝에 사용하지 않는다. 실제 드론 데이터도 촬영 시퀀스·장소·날짜 단위로 분리한다.

- supervised 학습/QAT와 PSNR/SSIM 품질 평가는 실제 정렬된 rainy/clean 쌍을 요구한다. clean-only 합성 rain·RainMix는 초기 범위에 추가하지 않는다. calibration은 명시한 `data.calibration`의 ****실제 rainy**** 이미지만 사용하며 train/val/test fallback·학습 증강을 적용하지 않는다. rainy-only calibration 폴더 또는 paired 구조의 rainy 쪽을 읽되 clean 쪽을 입력으로 사용하지 않는다. calibration/export/Orin 입력에는 동일하게 정의한 배포 전처리와 고정 shape를 적용한다. 공식 ceil16 resize와 별개인 고정 H/W resize를 선택하면 변경된 protocol로 기록하고 비교한다.

- 드론 정답 쌍이 없다면 paired Rain1400 등 검증된 derain 데이터로 QAT하고 드론 데이터는 calibration·시각·추론 평가에 사용한다. 정답 없는 드론 장면의 PSNR/SSIM을 보고하지 않는다.

- 실제 사용한 파일 목록·rainy/clean 대응·source ID와 경로/파일 크기/수정시간 fingerprint를 checkpoint·평가 결과에 ****자동 저장****한다. calibration 기록은 실제 소비한 subset만 남긴다. metadata fingerprint는 이미지 전체를 읽는 content hash가 아니며 변경 감지 범위를 문서화한다. 사용자가 목록을 다시 입력하도록 요구하지 않는다.

**## 가중치·참조 코드·artifacts 관리**

- 공식 원본 `.pth`는 최초 재이식과 변환 검증에 사용한다. 가중치의 값·dtype·key·shape를 보존하고 `state_dict`, `config.model`, 전처리 protocol, `provenance`를 담은 ********`weights/efficientderain-rain1400-v3.pt`********로 포장한다. 불필요한 원본 optimizer·pickle model 객체를 배포 자산에 넣지 않는다. FP16 변환·압축 근사·재학습·양자화를 단순 포장 변환과 섞지 않는다. PyTorch 로드는 `weights_only=True`를 기본으로 한다.

- 원본 URL·고정 upstream commit·원본 SHA-256·변환본 SHA-256·실제 파일 크기·변환 내용을 `weights/README.md`와 `THIRD_PARTY_NOTICES.md`에 기록한다. 원본과 변환본을 각각 strict load하여 모든 tensor 값·dtype과 FP32 출력을 비교한다. checkpoint의 모델·전처리 metadata가 현재 설정과 다른 경우 조용히 진행하지 않는다. 코드의 MIT나 프로젝트 AGPL만으로 외부 weight 이용 조건을 확정하거나 다시 라이선스하지 않는다.

- 소스의 기본 구조 기준 FP32 parameter payload는 약 ****104.5 MiB****로 추산된다. 이는 실제 `.pth`의 byte 수·dtype을 확인한 결과가 아니다. 변환 후 실제 크기를 확인하고, 이 기본 자산은 ****자체 GitHub Release****에 게시하는 것을 출발점으로 삼는다. 일반 Git의 100 MiB 제한을 피하려고 FP16으로 줄이거나 바이너리를 임의 분할하지 않는다. Git에는 `weights/README.md`·확정한 버전별 Release URL·SHA-256·작은 테스트 oracle을 보존한다. [모델 구조](https://github.com/tsingqguo/efficientderain/blob/465759da2428a4263eeee6ab464a0135fca2900c/network.py), [GitHub 파일 제한](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

- 기본 가중치가 `weights/`에 없으면 ****그 자산을 쓰는 명령 실행에서만**** 확정된 자체 Release URL로 내려받는다. stdlib 다운로드·기존 SHA helper를 재사용하고 임시 파일 → 크기/체크섬 검증 → atomic rename으로 저장한다. 기존 파일은 검증하며 다른 내용을 덮어쓰지 않는다. 실패한 임시 파일은 정리하고 오류를 명시한다. package import·`--help`에서는 다운로드하지 않고, `latest` URL·런타임 upstream 다운로드·불명확한 후보 URL 순회는 사용하지 않는다. 기본 제공 자산 이외에 사용자가 지정한 checkpoint 파일이 없으면 다른 모델을 받지 않고 오류로 처리한다. 이미 확보한 `--checkpoint`로 offline 실행할 수 있다.

- 원본 개발 oracle은 `tests/reference/EfficientDeRain/<UPSTREAM_COMMIT>/`에 테스트에 필요한 파일만 고정하고 원래 라이선스·파일별 provenance를 함께 Git에 보존한다. production 모델/runtime이 이를 import하지 않는다. `.gitattributes`로 원래 byte·줄바꿈을 유지하고 Ruff 대상에서는 해당 원본만 제외한다. 학습 SSIM의 별도 원저작물·고지도 보존한다. clone한 소스로 원본 동등성 테스트를 실행할 수 있게 하며, 큰 기본 weight는 앞의 Release 확보 정책을 따른다. 설치 wheel에는 프로젝트 Python 코드만 넣고 weight·oracle은 포함하지 않는다.

- ********`artifacts/`****는 실행으로 생성한 결과 전용****이다. 학습 checkpoint·복원 이미지·평가 JSON·ONNX/외부 가중치·환경 기록을 `artifacts/<run>/`에 둔다. 제공 FP32 weight와 개발 oracle의 원본 보관 위치로 사용하지 않는다. 적용된 config, 프로젝트 revision·upstream commit, 사용 checkpoint hash, 패키지·장비·SDK 버전, 전후처리·평가 protocol, 데이터 snapshot, 결과와 ONNX/engine hash·typing 설정을 기록한다. 외부 실험 추적 서버는 초기 범위에 넣지 않는다.

- 기존 `artifacts/checkpoints/`의 공식 자산과 `artifacts/reference/`의 oracle은 새 경로로 ****검증 후 이동****하고 config·CLI 예제·테스트·고지를 함께 갱신한다. 원본과 변환본의 동등성 및 새 경로 테스트를 확인한 뒤 중복 원본·stale 환경 기록·재생성 가능한 cache만 정리한다. 사용자 데이터·학습 결과를 일괄 삭제하지 않고 새 cleanup CLI나 cache framework를 만들지 않는다.

**## checkpoint와 학습 재개**

- FP32/PTQ/QAT checkpoint를 구분한다. 초기 제공 `.pt`는 fine-tune/infer/export용이며 학습 재개 상태가 아니다. 학습 checkpoint는 model·전처리·증강 설정, 데이터 snapshot, optimizer/scheduler/scaler, epoch·step과 checkpoint 출처를 보존한다. QAT에는 `modelopt_state + state_dict` 또는 검증된 ModelOpt 저장/복원 경로를 사용해 quantizer 구조·scale도 함께 보존한다.

- Python·NumPy·torch/CUDA·Albumentations 독립 RNG·DataLoader generator 상태를 저장·복원한다. 정확한 augmentation replay는 ********`num_workers=0`****인 epoch 경계****를 출발점으로 삼고 실제 검증한 재개 범위를 기록한다. worker/prefetch 상태가 없는 임의 step에서 exact resume를 보장하지 않는다. partial/interrupted checkpoint의 inference/export 가능성과 resume 가능성을 구분한다.

- `--resume`는 데이터 fingerprint·model·전처리·증강·sampler/batch·seed·worker 수·학습 schedule 조건을 비교하고 바뀐 조건을 거부한다. snapshot·재개 상태가 없는 과거 checkpoint는 `--checkpoint`로 가중치만 사용할 수 있다. 복원한 QAT checkpoint에는 원래 calibration 출처를 유지하며 현재 폴더로 provenance를 덮어쓰지 않는다. 평가·변환 출력은 입력·제공 weight·사용자 checkpoint·GT를 덮어쓰지 않게 검증한다.

**## 모델 기준**

- 출처는 [공식 EfficientDeRain 리포](https://github.com/tsingqguo/efficientderain)이며 참조 pin은 ********`465759da2428a4263eeee6ab464a0135fca2900c`********다. 모델·데이터·loss·metric·설정 파일별 출처와 재이식 이유를 기록하고 원래 MIT 라이선스·`Copyright (c) 2020 Qing Guo` 고지를 보존한다. checkpoint의 이용 조건은 코드의 MIT 표기만으로 대신하지 않는다. 실제 weight의 URL·SHA-256·dtype·호환성은 최초 확보 시 검증하며 확인 전에는 값을 만들지 않는다. [고정 원본 LICENSE](https://github.com/tsingqguo/efficientderain/blob/465759da2428a4263eeee6ab464a0135fca2900c/LICENSE)

- 기본 출발 체크포인트는 ****v3_rain1400****이다. 공개 기준은 ****32.30 dB / SSIM 0.9272****이며 원본 부동소수점 모델 수치다. Rain100H-old v4의 ****31.12 dB / 0.9099****는 별도 데이터셋·체크포인트 결과로 취급한다. 공개 수치를 이 리포의 재현·INT8 실측 결과로 보고하지 않는다. [공식 결과표](https://github.com/tsingqguo/efficientderain/blob/465759da2428a4263eeee6ab464a0135fca2900c/results/data/DERAIN.xlsx)

- `KPN`을 유지한다. 공식 평가 경로는 `kernel_size=[3]`이며 생성자 기본값 `[5]`와 다르다. `color=True`, `burst_length=1`, `blind_est=True`, `sep_conv=False`, `channel_att=False`, `spatial_att=False`, `upMode='bilinear'`, `core_bias=False`를 명시하고 checkpoint와 대조한다.

- 외부 인터페이스는 단일 RGB·NCHW·범위 `[0,1]` 텐서이며 내부 호출은 `KPN(rainy, rainy, white_level=1.0)`이다. CNN의 `data_with_est`와 동적 필터의 `data`라는 두 입력 의미를 유지한다. 27채널 core 하나를 dilation rate 1·2·3·4가 공유하고 네 RGB 결과를 순서대로 concat한 뒤 최종 `conv_final` 12→3 Conv를 적용한다. 임의의 softmax·커널 정규화·출력 sigmoid/clip을 모델 안에 추가하지 않는다. [공식 모델 코드](https://github.com/tsingqguo/efficientderain/blob/465759da2428a4263eeee6ab464a0135fca2900c/network.py)

- 공식 RGB 4D 호출에서 `KernelConv`는 `N=3, color=1`로 처리하고 core를 `[B,3,9,1,H,W]`로 view한다. 이를 `[B,1,9,3,H,W]`로 바꾸면 checkpoint의 커널·채널 정렬이 달라진다. 각 dilation의 zero padding, 행 우선 `i/j` slice 순서, 커널 축 합산과 `white_level` 나눗셈을 보존한다.

- 공식 검증은 H/W를 각각 `ceil(H/16)*16`, `ceil(W/16)*16`으로 OpenCV bilinear resize하고 출력과 GT를 원래 크기로 복원한다. 이미 16 배수이면 크기를 유지한다. zero/reflect padding으로 대체하지 않는다. 배포 padding이나 다른 고정 해상도는 별도 protocol로 품질을 측정한다. 최소 지원 크기·16 배수 제약을 검증한다. [공식 데이터 전처리](https://github.com/tsingqguo/efficientderain/blob/465759da2428a4263eeee6ab464a0135fca2900c/dataset.py)

- 공식 checkpoint는 이름·shape가 호환되는 모델에 `strict=True`로 load한다. upstream의 누락 키를 초기 가중치로 메우는 loader를 재사용하거나 `strict=False`·임의의 키 재매핑으로 구조 차이를 숨기지 않는다.

- 원본 저장 코드는 raw state dict를 사용하고 DataParallel 학습에서도 `generator.module.state_dict()`를 저장한다. loader는 검증한 원본 raw dict와 자체 `state_dict` envelope를 지원하며 `weights_only=True`, CPU load 후 strict 검증한다. 공개 `.pth`의 실제 컨테이너는 최초 확보 시 확인한다. 소스만 보고 임의의 `module.` prefix 제거·여러 checkpoint 형식 fallback을 추가하지 않는다. [원본 저장/로드](https://github.com/tsingqguo/efficientderain/blob/465759da2428a4263eeee6ab464a0135fca2900c/trainer.py), [원본 load_dict](https://github.com/tsingqguo/efficientderain/blob/465759da2428a4263eeee6ab464a0135fca2900c/utils.py)

**## 현대 PyTorch 이식과 학습 원칙**

- production KPN·KernelConv·학습 SSIM을 ****원본을 참조해 표준 PyTorch 연산으로 재구축****한다. 공식 Python 파일 전체나 오래된 학습·데이터·`utils.py`를 runtime에 복사하지 않는다. 원본은 테스트 전용 oracle로만 보존한다. 참조 파일·원래 고지·독립 구현의 변경 이유는 `THIRD_PARTY_NOTICES.md`와 `licenses/`에 기록한다. 별도 복원 리포의 코드를 import하거나 공통 패키지로 묶지 않는다.

- 구조·모듈/parameter 이름·연산 순서·pooling·bilinear resize·KernelConv 경계를 유지한다. `.data`·`Variable` 등 오래된 API는 필요한 부분만 표준 autograd 연산으로 정리한다. forward와 입력·core·모델 parameter gradient의 동등성을 먼저 확인한다. FP32의 dtype/shape별 허용오차, max absolute/relative error와 finite 여부를 기록한 뒤 FP16 정책을 적용한다.

- `Basic.conv1`의 Sequential Conv index `0/2/4`와 `conv1..8`, `outc`, `conv_final`의 parameter 이름을 유지한다. KernelConv는 parameter 없는 `reshape/pad/slice/stack/multiply/sum`으로 표현할 수 있다. 모델 내부 `F.interpolate(..., scale_factor=2, mode="bilinear", align_corners=False, antialias=False)`의 원래 좌표 의미를 유지하며 외부 OpenCV resize와 혼동하지 않는다. channel 수 1024/512를 고정 입력 크기로 해석하지 않는다.

- 원본 `trainer.py`에서 실제 사용하는 loss는 ********`L1(pred, clean) - 0.2 * SSIM(clean, pred)`********다. `network.py`의 `LossBasic/LossAnneal`이나 다른 리포의 PSNRLoss를 활성 학습 loss로 가정하지 않는다. 학습 SSIM은 upstream의 `pytorch_ssim.SSIM()` 구현·기본값을 확인하고 표준 PyTorch 연산으로 최소 이식해 출력·gradient를 비교한다. 평가용 scikit-image SSIM과 구분한다. Adam·학습률 schedule·crop·활성 증강도 공식 설정을 확인해 기록하며 QAT의 변경값은 별도로 설정한다. [공식 학습 코드](https://github.com/tsingqguo/efficientderain/blob/465759da2428a4263eeee6ab464a0135fca2900c/trainer.py)

- 학습 SSIM은 11×11 Gaussian window·sigma 1.5·zero padding 5·채널별 grouped Conv·population moments·전체 평균과 `C1=0.01²`, `C2=0.03²`를 보존한다. window 생성 수치와 입력/출력 gradient를 비교하며 `.data`·`Variable`·기존 window의 잘못된 GPU 재사용을 제거한다. parameter key를 늘리지 않도록 `register_buffer(..., persistent=False)` 또는 입력의 device/dtype에 맞는 window를 사용한다. AMP에서 SSIM loss의 누적 dtype과 finite gradient를 검증한다. [원본 학습 SSIM](https://github.com/tsingqguo/efficientderain/blob/465759da2428a4263eeee6ab464a0135fca2900c/pytorch_ssim/__init__.py)

- v3 Rain1400 학습은 rainy/clean 공통 256 crop과 RGB float `[0,1]`을 기준으로 한다. 공식 스크립트의 `rainaug=false`와 기본 `angle_aug=false`를 확인하고 v4 RainMix·D4 등 추가 증강을 기본 정책으로 가져오지 않는다. 원본 학습의 작은 이미지 padding은 주석 처리되어 있으므로 crop_size보다 작은 pair는 명시적으로 거부한다. crop 좌표 범위도 비교하며 별도 padding·증강을 추가하면 원본 대비 전처리 변경으로 기록한다. [Rain1400 학습 스크립트](https://github.com/tsingqguo/efficientderain/blob/465759da2428a4263eeee6ab464a0135fca2900c/train_rain1400.sh), [학습 설정](https://github.com/tsingqguo/efficientderain/blob/465759da2428a4263eeee6ab464a0135fca2900c/train.py)

- 공식 설정의 Adam `lr=2e-4`, `betas=(0.5,0.999)`, weight decay 0, batch size 16, 250 epochs·50 epoch부터 선형 LR 감쇠를 기준으로 한다. 원본은 optimizer step 뒤 매 batch LR을 갱신하므로 schedule의 최초 적용 시점·경계·epoch 의미를 비교한다. 메모리나 fine-tune 목적의 batch·epoch·LR 변경은 config에 명시하고 원본 학습 재현과 구분한다. NAFNet의 AdamW/cosine/PSNRLoss를 기본값으로 가져오지 않는다.

- FP32 동등성을 먼저 확보한다. 혼합정밀도 학습은 `torch.amp.autocast(device_type="cuda", ...)`와 `torch.amp.GradScaler("cuda", ...)`를 사용하고 동적 필터의 곱셈·합산 및 loss의 dtype·finite gradient를 확인한다. inference는 `model.eval()`과 `torch.inference_mode()`를 사용한다. [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp)

- 부동소수점 ONNX는 `torch.onnx.export(..., dynamo=True)`를 우선 검증한다. ModelOpt Q/DQ는 선택 버전의 quantizer/exporter 지원을 별도로 확인한다. legacy exporter가 필요하면 `dynamo=False`를 명시하고 버전·이유·Q/DQ·출력 검증을 기록한다. 현재 API에서 제거된 인자를 사용하거나 Q/DQ가 누락된 export를 성공으로 처리하지 않는다. [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx)

- `torch.compile`, channels-last, fusion은 이식 완료 조건으로 강제하지 않는다. 원본 동등성·QAT·export·Orin 기준선이 확보된 뒤 실측 병목에만 적용한다. 모델 폭·kernel size·dilation·attention·최종 Conv 변경은 API 현대화에 포함하지 않는다.

**## INT8/QAT 원칙**

- [NVIDIA ModelOpt 공식 경로](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html)를 사용한다: `INT8_DEFAULT_CFG` 기반 calibration → PTQ 품질·배포 검증 → QAT → ONNX Q/DQ → TensorRT. 긴 QAT 전에 실제 TensorRT parser/build/출력 경로가 실행되는지 확인한다.

- 커널 예측 CNN의 일반 Conv를 W8A8 대상으로 삼는다. static symmetric INT8, weight per-channel·activation per-tensor를 출발점으로 하고 실제 quantizer 삽입·scale·축·활성 상태를 summary와 그래프로 확인한다. ReLU·concat·resize·slice·곱셈·reduce가 자동으로 W8A8 실행된다고 가정하지 않는다.

- `KernelConv`는 픽셀별 동적 필터이며 고정 가중치 Conv가 아니다. 초기에는 예측 CNN INT8 + 동적 필터 FP16/FP32를 사용한다. 민감한 커널 출력 head(`outc`, 필요 시 `conv8`)는 품질 비교 후 고정밀도로 남기고 제외 모듈 이름·근거를 기록한다. 이를 전체 W8A8로 보고하지 않는다.

- 기본 Conv 변환에는 동적 필터 뒤 `conv_final`도 포함될 수 있다. `outc`만 제외하면 head 주변 모든 Q/DQ가 사라진다고 가정하지 않는다. `outc`의 input/weight/output quantizer, `KernelConv` 입출력, `conv_final` input/weight quantizer와 concat·resize 경계의 정밀도 변환을 확인해 최종 Conv의 정책도 명시한다.

- calibration은 실제 주간·야간·강우 장면과 배포 전처리·고정 입력 크기에서 수행하고 평가 데이터를 분리한다. calibration 후 quantizer scale을 고정한 QAT를 출발점으로 삼는다. 재calibration을 수행하면 시점·데이터·결과를 기록한다. QAT loss는 원본 L1−0.2×SSIM을 기준으로 한다.

- 저장은 ModelOpt의 `mto.save/restore` 또는 `modelopt_state + state_dict` 경로로 구조·quantizer·scale·가중치를 함께 복원한다. 가중치만 저장한 checkpoint를 QAT 재개/export용으로 취급하지 않는다. 학습 재개용 optimizer/scheduler/AMP scaler/epoch/step/RNG와 모델·전처리·quantizer 정책도 저장하고 저장 전후 출력·상태 및 재개 backward를 검증한다. [ModelOpt 저장/복원](https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html)

- Albumentations의 독립 난수 상태·DataLoader generator·데이터 snapshot을 포함한 재개 정책은 앞의 checkpoint 계약을 따른다. 증강 설정 파일만 저장해 진행 중 RNG까지 복원됐다고 판단하지 않는다.

- 최종 ONNX는 양자화한 CNN에 `QuantizeLinear`/`DequantizeLinear`를 포함해야 한다. 초기에는 batch 1과 고정 입력 크기를 사용한다. 그래프의 scale·axis·dtype과 Q/DQ 보존을 검사하고 PyTorch 양자화 모델·ONNX·TensorRT 출력을 비교한다. INT8 파일명·가중치 round/dequantize·ONNX 파일 존재만으로 완료를 판단하지 않는다.

- TensorRT는 explicit quantization 경로를 사용하며 typing 정책을 기록한다. strong typing을 선택한 경우 FP16/INT8 builder flag·layer precision 지정을 혼용하지 않는다. floating 구간의 dtype은 ONNX 타입·Cast와 엔진 분석·출력 비교로 검증하며 `.half()`만으로 동적 필터의 안전성을 판정하지 않는다. [TensorRT 10.x 정밀도 정책](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html#strongly-typed-networks)

- 학습/QAT 이후 성능 단계에서 `slice/stack/multiply/reduce`의 실행 시간·중간 메모리와 CNN/동적 필터 비중을 측정한다. 실제 병목일 때만 출력·입력/core gradient·경계가 동등한 fusion을 구현하고, 표준 연산으로 해결되지 않을 때 TensorRT plugin을 검토한다.

- 2:4 pruning은 dense INT8 검증 후 적합한 일반 Conv에만 적용한다. DLA는 대상 SDK에서 지원되는 큰 연속 CNN 구간만 후속 실험으로 시험하고 동적 필터의 GPU 실행·GPU↔DLA 전환 비용을 포함해 비교한다.

**## 공식 품질 평가 protocol**

- Rain1400의 동일 split·checkpoint·전후처리에서 PyTorch FP32·TensorRT FP16·PTQ·QAT INT8을 비교한다. 공식 원해상도 복원 평가와 배포 고정 해상도/padding 평가는 구분하며 같은 비교 안에서는 입력·GT·복원 조건을 같게 한다.

- 공식 `recover_process`의 순서는 모델 출력/GT ×255 → `[0,255]` clip → ****uint8 절삭(반올림 없음)**** → float32 → 채널 역순 → OpenCV 원크기 bilinear resize다. GT도 검증 입력 크기로 resize한 뒤 이 복원 경로를 거친다. 원본 GT나 저장 PNG를 바로 평가하면 protocol이 달라질 수 있다. 모델 입력은 RGB로 유지하고 metric 경로의 채널 처리·dtype을 fixture로 확인한다. [공식 복원·PSNR 코드](https://github.com/tsingqguo/efficientderain/blob/465759da2428a4263eeee6ab464a0135fca2900c/utils.py)

- 공식 PSNR은 이미지별 전체 3채널 MSE와 peak 255를 사용하고 MSE=0일 때 100 dB를 반환한다. PSNR/SSIM은 이미지별 점수의 산술 평균이다. 전체 데이터 MSE를 먼저 합쳐 PSNR로 변환하지 않는다. batch가 커져도 첫 샘플만 평가하는 upstream helper의 제한을 복사하지 않는다.

- `validation.py`는 `structural_similarity`를 import하고 `compare_ssim`을 호출하는 이름 오류가 있다. 현대 `structural_similarity(..., channel_axis=-1, data_range=255)`로 수정하고 원본 scikit-image 0.17.2 기준 `win_size=7`, `gaussian_weights=False`, `use_sample_covariance=True`, `K1=0.01`, `K2=0.03`의 채널별 2D SSIM 평균을 보존한다. 원본은 float64로 계산하지만 현대 API는 float32를 유지할 수 있으므로 ****복원된 pred/GT를 SSIM 입력에서 float64로 승격****한다. SSIM 내부 경계 처리·dtype과 fixture 수치도 비교한다. 다른 리포의 11×11×11 SSIM3d·Gaussian SSIM으로 교체하지 않는다. API 오류 수정만으로 공개 수치 재현이 완료됐다고 보고하지 않는다. [공식 평가 코드](https://github.com/tsingqguo/efficientderain/blob/465759da2428a4263eeee6ab464a0135fca2900c/validation.py), [원본 SSIM 구현](https://github.com/scikit-image/scikit-image/blob/v0.17.2/skimage/metrics/_structural_similarity.py), [현대 SSIM API](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity)

- TorchMetrics는 선택적 보조 지표다. `ssim_torchmetrics` 등 이름·range·clip·uint8 변환·집계를 구분하고 공식 지표를 대신하지 않는다. PSNR도 이미지별 functional 계산 또는 NCHW `dim=(1,2,3)`의 sum/count로 집계하며 전체 MSE 기반 기본 집계와 구분한다.

- Orin에 학습·평가 의존성을 설치하지 않고 실제 TensorRT 출력 배열과 sample ID·전처리 metadata를 보관해 x86 evaluator로 평가할 수 있다. 파일 저장으로 추가 양자화된 이미지의 지표는 raw engine 출력 비교와 구분한다. 새로운 CPU metric port가 필요하면 공식 evaluator와 수치 동등성을 검증한 뒤 사용한다.

**## 구현 순서**

1. `modern-derain` uv 프로젝트·라이선스·필요한 스캐폴드를 구성한다. 단일 JSON 설정과 Typer CLI, 폴더 기반 `paired/rain1400` 매칭·split 검증·자동 snapshot을 먼저 연결한다. 선택 extra 없는 import/help와 CLI override를 확인하고 라이브러리 후보를 작은 CUDA 실행 검증 후 고정한다.

2. pin한 원본 oracle을 `tests/reference/`에 보존하고 KPN·KernelConv·학습 SSIM을 표준 PyTorch로 재구축한다. 공식 checkpoint strict load, 원본/재구축 모델의 forward·gradient, 네 dilation 경계·커널 정렬을 확인한다. 동등한 FP32 `.pt`로 변환해 자체 GitHub Release와 `weights/` 계약을 확정하고 기본 자산을 재사용하게 한다. 기존 artifacts 자산·참조·설정·예제를 이관한 뒤 중복·stale 산출물을 정리한다.

3. `infer/eval`과 Albumentations paired pipeline·FP32 `train`을 구현한다. 원본 loss·crop·optimizer/schedule·RGB/range·checkpoint 저장/복원과 작은 epoch 재개를 검증한다. 실제 Rain1400의 공식 전후처리·PSNR/SSIM 재현은 데이터가 준비된 별도 검증으로 수행하며 단위 테스트 통과와 구분한다.

4. x86 `export`와 대상 Orin `trtexec`로 FP ONNX/TensorRT build·출력 동등성을 확인한다. 실제 이미지 비교에는 필요한 최소 runner를 사용하고 작은 patch뿐 아니라 배포 크기의 동적 필터·출력을 검증한다. 이 단계에서 FPS·병목·전력을 측정하지 않는다. 대상 parser/build 호환성을 확인한 뒤 배포 조합을 확정한다.

5. ModelOpt calibration/PTQ → 공통 QAT 루프 → Q/DQ export를 연결하고 저장/복원·backward·엔진 실행 경로를 확인한다. 실제 INT8 Conv와 floating 경계를 기록하며 긴 QAT 전에 parser/build·출력 호환성을 확인한다. 드론 도메인 추가 학습이 필요하면 같은 데이터/학습량의 FP32 대조군을 확보한다.

6. 학습/QAT 이후 별도 평가 단계에서 Orin의 최종 품질·지연·전력·온도와 end-to-end 성능을 측정한다. 해상도·P95 지연/FPS·전력 모드·메모리 목표를 실측에 따라 구체화한다. 확인된 병목만 최적화하고 실행 명령·결과를 README에 남긴다. 데이터·Orin이 없는 단계는 미검증으로 두며 독립적인 구현은 계속할 수 있다.

**## 검증과 완료 기준**

- 초기 품질 목표는 같은 평가 protocol의 양자화 출발 FP32 대비 PSNR 하락 ****0.3 dB 이내****이며 설정으로 조정할 수 있다. 공개 checkpoint FP32, calibration 직후 PTQ, QAT와 각각의 TensorRT 결과를 구분한다. 도메인 fine-tune의 개선과 양자화 손실을 분리해 보고한다.

- TensorRT engine inspector/레이어 프로파일로 실제 INT8 Conv, 동적 필터의 정밀도, 변환 비용을 확인한다. Q/DQ 노드 수를 INT8 실행 비율로 대신하지 않는다. CNN 양자화 비율과 전체 실행 시간의 개선을 구분한다.

- warm-up과 CUDA event 또는 필요한 CUDA 동기화를 사용해 batch 1의 engine 시간과 전처리·전송·후처리를 포함한 end-to-end 시간을 구분한다. P50/P95, FPS, 메모리, 입력 크기, 측정 횟수, 전력 모드·클록·온도·버전을 기록한다. 측정 방식에 필요한 동기화와 진행 표시의 추가 동기화를 구분한다.

- `trtexec`의 pipelined throughput·GPU compute time은 엔진 지표로 기록한다. 이를 이미지 전후처리가 포함된 application FPS나 end-to-end 지연으로 보고하지 않는다. 실제 Orin 출력의 품질 검증과 전체 처리 경로의 측정은 별도로 수행한다. [trtexec 측정 정의](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/reference/command-line-programs.html)

- 테스트는 ****pytest****를 사용하고 실행은 `uv run --locked --all-extras pytest`로 한다. 해당 단계에 필요한 작은 검증만 둔다: 선택 extra 없는 package import/각 CLI help, deep merge·미지정/`False`/`0`·잘못된 인자·입력 manifest 거부, 일반/Rain1400 폴더 대응·source ID leakage·누락/중복·decode/scan 호출 범위, paired 증강 정렬·RGB/stride, 공식 resize·절삭·float64 SSIM fixture, 원본/재구축 모델의 forward/gradient·네 dilation 경계·학습 SSIM, FP32 weight 변환·strict load·Release 다운로드의 checksum/atomic 저장·실패 복구, epoch 재개·데이터 변경 거부, PTQ/QAT gradient·저장/복원, FP/QDQ ONNX 및 엔진 출력 비교. 다운로드 테스트는 작은 로컬 fixture를 사용하고 네트워크나 100 MiB 이상 실파일을 매번 내려받지 않는다. CLI는 `typer.testing.CliRunner`로 확인하고 JSON stdout과 로그 stderr를 구분한다. [Typer 테스트](https://typer.tiangolo.com/tutorial/testing/)

- 문서 수정만을 위해 테스트·의존성을 추가하지 않는다. 데이터·CUDA GPU·Orin이 없으면 관련 검증을 미검증으로 표시하고 수치를 만들지 않는다. 학습 재개 재현 범위와 남은 export/runtime 제약도 명시한다.

- 해상도·FPS/P95·전력 모드가 미정이면 기능 구현과 검증 상태까지 보고하며 실시간 배포 목표 달성을 판정하지 않는다. 가변 입력·타일링·분산학습·외부 추적 서비스·2:4/DLA는 초기 완료 조건에 추가하지 않는다.
