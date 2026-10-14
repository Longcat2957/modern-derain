# Modern Derain

EfficientDeRain **v3 Rain1400 KPN**을 현재 PyTorch의 표준 연산으로 재구축한 독립 프로젝트다.
학습·calibration·QAT·ONNX export는 x86 CUDA에서 실행하며 ONNX를 실제 AGX Orin으로 전달해
그 장비에서 TensorRT 엔진을 빌드한다. 자체 코드는 **AGPL-3.0-only**다.
외부 source·weight·dataset의 조건은 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 구분했다.
[modern-nafnet](https://github.com/Longcat2957/modern-nafnet)은 사용 흐름의 참고 자료다.

**전체 Rain1400 품질 재현, Orin 엔진 실행 및 FPS·P95·전력 측정은 수행하지 않았다.**
구현/단위 테스트/합성 fixture 실행과 이 실기기 검증을 구분한다.

## 설치와 실행

Python 3.12와 uv를 사용하며 개별 pip 설치로 우회하지 않는다.

```bash
uv sync --locked --all-extras
uv run --locked modern-derain --help
uv run --locked --all-extras pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

기본 의존성은 NumPy·OpenCV headless·Typer·Rich다. `train`은 torch·AlbumentationsX,
`qat`은 여기에 ModelOpt, `export`는 torch·ONNX·ONNXScript, `metrics`는 scikit-image다.
pytest·Ruff·ONNX Runtime은 `dev` group에 있다. 필요에 따라 `uv sync --locked --extra train`처럼 설치한다.
PyTorch inference에는 `--extra train`, 공식 품질 평가에는 `--extra train --extra metrics`를 명시한다.
FP export에는 `--extra export`, Q/DQ export에는 `--extra export --extra qat`를 사용한다.

검증한 환경은 PyTorch **2.14.1+cu132**/CUDA wheel **13.2**, ModelOpt **0.47.0**,
AlbumentationsX **2.5.1**, ONNX **1.23.2**, ONNXScript **0.7.2**, Python **3.12.14**,
RTX 5000 Ada Generation Laptop GPU다. 실제 CUDA forward/backward·ModelOpt·export를 실행한 pin이다.
Linux x86_64 torch는 공식 `cu132` 인덱스, NVIDIA 배포 ModelOpt는 명시한 PyPI source를 사용한다.
ModelOpt base metadata에 빠진 `requests`·`huggingface-hub` import를 실행으로 확인해 `qat`에 추가했다.
torchvision·Transformers·ModelOpt `[all]`·TorchMetrics는 요구하지 않는다.

base만 설치한 별도 환경에서 package import와 루트·다섯 명령의 help를 확인했다.
선택 extra가 없으면 필요한 실행 프로필을 안내하고 nonzero 종료한다.
정상 stdout은 JSON 한 개이고 SDK 로그·진행 표시는 stderr로 보낸다.

## 설정과 폴더 입력

[`configs/train.json`](configs/train.json)에 경로와 바꿀 옵션만 적는다.
전체 기본값은 [`configs/default.json`](configs/default.json)이며 코드와 테스트로 비교한다.
적용 순서는 기본값 → 사용자 JSON → 명시한 CLI 옵션이다. `--no-amp`·`--steps 0`도 보존한다.
알 수 없는 키·잘못된 타입/shape·입력 manifest·`--manifest`는 거부한다.

```json
{
  "data": {"path": "data/Rain1400", "layout": "rain1400"},
  "training": {"checkpoint": "weights/efficientderain-rain1400-v3.pt"}
}
```

`data.path`·checkpoint·CLI 상대 경로는 CWD 기준이다. `data.train/val/test/calibration`은
그 루트 기준이며 절대 경로도 허용한다. split 기본값은 `train/val/test/calibration`이다.
명령에 필요한 split만 요구하고 데이터 목록 파일은 입력받지 않는다.

일반 `paired` 구조는 `data/paired/{train,val,test}/{rainy,clean}/`이며 동일 상대 경로·파일명·확장자를 매칭한다.
하위 시퀀스 폴더를 보존하고 누락·중복·모호한 구조·잘못된 decode·pair 크기 불일치를 거부한다.
시퀀스 하위 폴더 이름을 source ID로 사용한다. 실제 드론 데이터는 촬영 시퀀스·장소·날짜 단위로
폴더를 준비하고 train/held-out 사이에 같은 그룹을 섞지 않는다.

`rain1400`은 split 아래 `rainy_image/<clean_id>_<rain_id>.jpg`와 `ground_truth/<clean_id>.jpg`를 매칭한다.
선택한 배포본은 [Fu et al. 공식 데이터 링크](https://xueyangfu.github.io/projects/cvpr2017.html)의
[`rainy_image_dataset.zip`](https://drive.google.com/file/d/10cu6MA4fQ2Dz16zfzyrQWDDhOWhRdhpq/view)이다.
실제 전체 ZIP central directory를 확인했다: training GT 1–900/rainy 12,600개,
testing GT 901–1000/rainy 1,400개, 원본당 rain ID 1–14, JPG 확장자.
관측 출처와 hash는 [`tests/fixtures/rain1400.json`](tests/fixtures/rain1400.json)에 고정했다.
이 확인은 전체 이미지 payload를 내려받은 품질 평가가 아니다.

archive의 `training/testing` 이름을 그대로 쓰려면 `data.train/test`를 바꾼다.
train 내부 val/calibration은 **clean 원본 ID 단위로** 별도 폴더에 준비한다.
같은 clean의 rainy 변형을 split 사이에 섞지 않는다. 공식 test는 calibration·튜닝에 쓰지 않는다.
loader가 자동 split을 만들지 않는다. 원본 GT의 크기·ZIP CRC32로 ID를 확인하여
재번호/재인코딩된 미검증 배포본을 명확하게 거부한다. CRC32는 배포본 식별용이며 암호학적 hash가 아니다.
검증된 ID 901–1000은 공식 test에서만 허용하며 학습·val·calibration 유입을 거부한다.

calibration은 지정한 폴더의 **실제 rainy**만 사용한다. `rainy/` 또는 flat/시퀀스 폴더의 드론 이미지도
supervised layout과 독립적으로 지원한다. 원본 Rain1400 rainy-only `rainy_image/`/숫자 파일명은
archive의 rainy 크기·CRC32로 source ID를 검증하며 GT를 읽지 않는다.
일반 드론 파일이 Rain1400 숫자 이름과 겹치면 `rainy/` 아래에 두어 입력 종류를 명확히 한다.
clean-only 합성 rain·RainMix·학습 증강·다른 split fallback을 calibration에 적용하지 않는다.

## 가중치와 모델

[`weights/README.md`](weights/README.md)에 공식 URL·고정 upstream commit·실제 크기·SHA-256을 기록했다.
현재 `weights/efficientderain-rain1400-v3.pt`는 공식 `.pth`의 52개 FP32 tensor를
key·shape·dtype·값 변경 없이 포장한 **109,600,519-byte** 자산이다. optimizer·pickle model 객체는 없다.
큰 가중치는 Git/wheel에서 제외하고 [고정 `weights-v1` Release](https://github.com/Longcat2957/modern-derain/releases/tag/weights-v1)에 제공한다.
기본 파일이 없으면 그 파일을 사용하는 명령에서만 [검증된 FP32 파일](https://github.com/Longcat2957/modern-derain/releases/download/weights-v1/efficientderain-rain1400-v3.pt)을 내려받는다.
이미 확보한 `--checkpoint`로 offline 실행할 수 있다.
임의 upstream 다운로드·후보 URL 순회·`latest` fallback은 없다.
helper는 임시 파일의 크기·SHA-256 검증 후 원자적으로 설치하고 기존 다른 내용을 덮어쓰지 않는다.
import/help에서 다운로드하지 않는다. 외부 weight 이용 조건을 코드의 MIT/AGPL로 대신하지 않는다.

kernel 3×3·RGB·burst 1의 KPN을 유지한다. 27채널 core를 dilation 1·2·3·4가 공유하며
원래 채널/커널 정렬·zero padding·concat 순서와 `conv_final` 12→3을 보존한다.
외부 입력은 단일 RGB NCHW `[0,1]`이다. kernel softmax·출력 sigmoid·clip을 모델에 추가하지 않는다.
공식 입력은 H/W를 각각 ceil16으로 **OpenCV bilinear resize**한다. padding으로 대체하지 않는다.

## 추론·평가·학습

```bash
uv run --locked --extra train modern-derain infer \
  --input data/rainy.png --output artifacts/infer/restored.png --device cuda

uv run --locked --extra train --extra metrics modern-derain eval \
  --config configs/train.json --split test \
  --checkpoint weights/efficientderain-rain1400-v3.pt --output artifacts/eval/results.json

uv run --locked --extra train modern-derain train --config configs/train.json

uv run --locked --extra train modern-derain train --data data/paired \
  --checkpoint weights/efficientderain-rain1400-v3.pt --batch-size 1 --output artifacts/finetune
```

여러 CUDA GPU의 학습은 PyTorch DistributedDataParallel과 `torchrun`을 사용한다.
GPU당 프로세스 하나가 실행되며 아래 명령은 GPU 2장을 사용한다.
`--nproc-per-node`를 사용할 GPU 수에 맞추고 일부 GPU를 선택하려면 `CUDA_VISIBLE_DEVICES`를 지정한다.

```bash
uv run --locked --extra train torchrun --standalone --nproc-per-node=2 \
  --module modern_derain train --config configs/train.json \
  --batch-size 1 --output artifacts/train-2gpu
```

`training.batch_size`와 `--batch-size`는 **GPU별 배치 크기**다. 위 명령의 전체 배치는 2이며,
기본 배치 16을 GPU 2장에 사용하면 전체 배치는 32가 된다. 기존 전체 배치를 유지하려면
GPU별 배치를 나누어 설정한다. 학습률은 자동 변경하지 않는다.
`torchrun`이 설정한 local rank에 따라 CUDA 장치를 선택하며 분산 CPU 학습은 지원하지 않는다.
각 epoch에 sampler의 shuffle 순서를 갱신하고 GPU마다 서로 다른 sample을 학습한다.
sample 중복을 피하려고 GPU 수로 나누어떨어지지 않는 마지막 sample은 제외한다
(최대 GPU 수 − 1개). `training.drop_last=true`이면 GPU별 마지막 불완전 배치도 제외한다.
checkpoint·결과 JSON·진행 표시는 rank 0에서 작성하고 rank별 RNG 상태와 분산 실행 정보를 저장한다.
저장된 모델에는 DDP의 `module.` 접두사가 없으므로 단일 GPU inference/export에도 사용할 수 있다.

추론은 이미지/폴더를 받아 새 PNG와 metadata를 저장한다. 평가는 정렬된 GT pair가 필요하다.
GT 없는 드론 장면의 PSNR/SSIM을 보고하지 않는다.
공식 복원은 ×255 → clip → uint8 **절삭** → float32 → BGR → 원크기 bilinear resize다.
GT도 입력 크기로 resize한 뒤 같은 복원을 거친다. SSIM은 float64·7×7 uniform window·sample covariance·채널별 2D 평균이다.
PSNR은 전체 3채널 MSE/peak 255/동일 이미지 100 dB이며 결과는 이미지별 점수의 산술 평균이다.
저장 PNG의 추가 uint8 변환은 raw 출력 지표와 구분한다.
공개 **32.30 dB / 0.9272**는 원본 FP32 참고 수치이며 이 리포/INT8의 실측값이 아니다.

학습 기본값은 공식 uint8 공통 256 crop만 사용하고 작은 pair를 거부한다.
rainaug·angle augmentation·D4·합성 rain은 기본 정책에 없다.
paired Compose 한 번으로 crop한 뒤 contiguous CHW float `[0,1]`로 바꾼다.
Adam lr 2e-4/betas (0.5,0.999)/weight decay 0/batch 16/250 epochs,
50 epoch부터 optimizer step 이후의 원본 LR 갱신을 따른다. loss는 **L1 − 0.2×학습 SSIM**이다.
학습량·배치·LR 변경은 config에 기록하며 원본 재현과 구분한다.
AMP는 `torch.amp`를 사용하고 동적 필터·SSIM/loss는 FP32로 누적한다.

checkpoint는 optimizer/schedule/scaler·epoch/step·Python/NumPy/torch/CUDA RNG,
Albumentations 독립 RNG·DataLoader generator·실제 데이터 snapshot을 보존한다.
각 epoch 파일명은 결과 JSON에 반환한다. `--checkpoint`는 초기 가중치, `--resume`는 학습 상태다.
두 CLI 옵션을 함께 주면 오류이며 `--resume` 단독은 config의 초기 checkpoint를 대체한다.
재개는 동일 model/preprocessing/augmentation/sampler/batch/seed/worker/schedule과 데이터 fingerprint를 요구한다.
분산 재개는 동일 GPU 수도 요구하며 같은 `torchrun` 명령에 `--resume`를 사용한다.
정확한 crop/data-order replay는 `num_workers=0`인 epoch 경계다. CPU 작은 재개는 tensor 완전 일치를 검증했다.
CUDA backward 비결정성은 별도 오차로 기록한다. 중간 epoch QAT step 종료/중단 checkpoint는 inference/export용이며 exact resume를 허용하지 않는다.

```bash
uv run --locked --extra train modern-derain train --config configs/train.json \
  --resume artifacts/train/epoch-0001-step-00000010.pt --output artifacts/resumed
```

snapshot fingerprint는 절대 경로·파일 크기·mtime_ns이며 image content hash가 아니다.
반복 fine-tune은 이전 학습 snapshot도 보존하며 평가·calibration·held-out 데이터와 누수를 검사한다.
Rain1400 byte CRC ID 확인은 별도 준비 단계다. `artifacts/<run>/`은 실행 결과 전용이고
입력·GT·checkpoint·기존 산출물을 덮어쓰지 않는다.

## PTQ·QAT·ONNX

`export.shape`는 기본 null이다. 실제 배포 H/W를 batch 1·RGB·16 배수로 명시한다.
학습 crop을 배포 목표로 자동 선택하지 않는다. 아래 256은 예시다.
**긴 QAT 전에 대상 Orin의 FP/PTQ parser/build·출력 경로를 먼저 검증한다.**

```bash
uv run --locked --extra qat modern-derain qat --config configs/train.json \
  --height 256 --width 256 --steps 0 --output artifacts/ptq

uv run --locked --extra qat modern-derain qat --config configs/train.json \
  --checkpoint artifacts/ptq/ptq.pt --height 256 --width 256 --steps 100 --output artifacts/qat

uv run --locked --extra qat torchrun --standalone --nproc-per-node=2 \
  --module modern_derain qat --config configs/train.json \
  --checkpoint artifacts/ptq/ptq.pt --height 256 --width 256 \
  --batch-size 1 --steps 100 --output artifacts/qat-2gpu

uv run --locked --extra export modern-derain export \
  --height 256 --width 256 --output artifacts/export/fp32.onnx

uv run --locked --extra export --extra qat modern-derain export \
  --checkpoint artifacts/ptq/ptq.pt --quantized --height 256 --width 256 \
  --output artifacts/export/int8.onnx
```

`--steps 0`은 calibration만 수행하여 train 폴더를 요구하지 않는다.
QAT의 `--steps`는 모든 GPU가 함께 수행하는 optimizer step 수이며 GPU 수를 곱하지 않는다.
분산 calibration은 각 GPU에서 동일한 명시적 subset을 사용하며 학습용으로 분할하지 않는다.
ModelOpt가 calibration scale을 동기화하고 DDP가 학습 모델의 weight와 buffer를 동기화한다.
ModelOpt `INT8_DEFAULT_CFG`를 출발점으로 CNN의 static signed INT8 weight per-channel axis 0,
activation per-tensor를 사용한다. 초기에는 `outc`·`conv_final`을 고정밀도로 남기고 동적 필터는 FP32다.
24개 Conv의 W8A8 정책이며 전체 모델 W8A8 실행을 의미하지 않는다.
실제 quantizer/scale/축/enable·제외 정책을 기록하고 calibration 후 scale을 고정한다.
`modelopt_state + state_dict`로 strict 복원하며 원래 calibration provenance를 유지한다.

양자화 수치 정책은 calibration 시작부터 **`modern_derain_int8_qdq_divide_v2`**로 고정한다.
ModelOpt의 공개 backend 등록 API를 통해 `scale=float32(float64(amax)/127)`, `round(input/scale)`,
signed `[-128,127]` clip, scale 곱셈과 clipped STE를 사용한다.
고정 scale은 CPU·CUDA·export에서 동일한 FP32 값으로 반올림하도록 FP64 나눗셈 후 FP32로 저장한다.
ONNX에는 이 scale 값을 직접 Constant로 넣어 legacy exporter의 상수 접기가 값을 바꾸지 못하게 한다.
AMP 입력도 나눗셈·반올림은 FP32로 계산한 뒤 입력 dtype으로 복원하며,
STE gradient는 반올림 전 `input/scale`이 `[-128,127]` 안에 있을 때만 통과시킨다.
ModelOpt와 native fake-quant의 역수 곱셈은 반 bin 경계에서 ONNX Q/DQ와 다른 값을 만들 수 있어
실제 QAT checkpoint export에서 관측한 차이를 수정했다. backend·수식·range는 checkpoint에 기록하고
복원하며, 다른 정책의 checkpoint를 export 직전에 조용히 바꾸지 않는다.

Q/DQ export는 legacy symbolic 경로를 사용하며, 실제 CPU quantizer에 넣은 동일 tensor와
FP32 scale을 독립 Q/DQ 그래프에 넣어 ONNX Runtime과 bit 단위로 검증한다.
전체 그래프는 최적화를 끈 실행과 기본 INT8 fusion 실행을 각각 비교한다.
부동소수점 convolution의 누적 순서와 장치별 나눗셈 반올림은 작은 오차라도 INT8 반 bin을
넘어 출력을 바꿀 수 있다. 따라서 전체 출력이 허용오차를 벗어나면 `status=differs`와 실제 오차,
`quality_pending=true`를 기록한다. 이를 동등성 통과로 보고하지 않으며 허용오차를 늘리지 않는다.
실제 데이터의 품질과 대상 Orin 검증은 별도로 필요하다.

FP export는 dynamo를 사용한다. 실제 전체 KPN의 Pad 때문에 opset **18**을 검증했으며
요청한 opset이 조용히 다른 버전으로 바뀌면 실패한다.
Q/DQ는 지원되는 ModelOpt quantizer/exporter 경로를 명시하고 checker·signed INT8 scale/axis/dtype를 검사한다.
ONNX·외부 `.onnx.data`·JSON manifest를 함께 전달한다. manifest에는 shape/dtype/RGB/range,
resize/padding/복원·hash·SDK/package 버전·typing·calibration·검증 상태가 있다.
FP32 전체 출력 또는 동일 입력 Q/DQ 검증이 실패하면 전달 파일을 생성하지 않는다.
Q/DQ 전체 그래프의 수치 차이는 유효한 그래프의 전달과 구분하여 manifest에 남긴다.
quantized infer/eval은 `protocol=fixed`와 calibration shape의 일치를 요구한다.
초기 품질 목표는 동일 protocol의 양자화 출발 FP32 대비 PSNR 하락 0.3 dB 이내이며 설정으로 조정한다.

## 대상 Orin과 검증 범위

실제 SKU·설치 이미지의 [공식 JetPack 구성](https://developer.nvidia.com/embedded/jetpack/downloads)과
[TensorRT 지원 범위](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html)를 확인한다.
전체 학습 패키지는 Orin에 설치하지 않아도 된다. JetPack/L4T·CUDA·cuDNN·TensorRT·GPU SKU·driver,
사용한 Python ABI·전력 모드·클록을 대상 환경 manifest에 기록한다.
x86 TensorRT 버전/engine으로 Orin 지원을 추정하지 않는다.

초기 build·layer 분석은 시스템 `trtexec`를 사용한다. 다음은 **대상 SDK의 flag 지원을 확인할** 명령이다.

```bash
trtexec --onnx=fp32.onnx --saveEngine=fp32.engine --skipInference \
  --stronglyTyped --profilingVerbosity=detailed --exportLayerInfo=fp32-layers.json

trtexec --onnx=int8.onnx --saveEngine=int8.engine --skipInference \
  --stronglyTyped --profilingVerbosity=detailed --exportLayerInfo=int8-layers.json
```

Q/DQ strong typing에는 `--int8/--fp16`나 layer precision override를 혼용하지 않는다.
floating dtype은 ONNX 타입/Cast와 engine inspector로 확인한다.
[trtexec](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/reference/command-line-programs.html)와
[엔진 호환성](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html) 지침에 따라
엔진은 실제 대상 장비에서 빌드한다. FP16 대조 엔진에는 별도 precision 정책을 기록한다.

engine inspector로 실제 INT8 Conv·floating 동적 필터·변환 비용을 확인한다.
Q/DQ node 수는 INT8 실행 비율이나 application FPS가 아니다.
제품 연결에는 기존 runtime을 우선 사용하며 자체 TensorRT Python runner는 아직 필요하지 않아 추가하지 않았다.
품질 확인 후 engine/end-to-end P50/P95·FPS·메모리·전력·온도를 별도로 측정한다.
trtexec의 pipelined throughput/GPU compute time과 이미지 전후처리 포함 지연을 구분한다.
목표 shape/FPS/P95/전력 모드는 null이며 실시간 목표 달성을 판정하지 않는다.

`tests/reference/EfficientDeRain/465759da2428a4263eeee6ab464a0135fca2900c/`는 byte·license·파일별 provenance를
보존한 테스트 전용 원본이다. production에서 import하지 않으며 wheel에 포함하지 않는다.
공식 가중치의 CPU/TF32-off CUDA forward는 oracle와 일치했고 256×256도 확인했다.
원본 두 CUDA 인스턴스 사이에도 작은 backward 차이가 있어 gradient는 명시한 허용오차로 검증한다.
테스트는 CLI/config·data/누수/crop/RNG·공식 resize/recovery/SSIM·모델/loss gradient,
weight strict load/포장/download 실패 복구·epoch 재개·ModelOpt 복원/backward·FP/QDQ ONNX를 다룬다.
로컬 실행 evidence는 `artifacts/validation/`과 `artifacts/verification/`에 있다.
2026-10-14에는 RTX 4090 두 장에서 공식 가중치로 32×32 FP32/QAT 각 2 epoch·4 step과
epoch 재개, AMP 및 GPU별 DataLoader worker 1개, 256×256 FP32/QAT 각 1 step을 확인했다.
분산 gradient는 단일 global batch 기준과 비교했고 작은 모델의 rank별 crop/RNG 재개는
tensor 완전 일치를 확인했다. 전체 KPN의 CUDA backward 비결정성은 별도 기록한다.
검증 입력은 생성한 paired fixture이며 품질 평가나 속도 향상 측정이 아니다.
체크포인트의 단일 프로세스 strict 복원 결과는 `artifacts/validation-ddp/summary.json`에 보존한다.
검증용 checkpoint·입력 fixture·임시 스크립트·로그는 확인 후 정리했다.
전체 Rain1400 품질·대표 드론 calibration·Orin 품질/성능과 구분한다.
가변 입력·타일링·외부 추적·2:4 sparsity·DLA는 추가하지 않았다.
