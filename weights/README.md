# v3 Rain1400 FP32 weights

공식 원본은 EfficientDeRain의 고정 commit `465759da2428a4263eeee6ab464a0135fca2900c` README에 연결된 [모델 archive](http://www.xujuefei.com/models_effderain.zip)의 `models/v3_rain1400/v3_rain1400.pth`다. 실제 확보에는 동일 호스트의 [HTTPS URL](https://www.xujuefei.com/models_effderain.zip)을 사용했다. 2026-10-13에 ZIP central directory의 member 크기와 CRC32(`1774052226`)를 검증했다. 전체 archive SHA-256은 계산하지 않았으며 아래 원본 SHA-256은 추출한 `.pth` 전체에 대한 값이다.

| 자산 | 실제 byte 수 | SHA-256 |
|---|---:|---|
| `upstream-v3-rain1400.pth` | 109601091 | `b8aaa9b5cd01177ca22a5344813923f91f298515be6382cf8654799cab5fa676` |
| `efficientderain-rain1400-v3.pt` | 109600519 | `b7be892d6483e7e41b65f143ed1d6fc199cdaece8229f4a540e897a80abeeab6` |

원본은 `OrderedDict` raw state dict이며 52개 tensor, 27,395,406개 parameter, 전부 `torch.float32`다. DataParallel prefix가 없으며 원래 parameter key·shape·dtype·값을 모두 보존했다. 변환본은 `state_dict`, `config.model`, `config.preprocessing`, provenance만 추가한 FP32 envelope다. optimizer·pickle model·FP16 변환·근사 압축·학습·양자화를 포함하지 않는다. 변환에 사용한 PyTorch는 uv lock의 `2.14.1+cu132`다.

원본과 변환본을 각각 strict load하여 tensor 값·dtype의 완전 일치를 확인했다. 원본 oracle와 재구축 KPN의 RGB `[1,3,16,32]` FP32 CPU 출력은 절대/상대 오차 0으로 일치했다. 무작위 모델의 전체 parameter/input gradient, 네 dilation의 input/core gradient, 학습 SSIM 값/gradient는 `tests/test_model_reference.py`에서 검증한다. 이 검증은 공식 Rain1400 전체 품질 평가를 대신하지 않는다.

CUDA 검증 GPU는 NVIDIA RTX 5000 Ada Generation Laptop GPU다. TF32를 끈 FP32 공식 가중치 출력은 `[1,3,16,32]`와 `[1,3,256,256]` 모두 max absolute/relative error 0이었다. 네 dilation의 forward/input/core gradient도 오차 0이었다. 전체 KPN parameter gradient의 최대 절대 오차는 `3.8147e-6`, input gradient는 `2.3842e-7`였다. 동일 원본 모델 두 인스턴스도 CUDA backward에서 parameter 최대 절대 오차 `3.3379e-6`를 보였다. CUDA gradient 테스트는 `atol=1e-5, rtol=1e-4`를 명시하며 CPU 테스트는 더 좁은 허용오차를 쓴다. near-zero gradient의 최대 상대 오차는 `0.0361`이며 단독 상대 오차로 일치 여부를 판정하지 않는다. CUDA AMP는 FP16 CNN 출력, FP32 동적 필터 누적과 SSIM/loss를 사용해 출력 및 모든 gradient의 finite를 검증했다. 이 결과는 Orin/TensorRT 검증이나 성능 측정이 아니다.

두 큰 파일은 일반 Git 및 wheel에 넣지 않는다. 변환 FP32 파일과 `SHA256SUMS`는 [자체 `weights-v1` Release](https://github.com/Longcat2957/modern-derain/releases/tag/weights-v1)에 게시했다. 확정한 [파일 URL](https://github.com/Longcat2957/modern-derain/releases/download/weights-v1/efficientderain-rain1400-v3.pt)은 `models/assets.py`에 고정하며 `latest`나 upstream fallback을 사용하지 않는다. 기본 파일이 없으면 사용하는 명령에서만 다운로드하고 위의 실제 크기·SHA-256을 검증하여 원자적으로 설치한다. import/help에서는 다운로드하지 않으며 기존 검증본으로 offline 실행할 수 있다. 원본 `.pth`는 개발 검증용이고 Release에 중복 제공하지 않는다.

다른 checkout에서 원본을 확보한 뒤 동일 포장을 재생성할 경우:

```bash
curl -fL https://www.xujuefei.com/models_effderain.zip -o /tmp/models_effderain.zip
unzip -p /tmp/models_effderain.zip models/v3_rain1400/v3_rain1400.pth > weights/upstream-v3-rain1400.pth
uv run --locked --extra train python -c 'from modern_derain.models.kpn import package_checkpoint; from modern_derain.config import DEFAULT_PREPROCESSING; package_checkpoint("weights/upstream-v3-rain1400.pth", "weights/efficientderain-rain1400-v3.pt", DEFAULT_PREPROCESSING)'
```

포장 helper는 검증된 원본 SHA-256만 허용하며 기존 destination을 덮어쓰지 않는다. 다운로드 resolver는 임시 파일의 크기/hash 검증 후 원자적으로 설치하고 기존 내용이 다르면 오류로 처리한다. `package_checkpoint`는 `--resume`에 필요한 학습 재개 상태를 만들지 않는다.

코드 원저작물의 MIT 고지와 `Copyright (c) 2020 Qing Guo`는 `licenses/EfficientDeRain-MIT.txt`에 보존했다. 별도 checkpoint 이용/재배포 license는 archive에서 확인하지 못했다. upstream 코드 MIT나 프로젝트 AGPL을 외부 weights의 license로 대신하지 않는다. 이 Release는 외부 weights에 새 license를 부여하거나 별도 이용/재배포 조건의 검증 완료를 주장하지 않는다.
