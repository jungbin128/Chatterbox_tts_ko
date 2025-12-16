# Chatterbox TTS Evaluation

본 모듈은 파인튜닝된 Chatterbox TTS 모델로 생성된 음성의 품질을 정량적으로 평가하기 위한 도구이다.  
다음 두 가지 지표를 사용한다.

- MOS (Mean Opinion Score) 예측
- Whisper ASR 기반 WER (Word Error Rate)

본 evaluation 코드는 학습 및 추론 파트와 독립적으로 동작하며,
TTS 출력 품질 검증 및 실험 결과 정리에 사용된다.

---

## What is included

- 음성 폴더 단위 자동 평가
- WV-MOS 기반 MOS 예측
- Whisper ASR 기반 WER 계산
- 파일별 결과 및 전체 평균 통계 출력
- 평가 결과를 CSV 파일로 저장

본 모듈은 standalone evaluation 도구로 설계되어 있으며,
training / inference 코드 없이도 단독 실행 가능하다.

---

## My Environment

- Kubernetes Pod 기반 실행 환경
- NVIDIA CUDA 12.x + Ubuntu 22.04
- 단일 GPU 환경 (A100 40GB)
- Persistent Volume 기반 작업 디렉토리
- Whisper large-v2 ASR 모델
- WV-MOS pretrained 모델 사용

---

## Dependencies

시스템 패키지:
```bash
apt install ffmpeg

