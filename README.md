# 🖐️ Sign_Language_Generator
### Pose 기반 수어 동영상 생성 모델

[한국연구재단 과제]

기간 : 2024.03 ~ 2025.03

Temporal-aware Conditional Generation Framework

---

## Overview


본 프로젝트는 **수어 양방향 번역(Sign ↔ Text)** task로 수어 번역, text to gloss, gloss to pose, pose to image 총 4가지로 나누어 진행

[pipeline] : Spoken Language -> Text-to-Gloss -> Gloss-to-Pose -> Pose-to-Image -> Video

본 레포지토리는 전체 프로젝트 중 **Pose 정보를 기반으로 자연스러운 수어 이미지 프레임을 생성** 영역을 다룸

기존 수어 생성 연구는 단일 프레임 생성 중심이거나 Diffusion 기반 구조를 단순 적용하는 수준에 머물렀으며,  
시간적 일관성과 실시간성을 동시에 만족시키는 구조는 부족

본 연구에서는 수어 생성을 단순 이미지 생성 문제가 아닌  
**Temporal-Conditioned Generation 문제**로 재정의하고 통합 프레임워크를 제안


## Motivation


### 기존 한계

- 손가락 디테일 표현 부족
- 프레임 간 색상/외형 차이로 인한 flickering 발생
- 느린 추론 속도 (≈ 1 frame/sec)
- 시간 정보가 반영되지 않은 독립 프레임 생성
- GPU 한계

### 문제 인식

수어는 정적 이미지가 아닌 **연속 동작 기반 언어**
따라서 단일 프레임 품질보다 **시간적 안정성(Temporal Consistency)** 이 핵심 요소


## Key Insight

> 현재 Pose만을 조건으로 사용하는 것이 아니라, 이전 프레임과 스타일 정보를 함께 conditioning 하여 이미지들을 연결 -> 비디오 생성을 대체

즉,

- "이 프레임이 정확한가?" 가 아니라  
- "이 프레임이 이전 프레임과 자연스럽게 이어지는가?" 를 기준으로 설계



Spoken Language
↓
Text-to-Gloss
↓
Gloss-to-Pose
↓
Pose-conditioned Image Generator (Ours)
↓
Frame Sequence
↓
Video


## System Architecture


![Pipeline](그림1.png)

### Input

- Gloss 기반 Pose 정보
- 이전 프레임 이미지 (optional)
- 스타일 정보 (밝기, 채도, 색상)

### Output

- 시간 일관성이 확보된 수어 이미지 프레임
- Flickering이 완화된 영상 시퀀스

---

## Proposed Methods

### 1. Condition-based Pose Input Design

- 생성 대상 프레임의 Pose 정보 명시적 입력
- 손 형상 가정 정보 포함
- 외형 스타일 정보를 condition으로 제공

→ 손 모양 정밀도 향상  
→ 스타일 제어 가능


### 2. Temporal Consistency Modeling

#### Previous Frame Conditioning

- 직전 프레임 이미지를 condition으로 활용
- Inpainting 구조 적용

→ 자연스러운 프레임 연결  
→ 외형 왜곡 감소

#### Feature Rescale

- 밝기 / 채도 / 색상 정보 보정
- 스타일 drift 방지

→ 색상 flickering 완화  
→ 영상 안정성 확보


### 3. Efficient Inference Architecture

![Pipeline](그림2.png)

- Diffusion Transformer 기반 구조 적용
- Flow Matching 기법 도입
- 구조 경량화 실험

→ Inference 속도 대폭 개선


## Experimental Results

### Visual Quality

- Inpainting 적용 시 손 모양 표현 정확도 향상
- Previous frame condition 활용 시 품질 개선
- Feature rescale 적용 시 flickering 감소
- 외형 및 색감 일관성 확보

### Inference Time Comparison

| Method | Inference Time |
|--------|---------------|
| X-MDPT-S | 1.191s |
| X-MDPT-B | 1.299s |
| X-MDPT-L | 3.124s |
| **Ours (Medium, NFE=60)** | **0.489s** |
| Ours (Large, NFE=30) | 0.550s |
| Ours (Large, NFE=60) | 0.850s |

기존 Diffusion 기반 모델 대비 **최대 3.6배 속도 향상** 달성


## System Characteristics

- Pose-conditioned image generation 구조 설계
- Temporal-aware conditioning 전략 제안
- Flickering 완화 메커니즘 적용
- Diffusion Transformer 기반 고속 추론 구조 실험
- 실시간 수어 번역 시스템 확장 가능


## Research Contributions

- 수어 생성 문제를 Temporal Conditional Generation 문제로 재정의
- Previous-frame conditioning 전략 제안
- Feature rescale 기반 스타일 안정화
- Diffusion 기반 모델의 실시간화 구조 실험
