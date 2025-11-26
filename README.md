# HTF

## Multiple Instance Learning (MIL)
MIL은 CT 전체(=bag)에만 라벨이 있고, patch 개별(=instance)에는 라벨이 없는 상황에서 학습하는 방식
 - CT 단위에는 환자(1) / 정상(0)의 bag-level 라벨만 존재
 - patch 단위에서는 개별 병변 여부를 알 수 없음
 - 모델은 patch-level 라벨 없이 patch → feature → attention weight → bag feature → CT-level prediction을 통해 CT-level loss만으로 전체 네트워크를 end-to-end로 학습
 - 즉, bag 라벨만 가지고 patch encoder까지 학습하는 MIL 구조

## 학습 구조
① Patch Encoder (CNN patch embedding)
- 각 CT를 64×64 patch들로 분할
- CNN 3-layer Conv + FC 구조로 각 patch를 128차원 feature vector로 임베딩

② MIL Attention Pooling (Attention-based aggregation)
- 모든 patch embedding을 입력받아
  - 첫 번째 Linear(att_V) → tanh
  - 두 번째 Linear(att_U) → scalar score
  - softmax로 정규화 → attention weight Aᵢ 계산

- 이후 weighting 적용: bag_feature = Σ (Aᵢ * patch_featureᵢ)
- 즉, CT-level 병변 판단에 더 중요한 patch에 자동으로 높은 점수를 주는 구조

③ CT-level Classifier
- Attention으로 얻은 bag-level feature(128-D)을 입력받아 Linear classifier가 최종 CT-level 환자/정상 prediction 수행

④ End-to-end Backpropagation
- Loss는 CT-level 예측값만 계산
- Loss가 역전파되면 흐름은 다음 순서로 전달됨: CT-level classifier → Attention MIL → patch embedding CNN
- 결국 CT-level supervision만으로도 CNN patch encoder, Attention weights, CT classifier 모두 end-to-end 업데이트됨

## 데이터 흐름 요약
CT (bag)
<br/>→ CT에서 여러 개의 64×64 patch(instance) 추출
<br/>→ PatchEncoder(CNN) → patch embedding
<br/>→ AttentionMIL(att_V + att_U) → attention weight Aᵢ
<br/>→ 가중합 Σ(Aᵢ fᵢ) → bag feature
<br/>→ Classifier → CT-level prediction
<br/>→ CrossEntropyLoss(CT label)
<br/>→ Backprop → Attention + CNN encoder 모두 업데이트

## 성능 결과
- patch_size = 64, stride = 32 + 값이 전부 0인 patch는 filtering, MLP + epoch 50 + split 70:15:15
  - (split 70:15:15) AutoMIL : 0.5752 (Best : 0.5841, epoch 42)
  - (split 60:20:20) AutoMIL : 0.5695 (Best : 0.6060, epoch 16)
 
 
