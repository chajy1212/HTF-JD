# HTF

## Self-Supervised Learning (SSL)
- **CT 이미지에서 패치 단위로 일부 영역을 마스킹(masking)한 뒤, 가려진 패치를 복원하는 방식의 Self-supervised 학습**을 수행
- **라벨이 필요 없으며**, 모델이 **전역(context) + 국소(local) 구조 정보를 스스로 학습하도록** 설계되어 있음

## 학습 구조
### ① Patch Embedding & Positional Encoding
- 입력 CT slice(1 × 384 × 384)를 **PatchEmbed** 모듈로 일정 크기 패치로 변환
- 각 패치에 **2D sin-cos positional embedding**을 추가하여 공간적 정보를 유지
  - pos_embed, decoder_pos_embed가 모두 sin-cos 고정형(learnable=False)으로 세팅됨
- CLS token은 encoder에서 global representation 용도로 추가됨


### ② Masked Autoencoder (MAE)
**Encoder**
- 전체 패치 중 **mask_ratio(0.75)** 비율만큼 무작위로 제거하고 남은 패치들만 Transformer encoder에 입력
  - random_masking()에서 ids_shuffle/ids_restore로 복원 순서 관리
- 여러 Block을 거쳐 **latent representation** 생성
- Encoder는 **partial observation → global understanding**을 학습함

**Decoder**
- Encoder 출력(latent)을 decoder dimension에 projection 후 **mask token을 삽입해 원래 패치 길이로 복원함**
  - mask_token이 가려진 위치에 채워짐
- Decoder transformer가 전체 패치를 재구성하고 마지막 linear layer(decoder_pred)가 **각 패치 픽셀**을 직접 복원함

### ③ CT-level Classifier (Downstream Task)
- MAE 학습으로 얻은 latent representation 중 **encoder의 CLS token을 feature로 사용하여 CT-level classifier를 학습할 수 있는 구조**를 제공함
  - forward_latent()이 CLS token만 추출하여 downstream task에서 feature로 사용하도록 설계됨
- 이는 이후 disease classification 등 supervised fine-tuning에 활용됨

### ④ Reconstruction-based Loss
- target: 입력 이미지를 patchify한 결과
- pred: decoder가 복원한 패치
- mask 위치만 loss 계산(MSE 기반)
  - forward_loss()에서 **masked patches에 대한 복원 성능만으로 학습**됨

## 데이터 흐름 요약
CT Volume
<br/>→ **중간 slice 추출 (mid slice)**
<br/>→ **Brain region extraction + resize (clean_brain_region)**
<br/>→ **MAE 입력 이미지 (1×384×384)**
<br/>→ **Patchify → Masking → Encoder → Decoder → Reconstruction**

## 성능 결과
- MIL (patch_size = 64, stride = 32 + 값이 전부 0인 patch는 filtering, MLP + epoch 50)
  - (split 70:15:15) AutoMIL : 0.5752 (Best : 0.5841, epoch 42)
  - (split 60:20:20) AutoMIL : 0.5695 (Best : 0.6060, epoch 16)
 
- MAE (img_size=384*384, patch_size=32, Epoch 3000, lr=1e-4, batch_size=8, accum_steps=4)
  - (split 70:30) [Epoch 886] Train Loss=0.0136 | Train Acc=0.9264 | Val Loss=0.0193 | Val Acc=0.9229
 
