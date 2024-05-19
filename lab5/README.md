# DKP Lab5 - MaskGIT for Image Inpainting

---

## Introduction

### Quantilize features

### MaskGIT

---

## Implementation Details

### Multi-Head Attention

多頭注意力機制是為了要讓模型用更多參數學習到不同的特徵表示，將輸入投射成 `num_head` 個維度為 `dim//num_head` 的特徵向量，各自算過自注意力機制後，concatenate 回原本的 shape，再經過 output weight 得到最後的輸出。

在實作上，為了提升效率可以直接將輸入 `shape=(N, d_model)` 的特徵向量投射成維度三倍的特徵向量 `shape=(N, 3*d_model)`，將其重新排列為 `shape=(batch, N, num_head, 3*d_head)`，為了使計算自注意機制時只會計算各自head內的相似度，要將 `num_head` 的維度調整到前面得到 `shape=(batch, num_head, N, 3*d_head)`，再把特徵向量拆分成 `q, k, v` 計算自注意力，在這邊有加入 `attn_drop=0.1` 來調整自注意力機制的 Dropout 機率來訓練參數。

| Multi-Head Attention implement |
|              :---:             |
|  ![alt text](./att/MHA-2.png)  |
|  ![alt text](./att/MHA-1.png)  |

### Stage2 training: Masked Visual Token Modeling

MaskGIT 在訓練 BERT 時會隨機取樣需要被 mask 的 pixel，我是讓模型在 uniform distribution 上面取樣隨機點做為topk要取的個數 `K` 做為需要保留的 token 數量，再從隨機生成的 sample matrix 取得排名前面 `K` 個的索引值生成 `mask` 做為遮罩(需要保留的部分為 `True` )，將 `mask` 為 `False` 的部分換為 `mask_token_id` 與保留的 token 一起輸入進 BERT 內進行預測，得到機率值後返回 training loop 與 Ground True 計算 Cross Entropy Loss，使用 Adam 進行權重的更新。

|    MVTM training implement    |
|              :---:            |
|![alt text](./att/MaskGIT_forward.png)|
|![alt text](./att/MaskGIT_config.png)|

### Inpainting & Interative decoding

使用 MaskGIT 進行 Inpainting 的方法是進行 Interative decoding，剛開始會先將 testing image 輸入到 VQGAN encoder 取得 latent vector `z_indices`，再針對 latent vector 與 mask 進行迭代更新，其做法的 pipeline 為：

1. 根據輸入 `mask` ，將輸入 `z_indices` 不需保留的 token 換成 `mask_token_id` 輸入到 BERT 進行預測，並且取得預測最高的機率與對應的 token id `z_indices_predict_prob`, `z_indices_predict`
2. 在 gumbel distribution 上採樣更新 `z_indices_predict_prob` 得到 `confidence`，更新目的是為了避免 softmax 強者恆強、弱者恆弱的狀況，加上隨機採樣值可使得機率小的也有可能被選到
3. 將需要保留的 token 值設為 `torch.inf`，表示為不需要 mask 的位置(跟後面的演算法有關係)
4. 根據 mask scheduling，從 `step` 與 `total_iter` 得到的比值 `ratio` 輸入進 `self.gamma_func` 得到當前 step 還須保留的 masked token 比例，乘上總共的 masked token 數量 `N`，得到最終還需要 mask 的 token 數量 `n`
5. 利用 topK 取得 `confidence` 最小的第 `n` 個 token 的信心值，利用其數值判斷小於此數值的 token 得到 `mask_bc`
6. 更新 `z_indices_predict`
7. 迭代 1. ~ 6.，直到 step 到達設定的上限

|  Interative decoding implement  |
|               :---:             |
|![alt text](./att/inpainting.png)|

---

## Experimental results

### The best result

- **Best Fid score**: 39.39
- **Training strayegy**
  - `epoch`: 60 without warm-up
  - `batch size`: 24
  - `learning rate`: 1e-4
  - `attn_drop`: 0.1
  - `optimizer`: Adam with $\beta=(0.9, 0.96)$
  - `scheduler`: ReduceLROnPlateau with reduce factor=0.1

- **Inpainting mask scheduling parameters**
  - `mask-func` : cosine
  - `sweet_spot`: 10
  - `total_iter`: 10

|     Generation     |        mask        |  Fid score  |
|    :----------:    |    :----------:    | :---------: |
|![alt text](./att/cos_10-10_i.png)|![alt text](./att/cos_10-10_m.png)|![alt text](./att/cos_10-10_fid.png)|

### Comparison figures with different mask scheduling

#### Iterative generation

|            |                consine             |                linear              |                square              |
|:----------:|             :----------:           |             :----------:           |             :----------:           |
| t=10, T=10 | ![alt text](./att/cos_10-10_i.png)| ![alt text](./att/lin_10-10_i.png) | ![alt text](./att/squ_10-10_i.png) |
| t=20, T=20 | ![alt text](./att/cos_20-20_i.png) | ![alt text](./att/lin_20-20_i.png) | ![alt text](./att/squ_20-20_i.png) |
| t=30, T=30 | ![alt text](./att/cos_30-30_i.png) | ![alt text](./att/lin_30-30_i.png) | ![alt text](./att/squ_30-30_i.png) |

#### Iterative mask

|            |                consine             |                linear              |                square              |
|:----------:|             :----------:           |             :----------:           |             :----------:           |
| t=10, T=10 | ![alt text](./att/cos_10-10_m.png) | ![alt text](./att/lin_10-10_m.png) | ![alt text](./att/squ_10-10_m.png) |
| t=20, T=20 | ![alt text](./att/cos_20-20_m.png) | ![alt text](./att/lin_20-20_m.png) | ![alt text](./att/squ_20-20_m.png) |
| t=30, T=30 | ![alt text](./att/cos_30-30_m.png) | ![alt text](./att/lin_30-30_m.png) | ![alt text](./att/squ_30-30_m.png) |

#### Fid with different parameters

## Discussion

### some founded
