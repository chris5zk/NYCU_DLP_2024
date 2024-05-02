# DLP Lab4 - Conditional VAE Video Prediction

---

## Derivate CVAE formula

> picture

## Introduction

- **Video Prediction**
  
    在 

- **Teacher forcing**

    Teacher-Forcing 是在處理時序資料時，將 generator 輸入由前一個 time step 預測值改為 Ground True，避免序列生成的過程誤差逐漸放大，可以幫助模型加速收斂和增加穩定性。不過可能遇到 **Exposure Bias** 的問題，即 training stage 約束過重導致 inference stage 是從不同的分布中生成圖片，使得兩個階段之間有 gap 存在。因此本次實驗參考 2015 年 Google 發表的論文《Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks》的方式，在模型每一個 time steps 給定機率 $p$ 來選擇是否使用 teacher forcing，$(1-p)$ 機率來使用原本的 Autoregressive。

- **KL-annealing**

    在 VAE 訓練的過程中，為了最大化 reconstruction term 並最小化 KL-divergence term，很容易造成 KL-vanishing (posterior collapse) 的問題，這會使得 latent space distribution 直接與輸入 x 無關(因為最小化 KL-divergence 只需使得 posterior = prior)，也這會使得我們的 decoder 能力放大而不需仰賴 latent space distribution。因此為解決這項問題，需要使用 KL-annealing 循環或線性的調整 KL-divergence term 的係數，保證模型有多一點的時間學習輸入 x 的資訊到 latent space distribution 中。

---

## Implementation details

### Training protocol

訓練的方式會以三個方向來做調整：
- **KL annealing ratio**: 
- **Teacher forcing ratio**: 
- **Learning rate decay schedule**: 
  
> 等所有結果取最好的來給分數

### Reparameterization tricks

> 看完介紹後再來寫

### Teacher forcing strategy

先比較哪一種的 decay ratio 比較好
- 100   -> 0.01
- 50    -> 0.02
- 10    -> 0.1

再比較哪一種的 decay 結果比較好
- Linear decay - best ratio
- fix ratio - 0.5

> 說完 Tfr 的影響後，在分別驗證假設，根據影響來調整

### KL annealing ratio

- Monotonic
  - 0.1
  - 0.05
- Cyclical
  - 0.1
  - 0.05
- nKL

> 說完 KL annealing 的影響後，在分別驗證假設，根據影響來調整

---

## Analysis & Discussion

### Teacher forcing ratio

Analysis & compare with the loss curve，是否啟動Tfr與當下的ratio對loss的影響

### Training loss curve

|           |          |                 |
|   :---:   |   :---:  |      :---:      |
| Monotonic | Cyclical | No KL annealing |

Analysis the difference between them

### PSNR-per frame in validation set

> 不同 setting 所產生的 validation PSNR curve

### Other training strategy

> ???