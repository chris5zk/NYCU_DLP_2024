# DLP Lab6 - Generative Models

---

## Introduction

在本次作業中，需要實現 GAN 與 Diffusion model 兩種先後為主流的 Genrative AI model。GAN 主要是從 Gaussian noise 中一步生成圖像，透過對抗是的學習先是訓練 Discriminator 讓其能夠分辨真實影像與生成影像，再訓練 Generator 讓其生出更逼真更容易使 Discriminator 混淆的圖像。而 Diffusion model 則是逐步的加入 Gaussian Noise，為了保證最終的 timestep 為 Gaussian Noise，必須一次只加入一點點的雜訊，而在生成過程會是預測雜訊再做除噪，慢慢地往回採樣出生成圖像。

---

## Implementation details

在本次的作業中，我總共實作兩種 GAN: DCGAN 與 ACGAN，而 Diffusion model 是以最基礎的 DDPM 建構：

### DCGAN



### ACGAN

### Diffusion model

---

## Results and discussion

### Synthetic image grid

#### DCGAN results

#### ACGAN results

#### Diffusion model results

### Pons and cons

### Extra imexperiments
