# DLP Lab3 - Binary Semantic Segmentation

---

## Overview

在本次 Lab 中，我們需要建立兩種網路模型來完成影像分割的任務：(1) UNet, (2) Resnet34 + Unet，分別是基本的 Unet 架構以及把 encoder 換成 Resnet34 的版本。

|![alt text](./att/unet.png)|![alt text](./att/resnet34_unet.png)|
|:---:|:---:|
| **UNet**  | **Resnet34 + Unet** |

U-Net在分割任務上扮演很重要的角色，因為 down-sampling + up-sampling 的架構給予模型能夠從不同的 scale 來去學習影像細部的內容，而本次 Lab 要我們在 Oxford-IIIT Pet Dataset 上完成影像分割任務。Oxford-IIIT Pet Dataset 是一個常用於 Semantic Segmentation 和 Object Segmentation 的 Benchmark，包含 7349 張寵物的圖片(大多數是狗和貓)，我們需要以此資料集訓練並評估模型，最終用 DICE score 來衡量好壞。

$$ Dice\ score=\frac{2\times Intersection}{Pred\ mask + GT\ mask} $$

---

## Implemtation Details

A. Details of your training, evaluating, inferencing code

### Details of UNet & ResNet34_UNet

#### UNet

|![alt text](./att/unet_code.png)|
|------------------|
| Unet 主要由 Encoder, Bottleneck, Decoder 組成，Down-sampling 的部分我參照原始論文的方法，使用 `MaxPool2d` 來完成，而 Up-sampling 則使用 `ConvTranspose2d` 來實現，其中在 `_double_conv`內是由，兩組的 `Conv2d` + `BatchNorm2d` + `ReLU` 組成，在 Convolution 的設定上有加入 `padding=1` 避免在 skip-connection 的地方還需要做額外的 crop 以減少運算量。在輸出層的部分，因為考慮到我們要得到預測出來的 binary mask，所以輸出維度為1的 prediction mask ，再經過 `sigmoid` 得到最終每個 pixel 的機率值。 |

#### ResNet34_UNet

|![alt text](./att/resnet34_unet_code.png)|
|------------------|
| Resnet34_Unet 是由 Unet 改過來的，主要是把 Down-sampling 的過過程由 Resnet34 取代，因此原本的 `_double_conv` 以 `_residual_block` 取代，每一個 Convolution Layer 都由 `Conv2d` + `BatchNorm2d` + `ReLU` 實現，除了層數外沒有變動太多。 |

<!-- C. Anything more you want to mention -->

## Data Preprocessing

在資料前處理的過程中，我主要提供兩種不同處理方式的資料集，分別為簡單 Resize 的資料以及有做些許資料增強的資料集。在實驗結果分析中，會比較兩種不同資料集的表現。

### Simple Oxford Pet Dataset

|![alt text](./att/simpleDataset.png)|
|----------------------|
| 與助教提供的範例程式碼沒有做太大的修改，只有保留 `image`, `mask` 兩種形式的資料當成模型的輸入資料與標籤資料，輸入前都會 Resize 成 256x256，以避免模型在 sampling 的過程中因為尺寸問題導致有 pixel 被吃掉。|

### Data Augmentation Oxford Pet Dataset

|![alt text](./att/DAdataset.png)|
|----------------------|
| 在另一個資料集中，由於考慮到 image 與 mask 要做相同的 transform，我只有做一些簡單的資料增強，如水平翻轉、垂直翻轉等(`TF=torchvision.transforms.functional`)，另外正規化只會對 image 做而不會對 binary mask 做，以防在計算DICE Loss 時遇到值域跑掉的問題。|

## Analysis on the experiment results

### Training & validation stage

下圖為訓練階段時，兩個模型分別在有無資料集增強的方式下得到的 training & validation curve，綜觀而言，可以看到兩者在 Training 上的 DICE score 沒有太大的差距，但是在 Validation 上有資料增強的模型 DICE score 略高出一點，這樣的表現是可想而知的，因為訓練階段只會看到訓練集，無法學到比更複雜的情況，因此沒有資料增強的模型在驗證階段會比較低分。不過當 epoch 數越來越往上時，兩者的差異也越來越小，我認為有可能是資料集單純，因此模型最終會學到的東西趨近一致，若是在更複雜的資料集，兩者的表現差異會更明顯。

#### Simple dataset

|![alt text](./saved_models/unet_epoch100_acc.png)|![alt text](./saved_models/resnet34_unet_epoch100_acc.png)|
|:---: |     :---:     |
| UNet | Resnet34_UNet |

#### Data Augmentation dataset

|![alt text](./saved_models/unet_epoch100_acc_da.png)|![alt text](./saved_models/resnet34_unet_epoch100_acc_da.png)|
|:---: |     :---:     |
| UNet | Resnet34_UNet |

### Inference stage

#### Simple dataset

|||
|:---: |     :---:     |
| UNet | Resnet34_UNet |

#### Data Augmentation dataset

|||
|:---: |     :---:     |
| UNet | Resnet34_UNet |

C. Anything more you want to mention

## Execution command

### Training

```py
123

```

### Inference

```py
123

```

## Discussion

A. What architecture may bring better results?

B. What are the potential research topics in this task?

C. Anything more you want to mention
