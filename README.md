# Team Members
1. Bharath Gunasekaran
2. Tamanna Mehta
3. Stuti Agarwal
4. Riddhi Jain

# CMPE-297-Project: Plant Disease Classification using Contrastive Learning
## Introduction

In this project, We are trying to assess the health of plant leaves. Our labelled dataset is quite small. So, we have used Contrastive Learning technique using Simclrv2 for plant disease classification. For better performance of our model,Initially the ResNet 50 Model is trained using Plant village dataset.The pretrained model is finetuned and distilled using Plant Disease Dataset i.e. classify the health of plant leaf. Each image of the Plant Disease dataset belongs to one of the category- healthy, rust,scab and multi disease. We have also implemented tfx pipeline for our finetune model.

## Dataset Used: [Plant Disease Dataset](https://drive.google.com/drive/folders/1Rdhd0ngPeNVQM3ktU1Rp6k905MNgQN4_)


## References:
[Googgle-simclrv2] (https://github.com/google-research/simclr)
