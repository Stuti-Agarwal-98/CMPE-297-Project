# Team Members
1. Bharath Gunasekaran
2. Tamanna Mehta
3. Stuti Agarwal
4. Riddhi Jain

# CMPE-297-Project: Plant Disease Classification using Contrastive Learning
## Introduction

In this project, We are trying to assess the health of plant leaves. Our labelled dataset is quite small. So, we have used Contrastive Learning technique using Simclrv2 for plant disease classification. For better performance of our model,Initially the ResNet 50 Model is trained using Plant village dataset.The pretrained model is finetuned and distilled using Plant Disease Dataset i.e. classify the health of plant leaf. Each image of the Plant Disease dataset belongs to one of the category- healthy, rust,scab and multi disease. We have also implemented tfx pipeline for our finetune model.
![alt text](https://github.com/Stuti-Agarwal-98/CMPE-297-Project/blob/main/1_USmgYTlUc6D8XBh6kNXm4g.png)

## Dataset Used: [Plant Disease Dataset](https://drive.google.com/drive/folders/1Rdhd0ngPeNVQM3ktU1Rp6k905MNgQN4_)

## WebApp
### Screenshots:
### ![image](https://user-images.githubusercontent.com/71077352/145336003-519a524f-107b-42c8-807a-cf6aa31e6731.png)
### Application asks for a plant image we upload it from our device
### ![image](https://user-images.githubusercontent.com/71077352/145336081-3537dc68-be89-43a7-a19a-9ca73bb57d7f.png)
### Screenshot below shows the selected image
### ![image](https://user-images.githubusercontent.com/71077352/145336162-59bd01a5-76ac-4bef-8252-a0eb5a943fab.png)
### After clicking on the SUBMIT button We get the classified label of the disease
### ![image](https://user-images.githubusercontent.com/71077352/145336333-1343f83a-8e44-47e2-bb4c-16dee1eb5b73.png)



## References:
[Googgle-simclrv2](https://github.com/google-research/simclr)
