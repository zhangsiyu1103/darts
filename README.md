
## Requirements
```
Python >= 3.5.5, PyTorch >= 1.4.0, torchvision >= 0.5.0
```
This only works for cnn data, rnn still requires:
```
Python >= 3.5.5, PyTorch == 0.3.1, torchvision == 0.2.0
```


## Datasets
Instructions for acquiring PTB and WT2 can be found [here](https://github.com/salesforce/awd-lstm-lm). While CIFAR-10 can be automatically downloaded by torchvision, ImageNet needs to be manually downloaded (preferably to a SSD) following the instructions [here](https://github.com/pytorch/examples/tree/master/imagenet).

**Sampled CIFAR10** ([new_cifar.pth](https://docs.google.com/document/d/1f6EZSTxSAcVi_IQ1_vPLQB4p93By6jaN4y1vypf4Yms/edit?usp=sharing))
This file should be placed in directory ```cnn```


## Pretrained models
The easist way to get started is to evaluate our pretrained DARTS models.

**CIFAR-10** ([cifar10_model.pt](https://drive.google.com/file/d/1Y13i4zKGKgjtWBdC0HWLavjO7wvEiGOc/view?usp=sharing))
```
cd cnn && python test.py --auxiliary --model_path cifar10_model.pt
```
* Expected result: 2.63% test error rate with 3.3M model params.

**PTB** ([ptb_model.pt](https://drive.google.com/file/d/1Mt_o6fZOlG-VDF3Q5ModgnAJ9W6f_av2/view?usp=sharing))
```
cd rnn && python test.py --model_path ptb_model.pt
```
* Expected result: 55.68 test perplexity with 23M model params.

**ImageNet** ([imagenet_model.pt](https://drive.google.com/file/d/1AKr6Y_PoYj7j0Upggyzc26W0RVdg4CVX/view?usp=sharing))
```
cd cnn && python test_imagenet.py --auxiliary --model_path imagenet_model.pt
```
* Expected result: 26.7% top-1 error and 8.7% top-5 error with 4.7M model params.


## Architecture search (using small proxy models)
To carry out architecture search using 2nd-order approximation, run
```
cd cnn && python train_search.py --unrolled     # for conv cells on CIFAR-10
```
To use original darts code, run 
```
cd cnn && python train_search.py --unrolled --darts    # for conv cells on CIFAR-10
```

