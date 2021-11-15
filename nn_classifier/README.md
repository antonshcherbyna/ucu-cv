# Experiments with CIFAR-10

### Code
I rearranged code for training and evaluating the model a bit for better reproducibility and more convinient experimenting.
All code is located right [here](nn/) and has following structure:
* train.py - main script for training
* trainer.py - class for nn taining, convinient way to organize forward and backward passes
* model.py - class with pytorch nn.Module (simple ConvNet)
* utils.py - different handy functions (e.g., for saving loading the model)

### Experiments
I perform a couple of experiments with different optimizers, strategies for lr scheduling and network architectures, but the best configuration was:
* 3x[Conv2d(16-32-64) -> MaxPool2d -> BatchNorm2d -> Relu] -> Flatten -> 3x[Linear(256-128-10)] 
* Good old SGD with Nesterov momentum = 0.9
* Initial lr = 0.02 (I had a huge batch size - 1024, so big learning rate wasn't a problem), with decay on 0.5 after every 10 epochs

### BatchNorm: before or after nonlinearity?
I saw a lot of discussion about right place for batch norm in the network. Initially authors said that it should be used before nonlinearity. And it's make sense: as I understand batch norm it shifts the distribution prior to activation functuion closer to the domain of this activation function. But then there were papers with claims that batch norm should be used after activation. So I decided to run simple experiment with both variants: and I didn't see any signinficant difference between both methods.

### Results
I was able to achieve 65% accuracy with configuration described above and with both options for batch norm:
![](imgs/results.png)

### What can be improved
* Another experiment I want to try is to fine-tune the model on samples with the biggest losses. For this experiment maybe it's better to try bigger dataset to be sure that there is no overfitting, but still it looks promising to force teach hard cases.
* I'm interested in extracting representations from images (both supervised and unsupervised). One of the way to extract those representations is to train supervised classifier and then cut last linear layer. But such vectors have bad property, they tend to concentrate in one place in the feature space. But not so long I read a [paper](https://ydwen.github.io/papers/WenECCV16.pdf) about specific loss (center-loss), which we can add to classic Cross Entropy, which will force those representations for different classes to be distant. The idea is pretty similar to triplet-loss, but main advantage is that center-loss doesn't require triplets, which is computationally inefficient, and can be used in classic supervised manner. I want to check the quality of such representations and also check, whether this additional term can imrpove accuracy for classification task.
