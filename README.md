<img src="./cover.png"></img>


# Generative Deep Learning
Learning algorithms to generate similiar text, images, ...


---
### Introduction

Generative modeling is a machine learning process of training a model to produce new data which is similiar to the training data. A generative model is most likely probabilistic than deterministic, due to the fact that the output should be different every time.<br> The goal is to learn the underlying (unknown) probabilistic distribution/function of images which have a horse (for example) and so we can sample novel horse images from this distribution.<br>
A generative models like GANs are the counter part to discriminative models and don't have a specific goal during training but also self-improves themselves and so it is a technique between supervised and unsupervised.<br>
A GAN tries to learn the characteristics of the given data and gives a similiar novel output. It is possible that the human brain works like a GAN which let researches believe that a truly AI have something to do with GANs.
Most interesting is GAN implemented with Deep Learning, the current most promising approach in machine learning.

Representation learning is another important word for generative modeling. The goal is to learn a lower representation of higher dimensional data / unstructured data. Many generative models create a lower representation of the data, which is called latent space. The latent space contains a lower representation of the data. For example could an image of an human be brought into single facts like: black hair, green eyes, ... -> this would be the latent space. Every observation is/can be one point in the latent space and can be transform back to the higher domain.<br>
And why does many generative models use this latent space? Of course it is easier to detect and learn the underlying true stochastic distribution in this lower representation space than in the high dimensional space.

**Core probability concepts**<br>
Generative modeling is kind of statistical modeling of probability distributions using machine learning techniques.<br>
Here is some most important probabilistic/statistic knowledge for generative models.<br>
Some important definitions:
- **Sample Space**:<br>All possible values. Like all pixel-value combination in an image or all letter/word combinations in a NLP task. In a more complex environment/task the sample space also can get more complex. -> as more higher dimensional the data is, as bigger the sample space becomes.
- **Probability Density Function** (Density Function):<br>Is a function which describes the probability that a given variable X is a similiar observation. The probability is given through the interval (the field under the curve). The density function is never negative and the integral for the whole definition space is exactly 1. There is only one true density function -> **Pdata(x)**. But there are endless many density functions -> **Pmodel(x)** which estimate Pdata(x), so Pmodel(x) are approximations to the truth distribution and is learned/created from our generative model. 
- **Parametric Modeling**: A parametric model **Pθ(x)** is kind of density function using parameters θ1, θ2, ... to approximate the true density function.
- **Likelihood**:  The likelihood **$L$(θ|x)** describes how likely a given observation x is a similiar observation using the parameters θ to build a density function. Therefore applies: $L(θ|x) = pθ(x)$. Or for a multiple observations, to test if a whole dataset is similiar to the origin data: 
    $$
    L(θ|X) = \prod_{x \in X} pθ(x)
    $$ 
    Often a log likelihood is calculated, through the fact that very detail float values are very computional exhausting.
    $$
    l(θ|X) = \prod_{x \in X} log( pθ(x) ) 
    $$
    The likelihood function is made upon the parameters and not from the data! It is an approximation with parameters and may does not sum/integrate to 1. The goal of parametric modeling is to find optimal parameters $\theta$ to maximize the likelihood of the origin data.
- **Maximum Likelihood Estimation**: Is a technique to estimate $\hat{\theta}$ which are the parameters most likely explain some origin observed data $X$ by building a density function $p\theta(x)$.<br>$\hat{\theta}$ is called *maximum likelihood estimate* -> MLE. In order to achieve this $\hat{\theta}$ the likelihood function have to get maximized with given observations $X$ and adjusting the parameters $\theta$ to achieve the maximization. As math expression:
    $$
    \hat{\theta} = \arg\max_{\theta \in \Theta}( L(\theta)) = \arg\max_{\theta \in \Theta}(\ell(\theta))
    $$
    Maximation in this context means that the density function $p\theta(x)$ fits best to the given data observations $X$.<br><br>
    Another important fact: Neural networks try to minimize a loss-function. For generative modelling, the negated likelihood can be utilized to function as loss function which tries to get minimized.
    $$
    \hat{\theta} = \arg\min_{\theta}(-1* \ell(\theta|X)) = \arg\min_{\theta}(-1*\log(p\theta(X)))
    $$

<!-- \in for element of sign + \log_2(x) !-->

*Different generative model types


More informations and probably best ressource is: [Generative Deep Learning by David Foster](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/?_gl=1*1794xrw*_ga*MTE1MjU0NTY4OC4xNzAzNTE4OTky*_ga_092EL089CH*MTczMjEzMjUyNC4yLjEuMTczMjEzMjUzMC42MC4wLjA.)



---
### Functioning




---
### Installation

You can install a python environment as decribed as following:

1. Install Anaconda
2. Manual Env creation: -> in anaconda promt
   ```terminal
    conda create python=3.11 -n gen -c conda-forge -c nvidia
    conda activate gen
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 
    pip3 install tensorflow tensorboard 
    pip3 install mlflow  
   ```
3. OR Direct env installation: -> in anaconda promt + **cd** to project folder
   ```
   conda env create -f "./win_env.yml"
   ```






