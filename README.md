<img src="./cover.png"></img>


# Generative Deep Learning
Learning algorithms to generate similiar text, images, ...

Contents:
- [Generative Deep Learning](#generative-deep-learning)
    - [Basics of Generative Models](#basics-of-generative-models)
    - [Basics of Deep Learning](#basics-of-deep-learning)
    - [Variational Autoencoder (VAE)](#variational-autoencoder-vae)
    - [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
    - [Autoregressive Models](#autoregressive-models)
    - [Installation](#installation)
    - [Hardware Check](#hardware-check)
    - [Code](#code)





---
### Basics of Generative Models

Generative modeling is a machine learning process of training a model to produce new data which is similiar to the training data. A generative model is most likely probabilistic than deterministic, due to the fact that the output should be different every time.<br> The goal is to learn the underlying (unknown) probabilistic distribution/function of images which have a horse (for example) and so we can sample novel horse images from this distribution.<br>
A generative models like GANs are the counter part to discriminative models and don't have a specific goal during training but also self-improves themselves and so it is a technique between supervised and unsupervised.<br>
A GAN tries to learn the characteristics of the given data and gives a similiar novel output. It is possible that the human brain works like a GAN which let researches believe that a truly AI have something to do with GANs.
Most interesting is GAN implemented with Deep Learning, the current most promising approach in machine learning.

**Latent Space / Representation Learning**<br>
Representation learning is another important word for generative modeling. The goal is to learn a lower representation of higher dimensional data / unstructured data. Many generative models create a lower representation of the data, which is called latent space. The latent space contains a lower representation of the data. For example could an image of an human be brought into single facts like: black hair, green eyes, ... -> this would be the latent space. Every observation is/can be one point in the latent space and can be transform back to the higher domain.<br>
And why does many generative models use this latent space? Of course it is easier to detect and learn the underlying true stochastic distribution in this lower representation space than in the high dimensional space.

**Core probability concepts**<br>
Generative modeling is kind of statistical modeling of probability distributions using machine learning techniques.<br>
Here is some most important probabilistic/statistic knowledge for generative models.<br>
Some important definitions:
- **Sample Space**:<br>All possible values. Like all pixel-value combination in an image or all letter/word combinations in a NLP task. In a more complex environment/task the sample space also can get more complex. -> as more higher dimensional the data is, as bigger the sample space becomes.
- **Probability Density Function** (Density Function):<br>Is a function which describes the probability that a given variable $x$ is a similiar observation. The probability is given through the interval (the field under the curve). The density function is never negative and the integral for the whole definition space is exactly 1. There is only one true density function -> **$Pdata(x)$**. But there are endless many density functions -> **$Pmodel(x)$** which estimate Pdata(x), so Pmodel(x) are approximations to the truth distribution and is learned/created from our generative model. 
- **Parametric Modeling**: A parametric model **$P\theta(x)$** is kind of density function using parameters $\theta1, \theta2, ...$ to approximate the true density function.<br>
  Calculating the output/result/likelihood of the parameter model $P\theta(x)$ is a complicated task, which different approaches tries to solve in different ways. This is most likely caused through high dimensional data.
- **Likelihood**:  The likelihood **$L(θ|x)$** describes how likely a given observation x is a similiar observation using the parameters θ to build a density function. Therefore applies: $L(θ|x) = pθ(x)$. Or for a multiple observations, to test if a whole dataset is similiar to the origin data: 
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
    Another important fact: In neural networks $\theta$ can be thought as the weights and the biases which together and with the activation functions and other elements build a parametric model / density function.
    Neural networks try to minimize a loss-function. For generative modelling, the negated likelihood can be utilized to function as loss function which tries to get minimized.
    $$
    \hat{\theta} = \arg\min_{\theta}(-1* \ell(\theta|X)) = \arg\min_{\theta}(-1*\log(p\theta(X)))
    $$

<!-- \in for element of sign + \log_2(x) !-->

**Generative Model Families**<br>
The generative model families can be seperated in modelling the density function explicit and implicit. The implicit models do not try to build a density function directly. They more try to make a stochastic process to generate the data in the right form and build in that way implicitly a density function. <br>
The generative models with explicitly model the density function are again seperated in models which approximate the density function and models which make the density function in some way tractable/calculatable.<br>
Generative models have of course more family members and there are many hybrid models which can't be strictly sorted in this taxonomy. Still it can be helpful to sort the model types round about in these categories.

The generative models / family members belong to which category are listed below, with the most important generative models.

```
Generative Models
    |---- Explicit Density Function
              |---- Approximate Density Function
                        -> VAE (Variational Autoencoders)
                        -> Energy-Based Models
                        -> Diffusion Models
              |---- Tractable Density Function
                        -> AR (Autoregressive Models)
                        -> Normalizing Flow Models
    |---- Implicit Density Function
              -> GANs (Generative Adversarial Networks)
```

All types of generative models could be programmed in different ways. But all sophisticated models are implemented using neural networks/ deep learning, which allows much flexibility due to its bottom-up approach.





> More informations and probably best ressource to generative models is: [Generative Deep Learning by David Foster](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/?_gl=1*1794xrw*_ga*MTE1MjU0NTY4OC4xNzAzNTE4OTky*_ga_092EL089CH*MTczMjEzMjUyNC4yLjEuMTczMjEzMjUzMC42MC4wLjA.)






---
### Basics of Deep Learning

**Unstructed Data**<br>
The rise of deep learning is also a rise of unstructed data. 
Unstructured data have his information only implicit but not explicit saved. For example images, speech, texts, ...
Structured data have his information explicit in form from so called "features". Most likely a data table with columns and rows.<br>
The access of the information is much easier in structured data, but the world only have unstructured data which have to get transformed to structured data.<br>
Classic machine learning algorithms only can handle structured data which limits its use massively and transforming unstructured data to structured data in complex tasks is nearly impossible.<br>
Deep learning/neural networks instead are much more flexible and can internally learn to extract the information/features from the unstructured data by themselfs. And even more crucial: the extracted features can be very complex, so that the features can be used to predict if a cat is in a picture even if the cat is on another position on the image.

**What is a Neural Network?**<br>
A neural network or ANN = artificial neural network or deep neural network (which is often synonymous used for deep learning) is just a series of stacked layers.
Every layer consists of units (also called neurons) which takes a number from a previous unit or the summary of previous units. The units are connected with weights which multiply the number during the data transfer. If a unit "fires" (gives his value/number to the next unit) one number gets added to it's value, called bias. Many units additional have a so called activation function, which influences the output value of the unit -> it defines a behaviour when/how strong the unit fires. For example can a unit be changed with an activation function that it only fires if the value is greater than 0. There are many different activation functions which influences the effectness of neural networks. Activation functions are import to add non-linearity to the network which gives the network the ability to also approximate more complex functions. 
By adjusting the weights and the biases a neural network learns and changes his prediction.<br>
How the units are connected and how they work can vary from different kinds of layers.<br>
The fully connected layer (or dense layer) for example connects every unit/neuron in its layer with all units/neurons from the previous layer.<br>
Neural networks which consists only from dense layers called multilayer perceptrons (MLPs), caused through the fact that the perceptron was one of the first neural networks and it was fully connected (invented by Frank Rosenblatt).
By the way the number of units/neurons can also vary in a multilayer perceptron, the layers/untis just have to be fully connected.<br>
The input layer is always the first layer in a neural network, its untis contain the input data and give it with added bias and multiplied weight to the next unit/layer.<br>
The output layer is in contrast the opposite layer, the last layer, which contains the output of the network in probability or direct as "answer".<br> 
The layers in between are called hidden layers. <br>
If data is passing through the network from the input layer to the output layer it called "forward pass" or "inference" and if the network takes a output and gives it from the last to the first layer, then it called "backward pass", which is interesting to adjust the weights and biases or in other words: backward passes are used to let a neural network learnalso called training.<br>
<br>
In a global picture a neural network builds a approximated function to represent the input data $X$ to the output value(s) $y$.<br>
We assume that there is a function which behaves like: $f(X) = y$ and so the neural network builds a function by adjusting its weights and biases which estimate this function $f$, so we can use it for unseen data and hope it generalized well enough / approximated well enough to the data.<br>
It is important to mention that there is not always a function $f$, but in real tasks there is most likely a function $f$ because the nature is logical and most likely mathematical describable. The bigger challenge is to collect the right data, with enough amount and enough quality. Wrong data can harm the learning process and for many tasks it is difficult to collect (enough) data.<br>
To calculate a MLP it looks like that:

$$
f(X) = y = \sigma\left(\sum_{i=1}^n w_i x_i + b\right)
$$

With the definitions:
- $x_i$​: Inputvector (x_1,x_2,…,x_n​).
- $w_i$​: Weights from the current unit to the previous units.
- $w_i x_i$: Dot-Product of the vectors.
- $b$: The Bias of the current unit.
- $\sigma$: The activation function of the current unit (Sigmoid, ReLU, ...).


During the training an amount of data observations (given by the batch-size number) are passed through the network and the output is then after the batch compared to the true expected outputs (ground truths) with a so called loss-function or error-function.<br>
Then the network is backpropagated and the weights are adjusted to minimize the loss. This is done by calculating the negated gradient of the loss-function with the given weight and change the loss in this direction a little bit (learning rate regulates the great of the adjustment in this direction) -> in other words we find out in which direction the minimum (local or global) of the loss-function is and change the weight to go in this direction.<br>
This background knowledge explains why the choice of the loss-function is important and that the learning-rate should not be choosen too high or too low, else the adjustments are too big or too small in the "right" direction.<br>
The detail strategy to update the weights during the backpropagation using the gradient decides the so called 'optimizer'. There are for example RMSProp, ADAM and much more optimizer.<br>
All data observations/batches are called one epoch. A training consists of multiple epochs (complete dataset throughpasses).






---
### Variational Autoencoder (VAE)

Variational Autoencoders are one of the most basic generative models, introduced in the year 2013. It approximate the true density function and so build it explicitly.<br>
The Autoencoder (simpler version of the VAE) learns to encode the input (for example an image) into a low representation -> it learns to transform the input data to the latent space/embedding space. The decoder learns to decode/reconstruct an item/data (also called embedding -> because there is as much information as possible embedded) from the latent space (a lower dimensional representation) into the original domain/dimension.<br>
It simply can work as generator by decoding/recreating another/novel item/data point in the latent space.<br>
Problem with the standard autoencoder is the sampling from the latent space. It is not simple to draw/sample from the latent space randomly, due to the fact that the latent space might be different distributed with the different input classes (for example maybe there were many shoes in the training and the latent space defines now shoes for 50% of the defined latent space, sampling would now be not really random and often result in shoes). Another problem are gaps. It is not given that the whole latent-space (limited through the train-data embedding mins and maxs) is filled without gaps. It is good possible that the learned embedding is in a way that there are areas without any known data in the near by, which could result in a bad sampling or again in a unbalanced distribution (often sample a similiar item). VAE improves the sampling.<br>
For that the variational autoencoder changes the encoding from 1 input to one exact point in the lower space, to 1 input to one point in a gaussian/normal distribution (multivariate, to handle multiple dimension latent spaces).<br>
The VAE encodes the input now into a mean and a variance of each dimension (for better calculations with NN, we use the logarithm of the variance of each dimension).<br>
Sampling happens now through the mean value (of the latent space) + $\exp{log_var*0.5}$ (where log_var is the log of variance in the latent space variantion) multiplied with a random sampling/number from a standard normal distribution: 
$$
mean + e^{(\log(var) * 0.5)} * \epsilon \sim \mathcal{N}(0,1)

\\[10pt]

\begin{array}{l}
    \bullet \: \text{where } X \text{ is a random variable.} \\
    \bullet \: \text{where } mean \text{ is the mean point of the distribution.} \\
    \bullet \: \text{where } var \text{ is the variance of the distribution points.}\\
\end{array}
$$

This is given through the derivation of the normal distribution. This is called 'reparameterization trick' and allows us to simply sample from a standard normal distribution and then apply this sample to the latent space normal distribution. <br>
The VAE also uses an additional loss (to the reconstruction loss from standard autoencoder), the Kullback-Leibler divergence term, to make sure that the learned distribution for the input item is similiar to the standard normal distribution. Else the reparameterization trick would not be possible. So the encoder learns over time to create similiar distributions to the standard normal distribution. Therefore the KL divergence measures the similarity of 2 probability distributions. The KL divergence loss gets additionally weighted with a hyperparameter, so that the balance between the 2 losses can be controlled.<br>
Why this helps with the sampling? The encoder now ensures that nearby items in the latent-space are really similear through the encoding into an distibution. Also this is now a continues space without completly empty areas as in the standard autencoder (this is mostlikely given through the additional loss).<br>
The encoder from VAE learns to building similiar normal distributions for similiar inputs. You can imagine that it builds clusters/clouds on a point (the mean) and how big and how the propably to find the learned feature is for the given image is given by the variance.

$$
\mu = (\mu_1, \mu_2, ..., \mu_d), \; \sigma^2 = (\sigma^2_1, \sigma^2_2, ..., \sigma^2_d)
$$

At the end we need real values, for that we have to sample from the learned distribution. 

$$
z \sim \mathcal{N(\mu, \sigma^2)}
$$

The problem here is that this equation haven't a derivation due to the randomness (which is need to compute the gradient).<br>
So we use the reparametrization trick to sample from the founded distribution, so that we get a number from the learned input distribution. This is important because the trick allows us to build the derivation of the function which is important to calculate the gradient in backpropagation. The trick just takes the standard normal distribution and transform the output to the learned/latent-space distribution.

$$
z = \mu + e^{0.5 \cdot \log(var)}*\epsilon, \quad \epsilon \sim \mathcal{N(0, 1)} 
$$

The decoder learns to reconstruct the original input from a sample drawn from the learned distribution. The decoder does not "know" that the values come from whatever distribution, just like before it gets straigth (not probabilistic) values.






---
### Generative Adversarial Networks (GANs)

The idea behind GANs is simple. One part generates data from noise and the other part tries to tell, if the data is from the generator or from the real/original data distribution, called the discriminator or the critic. Both try now to improve. The generator tries to generator images, so that the discriminator does think it is from the real data and so uses the feedback of the discriminator. This is the reason why GANs only implicit learn the density function. There is no direct representation of the data/density function. It just tries to get not spotted from the discriminator and so learn the data distribution implicitly, without even knowing or seeing it. <br>
In comparison the VAE directly used the data distribution to learned an encoding and a decoding to build an explicit approximate probability distribution. A GAN only learnes a density function implicitly.<br>
Back to the GAN: The improvement of the discriminator is also straightforward due to the fact, that we know if we gave him a fake (from the generator) or a real data from the data. So the discriminator learnes to classify if data is from the original data or not -> it will predict in percentage (how likely is the data from the original data). <br>
It is a continuing process of improving. At the beginning the generator will only generates random noise and the discriminator will decide randomly but may adjust and learn that noisy images come from the generator, and so the generator get bad results and gets pushed to improve/change the generation process until it successfull fool the discriminator most of the times and so the discriminator adjust and this game continues.<br>
The first GAN is the DCGAN => Deep Convolutional GAN. <br>
Interestingly the generator does the same as the decoder from VAE. It learns to decode from a latent space. In the case of the GAN, we have a latent space and draw randomly from it. We does not learn to build the latent space, which shows again that the GAN is an implicit approach. <br>
<br>
Itcan always happen that the generator or the discriminator learned so good that it is overpowered and the trainings process can't continue, due the counterpart can't adjust anymore and the loss collapses.<br>
If the discriminator takes over and predicts to well:
- Increase the Droput rate to dampen the information flow
- reduce the learning rate of the discrimantor
- reduce the amount of filters in the discriminator
- add more noise to the labels of the discriminator
- random flipping of some labels (to confuse the discriminator)

How to find out that the discriminator overpowered the generator?<br>
-> The generated images look not good but different from each other. (The generator stopped adjusting/learning to generate well representive images, because the discriminator stopped challenging him -> too good in classifying and the generator is overwhelmed, does not know how to adjust.)


If the generator overtakes and generates to good images so that the discrimantor can't adjust his loss anymore following steps can help (notice that this does not mean that the images of the generator looks good but it is likely that the generator learned to create one image which is successfull in fooling the discriminator -> this image/representation called a 'mode'):
- Increase the Droput rate to dampen the information flow
- reduce the learning rate of the generator
- reduce the amount of filters in the generator
- add more noise to the labels of the generator
- random flipping of some labels (to confuse the generator)

How to find out that the generator overpowered the discriminator?<br>
-> The generated images look very similiar to each other and are not finsihed. (The generator stopped adjusting/learning to generate well representive images, because the generator found a 'gap' in the classification of the discrimantor and the discriminator is overwhelmed, does not know how to adjust.)

Through the fact that the generator and the discrimantor challenge each other, the losses are expect to not be high and go lower and lower, as by many other deep learning models. The losses are expect to change over time due to the fact that the losses depend on the other losses and if one loss will go lower, the other loss will increase but then adjust and maybe the losses change to the opposite.

GANs have many parameters like (where and how many) batch normalization layers, dropout layers, activation layers, convolutional layers, important the filters amount of the convolutional layers, size of the latent-space, learning rate, learning rate scheduler, epochs and optimizer.<br>
Trial and error is the way. The knowledge of how a GAN works and how to interpret the losses and the example images can be helpful through this process.

**WGAN-GP**<br>
The Wasserstein GAN - Gradient Penalty is another GAN implementation which is more stable as the DCGAN. It uses the Wasserstein loss function, which does not output in probability but in a score $(-\inf, \inf)$. The before used binary cross-entropy loss can be written as folowing:

$$
\underset{\text{D}}{min} -(\; \log(D(x))\; +\; \log(1-D(G(z))) \;)

\\[10pt]

\begin{array}{l}
    \bullet \: \text{where } D \text{ is the discriminator function.} \\
    \bullet \: \text{where } G \text{ is the generator function.} \\
    \bullet \: \text{where } x \text{ is a real image}\\
    \bullet \: \text{where } z \text{ is noise / point in the latent space}\\
\end{array}
$$

The discrimantor tries to minimize the wrong predictions.

$$
\underset{\text{D}}{min} -(\; \log(D(G(z))) \;)

\\[10pt]

\begin{array}{l}
    \bullet \: \text{where } D \text{ is the discriminator function.} \\
    \bullet \: \text{where } G \text{ is the generator function.} \\
    \bullet \: \text{where } z \text{ is noise / point in the latent space}\\
\end{array}
$$

The generator tries to fool the discriminator.

Let's now see the Wasserstein loss function in comparison.

$$
\underset{\text{D}}{min} -(\; D(x)\; -\; D(G(z)) \;)

\\[10pt]

\begin{array}{l}
    \bullet \: \text{where } D \text{ is the discriminator function.} \\
    \bullet \: \text{where } G \text{ is the generator function.} \\
    \bullet \: \text{where } x \text{ is a real image}\\
    \bullet \: \text{where } z \text{ is noise / point in the latent space}\\
\end{array}
$$

The discriminator with the wasserstein loss tries to maximize the difference between the predictions of the real and fakes, which is converted to a minimization task.

$$
\underset{\text{G}}{min} -(\; D(G(z)) \;)

\\[10pt]

\begin{array}{l}
    \bullet \: \text{where } D \text{ is the discriminator function.} \\
    \bullet \: \text{where } G \text{ is the generator function.} \\
    \bullet \: \text{where } z \text{ is noise / point in the latent space}\\
\end{array}
$$

The generator still tries to fool the discriminator and tries to get real labels for his fakes. Also converted to a minimization task. Notice that the log is removed which would make it probabilistic.

This continues losses are now bad, because we always want to keep the losses small, so that we can handle them well and smoothly and so the WGAN need a Lipschitz constraint. The discriminator have tomake sure that the difference of predictions of 2 images divided through the differences of the 2 images is smaller or equal 1:

$$
\frac{ | D(x_1) - D(x_2) | }{| x_1 - x_2 |}
$$

This is where the Gradient Penalty Loss come into the scene. The gradient penalty is the squared difference between the norm/length of the gradient of the predictions with respect to the input images and 1 (which are implemented as the interpolation of fake and real image). So the gradients tend towards a length/norm to 1, else the big number space $(-\inf, \inf)$ could lead to exploding gradients and to vanishing gradients.

**CGAN**<br>
Conditional GANs are another type of GANs which have the motivation to influence/control the output with an additional input. For example we could decide via another input if we want the output is a smiling face or a not smiling/sad face. To achieve that the generator gets an one-hot encoded latent space representation of our label (smile or not smile) and the discriminator/critic gets also an additional input, the label one-hot encoded (for example in this binary example: 1,0 ; 0,1). The discriminator/critic learns now to give a score if the image is real and is the labeling with the image right. Both give one score. In that way the generator learns that if the additional random latent space label vector is for example (0,1) the loss from the critic wants a smiling face. The opposite applies for the opposite. To simply adding these inputs, the label embedding gets added to the latent space vector and the critic just adds the label as another image channel, by reapeting the one-hot encoded label until it fits to the image shape.<br>
The labels of the generator will be given and not random drawn, else it would be unsure what the GAN got and it have to be equal to the labels of the critic. During generating the generator will be random draw a label.<br>
The CGAN can build on top of the WGAN-GP, the DCGAN or any other GAN implementation.



> Note that GANs in general need much more training than VAEs but than can achieve much satisfaction.


> You can give your GAN always additionally the label as input. This can improve the learning, but the sampling must be adjusted a bit.






---
### Autoregressive Models

...






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

**Alternativly** you can use online coding workstation, like [Google Colab](https://colab.research.google.com/).
There are most likely important libraries already installed, but if not, you always can install them by yourself:

```python
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 
!pip3 install tensorflow tensorboard  
!pip3 install mlflow 
```

Or sometimes:

```python
%pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 
%pip3 install tensorflow tensorboard  
%pip3 install mlflow 
```

In theory you also can install anaconda on a could workstation using commands and then proceed how described on top.






---
### Hardware Check

To **check your hardware** type following in your python cell:

```python
def get_hardware_info(use_in_notebook=True, install_packages=True):
    import sys
    import subprocess
    import importlib.util
    
    if install_packages:
        if importlib.util.find_spec("psutil") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
        if importlib.util.find_spec("gputil") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "gputil"], check=True)
        if importlib.util.find_spec("py-cpuinfo") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "py-cpuinfo"], check=True)

    # import needed packages
    import platform
    import psutil
    import GPUtil
    from cpuinfo import get_cpu_info

    if use_in_notebook:
        if install_packages and importlib.util.find_spec("ipython") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "ipython"], check=True)

        from IPython.display import clear_output
        clear_output()
    else:
        pass
        # os.system('cls' if os.name == 'nt' else 'clear')

    print("-"*32, "\nYour Hardware:\n")

    # General
    print("    ---> General <---")
    print("Operatingsystem:", platform.system())
    print("Version:", platform.version())
    print("Architecture:", platform.architecture())
    print("Processor:", platform.processor())

    # GPU-Information
    print("\n    ---> GPU <---")
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print("GPU Name:", gpu.name)
        print("VRAM Total:", gpu.memoryTotal, "MB")
        print("VRAM Used:", gpu.memoryUsed, "MB")
        print("Utilization:", gpu.load * 100, "%")

    # CPU-Information
    print("\n    ---> CPU <---")
    cpu_info = get_cpu_info()
    print("CPU-Name:", cpu_info["brand_raw"])
    print("CPU Kernels:", psutil.cpu_count(logical=False))
    print("Logical CPU-Kernels:", psutil.cpu_count(logical=True))
    print("CPU-Frequence:", psutil.cpu_freq().max, "MHz")
    print("CPU-Utilization:", psutil.cpu_percent(interval=1), "%")

    # RAM-Information
    print("\n    ---> RAM <---")
    ram = psutil.virtual_memory()
    print("RAM Total:", ram.total // (1024**3), "GB")
    print("RAM Available:", ram.available // (1024**3), "GB")
    print("RAM-Utilization:", ram.percent, "%")

    print(f"\n{'-'*32}")


get_hardware_info(use_in_notebook=True, install_packages=True)
```

Here are some alternatives...but there are more!

```python
!nvidia-smi
```

```python
import subprocess
subprocess.run(["nvcc", "--version"])
```






---
### Code


**TensorFlow Basics**
- [MLP with CIFAR10 dataset](./src/01_tensorflow_basics/MLP_CIFAR10.ipynb)
- [CNN with CIFAR10 dataset](./src/01_tensorflow_basics/CNN_CIFAR10.ipynb)


**Variational Autoencoder**
- []()



