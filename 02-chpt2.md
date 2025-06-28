---
layout: home
title: Chapter 2 - The ELBO Paradigm --- Proxy Objective for True Data Maximization
nav_order: 4
---

<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>


<div style="text-align: left; font-size: 1.3em;">
Chapter  2 - The ELBO Paradigm --- Proxy Objective for True Data Maximization
</div>
<br>
<div style="text-align: center;">
  <img src="./assets/images/fig_ch02.png" style="width: 45%; max-width: 400px; height: auto; margin: 0 auto;">
</div>

In previous chapter, we talked about finding the best modeled distribution via evaluating the likelihood \\(p(x)\\)  of the **observed** data, because that measures how well they explain the observation - higher likelihood suggests a better model.


Question now is: how to maximize the \\(p(x)\\), i.e. optimize \\(p(x)\\) to achieve maximal likelihood?


In this chapter, to get this game going, we will introduce two _deus ex machina_ magics, the first one being
> the latent variables, which is often denoted as \\(z\\).


The incorporation of latent variables in generative models represents a powerful paradigm grounded in critical insights about data structure. 
Real-world datasets often exhibit complex dependencies of **underlying, unobserved** factors.
>For instance, image data contains implicit attributes like illumination geometry and object orientation that are not explicitly encoded in pixel values but significantly influence the observed patterns. 


By explicitly representing these hidden factors, latent variables enable models to capture richer data structure.
Without latent variables, we would face the formidable challenge of modeling complex high-dimensional distributions directly - such as capturing intricate pixel-level correlations in images - which is both computationally intractable and statistically inefficient. 
Latent variables decompose the problem into more manageable components: 
- a simple _latent space distribution \\(p(z)\\)_, and 
- a _conditional data distribution \\(p(x\|z)\\)_, 

where \\(z\\) represents the latent factors and \\(x\\) the observed data. 
This framework offers a structured, interpretable, and scalable approach—transforming an intractable problem into one where hidden factors systematically explain observed phenomena.


The metaphor of Plato’s Cave from _The Republic_ provides a powerful analogy for understanding latent variables in generative models.
>In the allegory, prisoners are chained in a cave, seeing only shadows cast on a wall by objects they cannot directly observe. 

This mirrors the relationship between observed data and latent variables.
The shadows are like our raw data (e.g., pixel values in images), mere surface-level projections.
The true forms are the latent variables—the unobserved, higher-dimensional factors.
>e.g., lighting, pose, or semantic meaning that _generate_ the data, just as the objects outside the cave cast the shadows.

Reality is richer than what we directly perceive. 


**We now try maximizing \\(p(x)\\) by utilizing latents \\(z\\).**


---


We're here to find the distribution in which the observed data \\(x_1, x_2, ..., x_n\\) would be highest in probability compared with other distributions.
In modeling terms, this translates to finding a configuration of model parameters  such that the observed data have the highest  probability compared with other configuration.


Keep in mind that it has never been a problem of probability value arithmetic, as is suggested by its denotation format; it is a **distribution** (i.e. the best \\(p\\), in whichever form it takes, that we are actually looking for).


Inspired the above manageable components \\(p(z)\\) and \\(p(x\|z)\\), one way of linking latents \\(z\\) to \\(p(x)\\) is to think about: 

$$
p(x)=\int{p_\theta(x,z)dz}=\int{p_\theta(x|z)p(z)dz}
$$

where \\(p_\theta(x,z)\\) refers to a new modeled (hence the \\(_\theta\\)) probability distribution (hence the \\(p\\)) with new set of params (hence the \\(x,z\\) ).  


The decomposition of the integrand is directly from the chain of rule in probability.
- \\(p(x\|z)\\), i.e. _decoder_, describes how observations \\(x\\) are generated from latents \\(z\\);  
- \\(p(z\|x)\\), i.e. _encoder_, describes how latents \\(z\\) can be inferred from observations.

Note that the equation holds regardless of whether \\(x\\) and \\(z\\) are independent: independence would imply \\(p(x,z)=p(x)p(z)\\), but this is not necessary for the marginalization to be valid.

With this, one might propose the procedures of estimate \\(p(x\\)): 
- sample  \\(z\\) from  \\(p(z)\\) 
- for a given \\(\theta\\), calculate \\(p_\theta(x\|z)\\) , i.e. the probability of observing the data \\(x\\) we have observed given a sampled \\(z\\).
- do this with sufficient times of  \\(z\\) sampling, thus get to compute the integral.
- Therefore, by adjusting the model param \\(\theta\\), the integral gets bigger or smaller correspondingly, until the \\(\theta\\) corresponds to one high value of \\(p(x)\\) is found.


However, the intractability of the integration is a major challenge itself.


\\(z\\) is typically a high-dimensional vector. 
The integral is over all possible values of \\(z\\), which is computationally infeasible for even moderate size. 
>For example: if \\(z\\) has 100 dimensions and we discretize each dimension into just 10 values, the number of terms in the sum grows as much as \\(10^{100}\\).
The decoder \\(p_\theta(x|z)\\) , like other mathematical things in machine learning that is beyond explicit expression,  is usually modeled by a neural network. 
To such a complex nonlinear function,  there’s no analytical formula for integrating it over \\(z\\), even if \\(p(z)\\) is as simple as Gaussian.
Suppose we repeat  $$N$$ times to yield the approximation:

$$
p_{\boldsymbol{\theta}}(x) = \int p_{\theta}(x|z)p(z)dz \approx \frac{1}{N}\sum_{i=1}^N p_{\theta}(x|z^{(i)})
$$

This approximation converges to the true expectation as $$N \to \infty$$ by the law of large numbers. 
We optimize parameters $$\theta$$ to maximize $$p_{\theta}(x)$$, thereby improving the model's fit to the observed data while maintaining the learned low-dimensional structure.
However, it takes very large number to await the valid $$N$$ to come.
In addition, most \\(z\\) samples obtained in this approach will contribute negligibly to \\(p_\theta(x)\\), since \\(p(z)\\) is uninformed about \\(x\\), so most \\(z\\) values will lead to \\(p_\theta(x∣z)≈0\\), just consider how much naturalimages occupies the whole space.
Blindly sampling  \\(z \sim p(z)\\) provides little guidance to the decoder about which regions of \\(z\\)-space are relevant for generating meaningful \\(x\\).


----


A second approach of linking latents \\(z\\) to \\(p(x)\\)  is through chain rule of probability:  

$$
p(x)=\frac{p(x,z)}{p(z|x)}
$$

Since the latents  \\(z\\) are in both numerator and denominators, no direct observations can be made.
Given our goal is to maximize \\(p(x)\\), we're here faced with the problem of maximizing two unknown functions distributions simultaneously.


The denominator can be modeled as encoder network,  and it can be denoted as \\(p_\phi(z|x)\\).
It is quite instinctive to imagine a mapping from observed data \\(x\\) towards latents \\(z\\).
But in mainstream literature, the \\(p_\phi(z|x)\\) here is written as \\(q_\phi(z|x)\\), a notation that emphasizes it's an _approximation_ of the true posterior \\(p(z|x)\\).


Modeling the numerator  \\(p_\theta(x,z)\\) might seem conceptually straightforward: we could simply introduce another parameterized distribution. 
But in practice, designing a network that simultaneously takes both observed data \\(x\\) and latent variables \\(z\\) as inputs presents significant architectural challenges, making this approach less intuitive to implement.
We consider use chain rule of probability again to decompose \\(p_\theta(x,z)\\) into \\(p(z)p_\theta(x|z)\\).
Analogously, the \\(p_\theta(x|z)\\) can be modeled as a decoder network.


Now with the two networks are baptized, there's the term of \\(p(z)\\) left.




<!-- That dictates our prior to be 

$$p(z) \sim \mathcal{N}(\mu, \sigma^2).$$ -->

To wrap up our second approach, the new way to estimate \\(p(x)\\) can be like this: 
- sample one \\(z_i\\) (\\(i\\) being arbitrary integer as sampling index) from the variational posterior \\(q_\phi(z\|x)\\) by inputting one observed data \\(x_i\\) into the network. 
- one data point in true data dataset can be sampled more than one time.
- with the \\(z_i\\), evaluate \\(p(x)\\) as \\(\frac{p(z_i)p_\theta(x\|z_i)}{q_\phi(z_i\|x)}\\).

An expectation (even approximated with a few samples) gives a smoother, more stable gradient, because averaging over multiple samples would reduce variance.
So let there  be  multiple samples instead of only one, and average the sum of their values in the final \\(p(x)\\)'s, which is basically the idea of Monte Carlo estimating.


Wait，there is another concern is from \\(q_\phi(z|x)\\). 
All denominators are headaches for numerical computation. 
>It may lead to high variance if that is a poor approximation of the true encoder, which is a  source of numerical instability, notably when \\(q_\phi(z|x) \ll p(z)\\). 
>In such circumstance,  gradients during backpropagation can become extremely large (since gradients are inversely proportional to the denominator): e.g. for \\(f(x)=\frac{1}{x}\\), the gradient with respect to \\(x\\) is: \\(\frac{\partial f}{\partial x}=−\frac{a}{x^2}​\\).


One solution  is to transform \\(p(x)\\) into \\(\log p(x)\\). 
It won't  affect the finding of the right probability because the monotonicity of logarithm, and it moves denominator to the right of minus sign.
Numerically speaking, additivity is better than multiplictivity.
Thus, we have now 

$$
\log p(x)\simeq\mathbb{E}_{q_\phi(z|x)}[\log p(z) + \log p_\theta(x|z) - \log q_\phi(z|x)].
$$

However, it should be noted that with the introduction of \\(\theta\\) and \\(\phi\\) as the modeling effort, the equation of \\(p(x)\\) doesn't hold strictly.
We should still depart from the strictly-holing chain rule of probability and see what's the relationship between them two formulae: 

$$\begin{aligned}
\log p(x) 
  &=\log p(x)\int{q_\phi(z|x)dz} \quad \text{To introduce modeled sampler. Global Integral of probability  is 1.}\\
  &=\int{\log p(x)q_\phi(z|x)dz} \quad \text{p(x) is function of x, resembling a constant for the integral about z.}\\
  &=\mathbb{E}_{q_\phi(z|x)}[\log p(x)] \quad \text{Definition of expectation.}\\
  &=\mathbb{E}_{q_\phi(z|x)}[\log \frac{p(z)p(x|z)}{p(z|x)}]\quad \text{Chain rule of probability.} \\
  &=\mathbb{E}_{q_\phi(z|x)}[\log p(z) + \log p(x|z) - \log p(z|x)]\quad \text{Split summation.}
\end{aligned}$$


The sampler \\(q_\phi(z|x)\\) can be seen as a conditional probability density function over \\(z\\) given \\(x\\).
So far it is still holding strict as the true \\(\log p(x)\\).
Intuitively, we would like to replace \\(p(z|x)\\) with our modeled \\(q(z|x)\\), which obviously brings the 'cost' in so doing: the deviation.
In this case, we need to have a **measure of  distance between two distributions**, which should be a nonzero number and should be within the range of 0 to 1 for two normalized distributions to be compared.


There comes a second _deus ex machina_ invocation: 
> KL divergence:

>$$
>D_{KL}(P||Q)=\mathbb{E}_P[\log \frac{P(x)}{Q(x)}]=\mathbb{E}_P[\log P(x) -\log Q(x)],
>$$


which is originally used to measure the difference between two distributions \\(P\\) and \\(Q\\) in the formula across all values of the concerned variables.
Its most important feature is that it is never goes negative, which provides quantitative relationship between the true and simulated  \\(\log p(x)\\).
This is ensured by logarithm's concavity (i.e. Jensen's inequality): 

$$
D_{KL}(P||Q)=\mathbb{E}_P[\log \frac{P}{Q}] \geq -\log \mathbb{E}_P[\frac{Q}{P}]=-\log \int P(x)\frac{Q(x)}{P(x)}dx =-\log\int Q(x)dx = -\log(1)=0
$$

thanks to  

$$\log⁡(\sum_i\lambda _ix_i)\geq\sum_i\lambda_i\log⁡(x_i).$$

For a convex combination \\(\sum_i\lambda_i=1\\) and \\(\lambda_i\geq0\\), and the equality of \\(D_{KL}(P||Q)\\) holds only if \\(P=Q\\).
It can be understood by thinking of the secant line lies below the curve for logarithm function.

In fact, 

$$D_{KL}(P || Q)=H(P,Q)-H(P).$$

like  the gap between \\(H(P,Q)\\) vs. \\(H(P)\\).
And, given the unabandoned \\(\log\\), we come to notice that there is a asymmetry in \\(D_{KL}\\): 

$$D_{KL}(P||Q) \neq D_{KL}(Q||P)$$

So, \\(D_{KL}(P\|\|Q)\\) ignores regions where \\(Q(x)>P(x)\\) if \\(P(x)\approx 0\\), due to the weight \\(P\\).

Observe the simulated  \\(\log p(x)\\) again, we find that the sampler under expectation notation can form a \\(D_{KL}\\) with  \\(p(z\|x)\\).

$$
\begin{aligned}
  & \mathbb{E}_{q_\phi(z|x)}[\log p(z) + \log p(x|z) - \log p(z|x)] \\
  &=\mathbb{E}_{q_\phi(z|x)}[\log p(z)  \underbrace{- \log q_\phi(z|x) + \log q_\phi(z|x)}_{=0} + \log p(x|z) - \log p(z|x)] \\
  &=\mathbb{E}_{q_\phi(z|x)}[\log p(z) - \log q_\phi(z|x) + \log p(x|z) + (\log q_\phi(z|x)  - \log p(z|x))] \quad\text{Switch summation order.}\\
  &=\mathbb{E}_{q_\phi(z|x)}[\log p(z) - \log q_\phi(z|x) + \log p(x|z) ] + \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log q_\phi(z|x)  - \log p(z|x)]}_{i.e. D_{KL}(q_\phi(z|x) || p(x|z))\geq 0.} \\
  &\geq \mathbb{E}_{q_\phi(z|x)}[\log p(z) - \log q_\phi(z|x) + \log p(x|z) ] 
\equiv \mathbb{E}_{q_\phi(z|x)}[\log \frac{p(x,z)}{q_\phi(z|x)}]
\end{aligned}
$$

The last line of the equations is the expression of the so-called Evidence Lower Bound (ELBO).
Most literature online prefers the form in tight fraction.
The name Evidence comes again from the chain rule of probability:

$$\underbrace{p(x)}_{\text{evidence}}
= \frac{
    \underbrace{p(x, z)}_{\text{joint probability}}
  }{
    \underbrace{p(z|x)}_{\text{posterior}}
  }.$$

We have developed  the approaches of estimating the probability of the observed data and of comparing them with the true probability distribution in this framework all along.
And 

$$\log p(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p(z) - \log q_\phi(z|x) + \log p(x|z) ] \equiv \mathbb{E}_{q_\phi(z|x)}[\log \frac{p(x,z)}{q_\phi(z|x)}]$$

gives the name  to  Lower Bound.


We now use ELBO to be the measure so that the corresponding value of observing those \\(x\\) in modeled distributions can be lower than that in the true distribution just because the modeled distribution is not close to the true distribution.


The inequality about ELBO holds with a true \\(p(x|z)\\) being known, which is actually never the case in real life.
A straightforward method to get the true distribution is to guess all the possible distribution to see which one provides the highest ELBO, but this is surely unfeasible.
**ELBO is now almost completely practicable**, except for the numerator \\(p(x,z)\\) that can be further decomposed as \\(p(z)p_\theta(x|z)\\) where the \\(_\theta\\) indicates the modeling effort in decoder.
However, it should be noted that the decomposition brings in approximation error via the modeling \\(\theta\\), so we have in effect:

$$\mathbb{E}_{q_\phi(z|x)}[\log \frac{p(x,z)}{q_\phi(z|x)}] \sim \mathbb{E}_{q_\phi(z|x)}[\log \frac{p(z)p_\theta(x|z)}{q_\phi(z|x)}]$$

I put \\(\sim\\) there instead of equal sign.


----


In summary, people explore the hypothesizing structure of latents within the encoder-decoder methodology, hoping that by poking around the unknown universe of the mechanism of the true distribution of the observed data \\(x\\), some opportunities of improving tractability can be created in terms of modeling.
That is what we will discuss in the next chapter.
After all, we are relieved to know that our focus has by far transferred from the incomputable \\(\log p(x)\\) to the promising proxy objective ELBO now.

