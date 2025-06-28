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

In previous chapter, we have talked about finding the best modeled distribution via evaluating the likelihood \\(p(x)\\)  of observed or generated data, because that measures how well they explain the observation:  higher likelihood suggests a better model.
Question now is: how to maximize the \\(p(x)\\), i.e. optimize \\(p(x)\\) to achieve maximal likelihood?

In this chapter, to get this game going, we will introduce two _deus ex machina_ magics.
The first one in this chapter is
> the latent variables, denoted as \\(z\\).

The incorporation of latent variables in generative models represents a powerful paradigm grounded in critical insights about data structure. 
Real-world datasets often exhibit complex dependencies of underlying, unobserved factors.
For instance, image data contains implicit attributes like illumination geometry and object orientation that are not explicitly encoded in pixel values but significantly influence the observed patterns. 
By explicitly representing these hidden factors, latent variables enable models to capture richer data structure.
Without latent variables, we would face the formidable challenge of modeling complex high-dimensional distributions directly - such as capturing intricate pixel-level correlations in images - which is both computationally intractable and statistically inefficient. 
Latent variables decompose the problem into more manageable components: a simple latent space distribution \\(p(z)\\) and a conditional data distribution \\(p(x|z)\\), where \\(z\\) represents the latent factors and \\(x\\) the observed data. 
This framework offers a structured, interpretable, and scalable approach—transforming an intractable problem into one where hidden factors systematically explain observed phenomena.

In fact, the metaphor of Plato’s Cave—from _The Republic_—provides a powerful analogy for understanding latent variables in generative models.
In Plato’s allegory, prisoners are chained in a cave, seeing only shadows cast on a wall by objects they cannot directly observe. 
This mirrors the relationship between observed data and latent variables.
The shadows are like our raw data (e.g., pixel values in images), mere surface-level projections.
The true forms are the latent variables—the unobserved, higher-dimensional factors (e.g., lighting, pose, or semantic meaning) that _generate_ the data, just as the objects outside the cave cast the shadows.
In both cases, reality is richer than what we directly perceive. 
Latent variables act as the hidden causes behind the observable effects, allowing models to infer the underlying structure that shapes the data.


---


We now try maximizing \\(p(x)\\) by utilizing latents \\(z\\).
We're here to find the distribution in which the observed data \\(x_1, x_2, ..., x_n\\) would be highest in probability compared with other distributions.
In modeling terms, this translates to finding a configuration of model parameters  such that the observed data have the highest  probability compared with other configuration.
Keep in mind that it has never been a problem of probability value arithmetic, as is suggested by its denotation format; it is a distribution (i.e. the best \\(p\\), in whichever form it takes, that we are actually looking for).

One way of linking latents \\(z\\) to \\(p(x)\\) is to think about: 

$$p(x)=\int{p_\theta(x,z)dz}=\int{p_\theta(x|z)p(z)dz}$$

where \\(p_\theta(x,z)\\) refers to a new modeled (indicated by \\(_\theta\\)) probability distribution (indicated by \\(p\\)) with new set of params (indicated by \\(x,z\\) ).  
The decomposition of integrand is directly from the chain of rule in probability.
\\(p(x|z)\\), a.k.a. decoder, describes how observations \\(x\\) are generated from latents \\(z\\); and \\(p(z|x)\\), a.k.a. encoder, describes how latents can be inferred from observations.
Note that the equation holds regardless of whether \\(x\\) and \\(z\\) are independent: independence would imply \\(p(x,z)=p(x)p(z)\\), but this is not necessary for the marginalization to be valid.

With this, one might propose the procedures of estimate \\(p(x\\)): 
1. sample  \\(z\\) from  \\(p(z)\\) 
2. for a given \\(\theta\\), calculate \\(p_\theta(x\|z)\\) , i.e. the probability of observing the data \\(x\\) we have observed given a sampled \\(z\\).
3. do this with sufficient times of  \\(z\\) sampling, thus get to compute the integral.
4. Therefore, by adjusting the model param \\(\theta\\), the integral gets bigger or smaller correspondingly, until the \\(\theta\\) corresponds to one high value of \\(p(x)\\) is found.


However, the intractability of the integration a major challenge itself.
\\(z\\) is typically a high-dimensional vector. 
The integral is over all possible values of \\(z\\), which is computationally infeasible for even moderate size. 
Example: if \\(z\\) has 100 dimensions and we discretize each dimension into just 10 values, the number of terms in the sum grows as much as \\(10^{100}\\).
The decoder \\(p_\theta(x|z)\\) , like other mathematical things in machine learning that is beyond explicit expression,  is usually modeled by a neural network. 
To such a complex nonlinear function,  there’s no analytical formula for integrating it over \\(z\\), even if \\(p(z)\\) is as simple as Gaussian.
Suppose we repeat  $$N$$ times to yield the approximation:

$$p_{\boldsymbol{\theta}}(x) = \int p_{\theta}(x|z)p(z)dz \approx \frac{1}{N}\sum_{i=1}^N p_{\theta}(x|z^{(i)})$$

This approximation converges to the true expectation as $$N \to \infty$$ by the law of large numbers. 
We optimize parameters $$\theta$$ to maximize $$p_{\theta}(x)$$, thereby improving the model's fit to the observed data while maintaining the learned low-dimensional structure.
However, it takes very large number to await the valid $$N$$ to come.


In addition, most \\(z\\) samples obtained in this approach will contribute negligibly to \\(p_\theta(x)\\), since \\(p(z)\\) is uninformed about \\(x\\), so most \\(z\\) values will lead to \\(p_\theta(x∣z)≈0\\), just consider how much `natural`images occupies the whole space.
Blindly sampling  \\(z \sim p(z)\\) provides little guidance to the decoder about which regions of \\(z\\)-space are relevant for generating meaningful \\(x\\).


----


A second approach of linking latents \\(z\\) to \\(p(x)\\)  is through chain rule of probability:  

$$p(x)=\frac{p(x,z)}{p(z|x)}$$

Since the latents  \\(z\\) are in both numerator and denominators, no direct observations can be made.
Given our  goal is to maximize \\(p(x)\\), we're here faced with the problem of maximizing  two unknown functions distributions simultaneously.

The denominator can be modeled as encoder network,  and it can be denoted as \\(p_\phi(z|x)\\).
It is quite instinctive to imagine a mapping from observed data \\(x\\) towards latents \\(z\\).
But in many literature, the \\(p_\phi(z|x)\\) here is written as \\(q_\phi(z|x)\\), a notation that emphasizes it's an _approximation_ of the true posterior \\(p(z|x)\\).
Modeling the numerator  \\(p_\theta(x,z)\\) might seem conceptually straightforward: we could simply introduce another parameterized distribution. But in practice, designing a network that simultaneously takes both observed data xx and latent variables \\(z\\) as inputs presents significant architectural challenges, making this approach less intuitive to implement.
We consider use chain rule of probability again to decompose \\(p_\theta(x,z)\\) into \\(p(z)p_\theta(x|z)\\).
Analogously, the \\(p_\theta(x|z)\\) can be modeled as a decoder network.

Now there's the term of \\(p(z)\\) left.
For the time being, we want the distribution to be of as much entropy as possible with fixed mean and variance.
We're now proving that with the definition of Shannon's entropy, if there is  constraints of given fixed mean 

$$\mu = E[X]$$

and fixed variance 

$$\sigma^2 = E[(X-\mu)^2],$$

then it's Gaussian distribution that maximizes \\(H(X)\\):  

$$p^*(x) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

The problem is essentially another constraint optimization problem to maximize differential entropy:  

$$h(X) = -\int_{-\infty}^{\infty} p(x) \log p(x) \, dx$$

subject to:  
 
$$\int_{-\infty}^{\infty}p(x)dx = 1 \quad\text{(Normalization)},$$

$$\int_{-\infty}^{\infty} x p(x)dx = \mu \quad \text{(Mean)},$$

and 

$$\int_{-\infty}^{\infty} (x-\mu)^2 p(x)dx = \sigma^2 \quad \text{(Variance)}$$

Now the Lagrangian functional is:  



$$\begin{align*}
  \mathcal{L}[p] = &- \int p(x) \log p(x) \, dx \\
  &+ \lambda_1 \left( \int p(x) \, dx - 1 \right) \\
  &+ \lambda_2 \left( \int x p(x) \, dx - \mu \right) \\
  &+ \lambda_3 \left( \int (x-\mu)^2 p(x) \, dx - \sigma^2 \right)
  \end{align*} $$



As for a general functional of the form 

$$
F[p] = \int f(x, p(x), p'(x)) \, dx，
$$

the functional derivative is given by: 

$$
\frac{\delta F}{\delta p(y)} = \frac{\partial f}{\partial p}\bigg|_{x=y} - \frac{d}{dx}\left(\frac{\partial f}{\partial p'}\right)\bigg|_{x=y}.
$$

Term by term, we calculate:

$$\frac{\delta}{\delta p}\left(-\int p(x)\log p(x) \, dx\right) = -\left(\log p(x) + p(x)\cdot\frac{1}{p(x)}\right) = -\log p(x) - 1,$$

$$\frac{\delta}{\delta p}\left(\lambda_1 \int p(x) \, dx\right) = \lambda_1,$$

$$\frac{\delta}{\delta p}\left(\lambda_2 \int x p(x) \, dx\right) = \lambda_2 x ,$$

and 

$$\frac{\delta}{\delta p}\left(\lambda_3 \int (x-\mu)^2 p(x) \, dx\right) = \lambda_3 (x-\mu)^2.$$

Combining all terms, and setting it to zero:

$$\begin{align*}
 \frac{\delta\mathcal{L}}{\delta p} = -\log p(x) - 1 + \lambda_1 + \lambda_2 x + \lambda_3 (x-\mu)^2 &= 0 \\
 \implies 
 \log p(x) &= -1 + \lambda_1 + \lambda_2 x + \lambda_3 (x-\mu)^2 \\
  p(x) &= \exp\left(-1 + \lambda_1 + \lambda_2 x + \lambda_3 (x-\mu)^2\right)
\end{align*}$$

The exponential form can now be rewritten as:

$$p(x) = e^{\lambda_1 - 1} \cdot e^{\lambda_2 x} \cdot e^{\lambda_3 (x-\mu)^2} $$

After completing the square and enforcing the constraints, we find:

$$
\begin{align*}
\lambda_3 &= -\frac{1}{2\sigma^2} \\
\lambda_2 &= \frac{\mu}{\sigma^2} \\
 e^{\lambda_1 - 1} &= \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{\mu^2}{2\sigma^2}}
\end{align*}$$

Substituting these back gives the Gaussian distribution:

$$ p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

 Among all distributions with a given mean and variance, the Gaussian has the highest entropy (i.e., it makes the fewest assumptions). 
This makes it a natural default choice when no additional structure is assumed.
That dictates our prior to be 

$$p(z) \sim \mathcal{N}(\mu, \sigma^2).$$

To wrap up our second approach, the new way to estimate \\(p(x)\\) can be like this: 
1. sample one \\(z_i\\) (\\(i\\) being arbitrary integer as sampling index) from the variational posterior \\(q_\phi(z\|x)\\) by inputting one observed data \\(x_i\\) into the network. 
2. one data point in true data dataset can be sampled more than one time.
3. with the \\(z_i\\), evaluate \\(p(x)\\) as \\(\frac{p(z_i)p_\theta(x\|z_i)}{q_\phi(z_i\|x)}\\).

An expectation (even approximated with a few samples) gives a smoother, more stable gradient, because averaging over multiple samples would reduce variance.
So let there  be  multiple samples instead of only one, and average the sum of their values in the final \\(p(x)\\)'s, which is basically the idea of Monte Carlo estimating.


----


Another concern is from \\(q_\phi(z|x)\\): like all denominators, it is  a headache for numerical computation. 
It may lead to high variance if that is a poor approximation of the true encoder, which is a  source of numerical instability, notably when \\(q_\phi(z|x) \ll p(z)\\). 
In such circumstance,  gradients during backpropagation can become extremely large (since gradients are inversely proportional to the denominator): e.g. for \\(f(x)=\frac{1}{x}\\), the gradient with respect to \\(x\\) is: \\(\frac{\partial f}{\partial x}=−\frac{a}{x^2}​\\).

One solution to solve the  problem is to transform \\(p(x)\\) into \\(\log p(x)\\). 
It won't  affect the finding of the right probability because the monotonicity of logarithm, and it moves denominator to the right of minus sign.
Numerically speaking, additivity is better than multiplictivity.
Thus, we have now 

$$\log p(x)\simeq\mathbb{E}_{q_\phi(z|x)}[\log p(z) + \log p_\theta(x|z) - \log q_\phi(z|x)].$$

However, it should be noted that with the introduction of \\(\theta\\) and \\(\phi\\) as the modeling effort, the equation of \\(p(x)\\) doesn't hold strictly.
We should still depart from the strictly-holing chain rule of probability and see what's the relationship between them two formulae: 

$$\begin{aligned}
\log p(x) 
  &=\log p(x)\int{q_\phi(z|x)dz} \quad \text{Introduce modeled sampler. Global Integral of probability  is 1.}\\
  &=\int{\log p(x)q_\phi(z|x)dz} \quad p(x)\text{functions on x, resembling a constant for the integral about z.}\\
  &=\mathbb{E}_{q_\phi(z|x)}[\log p(x)] \quad \text{Definition of expectation.}\\
  &=\mathbb{E}_{q_\phi(z|x)}[\log \frac{p(z)p(x|z)}{p(z|x)}]\quad \text{Chain rule of probability.} \\
  &=\mathbb{E}_{q_\phi(z|x)}[\log p(z) + \log p(x|z) - \log p(z|x)]\quad \text{Split summation.}
\end{aligned}$$

You can see each one of deduction is simple up there.
The sampler \\(q_\phi(z|x)\\) can be seen as a conditional probability density function over \\(z\\) given \\(x\\).
So far it is still holding strict as the true \\(\log p(x)\\).
Intuitively, we would like to replace \\(p(z|x)\\) with our modeled \\(q(z|x)\\), which obviously brings the 'cost' in so doing: the deviation.
In this case, we need to have a measure of calculating the distance between two distributions.
The measuring result should be a nonzero number, and should be within the range of 0 to 1 for two normalized distributions to be compared.


There comes a second _deus ex machina_ invocation: 
> KL divergence,  taking form of 
$$D_{KL}(P||Q)=\mathbb{E}_P[\log \frac{P(x)}{Q(x)}]=\mathbb{E}_P[\log P(x) -\log Q(x)].$$

which is originally used to measure the difference between two distributions \\(P\\) and \\(Q\\) in the formula across all values of the concerned variables.
Its most important feature is it is never goes negative, which provides quantitative relationship between the true and simulated  \\(\log p(x)\\).
This is ensured by logarithm's concavity (i.e. Jensen's inequality): 

$$D_{KL}(P||Q)=\mathbb{E}_P[\log \frac{P}{Q}] \geq -\log \mathbb{E}_P[\frac{Q}{P}]=-\log \int P(x)\frac{Q(x)}{P(x)}dx =-\log\int Q(x)dx = -\log(1)=0$$

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
ELBO is now completely practicable, except for the numerator \\(p(x,z)\\) that can be further decomposed as \\(p(z)p_\theta(x|z)\\) where the \\(_\theta\\) indicates the modeling effort in decoder.
However, it should be noted that the decomposition brings in approximation error via the modeling \\(\theta\\), so we have in effect:

$$\mathbb{E}_{q_\phi(z|x)}[\log \frac{p(x,z)}{q_\phi(z|x)}] \sim \mathbb{E}_{q_\phi(z|x)}[\log \frac{p(z)p_\theta(x|z)}{q_\phi(z|x)}]$$

I put \\(\sim\\) there instead of equal sign.


----


People explore the hypothesizing structure of latents within the encoder-decoder methodology, hoping that by poking around the unknown universe of the mechanism of the true distribution of the observed data \\(x\\), some opportunities of improving tractability can be created in terms of modeling.
That is what we will discuss in the next chapter.
After all, we are relieved to know that our focus has by far transferred from the incomputable \\(\log p(x)\\) to the promising proxy objective ELBO now.

