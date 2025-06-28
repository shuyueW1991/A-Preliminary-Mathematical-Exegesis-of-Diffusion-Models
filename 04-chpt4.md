---
layout: home
title: Chapter 4 - Implementation on machine - get our hands dirty
nav_order: 6
---

<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>

<div style="text-align: left; font-size: 1.3em;">
Chapter 4 - Implementation on machine - get our hands dirty.
</div>
<br>
<div style="text-align: center;">
  <img src="./assets/images/fig_ch04.png" style="width: 45%; max-width: 400px; height: auto; margin: 0 auto;">
</div>


Implementation is one of the most beautiful thing that you can experience on earth, because it is the last push from idea on your mind to reality.


In the last chapter, we came to know that above all, the training is for noise prediction in the discussion of equivalent loss functions.
This chapter is aimed to push mathematics to executable level.


We will offer the specific _forward_ distribution transition formula, i.e. the _noise-adding_ procedure, as well as the _reverse_ distribution transition formula, i.e. the _denoising_ procedure in this chapter and in subsequent appendices.


---


We start from

$$
\mu(x,t) = -\frac{1}{2}\beta(t),\quad
\sigma(x,t) = \sqrt{\beta}
$$

as solution to our Itô equation.
That they push the initial distribution towards a converging gaussian is validated in Chapter 3.
<!-- The discrete update rules led by that provides more control for algorithm, e.g. accelerate and de-accelerate in noise schedule. -->

Start with the SDE:

$$
dx_t = -\frac{1}{2} \beta(t) x_t \, dt + \sqrt{\beta(t)} \, dW_t
$$

For a small time step \\(\Delta t = 1\\), the discretization gives:

$$
x_t \approx x_{t-1} - \frac{1}{2} \beta(t) x_{t-1} + \sqrt{\beta(t)} \, (W_t - W_{t-1})
$$

Since \\(W_t - W_{t-1} \sim \mathcal{N}(0, \Delta t = 1)\\), let \\(z_{t-1} = W_t - W_{t-1} \sim \mathcal{N}(0, I)\\), we have:

$$
x_t \approx \left(1 - \frac{1}{2} \beta(t)\right) x_{t-1} + \sqrt{\beta(t)} \, z_{t-1}
$$

For small \\(\beta(t)\\), take the first-order Taylor expansion of \\(\sqrt{1 - \beta(t)}\\):

$$
\sqrt{1 - \beta(t)} \approx 1 - \frac{1}{2} \beta(t)
$$

Substitute this into above, we then have:

$$
x_t = \sqrt{1 - \beta(t)} \, x_{t-1} + \sqrt{\beta(t)} \, z_{t-1}
$$

as the forward process. 
Notably, we can have \\( \beta(t) \\) to be written as \\( \beta_t \\), just as a new denotation.
Thus:

$$
x_t = \sqrt{1 - \beta_t} \, x_{t-1} + \sqrt{\beta_t} \, \varepsilon_{t-1}, \quad \varepsilon_{t-1} \sim \mathcal{N}(0, I).
$$

The latter is also known as the noise schedule.


---


As we recall, the reverse SDE in Chapter 3  claims

$$
d\mathbf{x} = \left[ \mathbf{\mu}(\mathbf{x}, t) - \sigma(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] dt + \sigma(t) d\bar{\mathbf{w}}_t.
$$

We then put our solution of \\(\mu(x,t)\\) and \\(\sigma(x,t)\\) in it, and thus we have:

$$
d\mathbf{x} = \left[ -\frac{1}{2}\beta(t) - \beta(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] dt + \sigma(t) d\bar{\mathbf{w}}_t
$$

where the \\(p_t(\mathbf{x})\\) is similar to the  noise-perturbed distribution that corrupts data with a known noise distribution (e.g., Gaussian) in Chapter 3:

$$q_\sigma(\tilde{x}) = \int q_{data}(x) q_\sigma(\tilde{x}|x) dx$$

We should establish the expression of \\(x_t\\) on \\(x_0\\), because the latter matches \\(q_{data}(x)\\) in essence.
However, we only have the expression of \\(x_t\\) on \\(x_{t-1}\\): \\(x_t = \sqrt{1 - \beta_t} \, x_{t-1} + \sqrt{\beta_t} \, \varepsilon_{t-1} \\), so we accumulate that till \\(t=0\\), then we have 

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \varepsilon,
$$

where $$\alpha_t = 1 - \beta_t$$, $$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$, and $$\varepsilon \sim \mathcal{N}(0, I)$$.

Importantly, we can also rewrite it as 

$$
q(x_t | x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I \right),
$$

where $$\alpha_t = 1 - \beta_t$$, and $$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$.

Make analogy to our analysis to 

$$\tilde{x} = x + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

in Chapter 3, according to 

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \varepsilon,
$$

here, similar to 

$$\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x) = - \frac{\varepsilon}{\sigma}$$

there, we have

$$
\nabla_{x_t} \log p_t(x_t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

here.


However, we wouldn't like a formalistic analogy to be the foundation of our theory.
Let's pause a little bit and discuss more on this.


---


First, let's restate the two scenarios, first being 

$$\tilde{x} = x + \sigma \epsilon$$

and second being:

$$
     x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \varepsilon
     $$

In the first scenario, it does no harm to assume \\( \tilde{x} = x + \sigma \varepsilon \\), then:

$$
\tilde{x} | x \sim \mathcal{N}(x, \sigma^2)
$$

The probability density function is:

$$
q_\sigma(\tilde{x} | x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(\tilde{x} - x)^2}{2\sigma^2} \right)
$$

Taking the logarithm:

$$
\log q_\sigma(\tilde{x} | x) = -\frac{(\tilde{x} - x)^2}{2\sigma^2} - \log(\sqrt{2\pi}\sigma)
$$

Now, taking the gradient with respect to \\( \tilde{x} \\):

$$
\nabla_{\tilde{x}} \log q_\sigma(\tilde{x} | x) = -\frac{2(\tilde{x} - x)}{2\sigma^2} = -\frac{\tilde{x} - x}{\sigma^2}
$$

Now, we are so certain that taking the gradient in the second scenario would definitely be 

$$
\nabla_{\tilde{x}} \log q(\tilde{x} | x) = -\frac{2(\tilde{x} - \sqrt{\bar{\alpha}} \, x)}{2(1 - \bar{\alpha})} = -\frac{\tilde{x} - \sqrt{\bar{\alpha}} \, x}{1 - \bar{\alpha}} = -\frac{\varepsilon}{\sqrt{1 - \bar{\alpha}}}
$$

by solid analogy of Tweedie's formula.


---


Ok, now we go back to our case by substituting the score approximation:

$$
dx_t = \left[ -\frac{1}{2} \beta(t) x_t + \frac{\beta(t)}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right] dt + \sqrt{\beta(t)} \, d\bar{W}_t.
$$

For small steps (\\(\Delta t = 1\\)), the reverse update becomes:

$$
x_{t-1} = x_t - \left[ -\frac{1}{2} \beta_t x_t + \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right] + \sqrt{\beta_t} \, z, \quad z \sim \mathcal{N}(0, I).
$$

Simplifying it, we have:

$$
x_{t-1} = \left(1 + \frac{\beta_t}{2}\right) x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) + \sqrt{\beta_t} \, z, \quad z \sim \mathcal{N}(0, I).
$$


**By far, all basic theoretical knowledge  for diffusion models  is illustrated.**
Among most of the diffusion models, all variants use the same  loss 

$$\mathbb{E}[|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)|^2].$$

All methods implicitly use Tweedie's formula to estimate the clean data \\(\hat{\mathbf{x}}_0\\) from the noisy observation \\(\mathbf{x}_t\\), where the relationship 

$$\boldsymbol{\epsilon}_\theta = -\sqrt{1-\bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$$

connects denoising to score matching.


---


Below in appendices, we will discuss the real code repository for the famous Latent Diffusion Model and other models.
Our discussion will include model architecture and its core building blocks, the underlying design rationale and key implementation details, and the connections to other leading diffusion-based approaches, etc.
This chapter will remain open-ended by design, allowing for future expansions as new advancements emerge in this rapidly evolving field.
It'll keep growing, fo' sho'.



