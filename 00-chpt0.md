---
layout: home
title: Chapter 0 - Preface
nav_order: 2
---

<head>
  <title>A Preliminary Mathematical Exegesis of Diffusion Models</title>
  <meta name="description" content="An in-depth mathematical analysis of diffusion models in machine learning.">
  <meta name="keywords" content="diffusion models, mathematics, machine learning, exegesis">
  <meta name="author" content="Shuyue Wang">
  <!-- Open Graph (for social sharing) -->
  <meta property="og:title" content="A Preliminary Mathematical Exegesis of Diffusion Models">
  <meta property="og:description" content="An in-depth mathematical analysis of diffusion models in machine learning.">
  <meta property="og:url" content="https://shuyuew1991.github.io/A-Preliminary-Mathematical-Exegesis-of-Diffusion-Models/">
  <meta property="og:type" content="website">
</head>

<div style="text-align: left; font-size: 1.3em;">
Chapter 0 - Preface
</div>

Generative AI has transformed countless aspects of our world. 
Among its many techniques, the diffusion model stands out as a groundbreaking framework—powering innovations like Stable Diffusion, whose magic captivates anyone (myself included) who remembers a time before the AI revolution.


Yet, for all its dazzling applications, the inner workings of diffusion models often elude clear understanding. 
For learners like me, most tutorials and code snippets in blogs and papers provide only fragmented insights—I had to piece together scattered, incomplete explanations by myself.
Even seemingly thorough articles often skims over core principles, leaving hurried exposition, sometimes with an unspoken assumption that they’re 'obvious.'
At times, the relentless onslaught of mathematical tricks might lure you into nodding along—"Sure, I follow"—only to realize, half an hour later, "But what does this actually mean?" Over time, these unresolved questions pile up, until frustration wins, and aspiring learners simply walk away.


I understand the frustration all too well.


Correct me if I was wrong, but I believe the fundamental idea behind diffusion models can be interpreted as this way: _like all generative AI, diffusion models attempt to aim to replicate what the nature offers—most often, images—in our own way._
(As for the term "diffusion", it derives from a specific component of its underlying mechanism, which will be examined in detail in Chapter 3 of this booklet.)


While preparing for a lecture on diffusion model and its SOTA applications at Emzan Technology Co. Ltd. for their department of [autoslide.cc](https://autoslide.cc/), I realized something: many of us needs a booklet on diffusion models that is mathematically rigorous yet accessible, one that strips away the noise and delivers the true, deep intuition behind the framework.


This booklet is my attempt to share what I’ve learned about diffusion models—both the math behind them and how they’ve evolved via logic.
I’ve tried to write it like something where ideas build naturally. 
The math is precise by my best, but I hope it feels more like following a story than reading a textbook.


The booklet’s theoretical progression relies on a few carefully chosen _deus ex machina_ elements—unavoidable but kept to a minimum.
The term _deus ex machina_ (Latin for "god from the machine") originates from ancient Greek theater, where an external intervention abruptly resolved a tangled plot. 
In this context, it refers to key assumptions or mathematical tools that enable deductions which might otherwise seem unmotivated.
Each such device will be explicitly introduced and justified:
- The use of Chebyshev inequality to bound the probability of high-dimensional data.
- The introduction of latent variables as a modeling component.
- The use of KL divergence to measure distances between distributions.
- The Itô SDE and Fokker-Planck equation that evolves the sampled data distribution in the long run.
- Central Limit Theorem, that provides a mathematically convincing path to the beautiful Gaussian distribution.

Only five of 'em!
By making these deliberate concessions explicit, the booklet ensures readers aren’t left puzzling over sudden leaps in reasoning.


The booklet is divided into four chapters:
- Chapter 1 dismantles the seemingly simple mantra _"maximize p(x)"_—an idea so deceptively complex that it demands an entire chapter.
- Chapter 2 offers a clear, concise derivation of ELBO.
- Chapter 3 unlocks the soul of diffusion model - the perspective of distribution transition that brings score into the game.
- Chapter 4 implements the learnt mathematics into robust, actionable practices.


For what it's worth, **don’t skim through this booklet**.
You won’t find rigid sections or numbered references within each chapter because our focus isn’t on compartmentalized facts, but on how the ideas connect. 
Nothing’s introduced without context, and no idea exists just for show.
And remember that this is not a cookbook with code snippets, but rather the mathematical bridge between ideas and codes.


As you explore this booklet, I hope you’ll feel the deep satisfaction of genuine intellectual engagement. 
I’ve written this with our community of practitioners in mind - learners who crave theoretical clarity. 
Every page is designed to equip you with knowledge that’s foundational to advancing our collective grasp of this field.
By the end, I hope you’ll grasp diffusion models so intuitively that you could explain them in your own words—not because you memorized the text, but because the ideas truly clicked. 
If I’ve done my job right, these concepts won’t feel like borrowed knowledge; they’ll feel like yours.

<br><br>

Далеко от Бишкека, Кыргызстан

First uploaded in June 2025.

