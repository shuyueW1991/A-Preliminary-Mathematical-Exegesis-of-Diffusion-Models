---
layout: home
title: Chapter 0 - Preface
nav_order: 2
---


<div style="text-align: left; font-size: 1.3em;">
Chapter 0 - Preface
</div>

Generative AI has transformed countless aspects of our world. 
Among its many techniques, the diffusion model stands out as a groundbreaking framework—powering innovations like Stable Diffusion, whose magic captivates anyone (myself included) who remembers a time before the AI revolution.


Yet, for all its dazzling applications, the inner workings of diffusion models often elude clear understanding. 
For learners like me, most tutorials and code snippets in blogs and papers provide only fragmented insights—forcing us to piece together scattered, incomplete explanations. 
Even seemingly thorough articles often rest on shallow foundations—skimming over core principles, whether from hurried exposition or an unspoken assumption that they’re 'obvious.'
Sometimes, relentless onslaught of mathematical tricks can leave you nodding along—"Sure, I follow"—only to later wonder, "But what does this actually mean to do?" 
Over time, these unresolved questions pile up, until the aspiring learners simply walk away.
I understand the frustration all too well.


The fundamental idea behind diffusion models is rarely stated outright in most tutorials: like all generative AI, diffusion models attempt to aim to replicate what the nature offers—most often, images—in our own way.
As for the term "diffusion", it derives from a specific component of its underlying mechanism, which will be examined in detail in Chapter III of this booklet.
Prior to delving into its technical foundations, we first establish the overarching objective shared by all generative AI paradigms: the development of models capable of synthesizing natural images. 
As the discussion progresses, the distinctive characteristics inherent to diffusion models will emerge.


While preparing for a lecture on diffusion model and their SOTA technology at Emzan Technology Co. for their department of [autoslide.cc](https://autoslide.cc/),  I realized something: many of us needs a booklet on diffusion models that is mathematically rigorous yet accessible, one that strips away the noise and delivers the true, deep intuition behind the framework.
This booklet is my attempt to share what I’ve learned about diffusion models—both the math behind them and how they’ve evolved via logic.
I’ve tried to write it like something where ideas build naturally. 
The math is precise by my best, but I hope it feels more like following a story than reading a textbook.
I’ve structured this booklet to guide readers naturally through diffusion models, like following a current. 
You won’t find rigid sections or numbered references because the focus isn’t on compartmentalized facts—it’s on how the ideas connect. 
Nothing’s introduced without context, and no idea exists just for show.
I will slow down where I once struggled, so the readers might have an easier time.


The booklet’s theoretical progression relies on a few carefully chosen _deus ex machina_ elements—unavoidable but kept to a minimum.
The term _deus ex machina_ (Latin for "god from the machine") originates from ancient Greek theater, where an external intervention abruptly resolved a tangled plot. 
In this context, it refers to key assumptions or mathematical tools that enable deductions which might otherwise seem unmotivated.
Each such device will be explicitly introduced and justified:
- The use of Chebyshev inequality to bound the dot product between high-dimensional data.
- The introduction of latent variables (denoted as *z*) as a modeling construct.
- The use of KL divergence to measure distributional distances, along with its mathematical properties and role in derivations.
- The Itô SDE and Fokker-Planck equation that evolves the sampled data distribution in the long run.
- Central Limit Theorem, that provides a mathematically convincing path to the beautiful Gaussian distribution.
By making these deliberate concessions explicit, the booklet ensures readers aren’t left puzzling over sudden leaps in reasoning.


The booklet is divided into four chapters:
- Chapter I dismantles the seemingly simple mantra _"maximize p(x)"_—an idea so deceptively complex that it demands an entire chapter.
- Chapter II offers a clear, concise derivation of the ELBO.
- Chapter III unlocks the soul of diffusion model, i.e. the perspective of distribution transition that brings score into the game.
- Chapter IV immediately implements mathematics into a robust, actionable practice.


As you explore this booklet, I hope you’ll feel the deep satisfaction of genuine intellectual engagement. 
I’ve written this with our community of practitioners in mind: learners who crave not just theoretical clarity, but practical mastery. 
Every page is designed to equip you with knowledge that’s both immediately useful in your work and foundational to advancing our collective grasp of this field.
By the end, I hope you’ll grasp diffusion models so intuitively that you could explain them in your own words—not because you memorized the text, but because the ideas truly clicked. 
If I’ve done my job right, these concepts won’t feel like borrowed knowledge; they’ll feel like yours.

<br><br>

At Бишкек, Кыргызстан

June, 2025

