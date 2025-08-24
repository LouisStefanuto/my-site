---
date: 2025-08-19
authors:
  - LouisStefanuto
categories:
  - SSL
  - Computer Vision
  - Embedding
draft: true
---

# **Overfit#12:** DINOv3

[![dinov3](./images/12/main.jpg)](./12_dinov3.md)

<!-- more -->

## Motivation

Self-Supervized Learning is a promising avenue for training large text/image/video general embedding models. SSL leverages huge datasets of unannotated data leading to more general models with emerging properties.

NLP scientists showed that scaling data and compute leads to stronger and more powerful models. Yet in the image field, despite datasets getting larger and larger, scaling beyond the 1B-parameter frontier has not been as successful (yet).

In fact, many scaling attempts showed that global features keep improving with scaling (image-level representations), but that **local features tend to degrade with scaling**. In other words, scaling leads to better embedding models for image-level downstream tasks like classification, but it doesn't work as well for patch-level downstream tasks like segmentation (see the figures below).

<figure markdown>
![scaling issues](./images/12/scaling_issues.png){width=600}
<figcaption>Image from DINOv3 paper.</figcaption>
</figure>

<figure markdown>
![scaling issues](./images/12/issues.jpg){width=600}
<figcaption>(pink) Classification accuracy on ImageNet1K.<br>(purple) Segmentation performance on VOC.<br>Image from DINOv3 paper.</figcaption>
</figure>

!!! note
    In this post, I will focus on the additions of DINOv3[^1] and give special attention to **Graw Anchoring**, a simple regularization loss, that helps mitigating these issues. I already covered DINOv2[^2] in details in a past post. If you are interested, I recommend having a look: **[DINOv2 post 🦖🦖](./09_dino_v2.md)**.

---

## DINOv3 is DINOv2's extension

!!! success "Let's start by summarizing DINOv2 ..."

DINOv2 uses a discriminative training procedure. At training, two random crops are extracted from each image. Each crop is fed through a student/teacher model. The student goal is to minimize the distance between the two view embeddings, at image-level $\mathcal{L}_{DINO}$ and at patch-level $\mathcal{L}_{iBOT}$. The teacher is derived from the student using an EMA of its weights. DINOv2 also uses a Koleo regularizer $\mathcal{L}_{koleo}$ to encourage features within a batch to spread, and SwAV centering.

This gives rise to a multi-term pretraining loss:

$$
\mathcal{Loss} = \mathcal{L}_{DINO} + \alpha \mathcal{L}_{iBOT} + \beta \mathcal{L}_{koleo}
$$

!!! note "The Meta team reuses the exact same DINOv2 loss to pretrain DINOv3. But ..."

DINOv3 is essentially applying scaling laws to DINOv2: 10x more data (~100M images $\rightarrow$ ~1000M images), and 7x more parameters (1B $\rightarrow$ 7B). When scaling up DINOv2, the research team made two observations:

- At mid-training (~200k steps), **patch-level similarities are already well structured**.
  
    > At this stage, similar patches have similar embeddings. They may not be perfect embeddings that compress all patch information, but their similarity relationships are meaningful.

- As training progresses toward 1M steps, **global features continue to improve**, but **local similarities start to collapse**.  

    > At end-training, similar patches have **less similar embeddings**. The feature map has somehow degraded.

<figure markdown>
![scaling issues](./images/12/scaling_issues3.png){width=600}
<figcaption>Image from DINOv3 paper.</figcaption>
</figure>

$\rightarrow$ What we need is a way to enforce similar patches to keep their embeddings similar.

!!! quote "But we have no labels... so how can we do that?"

## Gram Anchoring

In DINOv3, Meta introduces **Gram Anchoring**, a regularization technique to enforce feature maps to remain smooth, even when scaling up model size. As we'll see, Gram Anchoring is surprisingly simple. To explain it, let’s first introduce the **Gram Matrix**.

### Gram Matrix

Let $A$ be an image. Let's divide it into $N$ patches (like in Vision Transformers) and let's compute their embeddings $v_1, v_2, \dots, v_N$.

The **Gram matrix** $G \in \mathbb{R}^{N \times N}$ is defined as:

$$
G_{ij} = \langle v_i, v_j \rangle
$$  

This matrix captures the **relative similarities between patches**, independent of absolute positions.

<figure markdown>
![gram matrix](https://upload.wikimedia.org/wikipedia/commons/0/00/Gram_matrix.svg){width=400}
<figcaption>Gram Matrix.<br>Image from Wikipedia.</figcaption>
</figure>

### Intuition

!!! quote "Why is this Gram Matrix useful?"

At ~200k steps, **patch similarities are at their best**: they reflect meaningful relationships between parts of an image. As training continues, **global representations get stronger** (good for classification), but **patch-level consistency drifts** (bad for segmentation). Without constraints, the student forgets these useful local relations, even though they were already learned earlier.

So the idea is simple:

- Use the Gram matrix of an earlier checkpoint **as a reference**.  
- Regularize the student so that its patch similarity structure remains close to this anchor, even at later stages of training.  

This way, the student can continue improving features while preserving local consistency.

<figure markdown>
![scaling issues](./images/12/scaling_issues2.png){width=600}
<figcaption>Image from DINOv3 paper.</figcaption>
</figure>

### Gram Anchoring Loss

At a given step $t$, we store the teacher’s Gram matrix $G^{teacher}_t$ and enforce the student to maintain a similar patch similarity structure:

$$
\mathcal{L}_{gram} = \| G^{student}_t - G^{teacher}_t \|^2 = \left\| \mathbf{X}_S \cdot \mathbf{X}_S^{\top} - \mathbf{X}_G \cdot \mathbf{X}_G^{\top} \right\|^{2}
$$

This acts as an **anchor**: the student can freely learn new features, but **patch-level relations cannot drift too far**.  

In DINOv3, this loss is only added after 1M training steps, during what is called a post training phase. **Surprisingly the effect is immediate**: the model recovers local consistency in a few epochs, without any degradation in global features (orange).

Other techniques are combined on top of Gram Anchoring to boost performance on higher resolution images (green), while minimizing the training budget:

- **Training at higher resolution** (512, 768 instead of 256) helps learning how to embed larger images. It can be seen as context extension for LLMs.
- **Upscaling images and then interpolating Gram matrixes** gives smoother references for the regularization.
- **RoPE-box jittering**: basically adding noise in ROPE embeddings to get a more robust model at varying sizes

!!! quote ""
    <figure markdown>
    ![losses](./images/12/losses.png){width=700}
    </figure>

    <figure markdown>
    ![gram](./images/12/gram.png){width=700}
    <figcaption>Images from DINOv3 paper.</figcaption>
    </figure>

!!! success "Key takeaway"
    Gram Anchoring is a constraint that forces the student’s patch similarity to imitate that of a past checkpoint. The 1M-step student embeddings can evolve freely, but their **patch similarity structure remains close** to the reference, preserving local consistency.

## Model distillation

If you don't have the compute resources to run the 7B model, don't worry! Meta also released distilled ViT and ConvNeXT checkpoints in various model sizes (from 29M to 7B).

!!! quote ""
    <figure markdown>
    ![gram](./images/12/distilled_models.png){width=700}
    <figcaption>Image from DINOv3 paper.</figcaption>
    </figure>

An interesting addition of DINOv3 paper is their **multi-student distillation technique** to train these smaller models at a reduced cost. Usually student models are trained one at a time. In DINOv3, they distill multiple models **in parallel** to save calls to the teacher. This drastically reduces the compute cost. They further optimize the number of GPUs per student based on their size, to minimize GPU idle time.

!!! quote ""
    <figure markdown>
    ![distillation](./images/12/distillation.png){width=700}
    <figcaption>Image from DINOv3 paper.</figcaption>
    </figure>

## Text alignment

Aligning visual embedding models is a powerful technique to build CLIP-like models that benefit from pretrained image backbones. Using the LiT training paradigm[^3], they align their ViT-L model and evaluate it on global/local

## Benchmarking

I won't cover the benchmark, because it would be too long and because the paper does it better. In a nutshell they show that DINOv3 is SOTA in most benchmarks: classification, detection, segmentation, 3D tasks, ...

Yet it is important to mention the effort they put in benchmarking the 7B model and its distilled models.

The DINOv3 training recipe also shows strong performance on Geospatial Data and even sets the new state of the art on some benchmarks like SatLidar1M val, SatLidar1M test and Open-Canopy.

!!! quote ""
    <figure markdown>
    ![distillation](./images/12/benchmarks.png){width=700}
    <figcaption>Image from DINOv3 paper.</figcaption>
    </figure>

---

!!! note "Concluding and key takeaways"

    - DINOv3 is the result of scaling the DINOv2 recipe on 10x more data and compute.
    - Gram Anchoring appears as a simple yet efficient solution to the local embedding degradation observed in former scaling attempts. Its contrastive loss takes advantage of the high-quality early-stage local capabilities of the teacher.
    - Scaling resolution at the end of the training is a good tradeoff to boost model performance/usability while limiting training costs.
    - Meta's benchmarking work is seriously impressive.

## References

[^1]: Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., ... & Bojanowski, P. (2025). [DINOv3](https://arxiv.org/abs/2508.10104). arXiv preprint arXiv:2508.10104.
[^2]: **DINOv2**: Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., ... & Bojanowski, P. (2023). [Dinov2: Learning robust visual features without supervision](https://arxiv.org/pdf/2304.07193). arXiv preprint arXiv:2304.07193.
[^3]: **LiT**: Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas
Beyer. [LiT: Zero-shot transfer with locked-image text tuning](https://arxiv.org/abs/2111.07991). In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 18123–18133, 2022b.
