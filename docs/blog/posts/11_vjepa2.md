---
date: 2025-06-25
authors:
  - LouisStefanuto
categories:
  - SSL
  - Computer Vision
  - Embedding
  - Video
  - Robotics
---

# **Overfit#11:** V-JEPA 2

[![v jepa 2](./images/11/main.jpg)](./11_vjepa2.md)

<!-- more -->

## Motivation

Yann Lecun claims that LLMs are a dead end on the path towards AGI. According to his claims, we need new architectures to better mimic the way we think, learn, understand, plan.

In 2022, he wrote a position paper in which he introduces his vision of what an intelligent AI should look like. **JEPA**[^1] - Joint Embedding Predictive Architecture - is his attempt in that direction. Developed at Meta FAIR, JEPA is a family of non-generative models, trained in a novel SSL manner, to learn understanding world and planning.

Two weeks ago (june 2025), Meta released a new model in the JEPA family: **V-JEPA 2**, which is the focus of this post.

!!! quote "Why did this new JEPA model catch my attention?"

We have been hearing of JEPA models for two years now. I never took time to deep dive the papers, because it seemed still early-stage to me. I wanted to wait a bit, for the JEPA ecosytem to mature, and see how the research community reacts to these fresh ideas ...

V-JEPA 2 caught my attention because **it is the first JEPA model with real-world applications**. Image-JEPA and V-JEPA were mostly experiment papers, to see how far they could go with this SSL approach. V-JEPA 2 is the continuation of these papers and showcase 2 super cool applications: 1️⃣ **conditioning an LLM** for video Question Answering and 2️⃣ zero-shot **robot control**.

Finally, pretraining models on large amounts of text or images showcased interesting emerging properties. I was curious to study what properties emerged when a new temporal dimension is added into the mix. Time is in fact needed to learn concepts like motion, gravity, planning ... Useful concepts and properties for a world model.

!!! note "Revolutionary approach, or simply a new flavor of the good old auto-encoders? Let's find out! 👇"

<figure markdown>
![bench](./images/11/jepa.png){width=500}
<figcaption>JEPA diagrams are scary at first sight. <br>But no worries it is not that complicated. <br>Image from Meta AI blog post.</figcaption>
</figure>

!!! success "Recommended lectures"

    Plenty of super talented writers already covered JEPA in depth. Their content is top-notch. If you have some time, I highly recommend reading these first, and then come back to this post:

    - [What is JEPA?](https://www.turingpost.com/p/jepa) by the Turing Post
    - [Deep Dive into Yann LeCun’s JEPA](https://rohitbandaru.github.io/blog/JEPA-Deep-Dive/) by Rohit Bandaru

    If your time is limited, no worries. Let me give you a dense summary of JEPA, Image-JEPA and Video-JEPA.

## JEPA in a nutshell

> 👉 To keep things simple, I will explain JEPA by talking about I-JEPA, its image application. We will generalize back to JEPA and other modalities later.

!!! note "What are the limitations of current LLMs?"

Large Image models are mostly trained in the signal space (pixels), leading them to pay attention to irrelevant details during learning. For instance, most image embedding models are trained to reconstruct masked or blurred images (MAEs). Yet, asking a model to reconstruct a pixel-perfect image is an impossible task, because some details are almost random (the reflections of light on water, the position of bit of grass). Instead we would prefer them to focus on learning meaningful **higher level representations** (embeddings/concepts).

Moreover, LLMs have no **planning capabilities**. Given an observation of an environment (an image), they are poor predictors of what is coming next.

And finally, most methods have a single model to encode images and predict the missing parts. This is a weakness. In fact, the model tries to reconstruct based on the information it has at its disposal. Sometimes information is missing, so it has no other choice that to make an assumption by choosing one of the possible continuation of the image. Take the example of a car reaching an intersection, turning left and right are both plausible continuations, and you can't geuss which direction the car will take. To draw the continuation, you would have to make a choice. As a consequence, embeddings models tend to take into account this part of unknown in their embeddings. We would rather prefer to split the responsibilities: one model for encoding what is known, and one model for predicting what is missing.

!!! note "What is I-JEPA's solution?"

To build a powerful world model capable of abstraction, I-JEPA is trained in the latent space. If you are familiar with self-distillation techniques like DINOv2 (I wrote a [**dedicated post on it, check it out** 🌟](./09_dino_v2.md)), the model basically **learns to predict the embeddings of the masked tokens of an image**.

!!! quote ""
    <figure markdown>
    ![bench](./images/11/jepa.png){width=700}
    <figcaption>Image from Meta AI blog post.</figcaption>
    </figure>

The model is made of three ViT submodules:

- **a context encoder** $f_\theta$: The masked image $x$, also called the context, is encoded using a context ViT encoder. The outputs are $s_x$.
- **a target encoder** $f_\theta$: The original image $y$ (the bottom dog) is encoded using a target ViT encoder. The outputs are $s_y$.

    > Usually, the target encoder is an EMA of the context encoder. Like in DINO. This stabilizes training and avoids collapse.

- **a predictor** $g_\phi$ that predicts the embeddings of the masked tokens of the image. Its outputs are $\hat{s_y}$.

The regression loss is computed over the embeddings $\mathcal{L}(s_y, \hat{s_y})$.

---

As you can see the model respects the requirements we listed above in introduction:

- The models learns abstract concepts: it learns to reconstruct embeddings, not images. This encourages learning abstractions rather than focusing on details.
- The encoding logic (encoder) is separated from the reconstruction logic (predictor). The encoder focuses on compressing information of what is known, while the predictor's goal is to infer what is missing.

> For the moment, why the predictor is useful is maybe not straightforward, we'll see in Video-JEPA why it is powerful for planning.

I-JEPA showcases impressive performance on representation learning image benchmarks. It outperforms iBOT and MAE (two famous training techniques), for a training budget around one order of magnitude smaller. Learning seems more sample-efficient with JEPA than with traditional methods.

!!! quote ""
    <figure markdown>
    ![bench](./images/11/ijepa-perf.png){width=400}
    <figcaption>Image from I-JEPA paper [2].</figcaption>
    </figure>

As we saw in previous posts, **evaluating an embedding model is ambiguous**. The best way to validate the quality of the embeddings is usually to evaluate the performance of simple classification models (like a logisitc regression) trained on the frozen embeddings.

For V-JEPA 2, the authors also wanted to check if the features learned were just abstract, or if they still contained enough information to recover the masked pixels. Thus, they trained a decoder model, to reconstruct the masked pixels, from the latent embeddings. They observed that the trained decoder (only for evaluation, right) was able to reconstruct the images, even if the embeddings were never trained to perform a reconstruction task. Of course the decoder has poor reconstruction metrics compared to models pretrained to reconstruct, but this showcases that the features learnt are in fact meaningful.

!!! quote ""
    <figure markdown>
    ![vjepa](./images/11/ijepa-eval.png){width=400}
    <figcaption>Evaluation examples. Generations from the embeddings of I-JEPA.<br>Image from I-JEPA paper [4].</figcaption>
    </figure>

!!! success "Generalization to JEPA"

    As we saw, the I-JEPA is specific to images. But it can easily be generalized to any type of inputs, as long as they can be splitted into (masked) tokens.

    - Take an input sequence.
    - Mask some tokens.
    - Encode the two masked and unmasked sequences.
    - Predict the embeddings of the masked tokens using the predictor.

    <figure markdown>
    ![bench](./images/11/jepa-paper-diagram.png){width=600}
    <figcaption>Image Credit: Lecun's position paper.</figcaption>
    </figure>

## V-JEPA 2

> Now that you have the fundamentals of JEPA, let's jump direclty to V-JEPA 2, which is basically the same as V-JEPA 1[^3] but with more data and bigger ViT models.

V-JEPA 2[^4] is a world model trained on over a million hours of video data, with the JEPA architecture.

Like its little brother I-JEPA, the video frames are patchified. Each video tensor (Height, Width, Time), is splitted into a sequence of $L$ tokens called **tubelets** where each tubelet is of shape $(16, 16, 2)$.

> The 2 means that each token is made of two consecutive frames. As usual in transformers, they incorporate the position of the patches through 3D ROPE positional embeddings.

!!! quote ""
    <figure markdown>
    ![vjepa](./images/11/vjepa2.png){width=400}
    <figcaption>Image from V-JEPA 2 paper [4].</figcaption>
    </figure>

The training procedure is similar to I-JEPA. The predictor is trained to recover the embedding of masked tubelets, from the embeddings of the unmasked tubelets. The teacher model is an EMA of the student, the stop-gradient operation blocks the gradient flow to the teacher. The masking is very aggressive, with around 90% of the pixels masked. This high masking ratio may be due to the fact that information in video is redundant, so you need/can mask much more to reduce leakage and force the model to learn rather than copying.

After training, w ethus have an encoder and a predictor.

!!! quote "More about the paper ..."
    👾 Github: <https://github.com/facebookresearch/vjepa2>

    📚 Arxiv: <https://arxiv.org/abs/2506.09985>

## Dowstream applications

- Downstream applications
    - aligning with a LLM for video Question Answering (QA)
        - Used to condition an LLM, as an image encoder. Requires aligning the image model with the text model.
    - Robot control
        - small amount of robot data (62 hours)
        - zero-shot robotic planning for picking and placing objects using image goals, without task-specific data, training, or rewards
- images are patchified into **tubelets** ()

!!! quote ""
    <figure markdown>
    ![vjepa](./images/11/vjepa2-data.png){width=400}
    <figcaption>Image from V-JEPA 2 paper [4].</figcaption>
    </figure>

### VLM alignement

### Robotic control

## Unanswered questions

- Attentive Probes?

---

!!! note "Concluding remarks"
    That is all for the review of the JEPA model family.

    I hope you enjoyed reading this technical deep dive. If so, feel free to share it and connect 😊

## References

[^1]: **JEPA Paper**: [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf). Version 0.9.2, 2022-06-27
[^2]: **I-JEPA Paper**: Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., ... & Ballas, N. (2023). [Self-supervised learning from images with a joint-embedding predictive architecture](https://arxiv.org/abs/2301.08243). In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 15619-15629).
[^3]: **V-JEPA Paper**: Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., ... & Ballas, N. (2024). [Revisiting feature prediction for learning visual representations from video](https://arxiv.org/abs/2404.08471). arXiv preprint arXiv:2404.08471.
[^4]: **V-JEPA 2 Paper**: Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Muckley, M., ... & Ballas, N. (2025). [V-JEPA 2: Self-Supervised Video Models Enable Understanding](https://arxiv.org/abs/2506.09985), Prediction and Planning. arXiv preprint arXiv:2506.09985.
