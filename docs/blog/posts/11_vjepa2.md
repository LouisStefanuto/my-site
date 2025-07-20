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

In 2022, he wrote a position paper in which he introduces his vision of what a smarter AI could look like. **JEPA**[^1] - Joint Embedding Predictive Architecture - is his attempt in that direction. Developed at Meta FAIR, JEPA is a family of non-generative models, trained in a slightly different SSL manner, to learn world understanding and planning.

Two weeks ago (june 2025), Meta released a new model in the JEPA family: **V-JEPA 2**, which is the focus of this post.

!!! quote "Why did this new JEPA model catch my attention?"

We have been hearing of JEPA models for two years now. I never took time to deep dive the papers, because it seemed still early-stage. I wanted to wait a bit, for the JEPA ecosytem to mature, and see how the research community reacted to these fresh ideas ...

V-JEPA 2 caught my attention because **it is the first JEPA model with real-world applications**. Image-JEPA and V-JEPA were mostly experiment papers, to see how far they could go with this SSL approach. V-JEPA 2 is the continuation of these papers and showcase 2 super cool applications: 1️⃣ **conditioning an LLM** for video Question Answering and 2️⃣ zero-shot **robot control**.

Finally, pretraining models on large amounts of text or images showcased interesting emerging properties. I was curious to study what properties emerged when a new temporal dimension is added into the mix. Time is in fact needed to learn concepts like motion, gravity, planning ... Useful concepts and properties for a world model.

!!! note "Revolutionary approach, or simply a new flavor of the good old auto-encoders? Let's find out! 👇"

<figure markdown>
![video](https://video-cdg4-1.xx.fbcdn.net/o1/v/t2/f2/m69/AQN6DgLP8TuI-7z3giAEpJf-A15y2zeGoT3g6jsa3ibA6vZlEnUEfNJWpkGjGX9qQojwIIMABDmZpTdFVd9-zH1g.mp4?strext=1&_nc_cat=104&_nc_sid=5e9851&_nc_ht=video-cdg4-1.xx.fbcdn.net&_nc_ohc=BgCfX9EVQecQ7kNvwHH2H8q&efg=eyJ2ZW5jb2RlX3RhZyI6Inhwdl9wcm9ncmVzc2l2ZS5GQUNFQk9PSy4uQzMuMTI4MC5kYXNoX2gyNjQtYmFzaWMtZ2VuMl83MjBwIiwieHB2X2Fzc2V0X2lkIjo0MTM0NzIyNjIzNDc5NjI4LCJ2aV91c2VjYXNlX2lkIjoxMDYyNiwiZHVyYXRpb25fcyI6MTUsInVybGdlbl9zb3VyY2UiOiJ3d3cifQ%3D%3D&ccb=17-1&vs=b4f65e52e218632a&_nc_vs=HBkcFQIYOnBhc3N0aHJvdWdoX2V2ZXJzdG9yZS9HRDJHRng3dW9pMWdaZVFCQUp6Tnp1dEEzNHdXYnY0R0FBQUYVAALIARIAKAAYABsCiAd1c2Vfb2lsATEScHJvZ3Jlc3NpdmVfcmVjaXBlATEVAAAmmP28k8Sg2A4VAigCQzMsF0AuAAAAAAAAGBlkYXNoX2gyNjQtYmFzaWMtZ2VuMl83MjBwEQB1AmWEpgEA&_nc_zt=28&oh=00_AfR5F_Bb1KevslW_UUZ1I7306w5BGCmjVdDaWaVaulyk5Q&oe=6882E51A)
<figcaption>Video from Meta AI blog post.</figcaption>
</figure>

!!! success "Recommended lectures"

    Plenty of super talented writers already covered JEPA in depth. Their content is top-notch. If you have some time, I highly recommend reading these first, and then come back to this post:

    - [What is JEPA?](https://www.turingpost.com/p/jepa) by the Turing Post
    - [Deep Dive into Yann LeCun’s JEPA](https://rohitbandaru.github.io/blog/JEPA-Deep-Dive/) by Rohit Bandaru

    If your time is limited, no worries. Let me give you a dense summary of JEPA, Image-JEPA and Video-JEPA.

---

## JEPA in a nutshell

> 👉 To keep things simple, I will explain JEPA by introducing I-JEPA, its image application. We will generalize back to JEPA and other modalities later.

!!! note "What are the limitations of Large Image Models?"

Large Image models are mostly trained in the signal space (pixels), leading them to pay attention to irrelevant details during learning. For instance, most image embedding models are trained to reconstruct masked or blurred images (MAEs, iBOT). Yet, asking a model to reconstruct a pixel-perfect image is an impossible task, because some details are almost random (the reflections of light on water, the position of bit of grass). Instead we would prefer them to focus on learning meaningful **higher level representations** (embeddings/concepts).

Moreover, LLMs have no **planning capabilities**. Given an observation of an environment (an image), they are poor predictors of what is coming next.

And finally, most methods have a single model to encode images and predict the missing parts. This is a weakness. In fact, the model tries to reconstruct based on the information it has at its disposal. Sometimes information is missing, so it has no other choice than to make an assumption on the future, by choosing one of the possible continuation of the image. Take the example of a car reaching an intersection, turning left and right are both plausible continuations, and you can't guess which direction the car will take. To draw the continuation, you would have to make a choice. As a consequence, embeddings models tend to entangle these multiple possible futures when generating embeddings. We would rather prefer to split the responsibilities: one model for encoding what is known, and one model for predicting what is missing.

Doing so, we could first embed the known situation (the past), then sample possible actions (e.g. turning left/right) and then condition the second model to predict the embedding of the future based on the past + the sampled action.

!!! note "What is I-JEPA's solution?"

To build a powerful world model capable of abstraction, I-JEPA is trained in the latent space. If you are familiar with self-distillation techniques like DINOv2 (I wrote a [**dedicated post on it, check it out** 🌟](./09_dino_v2.md)), the model basically **learns to predict the embeddings of the masked tokens of an image**.

!!! quote ""
    <figure markdown>
    ![bench](./images/11/jepa.png){width=700}
    <figcaption>Image from Meta AI blog post.</figcaption>
    </figure>

I-JEPA is based on the the Vision Transformer (ViT) architecture and therefore processes images as sequences of patches. The model is made of three ViT submodules:

- **a context encoder** $f_\theta$: The masked image $x$, also called the context, is encoded using a context ViT encoder. The outputs are $s_x$.
- **a target encoder** $f_\theta$: The original image $y$ (the bottom dog) is encoded using a target ViT encoder. The outputs are $s_y$.

    > Usually, the target encoder is an EMA of the context encoder. Like in DINO. This stabilizes training and avoids collapse.

- **a predictor** $g_\phi$ that predicts the embeddings of the masked tokens of the image. Its outputs are $\hat{s_y}$.

!!! bug ""
    The regression loss is computed over the embeddings $\mathcal{L}(s_y, \hat{s_y})$.

---

As you can see the model respects the requirements we listed above in introduction:

- The models learn **abstract concepts**: they learn to reconstruct embeddings, not images. This encourages learning abstractions rather than focusing on details.
- The encoding logic (encoder) is **separated** from the reconstruction logic (predictor). The encoder focuses on compressing information of what is known, while the predictor's goal is to infer what is missing.

> For the moment, why the predictor is useful is maybe not straightforward, we'll see in Video-JEPA why the decoupling encoder/predictor is powerful for planning.

I-JEPA showcases impressive performance on representation learning image benchmarks. It outperforms iBOT and MAE (two famous training techniques), for a training budget around one order of magnitude smaller. Learning seems more sample-efficient with JEPA than with traditional methods.

!!! quote ""
    <figure markdown>
    ![bench](./images/11/ijepa-perf.png){width=400}
    <figcaption>Image from I-JEPA paper [2].</figcaption>
    </figure>

As we saw in previous posts, **evaluating an embedding model is ambiguous**. The best way to validate the quality of the embeddings is usually to evaluate the performance of simple classification models (like a logisitc regression) directly trained on its embeddings.

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
    - Encode both the masked and unmasked sequences.
    - Predict the embeddings of the masked tokens using the predictor. The predictor also gets the mask as input condition for guidance.
    - The loss is simply computed over the embeddings.

    <figure markdown>
    ![bench](./images/11/jepa-paper-diagram.png){width=600}
    <figcaption>Image Credit: Lecun's position paper.</figcaption>
    </figure>

## V-JEPA 2

> Now that you have the fundamentals of JEPA, let's jump direclty to V-JEPA 2, which is basically the same as V-JEPA 1[^3] but with more data and bigger ViT models.

V-JEPA 2[^4] is a billion-parameter model, pretrained on over a million hours of **video data**, using the JEPA's self-supervized training procedure.

Like its little brother I-JEPA, the video frames are patchified. Let $N$ the number of frames in a video. The video tensor $(N, H, W)$ is splitted into a sequence of $L$ tokens called **tubelets** where each tubelet is of shape $(2, 16, 16)$.

> The 2 means that each token is made of two consecutive frames. As usual in transformers, they incorporate the position of the patches through 3D ROPE positional embeddings.

!!! quote ""
    <figure markdown>
    ![vjepa](./images/11/vjepa2.png){width=400}
    <figcaption>Image from V-JEPA 2 paper [4].</figcaption>
    </figure>

The training procedure is similar to I-JEPA. The predictor is trained to recover the embedding of masked tubelets, from the embeddings of the unmasked tubelets. The teacher model is an EMA of the student, the stop-gradient operation blocks the gradient flow to the teacher. The masking is very aggressive, with around 90% of the pixels masked. This high masking ratio may be due to the fact that information in video is redundant, so you need/can mask much more to reduce leakage and force the model to learn rather than copying.

After training, we get a strong video patch embedder (encoder) and a versatile patch predictor.

!!! quote "More about the paper ..."
    👾 Github: <https://github.com/facebookresearch/vjepa2>

    📚 Arxiv: <https://arxiv.org/abs/2506.09985>

## Downstream applications

After SSL pre-training, V-JEPA 2 is essentially an embedding model. It is not useful out-of-the-box ... but it can easily become after post training.

The paper showcases multiple applications of V-JEPA 2. In fact (see diagram below), after a task-specific post-training (in green), one can evaluate its world Understanding, Prediction or Planning capabilities.

I will focus on the robotic application that I find super impressive. Check out the paper if you are interested by the other applications.

!!! quote ""
    <figure markdown>
    ![vjepa](./images/11/vjepa2-data.png){width=600}
    <figcaption>Image from V-JEPA 2 paper [4].</figcaption>
    </figure>

## V-JEPA 2-AC: Robotic control

> Let’s dive into the robotic application of V-JEPA 2, which I found particularly fascinating.

This experiment focuses on **zero-shot robotic planning for pick-and-place tasks using only image goals**. In simpler terms: you show the robot a picture of the goal (e.g. a ball inside a cup), and the model figures out by itself how to reach that state — **without any task-specific training, supervision, or reward function**.

<figure markdown>
![video](https://video-cdg4-1.xx.fbcdn.net/o1/v/t2/f2/m69/AQNACsVL3qCe48eGGe6HHokPD5LLldZ6rGFUCjmHg2ASRV4IT9aowjqr0ruYcUeUA4J_fOypR0Hhw1hcMZWJYjIS.mp4?strext=1&_nc_cat=102&_nc_sid=5e9851&_nc_ht=video-cdg4-1.xx.fbcdn.net&_nc_ohc=Xob21nQNmbcQ7kNvwEaf35p&efg=eyJ2ZW5jb2RlX3RhZyI6Inhwdl9wcm9ncmVzc2l2ZS5GQUNFQk9PSy4uQzMuMTkyMC5kYXNoX2gyNjQtYmFzaWMtZ2VuMl8xMDgwcCIsInhwdl9hc3NldF9pZCI6MTMzOTUzNjE5NDE5ODU0MCwidmlfdXNlY2FzZV9pZCI6MTA2MjYsImR1cmF0aW9uX3MiOjM0LCJ1cmxnZW5fc291cmNlIjoid3d3In0%3D&ccb=17-1&vs=189d57d89cbbec9f&_nc_vs=HBksFQIYOnBhc3N0aHJvdWdoX2V2ZXJzdG9yZS9HTW56Qmg2NGlzS1dVMlFFQUppQ3E4T1lvT2dOYnY0R0FBQUYVAALIARIAFQIYOnBhc3N0aHJvdWdoX2V2ZXJzdG9yZS9HTzlWRWg1SV9GY2JGOW9FQVA2ZlpyUVNlVnBYYnY0R0FBQUYVAgLIARIAKAAYABsCiAd1c2Vfb2lsATEScHJvZ3Jlc3NpdmVfcmVjaXBlATEVAAAmmPCmsKKT4QQVAigCQzMsF0BBTvnbItDlGBpkYXNoX2gyNjQtYmFzaWMtZ2VuMl8xMDgwcBEAdQJlhKYBAA&_nc_zt=28&oh=00_AfT8x5fYqHSJhmnQ66rKI2TenMddqKJzQQJCo2g5suOTmw&oe=6882DC64)
<figcaption>Video from Meta AI blog post.</figcaption>
</figure>

> Let's first see how we could use V-JEPA 2 to plan and control the robot. We'll then see how to post-train it to achieve our goals.

### Inference

Let’s say we are at time $t$ with a current image $x_t$ (e.g. the ball next to the cup), and we want the scene to become like $x_{\text{target}}$ (e.g. the ball inside the cup).

!!! quote ""
    1. **Encode the current and target frames** using the V-JEPA 2 encoder:
        - $s_t = f_\theta(x_t)$  
        - $s_{\text{target}} = f_\theta(x_{\text{target}})$

    2. **Sample a batch of candidate actions** (e.g. 100 random action trajectories like joint positions or movements).

    3. For each action $a_k$, use the **predictor** to simulate the future embedding:
        - $\hat{s}_{t+1}^{(k)} = g_\phi(s_t, a_k)$

    4. **Compute the distance** between the predicted embedding and the target one:
        - $d_k = \| \hat{s}_{t+1}^{(k)} - s_{\text{target}} \|$

    5. **Choose the trajectory** that minimizes this distance in N steps.

    6. Execute the first action of the best trajectory  $a^*$, observe the new frame, and repeat the process until convergence.

    > In practice, they sample actions using the Cross-Entropy Method (CEM). At each step, a distribution over actions is updated to concentrate on those that best match the target state. Only the first action of the best trajectory is executed before re-planning.

!!! quote ""
    <figure markdown>
    ![vjepa](./images/11/vjepa2ac-conditions.png){width=600}
    <figcaption>The goal is given via a target image. That's all!<br>Image from V-JEPA 2 paper [4].</figcaption>
    </figure>

!!! success ""
    What’s particularly interesting here is that **we don’t predict actions directly**. Instead, we **sample candidate actions** and use the **reconstruction loss as a proxy for energy** to evaluate them. The idea is simple but powerful: the better an action helps the model predict a future state close to the target, the lower its energy. This allows a **non-generative model to be used for planning**, by selecting actions that minimize prediction error—without ever explicitly generating the future.

### Training

!!! note "V-JEPA 2, as pretrained, is not immediately usable for robot control. Why?"

Because during its SSL pretraining, the predictor **never learns the causal impact of actions**. It only learns to predict the future frames of videos, not how actions modify the world.

To solve this:

- They **freeze the encoder** (trained on 1M hours of video).
- Then, they **post-train the predictor** on 62 hours of robot interaction data (from the DROID dataset), where each frame is paired with the robot's joint velocities and states.
- The updated predictor now takes as input:
    - an encoded state,
    - and a candidate action,
    - and predicts the embedding of the next frame.

!!! quote ""
    <figure markdown>
    ![vjepa](./images/11/vjepa2ac.png){width=600}
    <figcaption>Image from V-JEPA 2 paper [4].</figcaption>
    </figure>

!!! success "What makes this application stand out"

    > You end up with a **zero-shot robotic planning system** capable of solving pick-and-place tasks from image goals — without requiring dense rewards, task-specific demonstrations, or hand-labeled annotations.

    The only requirements are:

    - a pretrained V-JEPA 2 encoder (frozen),
    - a small dataset of robot interactions (videos + action labels),
    - and some clever inference using CEM.

    That’s **SSL with real-world impact**, and I find that pretty exciting.

!!! warning "Limitations"
    This SSL approach doesn't generalize to all robotic arms (yet?). This apporach still requires to collect videos and robot positions for the particular robot you want to control.

    This is cheaper than building a fully annotated dataset, but that remains a big limitation.

---

!!! note "Concluding remarks"
    That is all for the review of the JEPA model family.

    JEPA is an interesting approach that brings a bit of fresh air in the brute-force Transformer/Next-Token-Prediction wave. On my side, I am especially curious to see how the JEPA architectures will evolve over time to embrace the whole position paper of Yann Lecun. In fact, the current architectures only address understanding and planning. What about the configurator? ...

    I hope you enjoyed reading this technical deep dive. If so, feel free to share it and connect 😊

## References

[^1]: **JEPA Paper**: [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf). Version 0.9.2, 2022-06-27
[^2]: **I-JEPA Paper**: Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., ... & Ballas, N. (2023). [Self-supervised learning from images with a joint-embedding predictive architecture](https://arxiv.org/abs/2301.08243). In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 15619-15629).
[^3]: **V-JEPA Paper**: Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., ... & Ballas, N. (2024). [Revisiting feature prediction for learning visual representations from video](https://arxiv.org/abs/2404.08471). arXiv preprint arXiv:2404.08471.
[^4]: **V-JEPA 2 Paper**: Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Muckley, M., ... & Ballas, N. (2025). [V-JEPA 2: Self-Supervised Video Models Enable Understanding](https://arxiv.org/abs/2506.09985), Prediction and Planning. arXiv preprint arXiv:2506.09985.
