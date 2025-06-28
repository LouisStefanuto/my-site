---
date: 2025-06-25
authors:
  - LouisStefanuto
categories:
  - SSL
---

# **Overfit#11:** V-JEPA 2

[![v jepa 2](./images/11/main.jpg)](./11_vjepa2.md)

<!-- more -->

Video Joint-Embedding Predictive Architecture

Key points:

- SSL on video data + small amount of video data
- zero-shot robotic planning for picking and placing objects using image goals, without task-specific data, training, or rewards
- compared to V1, scaled data, model size, higher resolution videos
- Encoder, predictor
    - Masking
    - Stop gradient and EMA to stabilize training
- Planning
- Downstream applications
    - used to condition an LLM, as an image encoder. Requires aligning the image model with the text model.

Related Papers:

- JEPA
- V-JEPA

V-JEPA paper

- Representations (of signals) should be predictive of each others -> abstract from the pixel space, work in the latent space
    - no denoising in the signal space
- The goal is not to get the best possible model, it is to explore the unusual SSL objective
- $z = \Deltay$ -> not really the "choice", but the mask, to give the predictor a little bit of information to the predictor.
- loss = $\min_{\theta, \phi} || P_\phi (E_\theta(x), \Delta_y) - sg(\hat \E_\theta(y)) ||_1$
- masking 90% of the video
- predictor narrower
- data input: 16 x 16 x 2 x 16, 3D ROPE embeddings
- performance improvement in both frozen evaluation as well as end-to-end finetuning
- what is learned by the encoder? they then train a decoder (back to the pixel space), only for evaluation! (it will be less performant bc it was not trained for this task)
    - ask the decoder to reconstruct the masked parts ONLY, given ONLY the masked tokens, NOT the non-masked regions
