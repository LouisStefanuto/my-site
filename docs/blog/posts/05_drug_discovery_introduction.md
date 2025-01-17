---
date: 2024-03-30
authors:
  - LouisStefanuto
categories:
  - Drug Discovery
  - Chemistry
---

# **Overfit#5:** AI for drug discovery: An introduction

![main image post drug discovery](./images/5/main.jpg)

<!-- more -->

!!! note "Table of content"
    1. The broken market of drug discovery
    2. AI can improve the drug success ratio
    3. The main applications of AI in drug discovery

## The broken market of drug discovery

Bringing a new medicine to market is a tedious process that often spans **over a decade**. In fact, despite the considerable efforts (and money 💸) invested, only a handful of drug candidates manage to progress to the final stage of development, resulting in a significant waste of resources and time.

According to [J. Wouters (2020)](https://pubmed.ncbi.nlm.nih.gov/32125404/), the R&D investment needed to bring a new medicine to market doubled in the last decade, with recent estimates ranging **from $314 million to $2.8 billion**.

These massive costs were mainly due to the very low success rate of drug candidates in the preclinical and clinical phases. Pharmaceutical companies are thus looking for new techniques to **better distinguish** "good" candidate molecules (specific, efficient, cheap, hydrophilic) from "poor" ones (toxic, unspecific).

<figure markdown>
![frise](./images/5/frise.png){ width="600" }
<figcaption>Only a few drug candidates make it to the final stage.</figcaption>
</figure>

## AI can improve the drug success ratio

Decades of research have built up substantial datasets of molecules and proteins. By leveraging this data, the players of the pharmaceutical industry envision to predict the outcome of wet lab experiments ... **in silico**.

<!-- And to our greatest delight, their structures, chemical properties and mutual interactions [are mostly open-source](https://github.com/kjappelbaum/awesome-chemistry-datasets). -->

<!-- These datasets are a gateway for AI, which excels at **finding patterns** in complex data. This is precisely the ambition of many players in the pharmaceutical industry. -->

> 🧪 Searching drug candidates using algorithms is referred to **in silico**, in opposition to the traditional **in vitro/in vivo** approach, that relies on wet lab experiments.

!!! quote "Computational biology is nothing new. Why is AI a game-changer?"

For years, computational biologists have used **physics-driven** simulations to predict the properties of molecules and proteins. These simulations involve solving complex quantum mechanics equations, which is a time-consuming and high-computation task: it would take years to screen the whole database of compounds to find the best candidate for a single target protein.

With AI, drug discovery moves from a physics-driven approach to a **data-driven** approach. Using the available datasets, one can train models to implicitly learn the physics. Once trained on the millions of known molecules, neural networks give a rough estimation of how promising a molecule is, at a fraction of the time needed with the traditional methods.

Data-driven approaches unlock **screening capabilities at scale**. First, run a fast data-driven algorithm to filter a curated list of promising compounds. Then, run a slower and more precise physics-based algorithm to guess the best pick.

> | Approach | Accuracy | Inference speed |
> | - | - | - |
> | Physics-driven | ✅ Accurate | ❌ Slow |
> | Data-driven | ❌ Less precise | ✅ Fast |

<!-- Moreover, AI can help screening massive database of known molecules to check if they could be used for diseases they where first not designed for.

Artificial Intelligence presents a promising solution to expedite the drug discovery phase. AI models excel at finding patterns in the data, and are thus suitable solutions to estimate if a molecule is a good candidate or not.

By leveraging AI technologies, experts in biology can streamline the identification of potential drug candidates, primarily by targeting specific proteins and discerning promising compounds from less viable ones. This accelerated process aims to present superior candidates for preclinical trials, thereby increasing the likelihood of successful drug development while reducing the resources expended on less promising avenues. -->

<figure markdown>
![number of drugs per stage](./images/5/funnel.png){ width="600" }
<figcaption>AI-driven Drug Discovery aims to test less but better molecules in preclinical steps. Source:<a href="https://doi.org/10.23880/ipcm-16000208"> Kumar, M. (2020).</a></figcaption>
</figure>

---

## The main applications of AI in drug discovery

Now that we covered the competitive edge of AI, let's see how AI could accelerate research, with tangible examples[^1][^2].

Since the beginning of this post, I introduced in-silico drug discovery as a monolithic task. Yet, in practice, it is a **multi-step** process. The first challenge is to get a fine-grained knowledge of how the disease works, to then choose the best step in the biological process to act on. Next, we need to find a molecule that binds properly to the target protein. Finally, we need to double-check for potential side effects of the compound.

> 🔎 Hopefully, AI has the potential to accelerate each of these tasks. Let's zoom into the most relevant ones!

### Molecular property prediction

> 🧪 Given a molecule, predict its properties (solubility, toxicity ...)

This is maybe the most straightforward problem. It is fairly easy to turn this task into a Machine Learning formulation.

First, we need a vectorial representation of the molecule, i.e. we **embed the molecule** in a latent space, usually with a Graph Neural Network (GNN). Starting from this embedding, we then perform a **graph-level** classification (ex. toxic/not toxic) or regression (ex. solubility), using a fully-connected neural net.

Molecular property prediction is usually performed over large datasets. This process is called **virtual screening**. A good-performing property prediction model could discover new drugs, or new properties to existing drugs.

<figure markdown>
![property prediction](https://www.researcher-app.com/image/eyJ1cmkiOiJodHRwczovL3MzLWV1LXdlc3QtMS5hbWF6b25hd3MuY29tL3N0YWNrYWRlbWljL3Byb2R1Y3Rpb24vcGFwZXIvMTk4NDAxMC5wbmciLCJmb3JtYXQiOiJ3ZWJwIiwicXVhbGl0eSI6MTAwLCJub0NhY2hlIjp0cnVlfQ==.webp){ width="600" }
<figcaption>Property prediction is an embedding then prediction task. <a href="https://www.researcher-app.com/image/eyJ1cmkiOiJodHRwczovL3MzLWV1LXdlc3QtMS5hbWF6b25hd3MuY29tL3N0YWNrYWRlbWljL3Byb2R1Y3Rpb24vcGFwZXIvMTk4NDAxMC5wbmciLCJmb3JtYXQiOiJ3ZWJwIiwicXVhbGl0eSI6MTAwLCJub0NhY2hlIjp0cnVlfQ==.webp">Source</a>.</figcaption>
</figure>

### De novo drug design

> 🪄 The inverse of property prediction. Given some desired properties, generate a molecule with these properties.

De novo drug design is essentially **GenAI but for molecules** (🤯). You give a wishlist as input and get a molecule as output.

Depending on how you condition the generation, you can create really interesting models. For instance, if you condition the generation with the active site of a protein, you can then generate promising candidates for this target.

To my mind, this is the most fascinating and thrilling of all tasks, because it enables us to discover drug patterns, unknown to date, that can then be studied and improved by biologists and chemists.

<figure markdown>
![de_novo](./images/5/de_novo.png){ width="600" }
<figcaption>De novo drug design aims to generate the molecule that would fit some properties or protein site.</figcaption>
</figure>

### Binding site/interface prediction

> 🧬 Given a protein, guess where its active site is.

Just as some parts of our skin are more sensitive to touch than others, proteins have their "sensitive spots" called **active sites**. These are precise regions where molecules bind and reactions happen, essential for the protein's function.

Guessing where the active site lies is key because it conditions all the downstream tasks. The better we understand the target protein, the easier it is to find a relevant ligand to bind to its active site.

From an ML point of view, if we used Graph Neural Networks, this task could be a **node-level** classification problem.

<figure markdown>
![binding site prediction](https://www.creativebiomart.net/images/Protein-Binding-Site-Mapping.jpg){ width="500" }
<figcaption>Binding site prediction. <a href="https://www.creativebiomart.net/images/Protein-Binding-Site-Mapping.jpg"> Source.</a></figcaption>
</figure>

### Molecular docking

> ⚓️ Given a protein and a ligand (small molecule), predict the way they twist to fit one inside the other (their conformation).

To fit together, the protein's and the ligand's atoms move to **minimize the energy of the whole system** (protein+ligand). The lower the energy, the more stable the structure, and the better the match.

This task seems like a simple optimization problem. Yet, the issue is the **huge dimension** of the problem, as each atom adds some degrees of freedom.

AI usually performs better than gradient methods at these huge inverse problems. A good example is the AlphaFold deep learning model from Google DeepMind, which solved the protein folding problem. AI could thus help rank poses.

<figure markdown>
![docking](https://cbirt.net/wp-content/uploads/2023/04/DiffDock-1.png)
<figcaption>Illustration from <a href="https://cbirt.net/wp-content/uploads/2023/04/DiffDock-1.png"> DiffDock</a>. An example of Molecular docking.</figcaption>
</figure>

### Target identification

> 🎯 Given a disease, find the right biological process to target. Usually, a protein to activate or inhibit.

Last on my list, but usually the first step of the drug discovery journey. Target identification is the task of defining the protein to target to stop the disease.

Most of the time, a disease is the cause of a sequence of protein activations in our cells. To stop a disease, we need to find the most relevant step of the process to act on. It usually means finding the protein to activate or inhibit, that has little impact on other processes in the human body. This is to avoid side-effects, that are not a great property of a drug, to be honest.

Thanks to its pattern recognition capabilities, **AI could help in the identification of relevant proteins to target**. AI enables to better analyze multiomics data and is a bridge between the various modalities. It is thus a promising tool to identify better targets (i.e. better problems) to then find better drugs (i.e. solutions to those problems).

<figure markdown>
![de_novo](./images/5/target_identification.png){ width="600" }
<figcaption>Target identification. From [2].</figcaption>
</figure>

---

!!! success "To sum up ..."
    Drug discovery companies are working on each of these topics, building end-to-end pipelines to validate new chemical compounds. With these pipelines, they scrap the database of known molecules and seek promising candidates ...

## Challenges and limitations

- **Data is the limit** To train their models, researchers use open databases of all the known molecules and proteins to date. Yet these databases are expensive to expand and biased towards some types of molecules.
- **The feedback loop with the wet labs is slow** It takes years to validate a new drug. Only time will tell if all the investments in the field were worth it, or just a dead-end.
- **Misuses** It is pretty straightforward to use AI to generate molecules that harm (poisons) rather than molecules that cure. As always, AI is ambivalent.

---

## Conclusion

This is the end of our little journey at the intersection of AI and biology. I hope you enjoyed it! 🧪

As model architectures tailored for Computer Vision (ResNets) and NLP (RNN, Transformers) flourished in the last decade, I am convinced that new architectures are developing to crack these problems. I am especially curious about architectures that ensure rotation and translation invariances (SE3).

> I plan to post on this topic in future posts. More on that soon. So, stay connected! 👋

[^1]: Clemens Isert, Kenneth Atz, Gisbert Schneider, ["Structure-based drug design with geometric deep learning"](https://arxiv.org/abs/2210.11250) (2022): A survey of existing algorithms for each of the 4 problems.
[^2]: Frank W. Pun, Ivan V. Ozerov, Alex Zhavoronkov, ["AI-powered therapeutic target discovery"](https://pdf.sciencedirectassets.com/271047/1-s2.0-S0165614722X00102/1-s2.0-S0165614723001372/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFcaCXVzLWVhc3QtMSJGMEQCICPBVH8x89gAp68zo71uvUEVB7tIx4Dnhiz0cHSm8Hj5AiAUe2HmMU3Dv4Cc4tvmMRrGU3Qpf9VAgnQs1B34pgTEJCq8BQiA%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMtfSdlPlHXuX%2FJtE%2FKpAFfpiwz4Zy19EmaZfY%2FvOBLs0z%2Fj68S6irAENXMeqqlW%2FT9J6AFnJAfiYZarM488pou6GgA9OZy5B6VtZo2JeD0soURkxZyOOwcxy2lOWsLqnPPAvB0gZllvwPdHE%2Fz%2F8pCLYS%2FBczZJEvZraSHtMM6wAqw3sOEfVv%2BKfpR%2FU1bIse%2F1Pww76LB5m2yBAVV7%2FVxqy9siTVAd%2BdrHaSOr5d2Eg0i7gR7cYGyAnHoO32ZmRWq45pXPgRlPP9ObfmnPoUIRplpAZUGb6cvbryr97abwrQdh%2BkBCTH7lzQ65CfyH1w2JnZoFdvWBqMmTRu4hXnJVZrmf4HS45i67JUMn8ZGGOwjh9O3IduxQQrR1n8W23zOq%2BV91nPmlt6JRk3J3%2BKlsRmghLlx0UYN%2BiSTBe14y9VB%2FNqduEQaoLw0gHkhLVZVX%2Ffcl5RuZX%2BcT2DYtSEYzgXm15rkHWWK1yVOqe7XxqM1CARg4DkkPJ9apyefCu3z32fsXkimkItnqwF2xF3sH4GAf11sj9%2FfNORELhZ4fGZsmJBSsTZGh%2F5RQbc5fg2T3TEzspb5qw3Zzpo1YCu4%2BOlO8tj3UfuN1FOZL8Ax0AQPgyneUPGOBpT05pLxsL5rLWqTkscfPnLK0WosluNlEHYPq48CimGw7v3iyBtK2S5P%2B7zRDYdcwePvnwgHUIccZvlOyMnKhwtIxBFJZoGmj7WFYl7TNeT%2BZZrCnJCLSKDKr103ngRDGQ8OvCt08EXbq6qRAMt2ZvTc6E6BLfsHFF%2FZSyoy%2Fqen5XeDR2lMa4TKKINHqwWEaoBLehcXD2%2BcscJydTKD%2FumngDmNi7RXm%2Fy8ym9Vc21sQ8yM7S7BxbAEm4EyA4mMHCI87JbZiww17C3sAY6sgHXTkLQCDad7pAprJ5M3hVWLs2FkLlOD7MpfQArOApNoJsYvvoQP446BOGx%2FAP0LWUko3Gdy%2Fn9LmlPhAihlLTw9Mqqn5NvDd%2FWuihWsYr9UedydOqEwLBoPrXLz08HT92TGwwKnUOsnUCQXm%2FJLylVAzlJ9xdo2bctsO6whotMJPZjvJ9Jie0frMwU7g8%2BycHUC%2BP0Ms6CEOyZ8HAMWGVGzs1Z0Ue18cpmDW1mqCQCBIiU&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240403T230740Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7FSRUPZL%2F20240403%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=cae9d7c8c9c81f401c15625e530e53271d2e92cb7f7fafaa04270aa2a19c5fbb&hash=46d2e40234c30572716250097680015fb140d699eef56c565d250309b19d3220&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0165614723001372&tid=spdf-6f391880-a69c-412c-8190-1fc4d3f0c2b8&sid=e630497b5c6158449b2b724-beb1556ed526gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=00165750060203505000&rr=86ecb75c39049ea4&cc=fr) (2023)