---
date: 2024-05-16
authors:
  - LouisStefanuto
categories:
  - AlphaFold
  - Drug Discovery
  - Chemistry
---

# **Overfit#6:** AlphaFold2: Introduction

![main image post drug discovery](./images/6/main.jpg)

<!-- more -->

## The protein folding problem

Proteins are molecular machines that are **essential to life**. Just to name a few examples: hemoglobin (a protein) carries oxygen in our blood, our immune system fights bacteria using antibodies (proteins), our hair are made of keratine (a protein) ...

At the atom scale, proteins are **linear chains of amino acids**, small molecules which act as building blocks. Note that there is a limited number of amino acids (20). That means that with these 20 simple building blocks, you can build any possible protein. Amino-acids are often called **residues**.

> 🧬 In a nutshell, a protein is a word from a 20-letter alphabet.

Proteins have hydrophilic residues (that like water) and hydrophobic residues (that dislike water). To minimize the contact surface of the latter with water, **proteins fold and twist** to hide their hydrophobic regions. Thus, the linear chain linear chain of amino-acids takes a 3D complex structure, a bit like a spaghetti knot.

<figure markdown>
![structure](https://www.open.edu/openlearncreate/pluginfile.php/607232/mod_page/content/5/TP3.png){width="600"}
<figcaption>Proteins fold into a complex 3D shape. This shape defines their properties.</figcaption>
</figure>

!!! quote "Why is it so important to know how a protein folds?"

The 3D shape of a protein entirely defines how molecules and other proteins bind to it. In other words, **the 3D structure of a protein defines its properties**. To quote [The New York Times](https://www.nytimes.com/2022/07/28/science/ai-deepmind-proteins.html): "If scientists can pinpoint the shape of a particular protein, they can decipher how it operates."

Yet the **protein folding problem** is hard and expensive to solve. In 2021, the [UniProt](https://www.uniprot.org) knowledgebase, a large resource of protein sequences, contained over **60M sequences**. For comparison, the [Protein Data Bank](https://www.rcsb.org) (PDB), the biggest resource of known protein structures, contained barely over **200K structures**.

<figure markdown>
![](https://lh3.googleusercontent.com/pvN40nLfZI2M9iHd9MFPEb2v69Zuevb3R4NGCjZSMpW4VgbbxSwmQBdWvUW-U8s4lyHCdHqa-1R4NmuXNAZ5YfmaKFuKpqMtCnWedH7EGKpP1Oub=w2140-rw){width="600"}
<figcaption>From <a href="https://deepmind.google/technologies/alphafold/">DeepMind official webpage</a></figcaption>
</figure>

> 🤯 With the current techniques, finding the structures of all the known protein sequences would take decades ...

Here is for instance the go-to approach used for most of the structures known to date.

<figure markdown>
![crystal](./images/6/crystallo.png)
<figcaption>To determine the 3D structure, lots of work is needed.<br>Illustration replicated from the awesome presentation of Simon Kohl, DeepMind [2]</figcaption>
</figure>

!!! quote "The wet lab approach seems intractable at scale. Could we instead solve this problem computationally?"

Simulating proteins with physics-based approaches is **out-of-reach** for massive use. In fact, one has to compute the dynamics of the protein, take into account all the interactions between thousands of atoms, at the femtosecond scale. Computational chemists already struggle with small molecules, imagine with entire proteins ...

!!! note ""
    Now that the plot is in place, let's talk a bit about AI, and discover how **Google Deep Mind** (almost) cracked this 50-year problem.

## CASP14, the break-even

Protein folding is nothing new, scientists have been working on it for decades. Indeed, to accelerate scientific research, the protein folding community even has its own biannual competition: the Critical Assessment of Structure Prediction or **CASP**.

Every two years, the CASP organizers release the sequences of proteins whose 3D structures have been found but not published yet (it acts as a private test dataset). Any researcher is free to submit its structure predictions. The predictions are then compared to the wet lab results to determine which algorithm performs the best.

In 2020, at the 14th edition of CASP (CASP14) a team from Google DeepMind achieved incredible results, beating all their competitors by a **MASSIVE** margin.

> 🌟 Their algorithm, **AlphaFold2**, is the core topic of this post.

<figure markdown>
![evoformer](./images/6/casp.png)
<figcaption>Performance of the best submission over the last CASP editions. GDT is the evaluation metric. A 90-ish GDT is considered as an almost perfect prediction. From Simon Kohl [2]</figcaption>
</figure>

## AlphaFold2, a gemstone of engineering

With AlphaFold2, DeepMind introduced an end-to-end model that takes as input a sequence of residues, and returns as output the position in space of each atom of the protein. AlphaFold2 is a sandwich of 2 interleaved modules: the **Evoformer** and the **Structure network**, that refine two representations (MSA and pair) of the protein before predicting its structure.

![](https://borisburkov.net/static/aaf883e0fa115ba6ac38a9771251c69a/f4fee/AF2_bird_eye_view.webp)

In this series of posts, we will dive together in the architectural and training tricks that make AlphaFold2 such a success. Step-by-step, we will decipher the secrets of **AlphaFold2, a gemstone of engineering**.

!!! note "Table of content"
    1. [Intro: The protein folding problem](./06_alphafold2_part1.md) (you are here)
    2. [Part 1: The Evoformer network](./07_alphafold2_part2.md)
    3. [Part 2: The structure network](./08_alphafold2_part3.md)

[:material-arrow-left-circle: Menu](../index.md){ .md-button }
[The Evoformer :material-arrow-right-circle:](./07_alphafold2_part2.md){ .md-button .md-button--primary }

[^1]: Jumper, J., Evans, R., Pritzel, A. et al. Highly accurate protein structure prediction with AlphaFold. Nature 596, 583–589 (2021). <https://doi.org/10.1038/s41586-021-03819-2>
[^2]: Heidelberg AI Talk 5th of May 2022 | Highly Accurate Protein Structure Prediction with AlphaFold | Simon Kohl, DeepMind. <https://www.youtube.com/watch?v=tTN0MM2CQLU>
[^3]: DeepMind website, AlphaFold technology: <https://deepmind.google/technologies/alphafold/>
[^4]: Picture: from S. Bittrich & al., Structural relevance scoring identies the most informative entries of a contact map. <https://www.researchgate.net/publication/337787176_StructureDistiller_Structural_relevance_scoring_identifies_the_most_informative_entries_of_a_contact_map>
[^5]: Supplementary Information for: Highly accurate protein structure prediction with AlphaFold: <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>