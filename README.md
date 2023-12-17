# Generating Explanations to Understand and Repair Embedding-based Entity Alignment

This repository is the official implementation of ExEA, the method proposed in paper ***Generating Explanations to Understand and Repair***
***Embedding-based Entity Alignment.*** （ICDE 2024）

> Entity alignment is the task to seek identical entities in different knowledge graphs, which is a long-standing research topic in the database and Semantic Web fields. 
>
> The development of deep learning techniques motivates recent work to embed entities in vector space and find entity alignment via the nearest
> neighbor search.
>
>  Although embedding-based entity alignment has gained marked success in recent years, it still remains a “black box” with few explanations for entity alignment decisions.
>
>  In this paper, we present the first work that generates human-readable explanations for understanding and repairing embedding-based entity alignment results.
>
> We first compare the neighbor entities and relations of an entity alignment pair to build the subgraph matching graph as a local explanation.
>
>  We then construct an alignment dependency graph to understand the given pair from an abstract perspective.
>
> Finally, we repair the entity alignment results by using alignment dependency graphs to resolve three types of alignment conflicts. 
>
> Experiments on benchmark datasets demonstrate the effectiveness and generalization ability of the proposed framework in generating explanations and repairing results for embedding-based entity alignment.

## Environment

The essential packages and recommened version to run the code:

- python3 (>=3.7)
- pytorch (1.11.0+cu113)
- numpy   (1.21.5)
- torch-scatter (2.0.9, see the following)
- scipy  (1.7.3)
- tabulate  (0.8.9)
- sklearn

## Run ExEA

There is a folder for each model, organized as follows:

```python
 - model/     
     |- datasets/   
     |- saved_model/    
     |- src/  
```

`datasets` stores the DBP15K dataset and the generated explanation will be placed under the corresponding sub-dataset. The trained model is saved under the `saved_model`.  The code files are placed in `src`. The implementation of [LORE](https://github.com/riccotti/LORE) and [Anchor](https://github.com/marcotcr/anchor?tab=readme-ov-file) comes from the github repositories given in the original papers.

For example, To run Explanation Generation by ExEA on the zh -en dataset , enter `src/` and run:

```
$ python main.py zh EG
```

To run Entity Alignment Repair by ExEA on the zh -en dataset , enter `src/` and run:

```
$ python main.py zh repair
```

To run  Explanation Generation by baseline, such as lime, on the zh-en dataset , enter `src/` and run:

```
$ python main.py zh lime
```

Based on the above operations, important features that are sorted will be obtained. Then run the following command to get the selected topk features as explanations.

```
$ python select_exp.py zh lime --num k
```

