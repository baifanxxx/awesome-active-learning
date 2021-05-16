# Awesome Active Learning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

🤩 A curated list of awesome Active Learning ! 🤩

## Background
#### What is Active Learning?
Active learning is a special case of machine learning in which a learning algorithm can interactively query a oracle (or some other information source) to label new data points with the desired outputs.

There are situations in which unlabeled data is abundant but manual labeling is expensive. In such a scenario, learning algorithms can actively query the oracle for labels. This type of iterative supervised learning is called active learning. Since the learner chooses the examples, the number of examples to learn a concept can often be much lower than the number required in normal supervised learning. With this approach, there is a risk that the algorithm is overwhelmed by uninformative examples. Recent developments are dedicated to multi-label active learning, hybrid active learning and active learning in a single-pass (on-line) context, combining concepts from the field of machine learning (e.g. conflict and ignorance) with adaptive, incremental learning policies in the field of online machine learning.

(source: [Wikipedia](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)))

## Contributing

If you find the awesome paper/code/book/tutorial or have some suggestions, please feel free to [pull requests](https://github.com/baifanxxx/awesome-active-learning/pulls) or contact <baifanxxx@gmail.com> to add papers using the following Markdown format:

```
Year | Paper Name | Conference | [Paper](link) | [Code](link) | Tags | Notes |
```

Thanks for your valuable contribution to the research community. 😃


## Table of Contents
* [Books](#books)	
* [Surveys](#surveys)
* [Papers](#papers)
* [Turtorials](#turtorials)
* [Tools](#tools)


## Books
* [Active Learning](https://www.morganclaypool.com/doi/abs/10.2200/S00429ED1V01Y201207AIM018). Burr Settles. (CMU, 2012)

## Surveys
* [Active Learning Literature Survey](https://minds.wisconsin.edu/handle/1793/60660). Settles, Burr. (2009)
* [A Survey of Deep Active Learning](https://arxiv.org/abs/2009.00236). Pengzhen Ren et al. (2020)
* [From Model-driven to Data-driven: A Survey on Active Deep Learning](https://arxiv.org/abs/2101.09933). Peng Liu et al. (2021)


## Papers


### Tags
`Sur.`: survey               | `Cri.`: critics                     |
`Pool.`: pool-based sampling | `Str.`: stream-based sampling       | `Syn.`: membership query synthesize |
`Meta.`: meta learning       | `Semi.`: semi-supervised learning   | `RL.`: reinforcement learning       |
`FS.`: few-shot learning     |

### 2017
| Title        | Conf    |  Paper  |  Code  | Tags | Notes |
| --------     | :-----: |  :----: | :----: |----|----|
| Deep Bayesian Active Learning with Image Data         |ICML|[paper](http://proceedings.mlr.press/v70/gal17a)||   |     |
| Active One-shot Learning         |CoRR|[paper](https://arxiv.org/abs/1702.06559)|[code](https://github.com/markpwoodward/active_osl)|`FS.` `RL.`|     |
| A Meta-Learning Approach to One-Step Active-Learning         |AutoML@PKDD/ECML|[paper](https://arxiv.org/abs/1706.08334)||`Meta.`|     |
| Generative Adversarial Active Learning         |arXiv|[paper](https://arxiv.org/abs/1702.07956)||`Syn.`|     |


### 2018
| Title        | Conf    |  Paper  |  Code  | Tags | Notes |
| --------     | :-----: |  :----: | :----: |----|----|
| Adversarial Learning for Semi-Supervised Semantic Segmentation |BMVC|[paper](https://arxiv.org/abs/1802.07934)|[code](https://github.com/hfslyc/AdvSemiSeg)|`Semi.`|     |
| Meta-Learning Transferable Active Learning Policies by Deep Reinforcement Learning |arXiv|[paper](https://arxiv.org/abs/1806.04798)||`Meta.` `RL.`||


### 2019
| Title        | Conf    |  Paper  |  Code  | Tags | Notes |
| --------     | :-----: |  :----: | :----: |----|----|
| ViewAL: Active Learning with Viewpoint Entropy for Semantic Segmentation |CVPR|[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Siddiqui_ViewAL_Active_Learning_With_Viewpoint_Entropy_for_Semantic_Segmentation_CVPR_2020_paper.html)||| |
| Bayesian Generative Active Deep Learning    |ICML|[paper](http://proceedings.mlr.press/v97/tran19a.html)|[code](https://github.com/toantm/BGADL)|`Semi.`|   |
| Variational Adversarial Active Learning    |ICCV|[paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Sinha_Variational_Adversarial_Active_Learning_ICCV_2019_paper.html)||`Semi.`|   |
| Learning Loss for Active Learning   |CVPR|[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.html)|| |   |



### 2020
| Title        | Conf    |  Paper  |  Code  | Tags | Notes |
| --------     | :-----: |  :----: | :----: |----|----|
| Deep Reinforcement Active Learning for Medical Image Classification   |MICCAI|[paper](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_4)||`RL.`| |
| State-Relabeling Adversarial Active Learning  |CVPR|[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_State-Relabeling_Adversarial_Active_Learning_CVPR_2020_paper.html)||  | |
| Reinforced active learning for image segmentation |ICLR|[paper](https://arxiv.org/abs/2002.06583)|[code](https://github.com/ArantxaCasanova/ralis)|`RL.`| |


### 2021
| Title        | Conf    |  Paper  |  Code  | Tags | Notes |
| --------     | :-----: |  :----: | :----: |----|----|
| MedSelect: Selective Labeling for Medical Image Classification Combining Meta-Learning with Deep Reinforcement Learning   |arXiv|[paper](https://arxiv.org/abs/2103.14339)||`Meta.` `RL.`|  |



## Turtorials
* [Overview of Active Learning for Deep Learning](https://jacobgil.github.io/deeplearning/activelearning). Jacob Gildenblat.
* [Active Learning from Theory to Practice](https://www.youtube.com/watch?v=_Ql5vfOPxZU). Steve Hanneke, Robert Nowak. (ICML, 2019)


## Tools
* [ALiPy: Active Learning in Python](https://github.com/NUAA-AL/alipy). Ying-Peng Tang, Guo-Xiang Li, Sheng-Jun Huang. (NJU, 2019)



