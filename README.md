# Awesome Active Learning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

🤩 A curated list of awesome Active Learning ! 🤩

## Background

![](https://github.com/baifanxxx/awesome-active-learning/blob/main/fig/an_illustrative_AL_example.jpg)

(image source: [Settles, Burr](https://minds.wisconsin.edu/handle/1793/60660))

#### What is Active Learning?
Active learning is a special case of machine learning in which a learning algorithm can interactively query a oracle (or some other information source) to label new data points with the desired outputs.

![](https://github.com/baifanxxx/awesome-active-learning/blob/main/fig/active_learning_cycle.jpg)

(image source: [Settles, Burr](https://minds.wisconsin.edu/handle/1793/60660))

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
`Meta.`: meta learning       | `SSL.`: semi-supervised learning   | `RL.`: reinforcement learning       |
`FS.`: few-shot learning     |


### Before 2017
|Year| Title        | Conf    |  Paper  |  Code  | Tags | Notes |
|----| --------     | :-----: |  :----: | :----: |----|----|
|1994|Improving Generalization with Active Learning|Machine Learning|[paper](https://link.springer.com/content/pdf/10.1007/BF00993277.pdf)||   |     |
|2007|Discriminative Batch Mode Active Learning|NIPS|[paper](https://dl.acm.org/doi/pdf/10.1145/1390156.1390183)||   |     |
|2008|Active Learning with Direct Query Construction|KDD|[paper](https://dl.acm.org/doi/abs/10.1145/1401890.1401950)||   |  | 
|2008|An Analysis of Active Learning Strategies for Sequence Labeling Tasks|EMNLP|[paper](https://www.aclweb.org/anthology/D08-1112.pdf)||   |     |
|2008|Hierarchical Sampling for Active Learning|ICML|[paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.92.8661&rep=rep1&type=pdf)||   |  | 
|2010|Active Instance Sampling via Matrix Partition|NIPS|[paper](http://people.scs.carleton.ca/~yuhongguo/research/papers/activenips10figs.pdf)||   |     |
|2011|Ask Me Better Questions: Active Learning Queries Based on Rule Induction|KDD|[paper](https://dl.acm.org/doi/abs/10.1145/2020408.2020559)||   |     |
|2011|Active Learning from Crowds|ICML|[paper](https://openreview.net/pdf?id=yVemp8x6Av3y)||   |     |
|2011|Bayesian Active Learning for Classification and Preference Learning|CoRR|[paper](https://arxiv.org/abs/1112.5745)||   |     |
|2011|Active Learning Using On-line Algorithms|KDD|[paper](https://dl.acm.org/doi/abs/10.1145/2020408.2020553)||   |     |
|2012|Bayesian Optimal Active Search and Surveying|ICML|[paper](https://arxiv.org/abs/1206.6406)||   |     |
|2012|Batch Active Learning via Coordinated Matching|ICML|[paper](https://arxiv.org/abs/1206.6458)||   |     |
|2013|Active Learning for Multi-Objective Optimization|ICML|[paper](http://proceedings.mlr.press/v28/zuluaga13.html)||   |     |
|2013|Active Learning for Probabilistic Hypotheses Usingthe Maximum Gibbs Error Criterion|NIPS|[paper](https://eprints.qut.edu.au/114032/)||   |     |
|2014|Active Semi-Supervised Learning Using Sampling Theory for Graph Signals|KDD|[paper](https://dl.acm.org/doi/abs/10.1145/2623330.2623760)||   |     |
|2014|Beyond Disagreement-based Agnostic Active Learning|NIPS|[paper](https://arxiv.org/abs/1407.2657)||   |     |
|2016|Cost-Effective Active Learning for Deep Image Classification|TCSVT|[paper](https://arxiv.org/pdf/1701.03551.pdf)||   |     |
|2016|Active Image Segmentation Propagation|CVPR|[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Jain_Active_Image_Segmentation_CVPR_2016_paper.pdf)||   |     |



### 2017
| Title        | Conf    |  Paper  |  Code  | Tags | Notes |
| --------     | :-----: |  :----: | :----: |----|----|
|Active Decision Boundary Annotation with Deep Generative Models|ICCV|[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huijser_Active_Decision_Boundary_ICCV_2017_paper.pdf)||||
| Active One-shot Learning         |CoRR|[paper](https://arxiv.org/abs/1702.06559)|[code](https://github.com/markpwoodward/active_osl)|`Str.` `RL.` `FS.`|     |
| A Meta-Learning Approach to One-Step Active-Learning         |AutoML@PKDD/ECML|[paper](https://arxiv.org/abs/1706.08334)||`Pool.` `Meta.`|     |
| Generative Adversarial Active Learning         |arXiv|[paper](https://arxiv.org/abs/1702.07956)||`Pool.` `Syn.`|     |
|Active Learning from Peers|NIPS|[paper](http://papers.neurips.cc/paper/7276-active-learning-from-peers.pdf)||||
|Learning Active Learning from Data|NIPS|[paper](https://arxiv.org/abs/1703.03365)|[code](https://github.com/ksenia-konyushkova/LAL)|`Pool.`||
|Learning Algorithms for Active Learning|ICML|[paper](http://proceedings.mlr.press/v70/bachman17a.html)||||
| Deep Bayesian Active Learning with Image Data|ICML|[paper](http://proceedings.mlr.press/v70/gal17a)|[code](https://github.com/Riashat/Active-Learning-Bayesian-Convolutional-Neural-Networks/tree/master/ConvNets/FINAL_Averaged_Experiments/Final_Experiments_Run)|`Pool.`   |     |





### 2018
| Title        | Conf    |  Paper  |  Code  | Tags | Notes |
| --------     | :-----: |  :----: | :----: |----|----|
|The Power of Ensembles for Active Learning in Image Classification|CVPR|[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf)||||
| Adversarial Learning for Semi-Supervised Semantic Segmentation |BMVC|[paper](https://arxiv.org/abs/1802.07934)|[code](https://github.com/hfslyc/AdvSemiSeg)|`Pool.` `SSL.`|  |
|A Variance Maximization Criterion for Active Learning|Pattern Recognition|[paper](https://www.sciencedirect.com/science/article/pii/S0031320318300256)||||
|Meta-Learning Transferable Active Learning Policies by Deep Reinforcement Learning|ICLR-WS|[paper](https://arxiv.org/abs/1806.04798)|`Pool.` `Meta.` `RL.`|||
|Active Learning for Convolutional Neural Networks: A Core-Set Approach|ICLR|[paper](https://openreview.net/pdf?id=H1aIuk-RW)||||
|Adversarial Active Learning for Sequence Labeling and Generation|IJCAI|[paper](https://www.ijcai.org/proceedings/2018/0558.pdf)||||
|Meta-Learning for Batch Mode Active Learning|ICLR-WS|[paper](https://openreview.net/references/pdf?id=r1PsGFJPz)||||



### 2019
| Title        | Conf    |  Paper  |  Code  | Tags | Notes |
| --------     | :-----: |  :----: | :----: |----|----|
| ViewAL: Active Learning with Viewpoint Entropy for Semantic Segmentation |CVPR|[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Siddiqui_ViewAL_Active_Learning_With_Viewpoint_Entropy_for_Semantic_Segmentation_CVPR_2020_paper.html)||`Pool.`| |
| Bayesian Generative Active Deep Learning    |ICML|[paper](http://proceedings.mlr.press/v97/tran19a.html)|[code](https://github.com/toantm/BGADL)|`Pool.` `Semi.`|   |
| Variational Adversarial Active Learning    |ICCV|[paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Sinha_Variational_Adversarial_Active_Learning_ICCV_2019_paper.html)||`Pool.` `SSL.`|   |
|Integrating Bayesian and Discriminative Sparse Kernel Machines for Multi-class Active Learning|NeurIPS|[paper](https://papers.nips.cc/paper/2019/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html)|| |   |
|Active Learning via Membership Query Synthesisfor Semi-supervised Sentence Classification|CoNLL|[paper](https://www.aclweb.org/anthology/K19-1044/)|| |   |
|Discriminative Active Learning|arXiv|[paper](https://arxiv.org/pdf/1907.06347.pdf)|| |   |
|Semantic Redundancies in Image-Classification Datasets: The 10% You Don’t Need|arXiv|[paper](https://arxiv.org/pdf/1901.11409.pdf)|| |   |
|Bayesian Batch Active Learning as Sparse Subset Approximation|NIPS|[paper](http://papers.nips.cc/paper/8865-bayesian-batch-active-learning-as-sparse-subset-approximation.pdf)|| |   |
| Learning Loss for Active Learning   |CVPR|[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.html)|[code](https://github.com/Mephisto405/Learning-Loss-for-Active-Learning)|`Pool.` |   |
|Rapid Performance Gain through Active Model Reuse|IJCAI|[paper](http://www.lamda.nju.edu.cn/liyf/paper/ijcai19-acmr.pdf)|| |   |
|Parting with Illusions about Deep Active Learning|arXiv|[paper](https://arxiv.org/abs/1912.05361)||`Cri.` |   |
|BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning|NIPS|[paper](http://papers.nips.cc/paper/8925-batchbald-efficient-and-diverse-batch-acquisition-for-deep-bayesian-active-learning.pdf)|| |   |


### 2020
| Title        | Conf    |  Paper  |  Code  | Tags | Notes |
| --------     | :-----: |  :----: | :----: |----|----|
| Reinforced active learning for image segmentation |ICLR|[paper](https://arxiv.org/abs/2002.06583)|[code](https://github.com/ArantxaCasanova/ralis)|`Pool.` `RL.`| |
|[BADGE] Batch Active learning by Diverse Gradient Embeddings|ICLR|[paper](https://arxiv.org/abs/1906.03671)|[code](https://github.com/JordanAsh/badge)|`Pool.`| |
|Adversarial Sampling for Active Learning|WACV|[paper](https://openaccess.thecvf.com/content_WACV_2020/html/Mayer_Adversarial_Sampling_for_Active_Learning_WACV_2020_paper.html)||`Pool.`| |
|Online Active Learning of Reject Option Classifiers|AAAI|[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6019/5875)||| |
|Deep Active Learning for Biased Datasets via Fisher Kernel Self-Supervision|CVPR|[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Gudovskiy_Deep_Active_Learning_for_Biased_Datasets_via_Fisher_Kernel_Self-Supervision_CVPR_2020_paper.pdf)||| |
| Deep Reinforcement Active Learning for Medical Image Classification   |MICCAI|[paper](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_4)||`Pool.` `RL.`| |
| State-Relabeling Adversarial Active Learning  |CVPR|[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_State-Relabeling_Adversarial_Active_Learning_CVPR_2020_paper.html)|[code](https://github.com/Beichen1996/SRAAL)|`Pool.`  | |
|Towards Robust and Reproducible Active Learning Using Neural Networks|arXiv|[paper](https://arxiv.org/pdf/2002.09564)||`Cri.`| |
|Consistency-Based Semi-supervised Active Learning: Towards Minimizing Labeling Cost|ECCV|[paper](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_30)||`Pool.` `SSL.`| |




### 2021
| Title        | Conf    |  Paper  |  Code  | Tags | Notes |
| --------     | :-----: |  :----: | :----: |----|----|
| MedSelect: Selective Labeling for Medical Image Classification Combining Meta-Learning with Deep Reinforcement Learning   |arXiv|[paper](https://arxiv.org/abs/2103.14339)||`Pool.` `Meta.` `RL.`|  |



## Turtorials
* [Overview of Active Learning for Deep Learning](https://jacobgil.github.io/deeplearning/activelearning). Jacob Gildenblat.
* [Active Learning from Theory to Practice](https://www.youtube.com/watch?v=_Ql5vfOPxZU). Steve Hanneke, Robert Nowak. (ICML, 2019)


## Tools
* [ALiPy: Active Learning in Python](https://github.com/NUAA-AL/alipy). Ying-Peng Tang, Guo-Xiang Li, Sheng-Jun Huang. (NJU, 2019)



