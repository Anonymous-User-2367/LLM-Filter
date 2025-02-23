# One Filters All: A Generalist Filter For State Estimation

Official implementation: [One Filters All: A Generalist Filter For State Estimation]
<!-- (https://arxiv.org/abs/2402.02370).  -->
<!-- [[Slides]](https://cloud.tsinghua.edu.cn/f/7689d30f92594ded84f0/), [[Poster]](https://cloud.tsinghua.edu.cn/f/f2c18ae34fef4e74ad46/) -->

## Table of Contents

<!-- 1. [Updates](#updates)
2.  -->
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Experiments](#experiments)
4. [Quick Start](#quick-start)
<!-- 6. [Acknowledgement](#Acknowledgement) -->

<!-- ## Updates

:triangular_flag_on_post: **News** (2025.1) Scripts for the above tasks in our [paper](https://arxiv.org/pdf/2402.02370.pdf) are all available. -->


## Introduction

🌟 Estimating hidden states in dynamical systems, also known as optimal filtering, is a long-standing problem in various fields of science and engineering. To address the computational intractability of online filtering algorithms, learning-based filters have become the predominant approach. However, these methods are typically trained for specific systems, requiring repetitive retraining for different industrial applications. As a dual problem to filtering, control problems have led to the development of large control models that achieve both higher performance for specific control tasks and zero-shot capabilities across diverse control tasks.

💪 Inspired by the success of large control models, we propose a general large filtering model, **LLM-Filter**, designed to solve specific estimation tasks, including zero-shot tasks, by transferring knowledge from frozen large language models (LLMs).

🏆 Guided by prompts and leveraging pretraining knowledge, LLM-Filter outperforms specialized learning-based methods and demonstrates strong zero-shot generalization across various systems. This work introduces a novel perspective: state estimation can be reframed as a language task, effectively tackled using an LLM.

<p align="center">
<img src="./figures/motivation_3.png"  alt="" align=center width="60%"/>
</p>

## Methodology

* The process begins by embedding noisy observations as text prototypes, which are subsequently processed by a frozen LLM to harmonize disparate data modalities.
* To enable generalization across systems, we introduce System-as-Prompt (SaP), an in-context learning method that augments inputs with system-specific instructions, evolution equations, and task examples in natural language. 
* The outputs generated by the LLM are then mapped onto the final state estimates.

<p align="center">
<img src="./figures/framework_1.png" alt="" align=center />
</p>


## Experiments
### Canonical Estimation Task

We evaluate the fundamental filtering capabilities of the LLM-Filter on classical systems, and compare its performance with state-of-the-art filtering methods.

<!-- The best performances are marked in **bold**, and the second-best performances are <u>underlined</u>. Results labeled as *NaN* indicate cases where the models diverged during training. -->
<!-- <p align="center"><b>RMSE of State Estimation on Classical Systems</b></p> -->

| **Method**     | **LLM-Filter** | **MEstimator** | **RStateNet** | **ProTran**   | **KalmanNet** | **EnKF**      | **HubEnKF** |
| -------------- | -------------- | -------------- | ------------- | ------------- | ------------- | ------------- | ----------- |
| **Selkov**     | **0.4061**     | 0.8864         | 0.7202        | 1.0219        | 1.1662        | <u>0.5978</u> | 0.6676      |
| **Oscillator** | **0.5247**     | 0.8347         | 0.8493        | 0.8933        | 0.5665        | <u>0.5505</u> | 0.5540      |
| **Hopf**       | **0.5751**     | 0.8290         | 0.7282        | 0.7146        | 1.1984        | <u>0.6322</u> | 0.6982      |
| **Pendulum**   | **0.8348**     | 0.9354         | 0.9180        | <u>0.8456</u> | 2.7140        | 1.4117        | 1.4256      |
| **Lorenz96**   | **0.9149**     | <u>0.9649</u>  | 0.9762        | 0.9975        | NaN           | 6.6024        | 3.0602      |
| **VL20**       | **0.7717**     | 1.0014         | <u>0.9428</u> | 0.9902        | NaN           | 5.8633        | 10.1806     |
| **Average**    | **0.7139**     | 0.9219         | <u>0.8737</u> | 0.9152        | NaN           | 2.7448        | 2.8657      |


### Zero-shot Estimation Task
#### Model Mismatch

We evaluated LLM-Filter in **model mismatch** scenarios on the selkov, oscillator, and hopf systems, quantified by the **observation covariance expansion ratio (OCER)**. 

<p align="center">
<img src="./figures/cross_obs.png" alt="" align=center />
</p>

#### Across Systems

In this experiment, we evaluate the **zero-shot** performance of completely different systems.
Specifically, for LLM-Filter and the learning-based KalmanNet, MEstimator, RStateNet andv ProTran, we train the models on one system and test them on a different system. 
We design three zero-shot scenarios: (1) generalization between linear and nonlinear systems (tracking and pendulum), (2) generalization between nonlinear systems (oscillator and Hopf), and (3) generalization within high-dimensional chaotic systems (Lorenz96 and VL20).

<p align="center">
<img src="./figures/zero_shot.png" alt="" align=center />
</p>

### Scaling Behavior

In our study, we explore the **scaling behavior** of LLM-Filter in state estimation tasks by evaluating models of different parameter sizes as the LLM backbone. Our findings illustrate the scaling behavior of LLM-Filter: **as model parameters increase, RMSE decreases, boosting estimation accuracy, though at the cost of longer training times**. We also observed performance anomalies, such as OPT-1.3B outperforming OPT-6.7B, likely due to varying data distribution preferences across LLM architectures.

<p align="center">
<img src="./figures/param_count_1.png" alt="" align=center width="60%"/>
</p>


### Hyperparameter Sensitivity

We verify the **robustness** of the LLM-Filter with respect to hyperparameters, including the **window length `T`, the layer hidden dimension, and the number of layers of MLP in `ObsEmbedding` and `StateProjection`**.


<p align="center">
<img src="./figures/sensitive.png" alt="" align=center />
</p>

## Quick Start
### Installation

1. Create a Python environment
```
conda create -n llm-filter python=3.8.20 -y
conda activate llm-filter
```

2. Install necessary dependencies.

```
pip install -r requirements.txt
```

3. Download the datasets [[Google Drive]](https://drive.google.com/uc?export=download&id=1eo4fPo7OZAofrFj0yviBo6x5wGg0aokq) into the folder ```./dataset/```.

4. Download the large language models [GPT2](https://huggingface.co/openai-community/gpt2), [OPT Family](https://huggingface.co/facebook/opt-125m), [LLaMA-7B](https://huggingface.co/huggyllama/llama-7b) into the folder ```./model/```.

### Traning and Evaluation

1. Train the LLM-Filter. The default large language model is LLaMA-7B

```
bash ./scripts/LLMFilter.sh
```

2. We also provide a method to accelerate training by running multiple cards in parallel.

```
bash ./scripts/Accelerate_LLMFilter.sh
```

3. Evaluate the trained LLM-Filter.

```
bash ./scripts/LLMFilter_Eval.sh
```


<!-- ## Citation

If you find this repo helpful, please cite our paper. 

```
@article{liu2024autotimes,
  title={AutoTimes: Autoregressive Time Series Forecasters via Large Language Models},
  author={Liu, Yong and Qin, Guo and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2402.02370},
  year={2024}
}
``` -->

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- [AutoTimes](https://github.com/thuml/AutoTimes)
- [TimeLLM](https://github.com/KimMeen/Time-LLM)
- [KalmanNet](https://github.com/KalmanNet/KalmanNet_TSP)
- [weighted-likelihood-filter](https://github.com/gerdm/weighted-likelihood-filter)

