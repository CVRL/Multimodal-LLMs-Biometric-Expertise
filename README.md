# Generalist Multimodal LLMs Gain Biometric Expertise via Human Salience

Official repository for the IEEE Access paper: **IEEEXplore | [ArXiv]()**

## Abstract
> Iris presentation attack detection (PAD) is critical for secure biometric deployments, yet developing specialized models faces significant practical barriers: collecting data representing future unknown attacks is impossible, and collecting diverse-enough data, yet still limited in terms of its predictive power, is expensive. Additionally, sharing biometric data raises privacy concerns. Due to rapid emergence of new attack vectors demanding adaptable solutions, we thus investigate in this paper whether general-purpose multimodal large language models (MLLMs) can perform iris PAD when augmented with human expert knowledge, operating under strict privacy constraints that prohibit sending biometric data to public cloud MLLM services. Through analysis of vision encoder embeddings applied to our dataset, we demonstrate that pre-trained vision transformers in MLLMs inherently cluster many iris attack types despite never being explicitly trained for this task. However, where clustering shows overlap between attack classes, we find that structured prompts incorporating human salience (verbal descriptions from subjects identifying attack indicators) enable these models to resolve ambiguities. Testing on an IRB-restricted dataset of 224 iris images spanning seven attack types, using only university-approved services (Gemini 2.5 Pro) or locally-hosted models (e.g., Llama 3.2-Vision), we show that Gemini with expert-informed prompts outperforms both a specialized convolutional neural networks (CNN)-based baseline and human examiners, while the locally-deployable Llama achieves near-human performance. Our results establish that MLLMs deployable within institutional privacy constraints offer a viable path for iris PAD
> 

## Experimental Pipeline
<p align="center">
  <img src="Assets/teaser.png" width="1000" />
</p>
<br>

## Embedding Visualization
<p align="center">
  <img src="Assets/image_only.png" width="45%" />
  <img src="Assets/short_prompt.png" width="50%" />
</p>


## Dataset Overview
#### Summary
Each entry in the dataset contains the following:
* Reference to the iris sample in question
* Gemini MESH Description
* Llama MESH Description

#### Requesting a Copy of the Dataset
Instructions on how to obtain a copy of the dataset can be found at the [Notre Dame's Computer Vision Research Lab webpage](https://cvrl.nd.edu/projects/data/#autosight-2025-dataset) (Name TBD). Any questions can be directed to Adam Czajka at aczajka@nd.edu.

## Citation


## Acknowledgments

This work was supported by the U.S. Department of Defense (Contract No. W52P1J-20-9-3009). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the U.S. Department of Defense or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes, notwithstanding any copyright notation here on.
