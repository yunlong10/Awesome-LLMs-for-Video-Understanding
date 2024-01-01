# Awesome-LLMs-for-Video-Understanding [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

### 🔥Video Understanding with Large Language Models: A Survey
**[Paper](https://arxiv.org/pdf/2312.17432.pdf)** | **[Project Page](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding)**

![image](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding/blob/main/img/milestone.png)

<font size=5><center><b> Table of Contents </b> </center></font>
- [Awesome-LLMs-for-Video-Understanding ](#awesome-llms-for-video-understanding-)
    - [🔥Video Understanding with Large Language Models: A Survey](#video-understanding-with-large-language-models-a-survey)
  - [😎 Vid-LLMs: Models](#-vid-llms-models)
    - [🤖 LLM-based Video Agents](#-llm-based-video-agents)
    - [👾 Vid-LLM Pretraining](#-vid-llm-pretraining)
    - [👀 Vid-LLM Instruction Tuning](#-vid-llm-instruction-tuning)
      - [Fine-tuning with Connective Adapters](#fine-tuning-with-connective-adapters)
      - [Fine-tuning with Insertive Adapters](#fine-tuning-with-insertive-adapters)
      - [Fine-tuning with Hybrid Adapters](#fine-tuning-with-hybrid-adapters)
    - [🦾 Hybrid Methods](#-hybrid-methods)
  - [Tasks, Datasets, and Benchmarks](#tasks-datasets-and-benchmarks)
      - [Recognition and Anticipation](#recognition-and-anticipation)
      - [Captioning and Description](#captioning-and-description)
      - [Grounding and Retrieval](#grounding-and-retrieval)
      - [Question Answering](#question-answering)





## 😎 Vid-LLMs: Models 
![image](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding/blob/main/img/timeline.png)
### 🤖 LLM-based Video Agents
|  Title  |  Model   |   Date   |   Code   |   Venue  |
|:--------|:--------:|:--------:|:--------:|:--------:|
| [**Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language**](https://arxiv.org/abs/2204.00598)  |Socratic Models| 04/2022 | [project page](https://socraticmodels.github.io/) |arXiv|
| [**Video ChatCaptioner: Towards Enriched Spatiotemporal Descriptions**](https://arxiv.org/abs/2304.04227)[![Star](https://img.shields.io/github/stars/Vision-CAIR/ChatCaptioner.svg?style=social&label=Star)](https://github.com/Vision-CAIR/ChatCaptioner/tree/main/Video_ChatCaptioner) | Video ChatCaptioner | 04/2023 | [code](https://github.com/Vision-CAIR/ChatCaptioner/tree/main/Video_ChatCaptioner) | arXiv |
| [**VLog: Video as a Long Document**](https://github.com/showlab/VLog)[![Star](https://img.shields.io/github/stars/showlab/VLog.svg?style=social&label=Star)](https://github.com/showlab/VLog)  | VLog | 04/2023 | [code](https://huggingface.co/spaces/TencentARC/VLog) | - |
|[**ChatVideo: A Tracklet-centric Multimodal and Versatile Video Understanding System**](https://arxiv.org/abs/2304.14407) | ChatVideo | 04/2023 | [project page](https://www.wangjunke.info/ChatVideo/) | arXiv |
|[**MM-VID: Advancing Video Understanding with GPT-4V(ision)**](https://arxiv.org/abs/2310.19773)| MM-VID | 10/2023 | - | arXiv |
|[**MISAR: A Multimodal Instructional System with Augmented Reality**](https://arxiv.org/abs/2310.11699v1)[![Star](https://img.shields.io/github/stars/nguyennm1024/misar.svg?style=social&label=Star)](https://github.com/nguyennm1024/misar)| MISAR | 10/2023 | [project page](https://github.com/nguyennm1024/misar) | ICCV |

### 👾 Vid-LLM Pretraining
|  Title  |  Model   |   Date   |   Code   |   Venue  |
|:--------|:--------:|:--------:|:--------:|:--------:|
| **[Learning Video Representations from Large Language Models](https://arxiv.org/abs/2212.04501)** |LaViLa| 12/2022 | [code](https://github.com/facebookresearch/lavila) | CVPR |
|**[Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning](https://arxiv.org/abs/2302.14115)**|Vid2Seq|02/2023|[code](https://antoyang.github.io/vid2seq.html)|CVPR|
| **[VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset](https://arxiv.org/abs/2305.18500v1)**  | VAST | 05/2023 | [code](https://github.com/txh-mercury/vast) | NeurIPS|
|**[Merlin:Empowering Multimodal LLMs with Foresight Minds](https://arxiv.org/abs/2312.00589v1)**| Merlin | 12/2023| - | arXiv |

### 👀 Vid-LLM Instruction Tuning
#### Fine-tuning with Connective Adapters
|  Title  |  Model   |   Date   |   Code   |   Venue  |
|:--------|:--------:|:--------:|:--------:|:--------:|
| [**Video-LLaMA: An Instruction-Finetuned Visual Language Model for Video Understanding**](https://arxiv.org/abs/2306.02858) [![Star](https://img.shields.io/github/stars/DAMO-NLP-SG/Video-LLaMA.svg?style=social&label=Star)](https://github.com/DAMO-NLP-SG/Video-LLaMA) |Video-LLaMA | 06/2023 | [code](https://github.com/DAMO-NLP-SG/Video-LLaMA) | arXiv |
| [**VALLEY: Video Assistant with Large Language model Enhanced abilitY**](https://arxiv.org/abs/2306.07207)[![Star](https://img.shields.io/github/stars/RupertLuo/Valley.svg?style=social&label=Star)](https://github.com/RupertLuo/Valley)  | VALLEY | 06/2023 | [code](https://github.com/RupertLuo/Valley) | - | arXiv|
| [**Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models**](https://arxiv.org/abs/2306.05424)[![Star](https://img.shields.io/github/stars/mbzuai-oryx/Video-ChatGPT.svg?style=social&label=Star)](https://github.com/mbzuai-oryx/Video-ChatGPT)  | Video-ChatGPT | 06/2023 | [code](https://github.com/mbzuai-oryx/Video-ChatGPT) | arXiv |
| [**Macaw-LLM: Multi-Modal Language Modeling with Image, Audio, Video, and Text Integration**](https://arxiv.org/abs/2306.09093)[![Star](https://img.shields.io/github/stars/lyuchenyang/macaw-llm.svg?style=social&label=Star)](https://github.com/lyuchenyang/macaw-llm)  |Macaw-LLM| 06/2023 | [code](https://github.com/lyuchenyang/macaw-llm) | arXiv |
| [**LLMVA-GEBC: Large Language Model with Video Adapter for Generic Event Boundary Captioning**](https://arxiv.org/abs/2306.10354) [![Star](https://img.shields.io/github/stars/zjr2000/llmva-gebc.svg?style=social&label=Star)](https://github.com/zjr2000/llmva-gebc) | LLMVA-GEBC | 06/2023 | [code](https://github.com/zjr2000/llmva-gebc) | CVPR  |
|[**Youku-mPLUG: A 10 Million Large-scale Chinese Video-Language Dataset for Pre-training and Benchmarks**](https://arxiv.org/abs/2306.04362) [![Star](https://img.shields.io/github/stars/x-plug/youku-mplug.svg?style=social&label=Star)](https://github.com/x-plug/youku-mplug) | mPLUG-video | 06/2023 | [code](https://github.com/x-plug/youku-mplug) | arXiv |
|[**MovieChat: From Dense Token to Sparse Memory for Long Video Understanding**](https://arxiv.org/abs/2307.16449)[![Star](https://img.shields.io/github/stars/rese1f/MovieChat.svg?style=social&label=Star)](https://github.com/rese1f/MovieChat) | MovieChat | 07/2023 | [code](https://github.com/rese1f/MovieChat) | arXiv |


#### Fine-tuning with Insertive Adapters
|  Title  |  Model   |   Date   |   Code   |   Venue  |
|:--------|:--------:|:--------:|:--------:|:--------:|
| [**MIMIC-IT: Multi-Modal In-Context Instruction Tuning**](https://arxiv.org/abs/2306.05425)[![Star](https://img.shields.io/github/stars/luodian/otter.svg?style=social&label=Star)](https://github.com/luodian/otter) | Otter | 06/2023 | [code](https://github.com/luodian/otter) | arXiv |
| [**VideoLLM: Modeling Video Sequence with Large Language Models**](https://arxiv.org/abs/2305.13292)[![Star](https://img.shields.io/github/stars/cg1177/videollm.svg?style=social&label=Star)](https://github.com/cg1177/videollm)  |VideoLLM| 05/2023 | [code](https://github.com/cg1177/videollm) | arXiv |
#### Fine-tuning with Hybrid Adapters
|  Title  |  Model   |   Date   |   Code   |   Venue  |
|:--------|:--------:|:--------:|:--------:|:--------:|
|[**VTimeLLM: Empower LLM to Grasp Video Moments**](https://arxiv.org/abs/2311.18445v1)[![Star](https://img.shields.io/github/stars/huangb23/vtimellm.svg?style=social&label=Star)](https://github.com/huangb23/vtimellm)|VTimeLLM|11/2023|[code](https://github.com/huangb23/vtimellm)|arXiv|
|[**GPT4Video: A Unified Multimodal Large Language Model for lnstruction-Followed Understanding and Safety-Aware Generation**](https://arxiv.org/abs/2311.16511v1)|GPT4Video|11/2023|-|arXiv|

### 🦾 Hybrid Methods
|  Title  |  Model   |   Date   |   Code   |   Venue  |
|:--------|:--------:|:--------:|:--------:|:--------:|
| [**VideoChat: Chat-Centric Video Understanding**](https://arxiv.org/abs/2305.06355)[![Star](https://img.shields.io/github/stars/OpenGVLab/Ask-Anything.svg?style=social&label=Star)](https://github.com/OpenGVLab/Ask-Anything)  | VideoChat | 05/2023 | [code](https://github.com/OpenGVLab/Ask-Anything)  [demo](https://huggingface.co/spaces/ynhe/AskAnything) |arXiv|
|[**PG-Video-LLaVA: Pixel Grounding Large Video-Language Models**](https://arxiv.org/abs/2311.13435v2)[![Star](https://img.shields.io/github/stars/mbzuai-oryx/video-llava.svg?style=social&label=Star)](https://github.com/mbzuai-oryx/video-llava) | PG-Video-LLaVA | 11/2023 | [code](https://github.com/mbzuai-oryx/video-llava) | arXiv |

---

## Tasks, Datasets, and Benchmarks

#### Recognition and Anticipation
|  Title  |  Date   |   Code   |   Data   |   Venue   |
|:--------|:--------:|:--------:|:--------:|:--------:|
#### Captioning and Description

#### Grounding and Retrieval

#### Question Answering

<!-- ## Evaluation
|  Title  |  Date   |   Code   |   Data   |   Venue   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| [**Let’s Think Frame by Frame: Evaluating Video Chain of Thought with Video Infilling and Prediction**](https://arxiv.org/abs/2305.13903) [![Star](https://img.shields.io/github/stars/vaishnavihimakunthala/vip.svg?style=social&label=Star)](https://github.com/vaishnavihimakunthala/vip) | 05/2023 | [code](https://github.com/vaishnavihimakunthala/vip) | - |
| [**SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension**](https://arxiv.org/abs/2307.16125v1) [![Star](https://img.shields.io/github/stars/ailab-cvc/seed-bench.svg?style=social&label=Star)](https://github.com/ailab-cvc/seed-bench) | 07/2023 | [code](https://github.com/ailab-cvc/seed-bench) | - |
| [**Youku-mPLUG: A 10 Million Large-scale Chinese Video-Language Dataset for Pre-training and Benchmarks**](https://arxiv.org/abs/2306.04362v1) [![Star](https://img.shields.io/github/stars/x-plug/youku-mplug.svg?style=social&label=Star)](https://github.com/x-plug/youku-mplug) | 07/2023 | [code](https://github.com/x-plug/youku-mplug) | - |
| [**FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation**](https://arxiv.org/abs/2311.01813) [![Star](https://img.shields.io/github/stars/llyx97/fetv.svg?style=social&label=Star)](https://github.com/llyx97/fetv) | 11/2023 | [code](https://github.com/llyx97/fetv) | - |
| [**VLM-Eval: A General Evaluation on Video Large Language Models**](https://arxiv.org/abs/2311.11865)  | 11/2023 | - | - | -->


```
@misc{tang2023video,
      title={Video Understanding with Large Language Models: A Survey}, 
      author={Yunlong Tang and Jing Bi and Siting Xu and Luchuan Song and Susan Liang and Teng Wang and Daoan Zhang and Jie An and Jingyang Lin and Rongyi Zhu and Ali Vosoughi and Chao Huang and Zeliang Zhang and Feng Zheng and Jianguo Zhang and Ping Luo and Jiebo Luo and Chenliang Xu},
      year={2023},
      eprint={2312.17432},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
