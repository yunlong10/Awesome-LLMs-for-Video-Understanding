# Awesome-LLMs-for-Video-Understanding [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

### 🔥Video Understanding with Large Language Models: A Survey

> *Yunlong Tang<sup>1,\*</sup>, Jing Bi<sup>1,\*</sup>, Siting Xu<sup>2,\*</sup>, Luchuan Song<sup>1</sup>, Susan Liang<sup>1</sup> , Teng Wang<sup>2,3</sup> , Daoan Zhang<sup>1</sup> , Jie An<sup>1</sup> , Jingyang Lin<sup>1</sup> , Rongyi Zhu<sup>1</sup> , Ali Vosoughi<sup>1</sup> , Chao Huang<sup>1</sup> , Zeliang Zhang<sup>1</sup> , Feng Zheng<sup>2</sup> , Jianguo Zhang<sup>2</sup> , Ping Luo<sup>3</sup> , Jiebo Luo<sup>1</sup>, Chenliang Xu<sup>1,†</sup>.*  (\*Core Contributors, †Corresponding Authors)

> *<sup>1</sup>University of Rochester, <sup>2</sup>Southern University of Science and Technology, <sup>3</sup>The University of Hong Kong*

<h5 align="center">  
    
 **[Paper](https://arxiv.org/pdf/2312.17432v2.pdf)** | **[Project Page](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding)**

</h5>

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
    - [Video-based Large Language Models Benchmark](#video-llm-benchmark)
  - [Contributing](#contributing)




## 😎 Vid-LLMs: Models 

![image](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding/blob/main/img/timeline.png)

### 🤖 LLM-based Video Agents

| Title                                                        |        Model        |  Date   |                             Code                             | Venue |
| :----------------------------------------------------------- | :-----------------: | :-----: | :----------------------------------------------------------: | :---: |
| [**Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language**](https://arxiv.org/abs/2204.00598) |   Socratic Models   | 04/2022 |      [project page](https://socraticmodels.github.io/)       | arXiv |
| [**Video ChatCaptioner: Towards Enriched Spatiotemporal Descriptions**](https://arxiv.org/abs/2304.04227)[![Star](https://img.shields.io/github/stars/Vision-CAIR/ChatCaptioner.svg?style=social&label=Star)](https://github.com/Vision-CAIR/ChatCaptioner/tree/main/Video_ChatCaptioner) | Video ChatCaptioner | 04/2023 | [code](https://github.com/Vision-CAIR/ChatCaptioner/tree/main/Video_ChatCaptioner) | arXiv |
| [**VLog: Video as a Long Document**](https://github.com/showlab/VLog)[![Star](https://img.shields.io/github/stars/showlab/VLog.svg?style=social&label=Star)](https://github.com/showlab/VLog) |        VLog         | 04/2023 |    [code](https://huggingface.co/spaces/TencentARC/VLog)     |   -   |
| [**ChatVideo: A Tracklet-centric Multimodal and Versatile Video Understanding System**](https://arxiv.org/abs/2304.14407) |      ChatVideo      | 04/2023 |    [project page](https://www.wangjunke.info/ChatVideo/)     | arXiv |
| [**MM-VID: Advancing Video Understanding with GPT-4V(ision)**](https://arxiv.org/abs/2310.19773) |       MM-VID        | 10/2023 |                              -                               | arXiv |
| [**MISAR: A Multimodal Instructional System with Augmented Reality**](https://arxiv.org/abs/2310.11699v1)[![Star](https://img.shields.io/github/stars/nguyennm1024/misar.svg?style=social&label=Star)](https://github.com/nguyennm1024/misar) |        MISAR        | 10/2023 |    [project page](https://github.com/nguyennm1024/misar)     | ICCV  |

### 👾 Vid-LLM Pretraining

| Title                                                        |  Model  |  Date   |                        Code                        |  Venue  |
| :----------------------------------------------------------- | :-----: | :-----: | :------------------------------------------------: | :-----: |
| [**Learning Video Representations from Large Language Models**](https://arxiv.org/abs/2212.04501)[![Star](https://img.shields.io/github/stars/facebookresearch/lavila?style=social&label=Star)](https://github.com/facebookresearch/lavila) | LaViLa  | 12/2022 | [code](https://github.com/facebookresearch/lavila) |  CVPR   |
| [**Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning**](https://arxiv.org/abs/2302.14115) | Vid2Seq | 02/2023 |  [code](https://antoyang.github.io/vid2seq.html)   |  CVPR   |
| [**VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset**](https://arxiv.org/abs/2305.18500v1)[![Star](https://img.shields.io/github/stars/txh-mercury/vast?style=social&label=Star)](https://github.com/txh-mercury/vast) |  VAST   | 05/2023 |    [code](https://github.com/txh-mercury/vast)     | NeurIPS |
| [**Merlin:Empowering Multimodal LLMs with Foresight Minds**](https://arxiv.org/abs/2312.00589v1) | Merlin  | 12/2023 |                         -                          |  arXiv  |

### 👀 Vid-LLM Instruction Tuning

#### Fine-tuning with Connective Adapters

| Title                                                        |     Model     |  Date   |                         Code                         | Venue |
| :----------------------------------------------------------- | :-----------: | :-----: | :--------------------------------------------------: | :---: |
| [**Video-LLaMA: An Instruction-Finetuned Visual Language Model for Video Understanding**](https://arxiv.org/abs/2306.02858) [![Star](https://img.shields.io/github/stars/DAMO-NLP-SG/Video-LLaMA.svg?style=social&label=Star)](https://github.com/DAMO-NLP-SG/Video-LLaMA) |  Video-LLaMA  | 06/2023 |  [code](https://github.com/DAMO-NLP-SG/Video-LLaMA)  | arXiv |
| [**VALLEY: Video Assistant with Large Language model Enhanced abilitY**](https://arxiv.org/abs/2306.07207)[![Star](https://img.shields.io/github/stars/RupertLuo/Valley.svg?style=social&label=Star)](https://github.com/RupertLuo/Valley) |    VALLEY     | 06/2023 |     [code](https://github.com/RupertLuo/Valley)      |   -   |
| [**Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models**](https://arxiv.org/abs/2306.05424)[![Star](https://img.shields.io/github/stars/mbzuai-oryx/Video-ChatGPT.svg?style=social&label=Star)](https://github.com/mbzuai-oryx/Video-ChatGPT) | Video-ChatGPT | 06/2023 | [code](https://github.com/mbzuai-oryx/Video-ChatGPT) | arXiv |
| [**Macaw-LLM: Multi-Modal Language Modeling with Image, Audio, Video, and Text Integration**](https://arxiv.org/abs/2306.09093)[![Star](https://img.shields.io/github/stars/lyuchenyang/macaw-llm.svg?style=social&label=Star)](https://github.com/lyuchenyang/macaw-llm) |   Macaw-LLM   | 06/2023 |   [code](https://github.com/lyuchenyang/macaw-llm)   | arXiv |
| [**LLMVA-GEBC: Large Language Model with Video Adapter for Generic Event Boundary Captioning**](https://arxiv.org/abs/2306.10354) [![Star](https://img.shields.io/github/stars/zjr2000/llmva-gebc.svg?style=social&label=Star)](https://github.com/zjr2000/llmva-gebc) |  LLMVA-GEBC   | 06/2023 |    [code](https://github.com/zjr2000/llmva-gebc)     | CVPR  |
| [**Youku-mPLUG: A 10 Million Large-scale Chinese Video-Language Dataset for Pre-training and Benchmarks**](https://arxiv.org/abs/2306.04362) [![Star](https://img.shields.io/github/stars/x-plug/youku-mplug.svg?style=social&label=Star)](https://github.com/x-plug/youku-mplug) |  mPLUG-video  | 06/2023 |    [code](https://github.com/x-plug/youku-mplug)     | arXiv |
| [**MovieChat: From Dense Token to Sparse Memory for Long Video Understanding**](https://arxiv.org/abs/2307.16449)[![Star](https://img.shields.io/github/stars/rese1f/MovieChat.svg?style=social&label=Star)](https://github.com/rese1f/MovieChat) |   MovieChat   | 07/2023 |     [code](https://github.com/rese1f/MovieChat)      | arXiv |
| [**Large Language Models are Temporal and Causal Reasoners for Video Question Answering**](https://arxiv.org/abs/2310.15747)[![Star](https://img.shields.io/github/stars/mlvlab/Flipped-VQA.svg?style=social&label=Star)](https://github.com/mlvlab/Flipped-VQA) |   LLaMA-VQA   | 10/2023 |    [code](https://github.com/mlvlab/Flipped-VQA)     | EMNLP |
| [**Video-LLaVA: Learning United Visual Representation by Alignment Before Projection**](https://arxiv.org/pdf/2311.10122v2.pdf)[![Star](https://img.shields.io/github/stars/PKU-YuanGroup/Video-LLaVA.svg?style=social&label=Star)](https://github.com/PKU-YuanGroup/Video-LLaVA) |  Video-LLaVA  | 11/2023 | [code](https://github.com/PKU-YuanGroup/Video-LLaVA) | arXiv |
| [**Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding**](https://arxiv.org/abs/2311.08046v1)[![Star](https://img.shields.io/github/stars/pku-yuangroup/chat-univi.svg?style=social&label=Star)](https://github.com/pku-yuangroup/chat-univi) |  Chat-UniVi   | 11/2023 | [code](https://github.com/pku-yuangroup/chat-univi)  | arXiv |
| [**AutoAD: Movie Description in Context**](https://openaccess.thecvf.com/content/CVPR2023/papers/Han_AutoAD_Movie_Description_in_Context_CVPR_2023_paper.pdf) |    AutoAD     | 06/2023 |     [code](https://github.com/TengdaHan/AutoAD)      | CVPR  |
| [**AutoAD II: The Sequel - Who, When, and What in Movie Audio Description**](https://openaccess.thecvf.com//content/ICCV2023/papers/Han_AutoAD_II_The_Sequel_-_Who_When_and_What_in_ICCV_2023_paper.pdf) |   AutoAD II   | 10/2023 |                          -                           | ICCV  |
| [**Fine-grained Audio-Visual Joint Representations for Multimodal Large Language Models**](https://arxiv.org/abs/2310.05863v2)[![Star](https://img.shields.io/github/stars/the-anonymous-bs/favor.svg?style=social&label=Star)](https://github.com/the-anonymous-bs/favor) |     FAVOR     | 10/2023 |  [code](https://github.com/the-anonymous-bs/favor)   | arXiv |


#### Fine-tuning with Insertive Adapters

| Title                                                        |  Model   |  Date   |                    Code                    | Venue |
| :----------------------------------------------------------- | :------: | :-----: | :----------------------------------------: | :---: |
| [**MIMIC-IT: Multi-Modal In-Context Instruction Tuning**](https://arxiv.org/abs/2306.05425)[![Star](https://img.shields.io/github/stars/luodian/otter.svg?style=social&label=Star)](https://github.com/luodian/otter) |  Otter   | 06/2023 |  [code](https://github.com/luodian/otter)  | arXiv |
| [**VideoLLM: Modeling Video Sequence with Large Language Models**](https://arxiv.org/abs/2305.13292)[![Star](https://img.shields.io/github/stars/cg1177/videollm.svg?style=social&label=Star)](https://github.com/cg1177/videollm) | VideoLLM | 05/2023 | [code](https://github.com/cg1177/videollm) | arXiv |

#### Fine-tuning with Hybrid Adapters

| Title                                                        |   Model   |  Date   |                     Code                     | Venue |
| :----------------------------------------------------------- | :-------: | :-----: | :------------------------------------------: | :---: |
| [**VTimeLLM: Empower LLM to Grasp Video Moments**](https://arxiv.org/abs/2311.18445v1)[![Star](https://img.shields.io/github/stars/huangb23/vtimellm.svg?style=social&label=Star)](https://github.com/huangb23/vtimellm) | VTimeLLM  | 11/2023 | [code](https://github.com/huangb23/vtimellm) | arXiv |
| [**GPT4Video: A Unified Multimodal Large Language Model for lnstruction-Followed Understanding and Safety-Aware Generation**](https://arxiv.org/abs/2311.16511v1) | GPT4Video | 11/2023 |                      -                       | arXiv |

### 🦾 Hybrid Methods

| Title                                                        |        Model        |  Date   |                             Code                             | Venue |
| :----------------------------------------------------------- | :-----------------: | :-----: | :----------------------------------------------------------: | :---: |
| [**VideoChat: Chat-Centric Video Understanding**](https://arxiv.org/abs/2305.06355)[![Star](https://img.shields.io/github/stars/OpenGVLab/Ask-Anything.svg?style=social&label=Star)](https://github.com/OpenGVLab/Ask-Anything) |      VideoChat      | 05/2023 | [code](https://github.com/OpenGVLab/Ask-Anything)  [demo](https://huggingface.co/spaces/ynhe/AskAnything) | arXiv |
| [**PG-Video-LLaVA: Pixel Grounding Large Video-Language Models**](https://arxiv.org/abs/2311.13435v2)[![Star](https://img.shields.io/github/stars/mbzuai-oryx/video-llava.svg?style=social&label=Star)](https://github.com/mbzuai-oryx/video-llava) |   PG-Video-LLaVA    | 11/2023 |      [code](https://github.com/mbzuai-oryx/video-llava)      | arXiv |
| [**TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding**](https://arxiv.org/abs/2312.02051)[![Star](https://img.shields.io/github/stars/RenShuhuai-Andy/TimeChat.svg?style=social&label=Star)](https://github.com/RenShuhuai-Andy/TimeChat) |      TimeChat       | 12/2023 |     [code](https://github.com/RenShuhuai-Andy/TimeChat)      | arXiv |
| [**Video-GroundingDINO: Towards Open-Vocabulary Spatio-Temporal Video Grounding**](https://arxiv.org/pdf/2401.00901.pdf)[![Star](https://img.shields.io/github/stars/TalalWasim/Video-GroundingDINO.svg?style=social&label=Star)](https://github.com/TalalWasim/Video-GroundingDINO) | Video-GroundingDINO | 12/2023 |  [code](https://github.com/TalalWasim/Video-GroundingDINO)   | arXiv |

---

## Tasks, Datasets, and Benchmarks

#### Recognition and Anticipation

| Name              |                            Paper                             | Date |                            Link                             | Venue |
| :---------------- | :----------------------------------------------------------: | :--: | :---------------------------------------------------------: | :---: |
| **Charades**      | [**Hollywood in homes: Crowdsourcing data collection for activity understanding**](https://arxiv.org/abs/1604.01753v3) | 2016 |        [Link](http://vuchallenge.org/charades.html)         | ECCV  |
| **YouTube8M**     | [**YouTube-8M: A Large-Scale Video Classification Benchmark**](https://arxiv.org/abs/1609.08675v1) | 2016 | [Link](https://research.google.com/youtube8m/download.html) |   -   |
| **ActivityNet**   | [**ActivityNet: A Large-Scale Video Benchmark for Human Activity Understanding**](https://openaccess.thecvf.com/content_cvpr_2015/papers/Heilbron_ActivityNet_A_Large-Scale_2015_CVPR_paper.pdf) | 2015 |              [Link](http://activity-net.org/)               | CVPR  |
| **Kinetics-GEBC** | [**GEB+: A Benchmark for Generic Event Boundary Captioning, Grounding and Retrieval**](https://arxiv.org/abs/2204.00486v4) | 2022 |         [Link](https://github.com/showlab/geb-plus)         | ECCV  |
| **Kinetics-400**  | [**The Kinetics Human Action Video Dataset**](https://arxiv.org/abs/1705.06950) | 2017 |  [Link](https://paperswithcode.com/dataset/kinetics-400-1)  |   -   |

#### Captioning and Description


#### Grounding and Retrieval

#### Question Answering

#### Video-based Large Language Models Benchmark

| Title                                                        |  Date   |                            Code                            |              Venue               |
| :----------------------------------------------------------- | :-----: | :--------------------------------------------------------: | :------------------------------: |
| [**Video-Bench: A Comprehensive Benchmark and Toolkit for Evaluating Video-based Large Language Models**](https://arxiv.org/abs/2311.16103) | 11/2023 |    [code](https://github.com/PKU-YuanGroup/Video-Bench)    |                -                 |
| [**Perception Test: A Diagnostic Benchmark for Multimodal Video Models**](https://arxiv.org/abs/2305.13786) | 05/2023 | [code](https://github.com/google-deepmind/perception_test) | NeurIPS 2023, ICCV 2023 Workshop |
| [**Youku-mPLUG: A 10 Million Large-scale Chinese Video-Language Dataset for Pre-training and Benchmarks**](https://arxiv.org/abs/2306.04362v1) [![Star](https://img.shields.io/github/stars/x-plug/youku-mplug.svg?style=social&label=Star)](https://github.com/x-plug/youku-mplug) | 07/2023 |       [code](https://github.com/x-plug/youku-mplug)        |                -                 |
| [**FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation**](https://arxiv.org/abs/2311.01813) [![Star](https://img.shields.io/github/stars/llyx97/fetv.svg?style=social&label=Star)](https://github.com/llyx97/fetv) | 11/2023 |           [code](https://github.com/llyx97/fetv)           |                -                 |
| [**MoVQA: A Benchmark of Versatile Question-Answering for Long-Form Movie Understanding**](https://arxiv.org/abs/2312.04817) | 12/2023 |         [code](https://github.com/OpenGVLab/MoVQA)         |                -                 |
| [**MVBench: A Comprehensive Multi-modal Video Understanding Benchmark**](https://arxiv.org/abs/2311.17005) | 12/2023 |     [code](https://github.com/OpenGVLab/Ask-Anything)      |                -                 |

## Contributing
### 📑 Citation

We welcome everyone to contribute to this repository and help improve it. You can submit pull requests to add new papers, projects, and helpful materials, or to correct any errors that you may find. Please make sure that your pull requests follow the "Title|Model|Date|Code|Venue" format. Thank you for your valuable contributions!

If you find our survey useful for your research, please cite the following paper:

```bibtex
@article{vidllmsurvey,
      title={Video Understanding with Large Language Models: A Survey}, 
      author={Tang, Yunlong and Bi, Jing and Xu, Siting and Song, Luchuan and Liang, Susan and Wang, Teng and Zhang, Daoan and An, Jie and Lin, Jingyang and Zhu, Rongyi and Vosoughi, Ali and Huang, Chao and Zhang, Zeliang and Zheng, Feng and Zhang, Jianguo and Luo, Ping and Luo, Jiebo and Xu, Chenliang},
      journal={arXiv preprint arXiv:2312.17432},
      year={2023},
}
```

### 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yunlong10/Awesome-LLMs-for-Video-Understanding&type=Date)](https://star-history.com/#yunlong10/Awesome-LLMs-for-Video-Understanding&Date)

### ♥️ Contributors

<a href="https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=yunlong10/Awesome-LLMs-for-Video-Understanding" />
</a>
