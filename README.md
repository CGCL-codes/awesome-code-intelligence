# Awesome Deep Learning for Code Intelligence
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/zwang4/awesome-machine-learning-in-compilers/graphs/commit-activity)

This document presents a meticulously curated collection of awesome research papers, datasets, and tools dedicated to the application of machine learning techniques in code intelligence.

Code intelligence involves the application of machine learning techniques to extract knowledge from large-scale code repositories, with the aim of developing intelligent tools to improve the quality and productivity of computer programming. 

The list includes the publication year for each paper (or the submission year for pre-prints and arXiv articles), the name of the first author, and the publication venue.
Additionally, if the code associated with the research is available, it is linked via a corresponding hyperlink.

## Related Survey
| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2018 | [A Survey of Machine Learning for Big Code and Naturalness](https://arxiv.org/abs/1709.06182) |Allamanis et al.|CSUR| [Code](https://ml4code.github.io/)                 |
| 2021 | [A Systematic Literature Review on the Use of Deep Learning in Software Engineering Research](https://arxiv.org/abs/2009.06520) |Watson et al.|TOSEM| [Code](http://wm-semeru.github.io/dl4se/)                 |
| 2020 | [Synergy between Machine/Deep Learning and Software Engineering- How Far Are We?](https://arxiv.org/abs/2008.05515) |Wang et al.|arXiv| Code                 |
| 2020 | [A Survey on Deep Learning for Software Engineering](https://arxiv.org/abs/2011.14597) |Yang et al.|CSUR| Code                 |
| 2020 | [Deep Learning & Software Engineering- State of Research and Future Directions](https://arxiv.org/abs/2009.08525) |Devanbu et al.|arXiv| Code                 |
| 2021 | [CodeXGLUE- A Machine Learning Benchmark Dataset for Code Understanding and Generation](https://arxiv.org/abs/2102.04664) |Lu et al.|arXiv| [Code](https://github.com/microsoft/CodeXGLUE)                 |

## Code Representation
To represent source code, we need to first determine what to represent. Various work has proposed to
extract code features from multiple perspectives, including code tokens, intermediate representation,
abstract syntax tree, as well as many kinds of flow graphs. 
### *Code Tokens*
Code tokens, shaping the textual appearance of source code, are composed
of function name, keywords, and various variable identifiers. These tokens are simple yet effective
to represent the semantics of programs. The majority of approaches for processing code involve
breaking the program down into a sequence of tokens based on specific delimiters, such as spaces
or the capitalization patterns in identifiers (for identifiers like SortList and intArray).

| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2017 | [Synthesizing benchmarks for predictive modeling](https://ieeexplore.ieee.org/document/7863731) |Cummins et al.|CGO| [Code](https://github.com/ChrisCummins/clgen)                 |
| 2015 | [Toward deep learning software repositories](https://dl.acm.org/doi/10.5555/2820518.2820559) |White et al.|ICSE| Code                 |
| 2016 | [Summarizing source code using a neural attention model](https://aclanthology.org/P16-1195/) |Iyer et al.|ACL| [Code](https://github.com/sriniiyer/codenn)                 |
| 2016 | [A convolutional attention network for extreme summarization of source code](https://arxiv.org/abs/1602.03001) |Allamanis et al.|ICML| [Code](https://groups.inf.ed.ac.uk/cup/codeattention/)                 |
| 2019 | [Open Vocabulary Learning on Source Code with a Graph-Structured Cache](https://arxiv.org/abs/1810.08305) |Cvitkovic et al.|ICML| [Code](https://github.com/mwcvitkovic/Open-Vocabulary-Learning-on-Source-Code-with-a-Graph-Structured-Cache)                 |
| 2021 | [A Simple Approach for Handling Out-of-Vocabulary Identifiers in Deep Learning for Source Code](https://arxiv.org/abs/2010.12663) |Chirkova et al.|NAACL| [Code](https://github.com/bayesgroup/code_transformers)                 |
| 2020 | [Learning and Evaluating Contextual Embedding of Source Code](https://arxiv.org/abs/2001.00059) |Kanade et al.|ICML| Code                 |
| 2020 | [Codebert: A pre-trained model for programming and natural languages](https://arxiv.org/abs/2002.08155) |Feng et al.|EMNLP| [Code](https://github.com/microsoft/CodeBERT)                 |
| 2020 | [Big code!= big vocabulary: Open-vocabulary models for source code](https://arxiv.org/abs/2003.07914) |Karampatsis et al.|ICSE| [Code](https://github.com/mast-group/OpenVocabCodeNLM)                 |

### *API*
There have been multiple methods proposed to
analyze the API sequences in programs. One line of work is about mining API usage patterns from
a large code corpus to demonstrate how to use an API.  Another line of work is API recommendation, which aims to recommend or generate
a sequence of APIs for users. 

| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2015 | [How can I use this method?](https://dl.acm.org/doi/10.5555/2818754.2818860) |Moreno et al.|ICSE| Code                 |
| 2017 | [An unsupervised approach for discovering relevant tutorial fragments for APIs](https://dl.acm.org/doi/10.1109/ICSE.2017.12) |Jiang et al.|ICSE| Code                 |
| 2017 | [DeepAM: Migrate APIs with Multi-Modal Sequence to Sequence Learning](https://arxiv.org/abs/1704.07734) |Deepam et al.|IJCAI| Code                 |
| 2016 | [Deep API learning](https://arxiv.org/abs/1605.08535) |Gu et al.|FSE| [Code](https://guxd.github.io/deepapi/)                 |
| 2017 | [Exploring API Embedding for API Usages and Applications](https://ieeexplore.ieee.org/document/7985683) |Nguyen et al.|ICSE| Code                 |
| 2019 | [SAR: learning cross-language API mappings with little knowledge](https://arxiv.org/abs/1906.03835) |Bui et al.|FSE| [Code](https://github.com/bdqnghi/SAR_API_mapping)                 |

### *AST*
The Abstract Syntax Tree (AST) is a tree-structured intermediate representation of code
that describes the syntactic structure of a program. In an AST, the leaf
nodes typically correspond to the tokens of variables and method names in
the source code, while the non-leaf nodes represent the syntactic
structure of code, like function definition, branch functions. As a result, this representation allows
ASTs to be useful for both capturing the lexical information (e.g., variable number) and the syntactic
structure of the source code. In practice, we can extract ASTs using several open source tools,
e.g., tree-sitter parser, and LLVM Clang.

| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ | 
| 2016 | [Convolutional neural networks over tree structures for programming language processing](https://dl.acm.org/doi/10.5555/3015812.3016002) |Mou et al.|AAAI| [Code](https://sites.google.com/site/treebasedcnn/)                 |
| 2020 | [Modeling programs hierarchically with stack-augmented LSTM](https://arxiv.org/abs/2002.04516) |Liu et al.|JSS| Code                 |
| 2019 | [A novel neural source code representation based on abstract syntax tree](https://ieeexplore.ieee.org/document/8812062) |Zhang et al.|ICSE| [Code](https://github.com/apache/ctakes/blo/9c552c5c4f92af00d9d008b8c7f9e9d326a2450a/ctakes-core/src/main/java/org/apache/ctakes/core/resource/FileReadWriteUtil.java#L32)                 |
| 2018 | [Deep code comment generation](https://dl.acm.org/doi/10.1145/3196321.3196334) |Hu et al.|ICPC| Code                 |
| 2019 | [code2vec: Learning distributed representations of code](https://arxiv.org/abs/1803.09473) |Alon et al.|PLDI| Code                 |
| 2019 | [code2seq: Generating Sequences from Structured Representations of Code](https://arxiv.org/abs/1808.01400) |Alon et al.|ICLR| [Code](https://github.com/tech-srl/code2seq)                 |
| 2020 | [Structural language models of code](https://arxiv.org/abs/1910.00577) |Alon et al.|ICML| [Code](http://github.com/tech-srl/slm-code-generation/)                 |
| 2017 | [A syntactic neural model for general-purpose code generation](https://aclanthology.org/P17-1041/) |Yin et al.|ACL| Code                 |
| 2018 | [Tree-to-tree neural networks for program translation](https://arxiv.org/abs/1802.03691) |Chen et al.|ICLR| Code                 |

### *IR*
The Intermediate Representation (IR) is a well-formed structure that is independent of
programming languages and machine architectures. It is used by compilers to accurately represent
the source code during the translation process from the source code to low-level machine code. The
IR can express the operations of the target machine. It is natural to enhance the code embeddings
via utilizing IRs, with the benefit of limited vocabulary to significantly alleviate the OOV
issue.

| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2018 | [Neural code comprehension: A learnable representation of code semantics](https://arxiv.org/abs/1806.07336) |Ben et al.|Neurips| Code                 |
| 2020 | [IR2Vec: LLVM IR based Scalable Program Embeddings](https://arxiv.org/abs/1909.06228) |Venkatakeerthy et al.|TACO| Code                 |
| 2020 | [Compiler-based graph representations for deep learning models of code](https://dl.acm.org/doi/10.1145/3377555.3377894) |Brauckmann et al.|CC| Code                 |
| 2021 | [ProGraML: Graph-based Deep Learning for Program Optimization and Analysis](https://arxiv.org/abs/2003.10536) |Cummins et al.|ICML| Code                 |
| 2021 | [How could Neural Networks understand Programs?](https://arxiv.org/abs/2105.04297) |Peng et al.|ICML| Code                 |


### *Code Graphs*
Currently, many approaches have been proposed to convert programs into
graphs to better represent the rich structural information within the programs, including ControlFlow Graph (CFG), Data-Flow Graph (DFG) and Code Property Graph (CPG). 
The CFG represents the computation and control flow of a program. In this representation,
each node represents a basic block and each edge represents the transitions of control flow in the
program. The DFG is a directed graph that illustrates data relationships
among various functions. Each node in the DFG has input and output data ports, and each edge
links an output port to an input port on another node.

| Year | Title                                                        | Author                | Venue                | Code                                                         | 
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2018 | [Learning to represent programs with graphs](https://arxiv.org/abs/1711.00740) |Allamanis et al.|ICLR| Code                 |
| 2017 | [Smartpaste: Learning to adapt source code](https://arxiv.org/abs/1705.07867) |Allamanis et al.|arXiv| Code                 |
| 2018 | [Generative code modeling with graphs](https://arxiv.org/abs/1805.08490) |Brockschmidt et al.|ICLR| Code                 |
| 2020 | [Flow2Vec: value-flow-based precise code embedding](https://dl.acm.org/doi/10.1145/3428301) |Sui et al.|OOPSLA| Code                 |
| 2021 | [ProGraML: Graph-based Deep Learning for Program Optimization and Analysis](https://arxiv.org/abs/2003.10536) |Cummins et al.|ICML| Code                 |
| 2021 | [PLUR: A Unifying, Graph-Based View of Program Learning, Understanding, and Repair](https://proceedings.neurips.cc/paper/2021/hash/c2937f3a1b3a177d2408574da0245a19-Abstract.html) |Chen et al.|NeurIPS| [Code](https://github.com/google-research/plur)                 |
| 2017 | [Intelligent development environment and software knowledge graph](https://link.springer.com/article/10.1007/s11390-017-1718-y) |Lin et al.|NeurIPS| Code                 |
| 2020 | [Graph4code: A machine interpretable knowledge graph for code](https://arxiv.org/abs/2002.09440v2) |Abdelaziz et al.|arXiv| Code                 |
| 2020 | [Exploiting Code Knowledge Graph for Bug Localization via Bi-directional Attention](https://dl.acm.org/doi/abs/10.1145/3387904.3389281) |Zhang et al.|ICPC| Code                 |

### *Other Features of Code*
In addition to the aforementioned features of code that have already
been widely explored, there also exist several kinds of features that are used in some specific
scenarios.

| Year | Title                                                        | Author                | Venue                | Code                                                         |
|------| ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ | 
| 2018 | [Code vectors: Understanding programs through embedded abstracted symbolic traces](https://arxiv.org/abs/1803.06686) |Henkel et al.|FSE| [Code](https://github.com/jjhenkel/code-vectors-artifact)                 |
| 2019 | [Learning to Represent Edits](https://arxiv.org/abs/1810.13337) |Yin et al.|ICLR| [Code](https://github.com/Microsoft/msrc-dpu-learning-to-represent-edits)                 |
| 2019 | [Neural Networks for Modeling Source Code Edits](https://arxiv.org/abs/1904.02818) |Zhao et al.|arXiv| Code                 |
| 2020 | [Cc2vec: Distributed representations of code changes](https://arxiv.org/abs/2003.05620) |Hoang et al.|ICSE| [Code](https://github.com/CC2Vec/CC2Vec)                 |
| 2019 | [On Learning Meaningful Code Changes via Neural Machine Translation](https://arxiv.org/abs/1901.09102) |Tufano et al.|ICSE| Code                 |
| 2021 | [Copy that! Editing Sequences by Copying Spans](https://arxiv.org/abs/2006.04771) |Panthaplackel et al.|AAAI| Code                 |
| 2020 | [A Structural Model for Contextual Code Changes](https://arxiv.org/abs/2005.13209) |Brody et al.|OOPSLA| [Code](https://github.com/tech-srl/c3po/)                 |
| 2021 | [Learning Structural Edits via Incremental Tree Transformations](https://arxiv.org/abs/2101.12087) |Yao et al.|ICLR| Code                 |

### *Hybrid*
To leverage multiple code features, several approaches to representing
source code in a hybrid fashion have been developed.
| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2018 | [Deep code search](https://dl.acm.org/doi/10.1145/3180155.3180167) |Gu et al.|ICSE| [Code](https://github.com/guxd/deep-code-search)                 |
| 2016 | [Deep learning code fragments for code clone detection](https://dl.acm.org/doi/10.1145/2970276.2970326) |White et al.|ASE| Code                 |
| 2018 | [Deepsim: deep learning code functional similarity](https://dl.acm.org/doi/10.1145/3236024.3236068) |Zhao et al.|FSE| Code                 |
| 2018 | [Improving automatic source code summarization via deep reinforcement learning](https://arxiv.org/abs/1811.07234) |Wan et al.|ASE| Code                 |
| 2019 | [Multi-modal attention network learning for semantic source code retrieval](https://arxiv.org/abs/1909.13516) |Wan et al.|ASE| [Code](https://github.com/wanyao1992/mman_public)                 |


## Application
### *Code Classification*
Classifying source code into different classes (e.g., different functionalities and programming languages), is important for many tasks such as code categorization,
programming language identification, code prediction, and vulnerability detection. Various studies
have been conducted to classify code snippets into categories based on their functionalities. 

| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2016 | [Convolutional neural networks over tree structures for programming language processing](https://arxiv.org/abs/1409.5718) |Mou et al.|AAAI| [Code](https://sites.google.com/site/treebasedcnn/)                 |
| 2018 | [Adapting neural text classification for improved software categorization](https://ieeexplore.ieee.org/document/8530052) |Leclair et al.|ICSME| Code                 |
| 2019 | [Bilateral dependency neural networks for cross-language algorithm classification](https://ieeexplore.ieee.org/abstract/document/8667995) |Bui et al.|SANER| [Code](https://github.com/bdqnghi/)                 |
| 2018 | [SCC: Automatic classification of code snippets](https://arxiv.org/abs/1809.07945) |Alreshedy et al.|SCAM| [Code](https://github.com/Kamel773/SourceCodeClassification)                 |
| 2020 | [SCC++: predicting the programming language of questions and snippets of Stack Overflow](https://www.sciencedirect.com/science/article/abs/pii/S0164121219302791) |Alrashedy et al.|JSS| [Code](https://github.com/Kamel773/SourceCodeClassification-201)                 |

### *Vulnerability Detection and Bug Finding*
Detecting vulnerabilities or bugs in programs is
essential for assuring the quality of software, as well as saves much effort and time for software
development. Although many tools have been developed for vulnerability detection, e.g., Clang
Static Analyzer, Coverity, Fortify, Flawfinder, Infer, and SVF, most of them are based
on static analysis. Recently, a growing number of works employ deep learning to discover vulnerabilities.

| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2016 | [Automatically Learning Semantic Features for Defect Prediction](https://ieeexplore.ieee.org/abstract/document/7886912) |Wang et al.|ICSE| Code                 |
| 2017 | [Software defect prediction via convolutional neural network](https://ieeexplore.ieee.org/abstract/document/8009936) |Li et al.|QRS| Code                 |
| 2018 | [Automatic feature learning for predicting vulnerable software components](https://ieeexplore.ieee.org/abstract/document/8540022) |Dam et al.|TSE| Code                 |
| 2018 | [Vuldeepecker: A deep learning-based system for vulnerability detection](https://arxiv.org/abs/1801.01681) |Li et al.|NDSS| [Code](https://github.com/CGCL-codes/VulDeePecker)                 |
| 2019 | [μVulDeePecker: A Deep Learning-Based System for Multiclass Vulnerability Detection](https://ieeexplore.ieee.org/abstract/document/8846081) |Zou et al.|TPSC| [Code](https://github.com/CGCL-codes/VulDeePecker)                 |
| 2021 | [SySeVR: A framework for using deep learning to detect software vulnerabilities](https://ieeexplore.ieee.org/abstract/document/9321538) |Li et al.|TDSC| [Code](https://github.com/SySeVR/SySeVR)                 |
| 2018 | [Cross-project transfer representation learning for vulnerable function discovery](https://ieeexplore.ieee.org/abstract/document/8329207) |Lin et al.|TII| [Code](https://github.com/DanielLin1986/TransferRepresentationLearning)                 |
| 2018 | [Maximal divergence sequential autoencoder for binary software vulnerability detection](https://openreview.net/forum?id=ByloIiCqYQ) |Le et al.|ICLR| [Code](https://github.com/dascimal-org/MDSeqVAE)                 |
| 2019 | [Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks](https://arxiv.org/abs/1909.03496) |Zhou et al.|NeurIPS| [Code](https://github.com/epicosy/devign)                 |
| 2020 | [Combining graph-based learning with automated data collection for code vulnerability detection](https://ieeexplore.ieee.org/document/9293321) |Wang et al.|TIFS| [Code](https://github.com/HuantWang/FUNDED_NISL)                 |
| 2021 | [DeepWukong: Statically detecting software vulnerabilities using deep graph neural network](https://dl.acm.org/doi/fullHtml/10.1145/3436877) |Cheng et al.|TOSEM| [Code](https://github.com/DeepWukong/DeepWukong)                 |
| 2021 | [Combining Graph Neural Networks with Expert Knowledge for Smart Contract Vulnerability Detection](https://ieeexplore.ieee.org/abstract/document/9477066) |Liu et al.|TKDE| [Code](https://github.com/Messi-Q/GPSCVulDetector)                 |
| 2021 | [Vulnerability Detection with Fine-Grained Interpretations](https://arxiv.org/abs/2106.10478) |Li et al.|FSE| [Code](https://github.com/vulnerabilitydetection/VulnerabilityDetectionResearch)                 |
| 2021 | [Interpreting deep learning-based vulnerability detector predictions based on heuristic searching](https://dl.acm.org/doi/10.1145/3429444) |Zou et al.|TOSEM| Code                 |
| 2018 | [Deepbugs: A learning approach to name-based bug detection](https://arxiv.org/abs/1805.11683) |Pradel et al.|OOPSLA| [Code](https://github.com/michaelpradel/DeepBugs)                 |
| 2019 | [Improving bug detection via context-based code representation learning and attention-based neural networks](https://dl.acm.org/doi/10.1145/3360588) |Li et al.|OOPSLA| [Code](https://github.com/OOPSLA-2019-BugDetection/OOPSLA-2019-BugDetection)                 |
| 2020 | [Neural Attribution for Semantic Bug-Localization in Student Programs](https://arxiv.org/abs/1905.12454) |Gupta et al.|NeurIPS| [Code](https://bitbucket.org/iiscseal/nbl/src/master/)                 |
| 2021 | [Fault Localization with Code Coverage Representation Learning](https://arxiv.org/abs/2103.00270) |Li et al.|ICSE| [Code](https://github.com/deeprl4fl2021icse/deeprl4fl-2021-icse)                 |
| 2021 | [Learning to find naming issues with big code and small supervision](https://dl.acm.org/doi/abs/10.1145/3453483.3454045) |He et al.|PLDI| Code                 |

### *Code Completion*
Code completion is a core feature of most modern IDEs. It offers the
developers a list of possible code hints based on available information. 

| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2014 | [Code completion with statistical language models](https://dl.acm.org/doi/10.1145/2666356.2594321) |Raychev et al.|PLDI| Code                 |
| 2017 | [Neural code completion](https://openreview.net/forum?id=rJbPBt9lg) |Liu et al.|ICLR| Code                 |
| 2018 | [Code completion with neural attention and pointer networks](https://arxiv.org/abs/1711.09573) |Li et al.|IJCAI| [Code](https://github.com/jack57lee/neuralCodeCompletion)                 |
| 2016 | [Learning python code suggestion with a sparse pointer network](https://arxiv.org/abs/1611.08307) |Bhoopchand et al.|arXiv| [Code](https://github.com/uclnlp/pycodesuggest)                 |
| 2019 | [Pythia: Ai-assisted code completion system](https://arxiv.org/abs/1912.00742) |Svyatkovskiy et al.|SIGKDD| [Code](https://github.com/motykatomasz/Pythia-AI-code-completion)                 |
| 2021 | [Code prediction by feeding trees to transformers](https://arxiv.org/abs/2003.13848) |Kim et al.|ICSE| [Code](https://github.com/facebookresearch/code-prediction-transformer)                 |
| 2020 | [Structural language models of code](https://arxiv.org/abs/1910.00577) |Alon et al.|ICML| [Code](http://github.com/tech-srl/slm-code-generation/)                 |
| 2021 | [Code completion by modeling flattened abstract syntax trees as graphs](https://arxiv.org/abs/2103.09499) |Wang et al.|AAAI| Code                 |
| 2020 | [IntelliCode Compose: Code Generation Using Transformer](https://arxiv.org/abs/2005.08025) |Svyatkovskiy et al.|FSE| Code                 |
| 2020 | [A Self-Attentional Neural Architecture for Code Completion with Multi-Task Learning](https://arxiv.org/abs/1909.06983) |Liu et al.|ICPC| Code                 |
| 2020 | [Multi-task learning based pre-trained language model for code completion](https://arxiv.org/abs/2012.14631) |Liu et al.|ASE| Code                 |
| 2021 | [Fast and memory-efficient neural code completion](https://arxiv.org/abs/2004.13651) |Svyatkovskiy et al.|MSR| [Code](https://gist.github.com/mallamanis/ce633f79dddb12bd8cb915704bcaace3)                 |
| 2020 | [On-the-Fly Adaptation of Source Code Models using Meta-Learning](https://arxiv.org/abs/2003.11768) |Shrivastava et al.|arXiv| [Code](https://github.com/xennygrimmato/meta_learn_source_code)                 |
| 2019 | [Generative Code Modeling with Graphs](https://arxiv.org/abs/1805.08490) |Brockschmidt et al.|ICLR| [Code](https://github.com/Microsoft/graph-based-code-modelling)                 |
| 2018 | [A Retrieve-and-Edit Framework for Predicting Structured Outputs](https://arxiv.org/abs/1812.01194) |Hashimoto et al.|NIPS| [Code](https://worksheets.codalab.org/worksheets/0x1ad3f387005c492ea913cf0f20c9bb89/)                 |

### *Type Inference*
Programming languages with dynamic typing, like Python and JavaScript,
allow for rapid prototyping for developers and can save the time of software development dramatically. However, without the type information, unexpected run-time errors are prone to occur, which
may introduce bugs and produce low-quality code. Current works on type inference, with the aim
of automatically inferring variable types, mainly fall into two categories: the static-analysis-based
and learning-based. 

| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2018 | [MaxSMT-based type inference for Python 3](https://link.springer.com/chapter/10.1007/978-3-319-96142-2_2) |Hassan et al.|CAV| Code                 |
| 2004 | [Faster than C: Static type inference with Starkiller](http://michael.salib.com/writings/talks/Pycon2004/paper.pdf) |Salib et al.|PyCon Proceedings| Code                 |
| 2015 | [Predicting program properties from big code](https://dl.acm.org/doi/abs/10.1145/3306204) |Raychev et al.|Communications of the ACM| Code                 |
| 2016 | [Python probabilistic type inference with natural language support](https://dl.acm.org/doi/abs/10.1145/2950290.2950343) |Xu et al.|FSE| Code                 |
| 2018 | [Deep learning type inference](https://dl.acm.org/doi/abs/10.1145/3236024.3236051) |Hellendoorn et al.|FSE| [Code](https://github.com/DeepTyper/DeepTyper)                 |
| 2019 | [NL2Type: Inferring JavaScript Function Types from Natural Language Information](https://ieeexplore.ieee.org/abstract/document/8811893) |Malik et al.|ICSE| [Code](https://github.com/sola-da/NL2Type)                 |
| 2020 | [Typewriter: Neural type prediction with search-based validation](https://arxiv.org/abs/1912.03768) |Pradel et al.|FSE| Code                 |
| 2020 | [Lambdanet: Probabilistic type inference using graph neural networks](https://arxiv.org/abs/2005.02161) |Wei et al.|ICLR| [Code](https://github.com/MrVPlusOne/LambdaNet)                 |
| 2020 | [OptTyper: Probabilistic Type Inference by Optimising Logical and Natural Constraints](https://arxiv.org/abs/2004.00348) |Pandi et al.|arXiv| Code                 |
| 2020 | [Typilus: neural type hints](https://arxiv.org/abs/2004.10657) |Allamanis et al.|PLDI| [Code](https://github.com/typilus/typilus)                 |
| 2021 | [Type4Py: Deep Similarity Learning-Based Type Inference for Python](https://arxiv.org/abs/2101.04470) |Mir et al.|arXiv| [Code](https://github.com/saltudelft/type4py)                 |

### *Code Search*
Code search aims to retrieve a code snippet by a natural-language query (nl-tocode) or code query (code-to-code). 
The nl-to-code search refers to searching code fragments that have similar semantics to the natural-language query from a codebase.
In contrast to nl-to-code search, the input of code-to-code search is source code, rather than
natural-language description. The objective of the code-to-code search is to find code snippets that
are semantically related to an input code from a codebase. 

| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2015 | [Codehow: Effective code search based on api understanding and extended boolean model (e)](https://ieeexplore.ieee.org/abstract/document/7372014) |Lv et al.|ASE| Code                 |
| 2016 | [Relationship-aware code search for JavaScript frameworks](https://dl.acm.org/doi/abs/10.1145/2950290.2950341) |Li et al.|FSE| Code                 |
| 2018 | [Deep code search](https://ieeexplore.ieee.org/abstract/document/8453172) |Gu et al.|ICSE| [Code](https://github.com/guxd/deep-code-search)                 |
| 2019 | [Multi-modal attention network learning for semantic source code retrieval](https://arxiv.org/abs/1909.13516) |Wan et al.|ASE| [Code](https://github.com/wanyao1992/mman_public)                 |
| 2020 | [A Multi-Perspective Architecture for Semantic Code Search](https://arxiv.org/abs/2005.06980) |Haldar et al.|ACL| [Code](https://github.com/rajarshihaldar/codetextmatch)                 |
| 2020 | [OCoR: An Overlapping-Aware Code Retriever ](https://arxiv.org/abs/2008.05201) |Zhu et al.|ASE| [Code](https://github.com/pkuzqh/OCoR)                 |
| 2019 | [Coacor: Code annotation for code retrieval with reinforcement learning](https://arxiv.org/abs/1904.00720) |Yao et al.|WWW| [Code](https://github.com/LittleYUYU/CoaCor)                 |
| 2019 | [Aroma: Code recommendation via structural code search](https://arxiv.org/abs/1812.01158) |Luan et al.|OOPSLA| [Code](https://github.com/facebookresearch/aroma-paper-artifacts)                 |
| 2020 | [Deep Graph Matching and Searching for Semantic Code Retrieval](https://arxiv.org/abs/2010.12908) |Ling et al.|TKDD| [Code](https://github.com/kleincup/DGMS)                 |
| 2019 | [When deep learning met code search](https://arxiv.org/abs/1905.03813) |Cambronero et al.|FSE| Code                 |
| 2018 | [FaCoY: a code-to-code search engine](https://dl.acm.org/doi/abs/10.1145/3180155.3180187) |Kim et al.|ICSE| Code                 |
| 2021 | [Interactive Cross-language Code Retrieval with Auto-Encoders](https://ieeexplore.ieee.org/abstract/document/9678929) |Chen et al.|ASE| Code                 |

### *Code Clone Detection*
Numerous software engineering activities, including code reuse,
vulnerability detection, and code search, rely on detecting similar code snippets (or code clones).
There are basically four main types of code clones: Type-1 code clones are ones that are identical
except for spaces, blanks, and comments. Type-2 code clones denote identical code snippets except
for the variable, type, literal, and function names. Type-3 code clones denote two code snippets
that are almost identical except for a few statements that have been added or removed. Type-4 code
clones denote heterogeneous code snippets with similar functionality but differing code structures
or syntax. To handle different types of code clones, various works have been proposed.

| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2002 | [CCFinder: A multilinguistic token-based code clone detection system for large scale source code](https://ieeexplore.ieee.org/abstract/document/1019480) |Kamiya et al.|TSE| Code                 |
| 2008 | [NICAD- Accurate Detection of Near-Miss Intentional Clones Using Flexible Pretty-Printing and Code Normalization](https://ieeexplore.ieee.org/abstract/document/4556129) |Roy et al.|ICPC| Code                 |
| 2007 | [Deckard: Scalable and accurate tree-based detection of code clones](https://ieeexplore.ieee.org/abstract/document/4222572) |Jiang et al.|ICSE| Code                 |
| 2016 | [Sourcerercc: Scaling code clone detection to big-code](https://arxiv.org/abs/1512.06448) |Sajnani et al.|ICSE| Code                 |
| 2016 | [Deep learning code fragments for code clone detection](https://ieeexplore.ieee.org/abstract/document/7582748) |White et al.|ASE| Code                 |
| 2017 | [Supervised Deep Features for Software Functional Clone Detection by Exploiting Lexical and Syntactical Information in Source Code](https://dl.acm.org/doi/abs/10.5555/3172077.3172312) |Wei et al.|IJCAI| Code                 |
| 2018 | [Deepsim: deep learning code functional similarity](https://dl.acm.org/doi/abs/10.1145/3236024.3236068) |Zhao et al.|FSE| Code                 |
| 2020 | [SCDetector: Software Functional Clone Detection Based on Semantic Tokens Analysis](https://dl.acm.org/doi/abs/10.1145/3324884.3416562) |Wu et al.|ASE| Code                 |
| 2019 | [A novel neural source code representation based on abstract syntax tree](https://ieeexplore.ieee.org/abstract/document/8812062) |Zhang et al.|ICSE| [Code](https://github.com/zhangj1994/astnn)                 |
| 2019 | [Learning-based Recursive Aggregation of Abstract Syntax Trees for Code Clone Detection](https://ieeexplore.ieee.org/abstract/document/8668039) |Buch et al.|SANER| [Code](https://github.com/stanfordnlp/treelstm)                 |
| 2020 | [Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree ](https://arxiv.org/abs/2002.08653) |Wang et al.|SANER| [Code](https://github.com/jacobwwh/graphmatch_clone)                 |
| 2020 | [funcGNN: A Graph Neural Network Approach to Program Similarity](https://arxiv.org/abs/2007.13239) |Nair et al.|ESEM| [Code](https://github.com/aravi11/funcGNN)                 |
| 2021 | [Modeling Functional Similarity in Source Code with Graph-Based Siamese Networks](https://arxiv.org/abs/2011.11228) |Mehrotra et al.|TSE| [Code](https://sites.google.com/site/asegsecold/projects/seclone)                 |
| 2018 | [Deep Learning Similarities from Different Representations of Source Code](https://ieeexplore.ieee.org/abstract/document/8595238) |Tufano et al.|MSR| [Code](https://github.com/micheletufano/AutoenCODE)                 |

### *Code Summarization*
Inspired by the text generation work in NLP, many approaches have
been put forward to systematically generate a description or function name to summarize the
semantics of source code. 

| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2010 | [Supporting program comprehension with source code summarization](https://ieeexplore.ieee.org/abstract/document/6062165) |Haiduc et al.|ICSE| Code                 |
| 2013 | [Autocomment: Mining question and answer sites for automatic comment generation](https://ieeexplore.ieee.org/abstract/document/6693113) |Wong et al.|ASE| Code                 |
| 2015 | [Clocom: Mining existing source code for automatic comment generation](https://ieeexplore.ieee.org/abstract/document/7081848) |Wong et al.|SANER| [Code](http://asset.uwaterloo.ca/clocom/)                 |
| 2013 | [Evaluating source code summarization techniques: Replication and expansion](https://ieeexplore.ieee.org/abstract/document/6613829) |Eddy et al.|ICPC| Code                 |
| 2013 | [Natural Language Models for Predicting Programming Comments](https://aclanthology.org/P13-2007/) |Movshovitz et al.|ACL| Code                 |
| 2016 | [A convolutional attention network for extreme summarization of source code](https://arxiv.org/abs/1602.03001) |Allamanis et al.|ICML| Code                 |
| 2016 | [Summarizing source code using a neural attention model](https://aclanthology.org/P16-1195/) |Iyer et al.|ACL| [Code](https://github.com/sriniiyer/codenn)                 |
| 2018 | [Deep code comment generation](https://ieeexplore.ieee.org/abstract/document/8973050) |Hu et al.|ICPC| Code                 |
| 2019 | [code2seq: Generating Sequences from Structured Representations of Code](https://arxiv.org/abs/1808.01400) |Alon et al.|ICLR| [Code](https://github.com/tech-srl/code2seq)                 |
| 2019 | [Structured neural summarization](https://arxiv.org/abs/1811.01824) |Fernandes et al.|ICLR| [Code](https://github.com/CoderPat/structured-neural-summarization)                 |
| 2020 | [A transformer-based approach for source code summarization](https://arxiv.org/abs/2005.00653) |Ahmad et al.|ACL| [Code](https://github.com/wasiahmad/NeuralCodeSum)                 |
| 2021 | [SIT: Code Summarization with Structure-Induced Transformer](https://arxiv.org/abs/2012.14710v1) |Wu et al.|ACL| [Code](https://github.com/gingasan/sit3)                 |
| 2018 | [Improving automatic source code summarization via deep reinforcement learning](https://arxiv.org/abs/1811.07234) |Wan et al.|ASE| Code                 |
| 2020 | [Improved code summarization via a graph neural network](https://arxiv.org/abs/2004.02843) |Leclair et al.|ICPC| Code                 |
| 2021 | [CAST: Enhancing Code Summarization with Hierarchical Splitting and Reconstruction of Abstract Syntax Trees](https://arxiv.org/abs/2108.12987) |Shi et al.|EMNLP| [Code](https://github.com/DeepSoftwareAnalytics/CAST)                 |
| 2019 | [A Neural Model for Generating Natural Language Summaries of Program Subroutines](https://arxiv.org/abs/1902.01954) |Leclair et al.|ICSE| [Code](https://s3.us-east-2.amazonaws.com/icse2018/index.html)                 |
| 2020 | [Improved Automatic Summarization of Subroutines via Attention to File Context](https://arxiv.org/abs/2004.04881) |Haque et al.|MSR| [Code](https://github.com/Attn-to-FC/Attn-to-FC)                 |
| 2020 | [Suggesting Comment Completions for Python using Neural Language Models](https://ieeexplore.ieee.org/abstract/document/9054866) |Ciurumelea et al.|SANER| Code                 |
| 2020 | [Retrieval-based neural source code summarization](https://ieeexplore.ieee.org/abstract/document/9284039) |Zhang et al.|ICSE| Code                 |
| 2020 | [Retrieve and refine: exemplar-based neural comment generation](https://arxiv.org/abs/2010.04459) |Wei et al.|ASE| Code                 |
| 2021 | [Retrieval-Augmented Generation for Code Summarization via Hybrid GNN](https://arxiv.org/abs/2006.05405) |Liu et al.|ICLR| [Code](https://github.com/shangqing-liu/CCSD-benchmark-for-code-summarization)                 |
| 2021 | [EditSum: A Retrieve-and-Edit Framework for Source Code Summarization](https://ieeexplore.ieee.org/abstract/document/9678724) |Li et al.|ASE| [Code](https://conf.researchr.org/room/ase-2021/ase-2021-venue-kangaroo)                 |
| 2018 | [Summarizing source code with transferred api knowledge](https://dl.acm.org/doi/abs/10.5555/3304889.3304975) |Hu et al.|IJCAI| [Code](https://github.com/xinghu/TL-CodeSum)                 |
| 2019 | [Code generation as a dual task of code summarization](https://arxiv.org/abs/1910.05923) |Wei et al.|NeurIPS| Code                 |
| 2020 | [Leveraging Code Generation to Improve Code Retrieval and Summarization via Dual Learning](https://arxiv.org/abs/2002.10198) |Ye et al.|WWW| Code                 |
| 2019 | [Learning to Spot and Refactor Inconsistent Method Names](https://ieeexplore.ieee.org/abstract/document/8812134) |Liu et al.|ICSE| Code                 |
| 2021 | [Deep Just-In-Time Inconsistency Detection Between Comments and Source Code ](https://arxiv.org/abs/2010.01625) |Panthaplackel et al.|AAAI| Code                 |
| 2020 | [Suggesting Natural Method Names to Check Name Consistencies](https://dl.acm.org/doi/abs/10.1145/3377811.3380926) |Nguyen et al.|ICSE| Code                 |
| 2020 | [Learning to Update Natural Language Comments Based on Code Changes ](https://arxiv.org/abs/2004.12169) |Panthaplackel et al.|ACL| [Code](https://github.com/panthap2/LearningToUpdateNLComments)                 |
| 2020 | [Automating Just-In-Time Comment Updating](https://dl.acm.org/doi/abs/10.1145/3324884.3416581) |Liu et al.|ASE| [Code](https://github.com/tbabm/CUP)                 |
| 2021 | [Automating the removal of obsolete TODO comments](https://arxiv.org/abs/2108.05846) |Gao et al.|FSE| Code                 |

### *Program Translation*
Translating programs from a deprecated programming language to
a modern one is important for software maintenance. Many neural machine translation-based
methods have been proposed for program translation.

| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2013 | [Lexical statistical machine translation for language migration](https://dl.acm.org/doi/abs/10.1145/2491411.2494584) |Nguyen et al.|FSE| Code                 |
| 2015 | [Using machine translation for converting python 2 to python 3 code](https://peerj.com/preprints/1459/) |Aggarwal et al.|Technical Report| Code                 |
| 2015 | [Divide-and-conquer approach for multi-phase statistical migration for source code](https://ieeexplore.ieee.org/abstract/document/7372046) |Nguyen et al.|ASE| Code                 |
| 2018 | [Tree-to-tree neural networks for program translation](https://arxiv.org/abs/1802.03691) |Chen et al.|ICLR| Code                 |
| 2017 | [DeepAM: Migrate APIs with Multi-Modal Sequence to Sequence Learning](https://arxiv.org/abs/1704.07734) |Deepam et al.|IJCAI| Code                 |
| 2020 | [Unsupervised translation of programming languages](https://arxiv.org/abs/2006.03511) |Lachaux et al.|NeurIPS| Code                 |

### *Program Synthesis*
Program synthesis is a task for generating source code using high-level
specifications (e.g., program descriptions or input-output samples). Given the natural-language
inputs, current approaches resort to generating programs through machine translation.

| Year | Title                                                        | Author            | Venue  | Code                                                         |
|------| ------------------------------------------------------------ |-------------------|--------| ------------------------------------------------------------ |
| 2006 | [Learning for semantic parsing with statistical machine translation](https://aclanthology.org/N06-1056/) | Wong et al.       | NAACL  | Code                 |
| 2011 | [Automating string processing in spreadsheets using input-output examples](https://dl.acm.org/doi/abs/10.1145/1925844.1926423) | Gulwani et al.    | POPL   | Code                 |
| 2014 | [Structured Generative Models of Natural Source Code](https://arxiv.org/abs/1401.0514) | Maddison et al.   | ICML   | Code                 |
| 2015 | [Language to code: Learning semantic parsers for if-this-then-that recipes](https://aclanthology.org/P15-1085/) | Quirk et al.      | ACL    | Code                 |
| 2016 | [Language to logical form with neural attention](https://arxiv.org/abs/1601.01280) | Dong et al.       | ACL    | [Code](https://github.com/donglixp/lang2logic)                 |
| 2016 | [Latent attention for if-then program synthesis](https://arxiv.org/abs/1611.01867) | Liu et al.        | NIPS   | Code                 |
| 2016 | [Improved semantic parsers for if-then statements](https://aclanthology.org/P16-1069/) | Beltagy et al.    | ACL    | Code                 |
| 2016 | [Latent Predictor Networks for Code Generation](https://arxiv.org/abs/1603.06744) | Ling et al.       | ACL    | Code                 |
| 2017 | [A syntactic neural model for general-purpose code generation](https://arxiv.org/abs/1704.01696) | Yin et al.        | ACL    | Code                 |
| 2017 | [Abstract Syntax Networks for Code Generation and Semantic Parsing](https://arxiv.org/abs/1704.07535) | Rabinovich et al. | ACL    | Code                 |
| 2017 | [Neural Programming by Example](https://arxiv.org/abs/1703.04990) | Shu et al.        | AAAI   | Code                 |
| 2017 | [DeepCoder: Learning to write programs](https://arxiv.org/abs/1611.01989) | Balog et al.      | ICLR   | Code                 |
| 2017 | [RobustFill: Neural Program Learning under Noisy I/O](https://arxiv.org/abs/1703.07469) | Devlin et al.     | ICML   | Code                 |
| 2017 | [Seq2sql: Generating structured queries from natural language using reinforcement learning](https://arxiv.org/abs/1709.00103) | Zhong et al.      | arXiv  | Code                 |
| 2018 | [Mapping Language to Code in Programmatic Context](https://arxiv.org/abs/1808.09588) | Iyer et al.       | EMNLP  | Code                 |
| 2018 | [Selecting representative examples for program synthesis](https://arxiv.org/abs/1711.03243) | Pu et al.         | ICML   | [Code](https://github.com/evanthebouncy/icml2018_selecting_representative_examples)                 |
| 2018 | [NL2Bash: A Corpus and Semantic Parser for Natural Language Interface to the Linux Operating System](https://arxiv.org/abs/1802.08979) | Lin et al.        | LREC   | Code                 |
| 2018 | [An encoder-decoder framework translating natural language to database queries](https://arxiv.org/abs/1711.06061) | Cai et al.        | IJCAI  | Code                 |
| 2018 | [Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task](https://arxiv.org/abs/1809.08887) | Yu et al.         | EMNLP  | [Code](https://yale-lily.github.io/spider)                 |
| 2018 | [Syntaxsqlnet: Syntax tree networks for complex and cross-domain text-to-sql task](https://arxiv.org/abs/1810.05237) | Yu et al.         | EMNLP  | [Code](https://github.com/taoyds/syntaxsql)                 |
| 2019 | [Learning to infer program sketches](https://arxiv.org/abs/1902.06349) | Nye et al.        | ICML   | Code                 |
| 2019 | [AutoPandas: neural-backed generators for program synthesis](https://dl.acm.org/doi/abs/10.1145/3360594) | Bavishi et al.    | OOPSLA | [Code](https://github.com/rbavishi/autopandas)                 |
| 2019 | [Sparc: Cross-domain semantic parsing in context](https://arxiv.org/abs/1906.02285) | Yu et al.         | ACL    | [Code](https://yale-lily.github.io/sparc)                 |
| 2019 | [CoSQL: A conversational text-to-SQL challenge towards cross-domain natural language interfaces to databases](https://arxiv.org/abs/1909.05378) | Yu et al.         | EMNLP  | [Code](https://yale-lily.github.io/cosql)                 |
| 2019 | [A Grammar-Based Structural CNN Decoder for Code Generation](https://arxiv.org/abs/1811.06837) | Sun et al.        | AAAI   | [Code](https://github.com/zysszy/GrammarCNN)                 |
| 2019 | [Spoc: Search-based pseudocode to code](https://arxiv.org/abs/1906.04908) | Kulal et al.      | NIPS   | Code                 |
| 2020 | [HISyn: human learning-inspired natural language programming](https://dl.acm.org/doi/abs/10.1145/3368089.3409673) | Nan et al.        | FSE    | Code                 |
| 2021 | [Evaluating large language models trained on code](https://arxiv.org/abs/2107.03374) | Chen et al.       | arXiv  | Code | <center>N</center> |
| 2022 | [Competition-Level Code Generation with AlphaCode](https://storage.googleapis.com/deepmind-media/AlphaCode/competition_level_code_generation_with_alphacode.pdf) | Li et al.         | AI     | Code                 |
| 2022 | [CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis](https://arxiv.org/abs/2203.13474) | Nijkamp et al.    | arXiv   | Code                 |
| 2022 | [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) | Chowdhery et al.    | arXiv   | Code                 |
| 2023 | [InCoder: A Generative Model for Code Infilling and Synthesis](https://arxiv.org/abs/2204.05999) | Fried et al.    | ICLR   | Code                 |
| 2022 | [PanGu-Coder: Program Synthesis with Function-Level Language Modeling](https://arxiv.org/abs/2207.11280) | Christopoulou et al.    | arXiv   | Code                 |
| 2022 | [ERNIE-Code: Beyond English-Centric Cross-lingual Pretraining for Programming Languages](https://arxiv.org/abs/2212.06742) | Chai et al.    | ACL Findings   | Code                 |
| 2023 | [StarCoder: may the source be with you!](https://arxiv.org/abs/2305.06161) | Li et al.    | TMLR   | Code                 |
| 2023 | [Code Llama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950) | Roziere et al.    | arXiv   | Code                 |
| 2023 | [CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://arxiv.org/abs/2305.07922) | Wang et al.    | EMNLP   | Code                 |
| 2023 | [CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X](https://arxiv.org/abs/2303.17568) | Zheng et al.    | KDD   | Code                 |

### *Program Repair*
Automatically localizing and repairing bugs in programs can save much
manual effort in software development. One line of work is to learn the patterns of how
programmers edit the source code, which can be used to check syntax errors while compiling.
Another line of work is focusing on repairing programs by generating patches. 

| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2016 | [Automated Correction for Syntax Errors in Programming Assignments using Recurrent Neural Networks](https://arxiv.org/abs/1603.06129) |Bhatia et al.|arXiv| Code                 |
| 2018 | [Syntax and Sensibility: Using language models to detect and correct syntax errors](https://ieeexplore.ieee.org/abstract/document/8330219) |Santos et al.|SANER| Code                 |
| 2017 | [DeepFix: Fixing Common C Language Errors by Deep Learning](https://ojs.aaai.org/index.php/AAAI/article/view/10742) |Gupta et al.|AAAI| [Code](http://www.iisc-seal.net/)                 |
| 2021 | [SequenceR: Sequence-to-Sequence Learning for End-to-End Program Repair](https://arxiv.org/abs/1901.01808) |Chen et al.|TSE| [Code](https://github.com/KTH/chai)                 |
| 2018 | [Deep Reinforcement Learning for Programming Language Correction](https://arxiv.org/abs/1801.10467) |Gupta et al.|arXiv| Code                 |
| 2019 | [SampleFix: Learning to Correct Programs by Sampling Diverse Fixes](https://openreview.net/forum?id=6zafcLROWAd) |Hajipour et al.|arXiv| Code                 |
| 2019 | [Neural Program Repair by Jointly Learning to Localize and Repair](https://arxiv.org/abs/1904.01720) |Vasic et al.|ICLR| Code                 |
| 2020 | [Hoppity: Learning graph transformations to detect and fix bugs in programs](https://openreview.net/forum?id=SJeqs6EFvB) |Dinella et al.|ICLR| [Code](https://github.com/AI-nstein/hoppity)                 |
| 2014 | [Neural turing machines](https://arxiv.org/abs/1410.5401) |Graves et al.|arXiv| Code                 |
| 2019 | [DeepDelta: Learning to Repair Compilation Errors](https://dl.acm.org/doi/abs/10.1145/3338906.3340455) |Mesbah et al.|FSE| Code                 |
| 2020 | [Learning to Fix Build Errors with Graph2Diff Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3387940.3392181) |Tarlow et al.|ICSE| Code                 |
| 2020 | [Codit: Code editing with tree-based neural models](https://arxiv.org/pdf/1810.00314) |Chakraborty et al.|TSE| Code                 |
| 2021 | [A Syntax-Guided Edit Decoder for Neural Program Repair](https://dl.acm.org/doi/abs/10.1145/3468264.3468544) |Zhu et al.|FSE| Code                 |
| 2020 | [Graph-based, Self-Supervised Program Repair from Diagnostic Feedback ](https://proceedings.mlr.press/v119/yasunaga20a.htmll) |Yasunaga et al.|ICML| [Code](https://github.com/michiyasunaga/DrRepair)                 |
| 2021 | [TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer](http://proceedings.mlr.press/v139/berabi21a.html) |Berabi et al.|ICML| [Code](https://github.com/eth-sri/TFix)                 |
| 2020 | [Self-Supervised Bug Detection and Repair](https://proceedings.neurips.cc/paper/2021/hash/ea96efc03b9a050d895110db8c4af057-Abstract.html) |Allamanis et al.|NeurIPS| [Code](https://github.com/microsoft/neurips21-self-supervised-bug-detection-and-repair)                 |
| 2021 | [CURE: Code-Aware Neural Machine Translation for Automatic Program Repair](https://arxiv.org/abs/2103.00073) |Jiang et al.|ICSE| Code                 |
| 2018 | [An empirical investigation into learning bug-fixing patches in the wild via neural machine translation](https://dl.acm.org/doi/abs/10.1145/3238147.3240732) |Tufano et al.|ASE| Code                 |
| 2018 | [Learning to Generate Corrective Patches using Neural Machine Translation](https://arxiv.org/abs/1812.07170) |Hata et al.|arXiv| Code                 |
| 2018 | [Learning to Repair Software Vulnerabilities with Generative Adversarial Networks](https://proceedings.neurips.cc/paper/2018/hash/68abef8ee1ac9b664a90b0bbaff4f770-Abstract.html) |Harer et al.|NeurIPS| Code                 |
| 2020 | [Synthesize, execute and debug: Learning to repair for neural program synthesis](https://proceedings.neurips.cc/paper/2020/hash/cd0f74b5955dc87fd0605745c4b49ee8-Abstract.html) |Gupta et al.|NeurIPS| Code                 |
| 2020 | [DLFix: Context-based Code Transformation Learning for Automated Program Repair ](https://dl.acm.org/doi/abs/10.1145/3377811.3380345) |Li et al.|ICSE| Code                 |
| 2020 | [Evaluating Representation Learning of Code Changes for Predicting Patch Correctness in Program Repair](https://ieeexplore.ieee.org/abstract/document/9286101/) |Tian et al.|ASE| Code                 |
| 2004 | [At the end of synthesis: narrowing program candidates](https://ieeexplore.ieee.org/abstract/document/7966871/) |Shriver et al.|ICSE-NIER| Code                 |
| 2020 | [Human-in-the-loop automatic program repair](https://arxiv.org/abs/1912.07758) |Bohme et al.|ICST| Code                 |
| 2021 | [Interactive Patch Filtering as Debugging Aid](https://arxiv.org/abs/2004.08746) |Liang et al.|ICSME| Code                 |
| 2019 | [Learning to optimize halide with tree search and random programs](https://dl.acm.org/doi/abs/10.1145/3306346.3322967) |Adams et al.|TOG| Code                 |

### *Code Optimization*
| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2018 | [Learning to optimize tensor programs](https://proceedings.neurips.cc/paper/2018/hash/8b5700012be65c9da25f49408d959ca0-Abstract.html) |Chen et al.|NeurIPS| Code                 |
| 2020 | [FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System](https://dl.acm.org/doi/abs/10.1145/3373376.3378508) |Zheng et al.|ASPLOS| [Code](https://github.com/KnowingNothing/FlexTensor)                 |
| 2020 | [Ansor: Generating high-performance tensor programs for deep learning](https://dl.acm.org/doi/abs/10.1145/3373376.3378508) |Zheng et al.|OSDI| Code                 |
| 2013 | [Predictive modeling in a polyhedral optimization space](https://link.springer.com/article/10.1007/s10766-013-0241-1) |Park et al.|IJPL| Code                 |

### *Other Applications*
| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ | 
| 2021 | [ProGraML: Graph-based Deep Learning for Program Optimization and Analysis](https://arxiv.org/abs/2003.10536) |Cummins et al.|ICML| Code                 |
| 2020 | [Deep program structure modeling through multi-relational graph-based learning](https://dl.acm.org/doi/abs/10.1145/3410463.3414670) |Ye et al.|PACT| [Code](https://github.com/yeguixin/POEM)                 |
| 2020 | [Designing PairBuddy – A Conversational Agent for Pair Programming](http://sandeepkuttal.ens.utulsa.edu/Publications/Pairbuddy_Programming_Agent_UCD.pdf) |Robe et al.|arXiv| Code                 |
| 2021 | [On the Evaluation of Commit Message Generation Models: An Experimental Study](https://ieeexplore.ieee.org/abstract/document/9609189/) |Tao et al.|ICSME| [Code](https://github.com/DeepSoftwareAnalytics/CommitMsgEmpirical)                 |
| 2018 | [Large-scale and language-oblivious code authorship identification](https://dl.acm.org/doi/abs/10.1145/3243734.3243738) |Abuhamad et al.|CCS| Code                 |

## Dataset
| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2019 | [Codesearchnet challenge: Evaluating the state of semantic code search](https://arxiv.org/abs/1909.09436) |Husain et al.|arXiv| [Code](https://github.com/github/CodeSearchNet)                 |
| 2021 | [CoSQA: 20,000+ Web Queries for Code Search and Question Answering](https://arxiv.org/abs/2105.13239) |Huang et al.|ACL| [Code](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-WebQuery)                 |
| 2016 | [Probabilistic model for code with decision trees](https://dl.acm.org/doi/abs/10.1145/3022671.2984041) |Raychev et al.|OOPSLA| Code                 |
| 2017 | [A parallel corpus of Python functions and documentation strings for automated code documentation and code generation](https://arxiv.org/abs/1707.02275) |Barone et al.|IJCNLP| [Code](https://github.com/EdinburghNLP/code-docstring-corpus)                 |
| 2020 | [PyMT5: multi-mode translation of natural language and Python code with transformers](https://arxiv.org/abs/2010.03150) |Clement et al.|EMNLP| Code                 |
| 2018 | [Deep code comment generation](https://ieeexplore.ieee.org/abstract/document/8973050/) |Hu et al.|ICPC| Code                 |
| 2021 | [Retrieval-Augmented Generation for Code Summarization via Hybrid GNN](https://arxiv.org/abs/2006.05405) |Liu et al.|ICLR| [Code](https://github.com/shangqing-liu/CCSD-benchmark-for-code-summarization)                 |
| 2018 | [Deep learning type inference](https://dl.acm.org/doi/abs/10.1145/3236024.3236051) |Hellendoorn et al.|FSE| [Code](https://github.com/DeepTyper/DeepTyper)                 |
| 2021 | [CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding Tasks](https://arxiv.org/abs/2105.12655) |Puri et al.|arXiv| [Code](https://github.com/IBM/Project_CodeNet)                 |
| 2019 | [JuICe: A Large Scale Distantly Supervised Dataset for Open Domain Context-based Code Generation](https://arxiv.org/abs/1910.02216) |Agashe et al.|EMNLP| [Code](https://github.com/rajasagashe/juice)                 |
| 2021 | [ProGraML: Graph-based Deep Learning for Program Optimization and Analysis](https://arxiv.org/abs/2003.10536) |Cummins et al.|ICML| Code                 |
| 2019 | [Recommendations for Datasets for Source Code Summarization](https://arxiv.org/abs/1904.02660) |Leclair et al.|NAACL| [Code](http://leclair.tech/data/funcom/)                 |
| 2021 | [CoDesc: A Large Code-Description Parallel Dataset](https://arxiv.org/abs/2105.14220) |Hasan et al.|ACL| [Code](https://github.com/csebuetnlp/CoDesc)                 |
| 2021 | [Measuring Coding Challenge Competence With APPS](https://arxiv.org/abs/2105.09938) |Hendrycks et al.|NeurIPS| [Code](https://github.com/hendrycks/apps)                 |
| 2021 | [AVATAR: A Parallel Corpus for Java-Python Program Translation](https://arxiv.org/abs/2108.11590) |Ahmad et al.|arXiv| [Code](https://github.com/wasiahmad/AVATAR)                 |
| 2018 | [StaQC: A Systematically Mined Question-Code Dataset from Stack Overflow](https://dl.acm.org/doi/abs/10.1145/3178876.3186081) |Yao et al.|WWW| [Code](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset)                 |
| 2021 | [PyTorrent: A Python Library Corpus for Large-scale Language Models](https://arxiv.org/abs/2110.01710) |Bahrami et al.|arXiv| [Code](https://github.com/fla-sil/PyTorrent)                 |
| 2021 | [CodeQA: A Question Answering Dataset for Source Code Comprehension](https://arxiv.org/abs/2109.08365) |Liu et al.|EMNLP| [Code](https://github.com/jadecxliu/CodeQA)                 |
| 2021 | [CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation](https://arxiv.org/abs/2102.04664) |Lu et al.|NeurIPS| [Code](https://github.com/microsoft/CodeXGLUE)                 |


## CHALLENGES AND OPPORTUNITIES

### *Comprehensive Code Representation*
| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2019 | [Open Vocabulary Learning on Source Code with a Graph-Structured Cache](https://proceedings.mlr.press/v97/cvitkovic19b.html) |Cvitkovic et al.|ICML| [Code](https://github.com/mwcvitkovic/Open-Vocabulary-Learning-on-Source-Code-with-a-Graph-Structured-Cache)                 |
| 2020 | [Big code!= big vocabulary: Open-vocabulary models for source code](https://arxiv.org/abs/2003.07914) |Karampatsis et al.|ICSE| [Code](https://github.com/mast-group/OpenVocabCodeNLM)                 |
| 2021 | [A Simple Approach for Handling Out-of-Vocabulary Identifiers in Deep Learning for Source Code](https://arxiv.org/abs/2010.12663) |Chirkova et al.|NAACL| [Code](https://github.com/bayesgroup/code_transformers)                 |

### *Multi-Lingual and Cross-Language*
| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2021 | [Disentangled Code Representation Learning for Multiple Programming Languages](https://aclanthology.org/2021.findings-acl.391.pdf) |Zhang et al.|ACL| Code                 |
| 2022 | [Multilingual training for Software Engineering](https://arxiv.org/abs/2112.02043) |Ahmed et al.|ICSE| Code                 |
| 2019 | [Clcdsa: cross language code clone detection using syntactical features and api documentation](https://ieeexplore.ieee.org/abstract/document/8952189) |Nafi et al.|ASE| Code                 |
| 2019 | [Bilateral dependency neural networks for cross-language algorithm classification](https://ieeexplore.ieee.org/abstract/document/8667995) |Bui et al.|SANER| Code                 |
| 2019 | [SAR: learning cross-language API mappings with little knowledge](https://arxiv.org/abs/1906.03835) |Bui et al.|FSE| Code                 |
| 2021 | [Interactive Cross-language Code Retrieval with Auto-Encoders](https://ieeexplore.ieee.org/abstract/document/9678929) |Chen et al.|ASE| Code                 |
| 2022 | [Cross-Domain Deep Code Search with Few-Shot Meta Learning](https://arxiv.org/abs/2201.00150) |Chai et al.|ICSE| [Code](https://github.com/fewshotcdcs/CDCS)                 |
| 2022 | [Cross-Language Binary-Source Code Matching with Intermediate Representations](https://arxiv.org/abs/2201.07420) |Gui et al.|SANER| Code                 |

### *Model Interpretability*
| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2021 | [Vulnerability Detection with Fine-grained Interpretations](https://arxiv.org/abs/2106.10478) |Li et al.|FSE| [Code](https://github.com/vulnerabilitydetection/VulnerabilityDetectionResearch)                 |
| 2021 | [Interpreting deep learning-based vulnerability detector predictions based on heuristic searching](https://dl.acm.org/doi/abs/10.1145/3429444) |Zou et al.|TOSEM| Code                 |
| 2021 | [Interpretable Program Synthesis](https://dl.acm.org/doi/abs/10.1145/3411764.3445646) |Zhang et al.|CHI| Code                 |
| 2021 | [PyExplainer: Explaining the Predictions of Just-In-Time Defect Models](https://ieeexplore.ieee.org/abstract/document/9678763) |Pornprasit et al.|ASE| Code                 |


### *Robustness and Security*
| Year | Title                                                        | Author                | Venue                | Code                                                         |
| ---- | ------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------------------------------------ |
| 2017 | [Towards evaluating the robustness of neural networks](https://arxiv.org/abs/1608.04644) |Carlini et al.|SP| Code                 |
| 2018 | [Robust physical-world attacks on deep learning visual classification](https://store.computer.org/csdl/proceedings-article/cvpr/2018/642000b625/17D45W9KVJT) |Eykholt et al.|CVPR| Code                 |
| 2017 | [Towards evaluating the robustness of neural networks](https://arxiv.org/abs/1608.04644) |Carlini et al.|SP| Code                 |
| 2019 | [On evaluating adversarial robustness](https://arxiv.org/abs/1902.06705) |Carlini et al.|arXiv| Code                 |
| 2020 | [Adversarial attacks on deep-learning models in natural language processing: A survey](https://arxiv.org/abs/1901.06796) |Zhang et al.|TIST| Code                 |
| 2020 | [Semantic Robustness of Models of Source Code](https://arxiv.org/abs/2002.03043) |Ramakrishnan et al.|arXiv| Code                 |
| 2020 | [Adversarial Examples for Models of Code](https://dl.acm.org/doi/abs/10.1145/3428230) |Yefet et al.|OOPSLA| [Code](https://github.com/tech-srl/adversarial-examples)                 |
| 2021 | [Adversarial Attacks to API Recommender Systems: Time to Wake Up and Smell the Coffee?](https://ieeexplore.ieee.org/abstract/document/9678946) |Nguyen et al.|ASE| Code                 |
| 2020 | [Adversarial robustness for code](http://proceedings.mlr.press/v119/bielik20a.html) |Bielik et al.|ICML| [Code](https://github.com/eth-sri/robust-code)                 |
| 2021 | [Adversarial Robustness of Deep Code Comment Generation](https://arxiv.org/abs/2108.00213) |Zhou et al.|arXiv| [Code](https://github.com/zhangxq-1/ACCENT-repository)                 |
| 2019 | [Misleading Authorship Attribution of Source Code using Adversarial Learning](https://arxiv.org/abs/1905.12386) |Quiring et al.|USENIX Security| Code                 |
| 2021 | [A Practical Black-box Attack on Source Code Authorship Identiﬁcation Classiﬁers](https://ieeexplore.ieee.org/abstract/document/9454564) |Liu et al.|TIFS| Code                 |
| 2021 | [Backdoors in Neural Models of Source Code](https://arxiv.org/abs/2006.06841) |Ramakrishnan et al.|arXiv| Code                 |
| 2021 | [You Autocomplete Me: Poisoning Vulnerabilities in Neural Code Completion ](https://arxiv.org/abs/2007.02220) |Schuster et al.|USENIX Security| Code                 |
| 2021 | [Explanation-Guided Backdoor Poisoning Attacks Against Malware Classifiers](https://arxiv.org/abs/2003.01031) |Severi et al.|USENIX Security| [Code](https://github.com/ClonedOne/MalwareBackdoors)                 |
| 2020 | [Generating Adversarial Examples for Holding Robustness of Source Code Processing Models](https://ojs.aaai.org/index.php/AAAI/article/view/5469) |Zhang et al.|AAAI| [Code](https://github.com/SEKE-Adversary/MHM)                 |


