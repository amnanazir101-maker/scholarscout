# State Space Models vs. Transformers: A Comparative Literature Review

## TL;DR
Transformers rely on quadratic self-attention mechanisms that provide superior capabilities in in-context retrieval and string copying [1, 2]. Conversely, State Space Models (SSMs) leverage linear-time recurrent dynamics and fixed-size latent states to achieve high throughput and memory efficiency on long-context sequences [3, 4, 5]. Hybrid architectures combine both paradigms to balance attention-based memory with linear scaling [2, 4, 6].

## Key findings
### Architectural Mechanisms and Dynamics
Transformers rely on self-attention mechanisms that incur quadratic computational complexity ($O(N^2)$) and large key-value cache memory footprints [4, 7]. In contrast, State Space Models (SSMs) derive linear-time sequence modeling from continuous-time systems theory and structured recurrent dynamics [4, 5]. Early SSMs use structured diagonal parameterizations, while models like Mamba introduce selective state spaces where parameters become input-dependent functions to enable dynamic memory propagation [4, 5]. Mathematically, frameworks like structured state space duality connect SSMs to attention mechanisms using structured semiseparable matrices [3, 4]. Additionally, layer necessity and plasticity anti-align in Transformers, whereas they overlap in Mamba-style SSMs [8].

### Computational Efficiency and Scaling Behavior
Transformers face quadratic memory and computational bottlenecks ($O(N^2)$) from $N \times N$ attention matrix calculations, which restrict scaling on long contexts [3, 9]. SSMs process tokens recurrently using a fixed-size latent state ($h_t \in \mathbb{R}^{d_{\text{state}}}$), achieving linear $O(N)$ computational and memory scaling [3, 9]. While short sequences under 8,000 tokens may run up to 1.9x faster on Transformers, SSMs exhibit a performance inversion around 57,000 tokens, operating up to 4x faster with a ~64% lower memory footprint [9]. Despite these long-context advantages, custom SSM selective scan operators can account for over 55% of inference latency on edge platforms due to their sequential, element-wise nature [9]. Furthermore, the fixed-size latent state imposes fundamental limits on long-context copying and information retrieval [1].

### Cross-Domain Performance and Capabilities
In natural language processing, SSMs provide linear memory scaling and competitive or superior robustness relative to Transformers [3, 10]. For vision and language modeling, Mamba-based SSMs outperform Transformers up to 3B parameters on tasks such as captioning, question answering, and reading comprehension, though Transformers retain superior performance in visual grounding [11]. Across domains, Transformers significantly outperform pure SSMs on tasks requiring precise in-context retrieval, string copying, and associative recall, while SSMs rely on a fixed-size latent state or incremental representation updates [1, 2, 11]. Hybrid architectures bridge these gaps by combining SSMs and self-attention layers to match or exceed Transformer performance in data efficiency and retrieval tasks [2, 4]. SSMs are also evaluated alongside Transformers in audio processing and genomic sequence analysis [4].

## Comparison of approaches
| Approach | Core idea | Strengths | Limitations | Papers |
| :--- | :--- | :--- | :--- | :--- |
| Transformers | Quadratic self-attention over the full context ($O(N^2)$ complexity) [4, 7] | Superior in-context retrieval, precise position retrieval, and string copying capabilities [1, 2] | Quadratic memory/compute scaling and high key-value cache requirements [4, 7] | [1, 2, 3, 7, 8] |
| State Space Models (SSMs) | Linear-time sequence modeling via continuous-time systems and selective recurrences ($O(N)$) [4, 5] | Linear scaling, high inference throughput, and lower memory footprint on long sequences [3, 5, 9] | Poorer associative recall, copying, and retrieval; high edge-device latency from sequential scan kernels [1, 7, 9] | [3, 4, 5, 7, 9, 10, 11] |
| Hybrid Architectures | Integration of SSM layers with fractions of self-attention or priming mechanisms [2, 4, 6] | Combines linear-time scaling with attention-based eidetic memory; closes associative recall gaps [2, 4, 6] | Requires careful balancing of attention fractions and specialized architectural co-design [3, 4, 7] | [2, 3, 4, 6, 7] |

## Open problems
* Addressing the fundamental limitations of fixed-size latent states in pure SSMs regarding associative recall, precise position retrieval, and copying tasks [1, 2, 7].
* Mitigating runtime overhead and latency bottlenecks caused by sequential custom kernels like selective scans on edge platforms [9].
* Expanding multimodal and cross-domain extensions, such as evaluating visual grounding in vision-language models and optimizing genomic or audio modeling [4, 11].
* Optimizing hybrid architectures through strategies like priming and tuning the precise fraction of self-attention required to close downstream performance gaps [3, 6, 7].
* Analyzing structural characteristics related to representation flow and layer plasticity across deep sequences [3, 8].

## References
1. Samy Jelassi et al. (2024). Repeat After Me: Transformers are Better than State Space Models at Copying. *arXiv (cs.LG)*. https://arxiv.org/abs/2402.01032
2. Georgios Pantazopoulos et al. (2026). Retrievit: In-context Retrieval Capabilities of Transformers, State Space Models, and Hybrid Architectures. *arXiv (cs.AI)*. https://arxiv.org/abs/2603.02874
3. Abidemi Koledoye et al. (2026). Benchmarking the Computational and Representational Efficiency of State Space Models against Transformers on Long-Context Dyadic Sessions. *arXiv (cs.LG)*. https://arxiv.org/abs/2601.01237
4. Juby George (2026). Attention-Free Transformers: State Space Models as Scalable Alternatives. *Eduschool International Journal of Data Science and Machine learning (EIJDSML)*. https://doi.org/10.63090/eijdsml/3139.7344.0002
5. Albert Gu and Tri Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv (Cornell University)*. https://doi.org/10.48550/arxiv.2312.00752
6. Aditya Chattopadhyay et al. (2026). Priming: Hybrid State Space Models From Pre-trained Transformers. *arXiv (Cornell University)*. https://doi.org/10.48550/arxiv.2605.08301
7. Roger Waleffe et al. (2024). An Empirical Study of Mamba-based Language Models. *arXiv (Cornell University)*. https://doi.org/10.48550/arxiv.2406.07887
8. Istabrak Abbes et al. (2026). What Does Layer-Importance Reveal About Transformers and State-Space Models?. *arXiv (Cornell University)*. https://doi.org/10.48550/arxiv.2609.16537
9. Saptarshi Mitra et al. (2025). Characterizing State Space Model and Hybrid Language Model Performance with Long Context. *arXiv (cs.AR)*. https://arxiv.org/abs/2507.12442
10. Saron Tua Parsaoran L Tobing et al. (2025). Beyond Transformers: Evaluating the Robustness and Efficiency of State-Space Models for Next-Generation Natural Language Processing. *Jurnal Info Sains Informatika dan Sains*. https://ejournal.seaninstitute.or.id/index.php/InfoSains/article/view/7182
11. Georgios Pantazopoulos et al. (2024). Shaking Up VLMs: Comparing Transformers and Structured State Space Models for Vision & Language Modeling. *Conference on Empirical Methods in Natural Language Processing (EMNLP)*. https://arxiv.org/abs/2409.05395

---
*Generated by ScholarScout: 15 papers read, 11 cited, 26/26 cited statements confirmed by the critic agent.*
