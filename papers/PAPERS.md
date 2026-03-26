# Curated Reading List — EBM + RL + Transformers for Wind Farm Control

> 18 papers organized by topic. Each entry: citation, DOI, summary, key takeaway, and relevance to our transformer-based SAC wind farm agent + EBM research direction.

---

## A. Foundations: Maximum Entropy RL & Energy-Based Models

### Haarnoja et al. (2017) "Reinforcement Learning with Deep Energy-Based Policies"
**DOI:** [`10.48550/arxiv.1702.08165`](https://doi.org/10.48550/arxiv.1702.08165)

**Summary:** Proposes Soft Q-Learning, which expresses the optimal policy as a Boltzmann distribution over a learned energy function (the soft Q-function), enabling expressive multimodal policies in continuous action spaces. Uses amortized Stein Variational Gradient Descent (SVGD) to draw approximate samples from this energy-based policy, demonstrating improved exploration and compositionality (skill transfer between tasks) on simulated locomotion.

**Key Takeaway:** The optimal maximum-entropy policy is literally a Boltzmann distribution over the Q-function — making Q-learning equivalent to training an EBM where low-energy actions are high-probability actions.

**Relevance:** THE direct ancestor of our SAC codebase. The transformer critic already defines an energy landscape over per-turbine yaw actions, and the actor is an amortized sampler from that landscape. This paper provides the theoretical foundation for EBM-native training objectives (contrastive losses, Langevin sampling) as alternatives to SAC's reparameterization trick.

---

### Haarnoja et al. (2018) "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
**DOI:** [`10.48550/arxiv.1801.01290`](https://doi.org/10.48550/arxiv.1801.01290)

**Summary:** Reformulates Soft Q-Learning into a practical, stable actor-critic algorithm by replacing the SVGD sampler with a reparameterized squashed Gaussian actor and introducing clipped double-Q critics. Optimizes the maximum entropy objective J = E[sum(r + alpha * H(pi))], with automatic temperature alpha tuning. One of the most widely adopted deep RL algorithms (2700+ citations).

**Key Takeaway:** SAC trades the expressive multimodal energy-based policy of Soft Q-Learning for a tractable unimodal Gaussian, gaining massive stability and scalability — but this is precisely the approximation that EBM research could revisit.

**Relevance:** The exact algorithm our transformer wind farm agent uses. Understanding it as a "simplified EBM" (where the Gaussian actor is a restricted amortized sampler) clarifies what expressiveness is lost and what EBM extensions could recover for multi-turbine coordination.

---

### LeCun, Chopra & Hadsell (2006) "Energy-Based Models"
**DOI:** [`10.7551/mitpress/7443.003.0014`](https://doi.org/10.7551/mitpress/7443.003.0014)

**Summary:** The canonical 55-page tutorial defining the EBM framework: models that associate a scalar energy E(Y, X) to each configuration of variables, with inference by energy minimization and learning by shaping the energy surface. Systematically categorizes loss functions (contrastive, margin-based, negative log-likelihood) and architectures (explicit energy functions vs. latent-variable models).

**Key Takeaway:** Any model that maps inputs to a scalar compatibility score and performs inference by optimization over that score is an energy-based model — this framing is far more general than probabilistic models since EBMs need not compute normalized probabilities.

**Relevance:** Provides the theoretical lens to reinterpret our transformer critic: the Q-network is an energy function E(s,a), SAC's policy is inference by approximate energy minimization, and the Bellman loss is one particular way to shape the energy surface — EBM-native losses could offer alternatives that better exploit the transformer's attention structure over turbine tokens.

---

### Du & Mordatch (2019) "Implicit Generation and Generalization in Energy-Based Models"
**DOI:** [`10.48550/arxiv.1903.08689`](https://doi.org/10.48550/arxiv.1903.08689)

**Summary:** Demonstrates that continuous EBMs trained with MCMC-based contrastive divergence (using Langevin dynamics initialized from a replay buffer) can scale to high-dimensional domains including ImageNet and robotic trajectory prediction. Highlights unique EBM capabilities: compositionality (combining independently trained energy functions by addition), OOD detection, adversarial robustness, and continual learning.

**Key Takeaway:** Energy functions compose by simple addition — independently trained EBMs for different concepts can be combined at inference time to generate samples satisfying all concepts simultaneously, without retraining.

**Relevance:** The compositionality property is directly applicable to wind farm control: separate energy functions for power maximization, fatigue reduction, and wake steering could be trained independently and composed via energy addition at inference time. The trajectory prediction results show EBMs can model multi-step dynamics relevant to sequential wind farm decisions.

---

## B. Diffusion & Score-Based Policies

### Chi et al. (2023) "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
**DOI:** [`10.15607/rss.2023.xix.026`](https://doi.org/10.15607/rss.2023.xix.026)

**Summary:** Formulates robot visuomotor policies as conditional DDPMs over the action space. Instead of directly outputting actions, the policy infers action-score gradients conditioned on observations through K denoising iterations. Demonstrates strong multimodal action distribution handling and state-of-the-art performance across 12 tasks. Proposes a transformer-based diffusion network architecture.

**Key Takeaway:** Diffusion models as policy representations naturally handle multimodal action distributions with stable training, and the iterative denoising process is synergistic with position-control action spaces.

**Relevance:** Directly applicable to our architecture — per-turbine yaw action outputs could be replaced by a diffusion process that denoises yaw actions conditioned on turbine-token embeddings, naturally capturing multimodal yaw strategies from wake interactions. The paper's transformer-based architecture is especially relevant given the existing turbines-as-tokens design.

---

### Janner et al. (2022) "Planning with Diffusion for Flexible Behavior Synthesis" (Diffuser)
**DOI:** [`10.48550/arxiv.2205.09991`](https://doi.org/10.48550/arxiv.2205.09991)

**Summary:** Uses an unconditional diffusion model to generate full trajectories (sequences of states and actions) as the primary planning mechanism. Treats planning as iterative trajectory refinement, analogous to image generation. Uses classifier-guided sampling with a learned reward function to steer denoising toward high-return trajectories. Enables flexible composition of objectives at test time.

**Key Takeaway:** Trajectory-level diffusion planning replaces traditional policy-value decomposition with a single generative model, enabling flexible reward composition and constraint satisfaction at inference time through guided sampling.

**Relevance:** The trajectory-level generation paradigm maps well to wind farm control where the full joint yaw configuration across all turbines must be coordinated — Diffuser's approach could produce globally coherent farm-wide yaw plans rather than greedy per-turbine decisions. The composable reward guidance is appealing for balancing power maximization with fatigue constraints.

---

### Ajay et al. (2022) "Is Conditional Generative Modeling all you Need for Decision-Making?" (Decision Diffuser)
**DOI:** [`10.48550/arxiv.2211.15657`](https://doi.org/10.48550/arxiv.2211.15657)

**Summary:** Employs a conditional diffusion model with classifier-free guidance to generate state-only trajectories conditioned on returns, constraints, or skill descriptors. Unlike Diffuser, eliminates the need for a separate reward classifier. Uses an inverse dynamics model to extract actions from planned state trajectories. Enables composable conditioning on multiple objectives simultaneously.

**Key Takeaway:** Classifier-free guidance on diffusion trajectory models enables direct conditioning on returns, constraints, and skills that can be composed at inference time without retraining.

**Relevance:** The composable conditioning framework is highly relevant for wind farm operators who may want to simultaneously condition on target power output, fatigue constraints, and noise limits. State-only trajectory generation with inverse dynamics could produce interpretable farm-state plans from which yaw actions are derived.

---

### Wang et al. (2022) "Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning" (Diffusion-QL)
**DOI:** [`10.48550/arxiv.2208.06193`](https://doi.org/10.48550/arxiv.2208.06193)

**Summary:** Uses a diffusion model directly as an expressive policy class within an actor-critic framework for offline RL. The diffusion policy learns to generate actions via iterative denoising (implicit behavior regularization keeping actions near offline data), while Q-values guide the process toward high-return actions. Training combines diffusion loss with Q-value maximization.

**Key Takeaway:** Using a diffusion model as the policy in an actor-critic loop provides expressive behavior regularization for offline RL — the diffusion loss naturally constrains actions to the data support while Q-value guidance pushes toward optimality.

**Relevance:** Most directly relevant upgrade path for our SAC architecture. Since SAC already has the actor-critic structure and the EBM connection (pi proportional to exp(Q/alpha)), Diffusion-QL shows how to replace the Gaussian actor with a diffusion-based actor for richer action distributions. A near-drop-in upgrade adding multimodal expressiveness.

---

### Chen et al. (2024) "Boosting Continuous Control with Consistency Policy" (CPQL)
**DOI:** [`10.48550/arxiv.2310.06343`](https://doi.org/10.48550/arxiv.2310.06343)

**Summary:** Addresses the major practical limitation of diffusion-based RL: slow multi-step denoising inference. Inspired by consistency models, CPQL establishes a mapping from the reverse diffusion trajectory to the desired policy enabling single-step action generation from noise. Achieves state-of-the-art on 11 offline and 21 online RL tasks with ~45x speedup over Diffusion-QL.

**Key Takeaway:** Consistency distillation reduces diffusion policy inference from many denoising steps to a single step, making diffusion-based RL policies practical for real-time continuous control.

**Relevance:** Critical for practical wind farm deployment where real-time yaw adjustments are needed. If diffusion-based policies are adopted, CPQL solves the inference speed bottleneck — going from ~50 denoising steps to 1 step makes diffusion policies viable for both the online SAC training loop and real-time farm control.

---

## C. Implicit / EBM Policies

### Florence et al. (2021) "Implicit Behavioral Cloning"
**DOI:** [`10.48550/arxiv.2109.00137`](https://doi.org/10.48550/arxiv.2109.00137)

**Summary:** Replaces explicit (feedforward) policy networks in behavioral cloning with implicit models where a learned energy function E(observation, action) is trained via contrastive learning (InfoNCE-style loss) and actions are selected by minimizing energy using Langevin dynamics MCMC or derivative-free methods. Naturally handles multimodal action distributions. Demonstrates strong results on robotics manipulation tasks.

**Key Takeaway:** Representing policies as energy functions E(s,a) rather than direct mappings s→a enables naturally multimodal action distributions and avoids mode-averaging, at the cost of iterative optimization at inference time and known training instability.

**Relevance:** The most direct precedent for explicit EBM policies in a control setting. The training instability challenges documented here are critical to understand before pursuing explicit EBM policies. The SAC framework already has an implicit EBM interpretation (policy proportional to exp(Q/alpha)), and IBC shows what happens when this is made fully explicit.

---

### Cao et al. (2023) "Enhancing OOD Generalization in Offline RL with Energy-Based Policy Optimization" (EBPO)
**DOI:** [`10.3233/faia230288`](https://doi.org/10.3233/faia230288)

**Summary:** Introduces Energy-Based Policy Optimization (EBPO), which uses an energy function derived from the offline data distribution to evaluate and guide out-of-distribution generalization, rather than conservatively constraining the policy to in-distribution actions. The energy function quantifies how far a state-action pair is from the data support, enabling principled balance between return maximization and distribution shift risk. Incorporates episodic memory for sample efficiency.

**Key Takeaway:** Energy functions derived from offline data provide a more nuanced mechanism for managing OOD generalization than binary in/out-of-distribution constraints, enabling controlled generalization beyond training data.

**Relevance:** Directly addresses a core challenge for wind farm RL: the policy must generalize to unseen farm layouts and wind conditions. EBPO's energy-based data-distribution awareness could complement the transformer-SAC architecture's zero-shot layout generalization, preventing overconfident extrapolation to novel configurations while allowing useful generalization.

---

## D. Compositional Generation

### Du et al. (2020) "Compositional Visual Generation and Inference with Energy Based Models"
**DOI:** [`10.48550/arxiv.2004.06030`](https://doi.org/10.48550/arxiv.2004.06030)

**Summary:** Proposes composing multiple independently-trained EBMs by summing their energy functions (product of experts in probability space). Samples are obtained via composed Langevin dynamics on the summed energy landscape. Enables conjunction (AND), negation (NOT), and other logical operators over visual concepts at generation time without retraining. Demonstrates zero-shot combinatorial generalization.

**Key Takeaway:** Independent EBMs compose algebraically at inference time — energy addition corresponds to concept conjunction, enabling zero-shot combinatorial generalization to concept combinations never seen during training.

**Relevance:** The most direct analog to our compositional wind farm objectives: separate energy functions for power, loads, and wake steering could be trained independently and composed at deployment time, with adjustable weighting — exactly the operator-facing capability that makes EBMs uniquely valuable for wind farm control.

---

### Liu et al. (2022) "Compositional Visual Generation with Composable Diffusion Models"
**DOI:** [`10.1007/978-3-031-19790-1_26`](https://doi.org/10.1007/978-3-031-19790-1_26)

**Summary:** Extends the EBM compositionality framework to diffusion models by noting that score functions (gradient of log-probability) are analogous to energy gradients. Defines composition operators (conjunction via score addition, negation via score subtraction) over diffusion model score functions. Achieves improved compositional generation quality compared to monolithic text-to-image models.

**Key Takeaway:** Diffusion model score functions compose using the same algebraic operators as EBM energy gradients, establishing compositionality as a general property of energy/score-based models.

**Relevance:** Validates that EBM composition generalizes to score-based formulations. Since the SAC policy gradient is itself a score function (gradient of soft Q-value), per-objective Q-functions could be composed using the same conjunction/negation operators — enabling modular multi-objective wind farm control where new objectives can be added at deployment without retraining.

---

## E. World Models & Model-Based

### Moerland et al. (2017) "Learning Multimodal Transition Dynamics for Model-Based Reinforcement Learning"
**DOI:** [`10.48550/arxiv.1705.00470`](https://doi.org/10.48550/arxiv.1705.00470)

**Summary:** Systematically compares deep generative models (GANs, normalizing flows, VAEs) for learning stochastic, multimodal transition dynamics in model-based RL. Argues that conditional variational inference (CVAEs) is most suitable because it provides tractable sampling, latent-level planning, and proper density estimation. Evaluates on synthetic functions and stochastic gridworlds.

**Key Takeaway:** Conditional variational inference is the most suitable generative framework for learning multimodal transition models in RL, uniquely combining tractable density estimation, easy sampling, and latent-space planning.

**Relevance:** Wind farm dynamics are inherently stochastic and multimodal (turbulent wind fields create branching wake interactions). The CVAE-based world model approach could serve as a foundation for EBM-style transition models over turbine-token states, where the latent variable captures multimodal uncertainty in wake propagation.

---

### Peng et al. (2024) "Implicit Policy Constraint for Offline Reinforcement Learning"
**DOI:** [`10.1049/cit2.12304`](https://doi.org/10.1049/cit2.12304)

**Summary:** Proposes Implicit Policy Constraint (IPC), which uses EBMs to implicitly represent the behavior policy in offline RL. The EBM assigns low energy to in-distribution state-action pairs and high energy to OOD ones, naturally handling multimodal behavior policies. Built on SAC, the method derives an energy-based policy constraint that combines entropy maximization with energy minimization. The learned energy function additionally serves as an OOD detector at deployment.

**Key Takeaway:** EBMs can replace explicit behavior policy modeling in offline RL by providing unnormalized probability estimates that naturally distinguish in-distribution from OOD state-action pairs, yielding both better learning and a built-in safety signal.

**Relevance:** Directly demonstrates EBMs integrated with SAC — the same RL algorithm we use. The EBM's ability to handle multimodal data from diverse sources maps to learning from multiple farm layouts, and OOD detection is valuable for safety-critical turbine control. Reinforces the "completing the circle" narrative between SQL, SAC, and explicit EBM policies.

---

## F. Wind Farm RL Context

### Stanfel et al. (2020) "A Distributed Reinforcement Learning Yaw Control Approach for Wind Farm Energy Capture Maximization"
**DOI:** [`10.23919/acc45564.2020.9147946`](https://doi.org/10.23919/acc45564.2020.9147946)

**Summary:** One of the first RL-based approaches to wind farm yaw control for wake steering. Proposes a distributed multi-agent temporal difference RL algorithm where individual turbines act as agents optimizing farm-level power output, comparing against FLORIS-based lookup table baselines. Demonstrates that model-free RL can match or exceed model-based yaw schedules.

**Key Takeaway:** Distributed per-turbine RL agents can learn cooperative yaw misalignment strategies that maximize farm-level power without requiring an analytical wake model.

**Relevance:** Foundational paper for RL-based yaw control. Our transformer-SAC agent extends this by replacing independent per-turbine agents with a single transformer policy that processes all turbines as tokens — enabling zero-shot layout generalization that per-agent approaches cannot achieve.

---

### Dong & Zhao (2023) "Data-Driven Wind Farm Control via Multiplayer Deep Reinforcement Learning"
**DOI:** [`10.1109/tcst.2022.3223185`](https://doi.org/10.1109/tcst.2022.3223185)

**Summary:** Introduces multiplayer deep RL (MPDRL) for wind farm power maximization under time-varying winds. Entirely data-driven and model-free, frames the problem as a multi-player game where each turbine has its own DNN policy. Achieves 30%+ power gains over greedy baselines on WFSim. Explicitly addresses time-delayed wake effects.

**Key Takeaway:** Deep RL achieves substantial farm-level power gains in dynamic wind conditions by treating each turbine as an independent learner, without any wake model.

**Relevance:** Closest prior work to our approach in terms of using deep neural networks for per-turbine control. The key limitation our transformer architecture addresses: Dong & Zhao train separate networks per turbine (requiring retraining for new layouts), whereas our turbines-as-tokens transformer shares a single policy across all farm sizes.

---

### Sheehan et al. (2024) "Graph-based Deep Reinforcement Learning for Wind Farm Set-Point Optimisation"
**DOI:** [`10.1088/1742-6596/2767/9/092028`](https://doi.org/10.1088/1742-6596/2767/9/092028)

**Summary:** Combines Graph Neural Networks with DDPG for wind farm wake steering, representing inter-turbine wake interactions as a directed graph. Edges encode potential wake connections under varying wind directions. Achieves 6.5% additional power over greedy control on a nine-turbine array. Makes topology-awareness explicit in the architecture.

**Key Takeaway:** Encoding inter-turbine wake topology as a graph structure enables the RL agent to reason about spatial turbine relationships rather than treating observations as flat vectors.

**Relevance:** Most architecturally relevant prior work. Both approaches recognize that structured representation of turbine interactions improves control — Sheehan et al. use GNNs with explicit graph edges, while our project uses a transformer with positional encodings (14+ variants including spatial, directional, and GAT-based) that let attention implicitly learn interaction patterns. Our approach is more flexible since it does not require pre-defined graph topology.

---

## Additional References

Papers encountered during the literature search that may be useful for future reference:

- **Bizon Monroc et al. (2025)** "WFCRL: A Multi-Agent Reinforcement Learning Benchmark for Wind Farm Control" — DOI: [`10.48550/arxiv.2501.13592`](https://doi.org/10.48550/arxiv.2501.13592) — NeurIPS 2024 benchmark for MARL in wind farm control; useful for standardized evaluation.
- **Dong & Zhao (2022)** "Intelligent Wind Farm Control via Grouping-Based Reinforcement Learning" — DOI: [`10.23919/ecc55457.2022.9838151`](https://doi.org/10.23919/ecc55457.2022.9838151) — Addresses scalability of DRL to large farms via turbine grouping.

---

## Reading Order Recommendation

For the EBM+RL research direction, read in this order:

1. **Haarnoja et al. (2017)** — Understand the SAC→EBM connection (Soft Q-Learning IS an EBM)
2. **Haarnoja et al. (2018)** — Understand what SAC kept/dropped from SQL
3. **Du & Mordatch (2019)** — Understand energy compositionality
4. **Chi et al. (2023)** — Understand state-of-the-art diffusion policy
5. **Wang et al. (2022)** — Understand diffusion+Q-learning integration (Diffusion-QL)
6. **Chen et al. (2024)** — Understand single-step inference via consistency models
7. **Florence et al. (2021)** — Understand EBM policy training challenges
8. **Du et al. (2020)** — Understand compositional generation with EBMs

Then domain-specific:
9. **Stanfel et al. (2020)** and **Dong & Zhao (2023)** — Baseline wind farm RL approaches
10. **Sheehan et al. (2024)** — Structured (graph) architectures for wind farm RL
