## Spectral and Structural Analysis of Network Fine-Tuning

### 1. Introduction: Beyond Scalar Loss

The process of fine-tuning a pre-trained model on a downstream task is a cornerstone of modern machine learning. While success is often measured by a single scalar metric, such as accuracy or loss, this provides little insight into the underlying mechanics of adaptation. How does a network truly change? Does it make subtle refinements to its existing representations, or does it learn fundamentally new features? To answer these questions, we must move beyond scalar performance and develop tools to dissect the changes in the network's weights.

This chapter introduces a suite of techniques rooted in linear algebra to analyze the weight difference matrix, $\Delta W$, between a fine-tuned model and its base checkpoint. For a given layer, we define this as:

$$
\Delta W = W_{\text{finetuned}} - W_{\text{base}}
$$

In architectures like the Transformer Networks, which are built upon residual connections, this $\Delta W$ matrix represents the learned **correction term** applied to the network's internal representations. By characterizing the magnitude, structure, and direction of $\Delta W$, we can formulate a precise and quantitative understanding of the learning process.

### 2. Global Analysis: Visualizing and Quantifying the Change

Before dissecting the change's interaction with the base model's structure, we first analyze the properties of the difference matrix, $\Delta W$, in a global sense. This initial step provides a high-level "fingerprint" of the update, answering fundamental questions about its scale and intrinsic complexity.

#### 2.1. The Normalized Singular Value Spectrum

The most direct and informative visualization of the change is the **Normalized Singular Value Spectrum**. This is a log-log plot of the singular values of the change matrix, $\sigma_i(\Delta W)$, normalized by the Frobenius norm of the base weight matrix, $\|W_{\text{base}}\|_F$.

$$
\text{Plot: } \quad \frac{\sigma_i(\Delta W)}{\|W_{\text{base}}\|_F} \quad \text{vs.} \quad i
$$

This single plot reveals two key properties simultaneously:

1.  **Magnitude of Change:** The y-intercept of the curve (the value for $i=1$) shows the size of the largest principal component of the change relative to the total scale of the original layer. It gives an immediate sense of the update's significance.
2.  **Structure of Change:** The rate of decay of the curve (its slope on the log-log plot) reveals the intrinsic structure of the update. A steep decay indicates a highly structured, low-rank change concentrated in a few directions. A slow decay suggests a more diffuse, complex, or noisy change spread across many directions.

#### 2.2. Quantifying the Spectrum: Summary Metrics

While magnitude metrics capture the overall scale of the update, structural metrics aim to quantify the _shape_ of the spectral decay curve. They answer questions about the nature of the change: Was it a simple, low-rank update concentrated in a few directions, or a complex, high-rank update spread across many? We can probe this structure from two distinct perspectives: by analyzing the change's internal properties (self-normalization) or by measuring its impact relative to the original layer (base-normalization).

- **Effective Rank:** This metric provides a single, elegant measure of the change's intrinsic complexity. It is inherently self-normalized, answering the question: "How 'spread out' is the energy of the change among its principal components?" A low effective rank indicates that the change's energy is highly concentrated in its first few singular values, corresponding to a steep spectral decay and a simple, low-rank update. Conversely, a high effective rank, approaching the matrix's full rank, signifies a complex or noisy update whose energy is distributed more evenly across many directions.

  $$
  r_{\text{eff}}(\Delta W) = \frac{\|\Delta W\|_F^2}{\sigma_{\max}^2(\Delta W)} = \frac{\sum_i \sigma_i^2(\Delta W)}{\sigma_1^2(\Delta W)}
  $$

- **Energy Concentration Rank ($k_{p}$):** This metric offers a more granular and interpretable view of the spectral decay by asking: "How many principal components are required to capture a percentage $p$ of a certain total energy?" The choice of what this total energy represents—the energy of the change itself or the energy of the original layer—gives rise to two different but complementary metrics.

  1.  **Self-Normalized $k_{p}$ (Internal Structure):** In this formulation, the cumulative energy of the change is normalized by the total energy of the change itself.

      $$
      k_{p}^{\text{self}}(\Delta W) = \min \left\{ k \;\bigg|\; \frac{\sum_{i=1}^k \sigma_i^2(\Delta W)}{\|\Delta W\|_F^2} \ge \frac{p}{100} \right\}
      $$

      **Interpretation:** This metric characterizes the _internal structure_ of the update, independent of its overall size. A small $k_{90}^{\text{self}}$ signifies that the change, as a standalone operation, is simple and low-rank. It tells us whether the update was a "targeted strike" (low $k_{90}$) or a "diffuse wave" (high $k_{90}$), providing a pure measure of the update's shape.

  2.  **Base-Normalized $k_{p}$ (Relative Impact):** Here, the cumulative energy of the change is normalized by the total energy of the original base layer.

      $$
      k_{p}^{\text{base}}(\Delta W) = \min \left\{ k \;\bigg|\; \frac{\sum_{i=1}^k \sigma_i^2(\Delta W)}{\|W_{\text{base}}\|_F^2} \ge \frac{p}{100} \right\}
      $$

      **Interpretation:** This metric measures the _disruptive power_ or _relative impact_ of the change. It answers a fundamentally different and pragmatically crucial question: "How many principal directions of change are needed to accumulate an energy equivalent to $p\%$ of the original layer's entire energy?" This normalization provides a powerful tool for isolating the most significant components of the change, as it anchors the notion of "significance" to an external, meaningful reference: the scale of the original layer. A component of the update is only considered important if it is "loud enough" to be heard over the existing machinery of the layer.

      Consequently, this metric is exceptionally well-suited for guiding the construction of an efficient, filtered update. When creating a compressed representation of the change, the primary goal is to preserve the components that have a meaningful impact on the network's function. The base-normalized metric provides a principled threshold for this task; by setting a small target like $p=1$, $k_{1}^{\text{base}}$ directly tells us how many components we must keep to capture a non-trivial (1%) portion of the original layer's energy.

By contrast, the self-normalized metric, $k_{p}^{\text{self}}$, evaluates the update in a vacuum. While excellent for describing the update's intrinsic shape, it is agnostic to its overall importance. Using it to guide filtering could be misleading, as it might identify the "top 90%" of a change that is, in its entirety, completely negligible compared to the base weights. This would be akin to meticulously analyzing the structure of a whisper while ignoring the roar of a hurricane happening in the same room. The base-normalized metric avoids this pitfall by ensuring that only components contributing to a meaningful, absolute level of energy are deemed significant, making it the more reliable guide for creating impactful, low-rank approximations of the fine-tuning update.

#### 2.3. Practical Considerations: Randomized SVD

The computation of the singular value spectrum for the weight matrices found in large models like Transformers Networks presents a significant computational challenge. A full SVD is often prohibitively expensive in terms of both time and memory.

Fortunately, we are typically interested in the "head" of the spectrum, as the singular values of neural network weight matrices tend to decay rapidly. This structure makes them amenable to **Randomized SVD** algorithms. These methods use randomized projections to create a much smaller "sketch" of the large matrix, from which the leading singular values and vectors can be computed with high accuracy and dramatic gains in efficiency. For all the analyses proposed in this chapter, the use of a randomized SVD to compute the spectrum of both $W_{\text{base}}$ (if needed) and $\Delta W$ is not just recommended; it is a practical necessity.

### 3. Decomposing Change in the Base Model's Basis

To understand how the change $\Delta W$ interacts with the pre-existing computational structure of the network, we must analyze it from the perspective of the base model. This is achieved by changing the basis of our analysis from the standard Euclidean basis to the singular basis of $W_{\text{base}}$.

Let the Singular Value Decomposition (SVD) of the base matrix be $W_{\text{base}} = U_{\text{base}} \Sigma_{\text{base}} V_{\text{base}}^T$. The columns of $V_{\text{base}}$ form an orthonormal basis of principal input directions, and the columns of $U_{\text{base}}$ form an orthonormal basis of principal output directions. We project $\Delta W$ into this new coordinate system to get the rotated change matrix, $\Delta \tilde{W}$:

$$
\Delta \tilde{W} = U_{\text{base}}^T \Delta W V_{\text{base}}
$$

A critical property of this transformation is that, because $U_{\text{base}}$ and $V_{\text{base}}$ are unitary matrices, they preserve the Frobenius norm. This gives us the fundamental identity linking our global analysis to this new structural view:

$$
\|\Delta W\|_F^2 = \|\Delta \tilde{W}\|_F^2
$$

This identity is the cornerstone of our detailed analysis. It tells us that the total energy of the update, which we measured globally in Section 2, is conserved in this new basis. The matrix $\Delta \tilde{W}$ is therefore not a new object, but a **disaggregation of the total change energy** across the base model's principal modes.

An element $(\Delta \tilde{W})_{ij}$ quantifies how the fine-tuning process couples the base model's $j$-th principal input direction to its $i$-th principal output direction. The total energy can now be partitioned:

$$
\|\Delta W\|_F^2 = \|\Delta \tilde{W}\|_F^2 = \sum_{i,j} (\Delta \tilde{W})_{ij}^2 = \underbrace{\sum_{i} (\Delta \tilde{W})_{ii}^2}_{\text{Diagonal Energy}} + \underbrace{\sum_{i \neq j} (\Delta \tilde{W})_{ij}^2}_{\text{Off-Diagonal Energy}}
$$

This decomposition is profoundly insightful. It allows us to attribute the total change energy to two distinct, interpretable mechanisms:

1.  **Re-weighting (Diagonal Energy):** Energy on the diagonal represents changes that act along the original singular modes, amplifying or suppressing existing feature transformations.
2.  **Re-wiring (Off-Diagonal Energy):** Energy off the diagonal represents the creation of new couplings between previously orthogonal input and output modes.

Furthermore, we can analyze how this energy is distributed across the hierarchy of the base model's modes. By summing the energy in the first $k$ rows or columns of $\Delta \tilde{W}$, we can precisely measure the fraction of the change that impacts the top-$k$ most important output or input subspaces of the base model. This provides the foundation for the detailed structural metrics introduced in the following sections.

### 4. Structural Analysis: Re-weighting vs. Re-wiring

The rotated change matrix, $\Delta \tilde{W}$, is a rich diagnostic tool. From it, we can answer two fundamental questions about the fine-tuning process:

1.  Is the change primarily **re-weighting** existing computational pathways, or is it **re-wiring** them to create new ones?
2.  Is the energy of the change concentrated on the most important input/output modes of the base model, or is it happening in the periphery?

#### 4.1. Re-weighting vs. Re-wiring: The Diagonal and Off-Diagonal

The distinction between re-weighting and re-wiring is encoded in the distribution of energy within $\Delta \tilde{W}$.

- **Structural Alignment:** This metric quantifies the degree to which fine-tuning acts along the original singular modes. It is calculated as the fraction of the change's total energy that lies on the diagonal of $\Delta \tilde{W}$. A value near 1 indicates that the update is almost exclusively **re-weighting**—amplifying or suppressing the base model's existing feature transformations.

  $$
  \text{Structural Alignment} = \frac{\sum_i (\Delta \tilde{W})_{ii}^2}{\|\Delta W\|_F^2}
  $$

- **Mode Mixing Ratio:** Complementary to alignment, this metric measures the energy off the diagonal. A high value signifies **re-wiring**, where fine-tuning learns new, non-trivial couplings between input and output modes that were previously orthogonal in the base model's coordinate system.
  $$
  \text{Mode Mixing Ratio} = \frac{\sum_{i \neq j} (\Delta \tilde{W})_{ij}^2}{\|\Delta W\|_F^2} = 1 - \text{Structural Alignment}
  $$

#### 4.2. Concentration of Change: The Subspace Alignment Curve

Beyond the re-weighting/re-wiring dynamic, a critical question is _where_ in the base model's feature hierarchy the change is concentrated. Does fine-tuning primarily alter the most dominant, high-energy features of the base model—the "head" of its singular value spectrum? Or does it operate on the long tail of previously insignificant features to learn new capabilities? The former implies a refinement of core competencies, while the latter suggests the acquisition of novel functions.

To answer this, we introduce the **Subspace Alignment Curve**, a powerful diagnostic plot. The curve is generated by plotting the fraction of the total change energy that is captured by the top-$k$ principal modes of the base model, as a function of $k$. Since the basis vectors in $U_{\text{base}}$ and $V_{\text{base}}$ are ordered by the magnitude of their corresponding singular values $\sigma_i(W_{\text{base}})$, the first few rows and columns of $\Delta \tilde{W}$ correspond to the "head" of the base model, while the later rows and columns represent the "tail."

We can construct this curve from either the input or output perspective:

- **Input Subspace Alignment:** This metric quantifies how much of the update is driven by the base model's top-$k$ input features. It is calculated as the fraction of total energy contained within the first $k$ columns of $\Delta \tilde{W}$.

  $$
  \text{Input Alignment}(k) = \frac{\sum_{j=1}^k \sum_i (\Delta \tilde{W})_{ij}^2}{\|\Delta W\|_F^2}
  $$

- **Output Subspace Alignment:** Symmetrically, this measures how much of the update's effect manifests along the base model's top-$k$ output directions, calculated from the first $k$ rows of $\Delta \tilde{W}$.
  $$
  \text{Output Alignment}(k) = \frac{\sum_{i=1}^k \sum_j (\Delta \tilde{W})_{ij}^2}{\|\Delta W\|_F^2}
  $$

#### 4.3 The Adaptation Matrix: A Unified Diagnostic Space

The metrics developed thus far provide two largely orthogonal lenses through which to view the fine-tuning update. The **Subspace Alignment Curve** reveals the **locus** of the change—_where_ in the base model's feature hierarchy the update occurs. The **Structural Alignment** metric reveals the **mechanism** of the change—_how_ the update is implemented, whether by re-weighting existing pathways or re-wiring them. By combining these two perspectives, we move from a simple list of metrics to a powerful diagnostic space that allows for a nuanced classification of a layer's adaptation strategy.

First, we determine the locus of change by examining the shape of the **Input or Output Subspace Alignment Curve**, $\text{Alignment}(k)$. This curve reveals the concentration of the update's energy:

- **Head-Focused Adaptation:** The curve is concave, rising sharply for small $k$. This signature indicates that the change is concentrated within the most dominant, high-energy modes of the base model. The adaptation is happening at the "head" of the spectrum, modifying the layer's core, pre-existing functionalities.
- **Tail-Focused Adaptation:** The curve is convex, remaining flat for small $k$ before rising. This demonstrates that the change is occurring in the "tail" of the spectrum, building upon directions that were previously unimportant or unused. This is the signature of learning something fundamentally new, orthogonal to the base model's primary operations.

Second, we determine the mechanism of change using the **Structural Alignment** metric, which measures the fraction of energy on the diagonal of $\Delta \tilde{W}$. This distinguishes between two fundamental mechanisms:

- **Re-weighting:** A high Structural Alignment score indicates the change is primarily on the diagonal. The adaptation works by adjusting the gain on existing singular modes.
- **Re-wiring:** A low Structural Alignment score indicates the energy is off-diagonal. The adaptation works by creating new couplings between previously orthogonal input and output modes.

By plotting a layer on a 2D grid defined by these two axes, we create the **Adaptation Matrix**. This framework classifies the update into one of four distinct strategic quadrants, providing a rich, descriptive narrative of the learning process.

|                                                       | **Mechanism: Re-weighting**<br/>(High Structural Alignment)                                                                     | **Mechanism: Re-wiring**<br/>(Low Structural Alignment)                                                                            |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Locus: Head-Focused**<br/>(Concave Alignment Curve) | **1. Refinement**<br/>The model fine-tunes the gains on its most important, pre-existing feature pathways.                      | **2. Re-purposing**<br/>The model leverages its core features but learns to combine them in novel ways to solve a new task.        |
| **Locus: Tail-Focused**<br/>(Convex Alignment Curve)  | **3. Feature Emergence**<br/>The model activates simple, previously dormant features by amplifying them without complex mixing. | **4. Novelty Acquisition**<br/>The model builds entirely new functional pathways from previously unimportant or unused dimensions. |

This matrix transforms our analysis. A layer is no longer just a set of numbers; it is a point in a strategic space. A layer in Quadrant 1 is undergoing **Refinement**, making conservative adjustments to its core competencies. A layer in Quadrant 2 is **Re-purposing** its strongest features for new ends—a sign of flexible knowledge transfer. A layer in Quadrant 4 is engaged in **Novelty Acquisition**, the most profound form of learning, building new knowledge from the ground up in the unused dimensions of its weight space. Finally, Quadrant 3, **Feature Emergence**, captures the interesting case where a simple, previously dormant capability is identified and "switched on." By locating each layer of a network within this diagnostic space, we can construct a granular and powerful narrative of the entire fine-tuning process.

#### Interpreting the Shape of the Alignment Curve

Plotting $\text{Alignment}(k)$ vs. $k$ (from 1 to the full rank) provides a visual signature of the adaptation strategy. The shape of this curve tells a story, typically falling into one of three archetypes:

1.  **Head-Focused Update (Refinement):** The curve exhibits a sharp, concave rise, quickly approaching 1.0 for small values of $k$. This signature indicates that the vast majority of the change energy is concentrated within the most dominant, pre-existing modes of the base layer. This is the hallmark of **refinement**, where the model is fine-tuning its core, established functionalities.

2.  **Tail-Focused Update (Novelty Acquisition):** The curve remains flat and close to zero for small $k$, only beginning to rise significantly for larger values of $k$, resulting in a convex shape. This demonstrates that the change is explicitly orthogonal to the base model's primary features and is instead happening in the "tail" of the spectrum. This is the signature of **novelty acquisition**, where the model learns new capabilities by activating and building upon directions that were previously unimportant.

3.  **Diffuse Update (Uniform Recalibration):** The curve rises steadily and roughly linearly with $k$ (i.e., $\text{Alignment}(k) \approx k/rank$). This signature indicates that the change energy is not concentrated in any particular part of the spectrum but is spread uniformly across all modes—head, body, and tail. This suggests a **global recalibration** of the layer rather than a targeted change.

By characterizing the shape of this curve for each layer, we can build a detailed, data-driven narrative of which parts of the network are being refined and which are learning entirely new functions.

### 5. Normalization: A Component-wise Perspective

To understand the significance of individual re-wiring events $(\Delta \tilde{W})_{ij}$, we must normalize them against a meaningful baseline. The choice of denominator defines the analytical perspective.

- **Input-Normalized Change:** This normalization assesses the strength of a new coupling relative to the original importance of its source input mode. It is ideal for understanding how the processing of specific input features is altered.

  $$
  (\hat{W}_{\text{input}})_{ij} = \frac{(\Delta \tilde{W})_{ij}}{\sigma_j(W_{\text{base}})}
  $$

- **Output-Normalized Change:** This assesses the new coupling's contribution relative to the magnitude of its destination output mode. It is best for analyzing how the composition of the layer's output is modified.

  $$
  (\hat{W}_{\text{output}})_{ij} = \frac{(\Delta \tilde{W})_{ij}}{\sigma_i(W_{\text{base}})}
  $$

- **Symmetric-Normalized Change:** This provides a balanced view, accounting for the importance of both the input and output modes involved. It measures the intrinsic strength of the re-wiring, preventing changes from appearing artificially large if they involve very weak modes.
  $$
  (\hat{W}_{\text{symm}})_{ij} = \frac{(\Delta \tilde{W})_{ij}}{\sqrt{\sigma_i(W_{\text{base}}) \sigma_j(W_{\text{base}})}}
  $$

### 6. A Prescriptive Framework for Analyzing Transformers Networks

The true power of this analytical framework emerges when we move from theory to a prescriptive application. In architectures like the Transformer, different layers serve distinct functions. By tailoring our analysis to the role of each layer, we can construct a precise, data-driven narrative of the fine-tuning process. This section provides a concrete workflow for doing so.

#### 6.1. Analyzing Input-Centric Layers: $W_Q, W_K, W_V$ and MLP $W_1$

**Functional Role:** These four weight matrices—the attention query, key, and value projections, along with the MLP's up-projection—share a common function: they are **input-centric**. Their primary role is to consume the representation vector from the residual stream and project it into a new space (for scoring, for value aggregation, or for non-linear processing). The central question for these layers is: "How has fine-tuning changed the way the model _uses_ its existing features?"

**Primary Diagnostic Tools:**
To answer this, we must adopt the input perspective. For these layers, we prioritize:

1.  The **Input Subspace Alignment Curve**: $\text{Input Alignment}(k)$ vs. $k$. This curve will reveal whether the change is focused on refining the most important pre-existing input features (head-focused) or on activating previously dormant ones (tail-focused).
2.  The **Input-Normalized Change Matrix**: $\hat{W}_{\text{input}}$, where $(\hat{W}_{\text{input}})_{ij} = \frac{(\Delta \tilde{W})_{ij}}{\sigma_j(W_{\text{base}})}$. This allows a detailed look at how the processing of specific input modes is being altered.

**Interpreting the Signatures:**

- **Refinement Signature:** A layer exhibits a head-focused `Input Subspace Alignment Curve` and a high `Structural Alignment` (energy concentrated on the diagonal of $\Delta \tilde{W}$). This indicates that the layer is adapting by re-weighting its most important, pre-existing input features. The model is learning to "pay more or less attention" to what it already knows.

- **Re-purposing Signature:** A layer shows a head-focused `Input Subspace Alignment Curve` but a low `Structural Alignment` (high mode mixing). This is a more complex adaptation. The model still leverages its most important input features, but it re-wires them to produce new combinations, effectively re-purposing them for the new task.

#### 6.2. Analyzing Output-Centric Layers: Attention $W_O$ and MLP $W_2$

**Functional Role:** The attention output projection ($W_O$) and the MLP down-projection ($W_2$) are **output-centric**. They take the processed representations from their respective sub-blocks and project them back into the residual stream's dimensionality. Their role is to construct the final $+ Layer(x)$ correction vector. The central question for these layers is: "What is the structure of the _new information_ being added back to the model's representation?"

**Primary Diagnostic Tools:**
Here, we must adopt the output perspective. For these layers, we prioritize:

1.  The **Output Subspace Alignment Curve**: $\text{Output Alignment}(k)$ vs. $k$. This reveals whether the final correction vector is being constructed along the same directions as the base model's primary outputs (head-focused) or if it's creating entirely new output directions (tail-focused).
2.  The **Output-Normalized Change Matrix**: $\hat{W}_{\text{output}}$, where $(\hat{W}_{\text{output}})_{ij} = \frac{(\Delta \tilde{W})_{ij}}{\sigma_i(W_{\text{base}})}$. This helps analyze the composition of the new output correction.

**Interpreting the Signatures:**

- **Novelty Acquisition Signature:** A layer exhibits a tail-focused `Output Subspace Alignment Curve`. This is a powerful and unambiguous finding. It means the fine-tuning process is creating corrections in directions that were orthogonal to the original layer's most important outputs. The model is not merely refining its output; it is learning to produce entirely new kinds of information required for the downstream task that was absent in its pre-trained state. This is often accompanied by a large `Relative Change Norm` and a higher `Effective Rank`, indicating a significant and complex update.

By systematically applying this targeted analysis to each layer, one can move beyond a monolithic view of fine-tuning and paint a rich, multi-faceted picture of network adaptation, identifying which layers are being subtly refined, which are re-purposing their knowledge, and which are undergoing the difficult work of learning truly novel capabilities.
