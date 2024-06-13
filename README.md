# Kmean

### Summary

Federated Learning (FL) allows multiple clients (e.g., mobile or IoT devices) to collaboratively train a deep learning model without sharing their private data. The process involves:

1. **Synchronizing with the Server**: Clients download the latest master model from the server.
2. **Local Training**: Each client performs local training using its data, resulting in a weight update.
3. **Aggregation**: Clients upload their updates to the server, where a new master model is created via weighted averaging.

This method assumes a single model can fit all clients' data distributions, but this assumption often fails when client data distributions differ significantly.

### Detailed Explanation

1. **FL Protocol**: The FL protocol involves iterative steps:
   - **Download Model**: Clients download the current global model.
   - **Local Update**: Each client improves the model using its local data via stochastic gradient descent (SGD), resulting in a weight update.
   - **Upload and Aggregate**: Clients upload their updates to the server, where they are aggregated to form a new global model.

2. **Assumption in FL**: It assumes that one model can minimize the risk across all clients' data distributions simultaneously, formalized as:
   \[
   Ri(θ^*) ≤ Ri(θ) \quad ∀θ ∈ Bε(θ^*), \quad i = 1, \dots, m
   \]
   where \( Ri(θ) \) is the risk function for client \( i \).

3. **Violation of Assumption**: The assumption fails if:
   - Clients have different conditional data distributions (\( \varphi_i(y|x) \neq \varphi_j(y|x) \)).
   - The model cannot express all distributions simultaneously.

4. **Congruent vs. Incongruent**: Clients are considered congruent if their data distributions can be modeled by a single model under FL, and incongruent if they cannot.

This highlights the need for approaches like CFL to handle data heterogeneity in FL.


In the context of the paper, \(R_i(\theta^*)\) represents the risk function, or the expected loss, of the model with parameters \(\theta^*\) on the data distribution of client \(i\). Formally, it is defined as:

\[ R_i(\theta^*) = \int l(f_{\theta^*}(x), y) d\varphi_i(x, y) \]

where \(l\) is the loss function, \(f_{\theta^*}\) is the model parameterized by \(\theta^*\), and \(\varphi_i\) is the data distribution for client \(i\). This risk function measures how well the model \(f_{\theta^*}\) performs on client \(i\)'s data.



The risk function, \( R_i(\theta) \), in machine learning is a measure of the expected loss (error) of a model with parameters \(\theta\) on the data distribution of client \(i\). Formally, it is defined as:

\[ R_i(\theta) = \int l(f_{\theta}(x), y) d\varphi_i(x, y) \]

Here:
- \( l(f_{\theta}(x), y) \) is the loss function, which quantifies the error between the model's prediction \( f_{\theta}(x

The risk function \( R_i(\theta) \) in machine learning is a measure of the expected loss or error of a model with parameters \(\theta\) on the data distribution specific to client \(i\). Mathematically, it is defined as:

\[ R_i(\theta) = \int l(f_{\theta}(x), y) \, d\varphi_i(x, y) \]

Where:
- \( l(f_{\theta}(x), y) \) is the loss function, which measures the error between the model's prediction \( f_{\theta}(x) \) and the true label \( y \).
- \( f_{\theta}(x) \) is the model's prediction for input \( x \) using parameters \(\theta\).
- \( \varphi_i(x, y) \) is the data distribution for client \(i\).

The risk function quantifies the performance of the model on client \(i\)'s data, with lower values indicating better performance.




The formula \(\theta_{t+1} = \theta_t + \sum_{i=1}^m \frac{|D_i|}{|D|} \Delta \theta_{t+1,i}\) describes the update rule for the global model in Federated Learning. Here’s a breakdown:

- \(\theta_t\) is the current global model's parameters.
- \(\Delta \theta_{t+1,i}\) is the weight update computed by client \(i\) after local training.
- \(|D_i|\) is the size of the dataset for client \(i\).
- \(|D|\) is the total size of all clients' datasets.

The updates from all clients are aggregated in a weighted manner, proportional to the size of each client's dataset, to form the next global model \(\theta_{t+1}\).


### Empirical Risk Function

The empirical risk function \( r_i(\theta) \) measures the average loss of a model on a finite dataset \( D_i \) for client \( i \). It is defined as:
\[ r_i(\theta) = \frac{1}{|D_i|} \sum_{(x,y) \in D_i} l(f_\theta(x), y) \]
where:
- \( \theta \) are the model parameters,
- \( l(f_\theta(x), y) \) is the loss function,
- \( D_i \) is the dataset of client \( i \).

### True Risk

The true risk \( R_i(\theta) \) represents the expected loss of the model over the entire data distribution \( \varphi_i \) for client \( i \). It is defined as:
\[ R_i(\theta) = \int l(f_\theta(x), y) \, d\varphi_i(x, y) \]
where \( \varphi_i \) is the true underlying distribution of the data for client \( i \).

### Key Differences
- **Empirical Risk**: Calculated using a finite sample of data points from the client's dataset.
- **True Risk**: Theoretical measure over the entire data distribution, often unknown but approximated by empirical risk.

Empirical risk approximates true risk as the dataset size increases.


### Explanation of Formulas

#### 1. **Client Mapping to Data Distribution**
\[ I : \{1, .., m\} \rightarrow \{1, .., k\}, i \mapsto I(i) \]

- **Description**: This function \(I\) maps each client \(i\) to its corresponding data generating distribution \(\varphi_{I(i)}\).
- **Purpose**: It helps identify which data distribution a particular client belongs to.

#### 2. **Correct Bi-Partitioning**
\[ c_1 \cupdot c_2 = \{1, .., m\} \quad \text{with} \quad c_1 \neq \emptyset \quad \text{and} \quad c_2 \neq \emptyset \]
\[ I(i) \neq I(j) \quad \forall i \in c_1, j \in c_2 \]

- **Description**: A bi-partitioning is correct if it splits clients into two non-empty sets \(c_1\) and \(c_2\) such that clients within the same set share the same data generating distribution.
- **Purpose**: Ensures clients with similar data are grouped together.

#### 3. **Client Data Sampling**
\[ D_i \sim \varphi_{I(i)}(x, y) \]

- **Description**: Data \(D_i\) for client \(i\) is sampled from the distribution \(\varphi_{I(i)}\).
- **Purpose**: Establishes that each client's data comes from a specific distribution.

#### 4. **Empirical Risk Function**
\[ r_i(\theta) = \sum_{x \in D_i} l_\theta(f(x), y) \]

- **Description**: The empirical risk function \(r_i(\theta)\) is the sum of losses for each data point \((x, y)\) in the client’s dataset \(D_i\).
- **Purpose**: Measures the average loss on a client's data.

#### 5. **True Risk Function Approximation**
\[ r_i(\theta) \approx R_{I(i)}(\theta) := \int l_\theta(f(x), y) d\varphi_{I(i)}(x, y) \]

- **Description**: The empirical risk \(r_i(\theta)\) approximates the true risk \(R_{I(i)}(\theta)\) when the number of data points is large.
- **Purpose**: Connects empirical risk to the expected true risk over the data distribution.

#### 6. **Federated Learning Objective**
\[ F(\theta) := \sum_{i=1}^m \frac{|D_i|}{|D|} r_i(\theta) = a_1 R_1(\theta) + a_2 R_2(\theta) \]
\[ a_1 = \frac{\sum_{i, I(i)=1} |D_i|}{|D|}, \quad a_2 = \frac{\sum_{i, I(i)=2} |D_i|}{|D|} \]

- **Description**: The FL objective function \(F(\theta)\) is the weighted sum of empirical risks across all clients, with \(a_1\) and \(a_2\) being the proportions of data from each distribution.
- **Purpose**: Aggregates individual client losses into a global objective.

#### 7. **Stationary Point in Federated Learning**
\[ 0 = \nabla F(\theta^*) = a_1 \nabla R_1(\theta^*) + a_2 \nabla R_2(\theta^*) \]

- **Description**: At the stationary point \(\theta^*\), the gradient of the FL objective function \(F(\theta)\) is zero, indicating a local minimum.
- **Purpose**: Demonstrates convergence of the optimization process.

#### 8. **Two Situations at Stationary Point**
- **Case 1**: \(\nabla R_1(\theta^*) = \nabla R_2(\theta^*) = 0\)
  - **Description**: Both gradients are zero, indicating that risks for both distributions are minimized.
  - **Implication**: \(\varphi_1\) and \(\varphi_2\) are congruent, and the distributed learning problem is solved.

- **Case 2**: \(\nabla R_1(\theta^*) = -\frac{a_2}{a_1} \nabla R_2(\theta^*) \neq 0\)
  - **Description**: The gradients are not zero but are balanced in proportion to \(a_1\) and \(a_2\).
  - **Implication**: Indicates different distributions \(\varphi_1\) and \(\varphi_2\) are present, requiring clustering.



### Explanation of Formulas

#### 1. **Cosine Similarity Between Gradients**
\[ \alpha_{i,j} := \alpha(\nabla r_i(\theta^*), \nabla r_j(\theta^*)) := \frac{\langle \nabla r_i(\theta^*), \nabla r_j(\theta^*) \rangle}{\|\nabla r_i(\theta^*)\| \|\nabla r_j(\theta^*)\|} \]

- **Description**: This formula calculates the cosine similarity between the gradient updates of two clients \(i\) and \(j\) at the stationary point \(\theta^*\).
- **Purpose**: To measure the alignment of the gradients between two clients. A cosine similarity of 1 means the gradients are perfectly aligned (indicating the same data distribution), and -1 means they are diametrically opposed (indicating different data distributions).

\[ = \frac{\langle \nabla R_{I(i)}(\theta^*), \nabla R_{I(j)}(\theta^*) \rangle}{\|\nabla R_{I(i)}(\theta^*)\| \|\nabla R_{I(j)}(\theta^*)\|} \]
\[ = \begin{cases} 
1 & \text{if } I(i) = I(j) \\ 
-1 & \text{if } I(i) \neq I(j) 
\end{cases} \]

- **Detailed Breakdown**:
  - \(\alpha_{i,j}\) represents the cosine similarity between the gradients of the empirical risk functions of clients \(i\) and \(j\).
  - \(\frac{\langle \nabla r_i(\theta^*), \nabla r_j(\theta^*) \rangle}{\|\nabla r_i(\theta^*)\| \|\nabla r_j(\theta^*)\|}\) is the formula for cosine similarity, where \(\langle \cdot, \cdot \rangle\) denotes the dot product and \(\|\cdot\|\) denotes the norm of a vector.
  - The expression \(\frac{\langle \nabla R_{I(i)}(\theta^*), \nabla R_{I(j)}(\theta^*) \rangle}{\|\nabla R_{I(i)}(\theta^*)\| \|\nabla R_{I(j)}(\theta^*)\|}\) translates the cosine similarity into terms of the true risk gradients. This shows that the empirical risk function approximates the true risk function well.
  - The result \(\alpha_{i,j} = 1\) if \(I(i) = I(j)\) and \(\alpha_{i,j} = -1\) if \(I(i) \neq I(j)\) tells us that the cosine similarity can identify whether two clients have the same data distribution.

#### 2. **Bi-Partitioning Clients Based on Cosine Similarity**
\[ c_1 = \{i \mid \alpha_{i,0} = 1\}, \quad c_2 = \{i \mid \alpha_{i,0} = -1\} \]

- **Description**: This formula defines how to partition clients into two clusters based on their cosine similarity \(\alpha_{i,0}\) with a reference client (denoted as client 0).
- **Purpose**: To create two groups of clients where each group has clients with the same data distribution. Clients in \(c_1\) have the same distribution as the reference client, and those in \(c_2\) have a different distribution.

- **Detailed Breakdown**:
  - \(c_1 = \{i \mid \alpha_{i,0} = 1\}\): This cluster includes clients whose gradients have a cosine similarity of 1 with the reference client, indicating they share the same data distribution.
  - \(c_2 = \{i \mid \alpha_{i,0} = -1\}\): This cluster includes clients whose gradients have a cosine similarity of -1 with the reference client, indicating they have a different data distribution.

### Insights

- **Cosine Similarity**: Cosine similarity measures the angle between two vectors (gradients here), with values ranging from -1 to 1. A value of 1 means vectors are identical in direction, and -1 means they are opposite.
- **Correct Clustering**: By examining the cosine similarities at the stationary point \(\theta^*\), we can distinguish clients based on their underlying data distributions, facilitating effective clustering in federated learning environments.



### Detailed Explanation of Formulas

#### 1. **Assumption on Risk Gradient Approximation**
\[ \|\nabla R_{I(i)}(\theta^*)\| > \|\nabla R_{I(i)}(\theta^*) - \nabla r_i(\theta^*)\| \]

- **Description**: This inequality ensures that the norm (magnitude) of the gradient of the true risk function \( R_{I(i)} \) for client \( i \) is greater than the norm of the difference between the true risk gradient and the empirical risk gradient \( r_i \).
- **Purpose**: This assumption implies that the true risk gradient is closer to itself than to the empirical risk gradient, validating that empirical risk approximates the true risk well at the stationary point \(\theta^*\).

#### 2. **Definition of \(\gamma_i\)**
\[ \gamma_i := \frac{\|\nabla R_{I(i)}(\theta^*) - \nabla r_i(\theta^*)\|}{\|\nabla R_{I(i)}(\theta^*)\|} \in [0, 1) \]

- **Description**: \(\gamma_i\) represents the relative approximation error of the empirical risk gradient for client \( i \). It is the ratio of the norm of the difference between the true and empirical risk gradients to the norm of the true risk gradient.
- **Purpose**: \(\gamma_i\) quantifies how well the empirical risk gradient approximates the true risk gradient. Values close to 0 indicate a good approximation, while values close to 1 indicate poor approximation.

#### 3. **Maximum Cross-Cluster Similarity**
\[ \alpha_{\text{max}}^{\text{cross}} := \min_{c_1 \cup c_2 = \{1, .., m\}} \max_{i \in c_1, j \in c_2} \alpha(\nabla r_i(\theta^*), \nabla r_j(\theta^*)) \]

- **Description**: This defines the maximum cosine similarity between the gradients of clients from different clusters. It is the maximum similarity observed between any pair of clients from two different clusters, minimized over all possible bi-partitions.
- **Purpose**: Measures the upper bound of similarity between clients in different clusters, ensuring clusters are well-separated.

#### 4. **Calculation of \(\alpha_{\text{max}}^{\text{cross}}\)**
\[ \alpha_{\text{max}}^{\text{cross}} = \max_{i \in c^*_1, j \in c^*_2} \alpha(\nabla r_i(\theta^*), \nabla r_j(\theta^*)) \]
\[ \leq \begin{cases} 
\cos\left(\frac{\pi}{k-1}\right) H_{i,j} + \sin\left(\frac{\pi}{k-1}\right) \sqrt{1 - H_{i,j}^2} & \text{if } H_{i,j} \geq \cos\left(\frac{\pi}{k-1}\right) \\
1 & \text{else}
\end{cases} \]

- **Description**: This provides an upper bound on the maximum cross-cluster similarity using cosine and sine terms that depend on the number of clusters \(k\) and the term \(H_{i,j}\).
- **Purpose**: Ensures that the cross-cluster similarity does not exceed a certain threshold, maintaining good separation between clusters.

#### 5. **Definition of \( H_{i,j} \)**
\[ H_{i,j} = -\gamma_i \gamma_j + \sqrt{1 - \gamma_i^2} \sqrt{1 - \gamma_j^2} \in (-1, 1] \]

- **Description**: \(H_{i,j}\) is a term that adjusts the cosine similarity calculation based on the approximation errors \(\gamma_i\) and \(\gamma_j\).
- **Purpose**: It helps to determine how the errors in gradient approximations affect the similarity bounds.

#### 6. **Minimum Intra-Cluster Similarity**
\[ \alpha_{\text{min}}^{\text{intra}} := \min_{i,j | I(i) = I(j)} \alpha(\nabla r_i(\theta^*), \nabla r_j(\theta^*)) \]
\[ \geq \min_{i,j | I(i) = I(j)} H_{i,j} \]

- **Description**: This defines the minimum cosine similarity between the gradients of clients within the same cluster. It is the minimum similarity observed between any pair of clients from the same cluster.
- **Purpose**: Ensures clients within the same cluster have high similarity, indicating they share the same data distribution.

#### 7. **Special Case for Two Distributions (\(k = 2\))**
\[ \alpha_{\text{max}}^{\text{cross}} = \max_{i \in c^*_1, j \in c^*_2} \alpha(\nabla r_i(\theta^*), \nabla r_j(\theta^*)) \leq \max_{i \in c^*_1, j \in c^*_2} -H_{i,j} \]
\[ \alpha_{\text{min}}^{\text{intra}} = \min_{i,j | I(i) = I(j)} \alpha(\nabla r_i(\theta^*), \nabla r_j(\theta^*)) \geq \min_{i,j | I(i) = I(j)} H_{i,j} \]

- **Description**: Simplifies the bounds for the case of exactly two data generating distributions.
- **Purpose**: Makes the analysis easier to understand and apply for scenarios with two clusters.

#### 8. **Correct Separation Rule (Corollary 1)**
\[ \alpha_{\text{min}}^{\text{intra}} > \alpha_{\text{max}}^{\text{cross}} \]
\[ c_1, c_2 \leftarrow \arg \min_{c_1 \cup c_2 = c} \left( \max_{i \in c_1, j \in c_2} \alpha_{i,j} \right) \]

- **Description**: This condition and rule provide a method for correctly separating clients into clusters based on their gradient similarities. If the minimum intra-cluster similarity is greater than the maximum cross-cluster similarity, the partitioning is considered correct.
- **Purpose**: Ensures that the partitioning of clients into clusters is correct, following the definition that clients with the same data distribution end up in the same cluster.






### Detailed Explanation of Remark 1 Formulas

#### 1. **Maximum Cross-Cluster Similarity for Two Distributions**
\[ \alpha_{\text{max}}^{\text{cross}} = \max_{i \in c^*_1, j \in c^*_2} \alpha(\nabla r_i(\theta^*), \nabla r_j(\theta^*)) \leq \max_{i \in c^*_1, j \in c^*_2} -H_{i,j} \]

- **Description**: For the case of exactly two data generating distributions (\(k = 2\)), this formula simplifies the calculation of the maximum cosine similarity between gradients of clients from different clusters.
- **Purpose**: It shows that the maximum cross-cluster similarity is bounded above by the maximum value of \(-H_{i,j}\) for clients \(i\) and \(j\) in different clusters \(c^*_1\) and \(c^*_2\).
- **Interpretation**: This simplification helps in understanding and applying the separation theorem in scenarios with only two clusters. Since \(H_{i,j} \in (-1, 1]\), the term \(-H_{i,j}\) ensures the similarity is negative, indicating distinct clusters.

#### 2. **Minimum Intra-Cluster Similarity for Two Distributions**
\[ \alpha_{\text{min}}^{\text{intra}} = \min_{i,j \mid I(i) = I(j)} \alpha(\nabla r_i(\theta^*), \nabla r_j(\theta^*)) \geq \min_{i,j \mid I(i) = I(j)} H_{i,j} \]

- **Description**: This formula gives the minimum cosine similarity between gradients of clients within the same cluster for the case of two data generating distributions.
- **Purpose**: It ensures that the minimum intra-cluster similarity is bounded below by the minimum value of \(H_{i,j}\) for clients \(i\) and \(j\) within the same cluster.
- **Interpretation**: This indicates that the gradients within the same cluster are positively correlated (since \(H_{i,j}\) is positive), reinforcing that clients within the same cluster share the same data distribution.

### Summary of Simplified Results
- For two data generating distributions, the separation conditions are easier to apply:
  - The maximum similarity between clients in different clusters is bounded by the maximum negative value of \(H_{i,j}\).
  - The minimum similarity within the same cluster is bounded by the minimum positive value of \(H_{i,j}\).

This simplification makes it more straightforward to verify whether a bi-partitioning of clients correctly separates different data distributions.



### Simplified Explanation of Corollary 1

**Corollary 1** states that if the minimum similarity within clusters is greater than the maximum similarity between clusters, then we can correctly separate clients into clusters with distinct data distributions.

**Steps to Understand**:

1. **Key Condition**:
   - Ensure that the minimum similarity within any cluster (\(\alpha_{\text{min}}^{\text{intra}}\)) is greater than the maximum similarity between different clusters (\(\alpha_{\text{max}}^{\text{cross}}\)).

2. **Correct Partitioning**:
   - Partition the clients such that you minimize the maximum similarity between any two clients in different clusters.
   - This means finding clusters where clients in different clusters are as different as possible.

3. **Result**:
   - If you achieve this partitioning, then clients within the same cluster have similar data, and clients in different clusters have different data.

**Proof Summary**:

1. Define the partition to minimize the maximum cross-cluster similarity.
2. By the given condition, the maximum similarity between clusters will be less than the minimum similarity within clusters.
3. Therefore, clients in different clusters must have different data distributions.

### Visualizing the Concept

Imagine you have a set of clients. You want to divide them into two groups (clusters) such that:

- Clients within the same group are very similar (intra-cluster similarity is high).
- Clients in different groups are very different (cross-cluster similarity is low).

If you can do this successfully, you’ve correctly separated the clients based on their data distributions.




### Explanation of Definition 2 and Related Content

#### 1. **Separation Gap Definition**
\[ g(\alpha) := \alpha_{\text{min}}^{\text{intra}} - \alpha_{\text{max}}^{\text{cross}} \]
\[ = \min_{i,j \mid I(i) = I(j)} \alpha_{i,j} - \min_{c_1 \cup c_2 = c} \left( \max_{i \in c_1, j \in c_2} \alpha_{i,j} \right) \]

- **Description**: The separation gap \(g(\alpha)\) is defined as the difference between the minimum intra-cluster similarity and the maximum cross-cluster similarity.
- **Purpose**: It measures how well the clients can be separated into distinct clusters based on their cosine similarity. A positive gap indicates that clients within the same cluster are more similar to each other than to clients in different clusters.

#### 2. **Remark 2: Correct Bi-Partitioning Condition**
- **Statement**: By Corollary 1, the bi-partitioning will be correct if and only if the separation gap is greater than zero.
- **Explanation**: If \(g(\alpha) > 0\), the partitioning correctly separates clients into clusters with distinct data distributions.

#### 3. **Practical Implications**
- **Theoretical Insight**: Theorem 1 provides an estimate for similarities in the worst-case scenario.
- **Practical Observation**: In practice, \(\alpha_{\text{min}}^{\text{intra}}\) is usually much larger and \(\alpha_{\text{max}}^{\text{cross}}\) is much smaller, especially in high-dimensional spaces (large \(d\)).
- **Experiment Findings**: Even with high values of \(k\) (number of clusters) and large approximation errors (\(\gamma_i\)), the probability of correct clustering is nearly 1.

### Summary
The separation gap \(g(\alpha)\) measures the effectiveness of clustering based on cosine similarity. A positive gap ensures correct partitioning. Despite theoretical worst-case scenarios, practical experiments show that correct clustering is highly probable, even with many clusters and large approximation errors. This makes the cosine similarity criterion a robust method for clustering in federated learning environments.


### Explanation of Distinguishing Congruent and Incongruent Clients

#### Context
In Federated Learning (FL), it's important to decide whether to split clients into clusters based on their data distributions. Splitting clients with congruent (similar) distributions can degrade performance because it restricts knowledge transfer between clusters.

#### Key Observations

1. **Incongruent Distributions**:
   - If clients have different data distributions, the global FL solution \(\theta^*\) won't be stationary for individual clients.
   - This means the gradients of individual clients won't be zero: \(\|\nabla r_i(\theta^*)\| > 0\).

2. **Congruent Distributions**:
   - If clients have similar data distributions, FL can jointly optimize all clients' risk functions.
   - This results in gradients approaching zero at the stationary point: \(\|\nabla r_i(\theta^*)\| \rightarrow 0\).

#### Criteria for Splitting

To decide whether to split clients into clusters, the following conditions must be checked:

1. **Near Stationary Point of FL Objective**:
   \[ 0 \leq \left\| \sum_{i=1}^m \frac{D_i}{|D_c|} \nabla r_i(\theta^*) \right\| < \epsilon_1 \]
   - Ensures the overall FL objective is near a stationary point.

2. **Far from Stationary Point for Individual Clients**:
   \[ \max_{i=1,..,m} \|\nabla r_i(\theta^*)\| > \epsilon_2 > 0 \]
   - Ensures individual clients' gradients are not zero, indicating incongruent distributions.

### Practical Approach

- **Post-Split Evaluation**: After splitting, if the model performance degrades, revert to the original FL solution. This guarantees that Clustered Federated Learning (CFL) either improves or maintains performance.

### Summary

In CFL, split clients into clusters only if:
- The global model is near a stationary point.
- Individual client gradients are not zero.
This approach ensures that only clients with different data distributions are separated, maintaining or improving model performance.


### Explanation of CFL Algorithm Steps

#### Initial Setup
1. **Start with Clients and Initial Parameters**:
   - Clients: \( c = \{1, .., m\} \)
   - Initial parameters: \( \theta_0 \)

2. **Perform Federated Learning**:
   - Use Algorithm 2 to achieve a stationary solution \( \theta^* \).

#### Evaluate Stopping Criterion
3. **Check Gradients**:
   - Evaluate \( 0 \leq \max_{i \in c} \|\nabla r_i(\theta^*)\| < \epsilon_2 \)
   - If true, all clients are near a stationary point, CFL terminates and returns \( \theta^* \).

4. **Handle Incongruent Clients**:
   - If false, clients are incongruent. Compute pairwise cosine similarities \( \alpha \) using equation (13).

#### Cluster Clients
5. **Optimal Bi-Partitioning**:
   - Separate clients into two clusters to minimize maximum cross-cluster similarity:
     \[ c_1, c_2 \leftarrow \arg \min_{c_1 \cup c_2 = c} \left( \max_{i \in c_1, j \in c_2} \alpha_{i,j} \right) \]
   - This problem is solved in \( O(m^3) \) using Algorithm 1.

#### Ensuring Correct Bi-Partitioning
6. **Ensure Correct Clustering**:
   - Use condition:
     \[ \alpha_{\text{min}}^{\text{intra}} > \alpha_{\text{max}}^{\text{cross}} \]
   - Estimate \(\alpha_{\text{min}}^{\text{intra}}\) using:
     \[ \alpha_{\text{min}}^{\text{intra}} \geq \min_{i,j \mid I(i) = I(j)} -\gamma_i \gamma_j + \sqrt{1 - \gamma_i^2} \sqrt{1 - \gamma_j^2} \geq 1 - 2 \gamma_{\text{max}}^2 \]
   - Criterion for correct bi-partitioning:
     \[ \gamma_{\text{max}} < \sqrt{\frac{1 - \alpha_{\text{max}}^{\text{cross}}}{2}} \]

#### Recursive Application
7. **Recursive Splitting**:
   - If the criterion is satisfied, apply CFL recursively to each of the two groups starting from \( \theta^* \).
   - Continue splitting until no sub-clusters violate the stopping criterion.

### Conclusion
The entire process identifies groups of congruent clients, solving the Clustered Federated Learning problem as per Assumption 2. The recursive procedure ensures that clients with similar data distributions are grouped together, optimizing the federated learning model.



### Algorithm 1: Optimal Bipartition

1. **Input**:
   - \(\alpha \in [-1, 1]^{m \times m}\): Similarity matrix between clients.
2. **Output**:
   - \(c_1, c_2\): Two clusters of clients.

#### Steps:
1. **Sort Similarities**:
   - \[ s \leftarrow \text{argsort}(-\alpha[:]) \]
   - **Explanation**: Flattens the similarity matrix and sorts indices in descending order of similarity values.
2. **Initialize Clusters**:
   - \[ C \leftarrow \{\{i\} | i = 1, .., m\} \]
   - **Explanation**: Start with each client in its own cluster.
3. **Merge Clusters**:
   - \[ \text{for } i = 1, .., m^2 \text{ do} \]
   - Loop through sorted similarities.
   - \[ i_1 \leftarrow s_i \div m; i_2 \leftarrow s_i \mod m \]
     - **Explanation**: Get row and column indices from flattened index.
   - \[ c_{\text{tmp}} \leftarrow \{\} \]
     - **Explanation**: Initialize a temporary cluster.
   - \[ \text{for } c \in C \text{ do} \]
     - Loop through existing clusters.
     - \[ \text{if } i_1 \in c \text{ or } i_2 \in c \text{ then} \]
       - Check if either client is in the cluster.
     - \[ c_{\text{tmp}} \leftarrow c_{\text{tmp}} \cup c \]
       - **Explanation**: Merge clusters containing the current clients.
     - \[ C \leftarrow C \setminus c \]
       - **Explanation**: Remove merged clusters from the set.
   - \[ C \leftarrow C \cup \{c_{\text{tmp}}\} \]
     - **Explanation**: Add the new merged cluster to the set.
   - \[ \text{if } |C| = 2 \text{ then} \]
     - Check if only two clusters remain.
   - \[ \text{return } C \]
     - **Explanation**: Return the two clusters if the condition is met.
4. **Return Clusters**:
   - The algorithm continues until exactly two clusters remain.

### Algorithm 2: Federated Learning (FL)

1. **Input**:
   - Initial parameters \(\theta\)
   - Set of clients \(c\)
   - Convergence threshold \(\epsilon_1\)
2. **Output**:
   - Updated parameters \(\theta\)

#### Steps:
1. **Repeat Until Convergence**:
   - \[ \text{repeat} \]
   - Loop until convergence.
   - \[ \text{for } i \in c \text{ in parallel do} \]
     - Loop through each client in parallel.
     - \[ \theta_i \leftarrow \theta \]
       - **Explanation**: Initialize local model with global parameters.
     - \[ \Delta \theta_i \leftarrow \text{SGD}_n(\theta_i, D_i) - \theta_i \]
       - **Explanation**: Compute weight update using local data \(D_i\).
   - \[ \theta \leftarrow \theta + \sum_{i \in c} \frac{|D_i|}{|D_c|} \Delta \theta_i \]
     - **Explanation**: Aggregate updates to form the new global model.
   - \[ \text{until} \left\| \sum_{i \in c} \frac{|D_i|}{|D_c|} \Delta \theta_i \right\| < \epsilon_1 \]
     - **Explanation**: Check if the change in the global model is less than \(\epsilon_1\).
2. **Return \(\theta\)**:
   - Once convergence is achieved, return the updated parameters.

### Algorithm 3: Clustered Federated Learning (CFL)

1. **Input**:
   - Initial parameters \(\theta\)
   - Set of clients \(c\)
   - Maximum allowed approximation error \(\gamma_{\text{max}} \in [0, 1]\)
   - Convergence threshold for individual clients \(\epsilon_2\)
2. **Output**:
   - Updated parameters for each client \(\theta_i\)

#### Steps:
1. **Perform Federated Learning**:
   - \(\theta^* \leftarrow \text{FederatedLearning}(\theta, c)\)
   - **Explanation**: Run the standard federated learning to get a stationary solution \(\theta^*\).
2. **Compute Cosine Similarities**:
   - \(\alpha_{i,j} \leftarrow \frac{\langle \nabla r_i(\theta^*), \nabla r_j(\theta^*) \rangle}{\|\nabla r_i(\theta^*)\| \|\nabla r_j(\theta^*)\|}\)
   - **Explanation**: Calculate pairwise cosine similarities between gradients.
3. **Optimal Bi-Partitioning**:
   - \(c_1, c_2 \leftarrow \arg \min_{c_1 \cup c_2 = c} (\max_{i \in c_1, j \in c_2} \alpha_{i,j})\)
   - **Explanation**: Find clusters that minimize the maximum similarity between different clusters.
   - \(\alpha_{\text{max}}^{\text{cross}} \leftarrow \max_{i \in c_1, j \in c_2} \alpha_{i,j}\)
   - **Explanation**: Compute the maximum similarity between different clusters.
4. **Check Splitting Criteria**:
   - \(\text{if } \max_{i \in c} \|\nabla r_i(\theta^*)\| \geq \epsilon_2 \text{ and } \sqrt{\frac{1 - \alpha_{\text{max}}^{\text{cross}}}{2}} > \gamma_{\text{max}} \text{ then}\)
     - **Explanation**: Check if the maximum gradient norm is greater than \(\epsilon_2\) and the criterion for \(\gamma_{\text{max}}\) is satisfied.
     - \(\theta^*_i, i \in c_1 \leftarrow \text{ClusteredFederatedLearning}(\theta^*, c_1)\)
     - \(\theta^*_i, i \in c_2 \leftarrow \text{ClusteredFederatedLearning}(\theta^*, c_2)\)
     - **Explanation**: Recursively apply CFL to the two clusters.
   - \(\text{else}\)
     - \(\theta^*_i \leftarrow \theta^*, i \in c\)
     - **Explanation**: If criteria are not met, return the current model for all clients.
5. **Return \(\theta^*_i, i \in c\)**:
   - The final updated parameters for each client are returned.

This detailed explanation breaks down each line of the algorithms, clarifying the purpose and steps involved in each part of the process.


### Theorem 1 (Separation Theorem)

Theorem 1 provides a formal way to separate clients into clusters based on the cosine similarity of their gradient updates. Here are the key points:

1. **Assumptions**:
   - Local training data \(D_1, .., D_m\) for \(m\) clients, each sampled from one of \(k\) different data generating distributions \(\varphi_1, .., \varphi_k\).
   - Empirical risk function \(r_i(\theta)\) approximates true risk \(R_{I(i)}(\theta)\).

2. **Gradient Norm Condition**:
   \[ \|\nabla R_{I(i)}(\theta^*)\| > \|\nabla R_{I(i)}(\theta^*) - \nabla r_i(\theta^*)\| \]
   - This ensures that the true gradient's magnitude is larger than the difference between the true gradient and the empirical gradient.

3. **Separation Gap \(\gamma_i\)**:
   \[ \gamma_i := \frac{\|\nabla R_{I(i)}(\theta^*) - \nabla r_i(\theta^*)\|}{\|\nabla R_{I(i)}(\theta^*)\|} \in [0, 1) \]
   - \(\gamma_i\) quantifies the approximation error of the empirical gradient relative to the true gradient.

4. **Bi-Partitioning**:
   - A correct bi-partitioning separates clients such that the maximum similarity between clients from different clusters is minimized:
     \[ \alpha_{\text{max}}^{\text{cross}} := \min_{c_1 \cup c_2 = \{1, .., m\}} \max_{i \in c_1, j \in c_2} \alpha(\nabla r_i(\theta^*), \nabla r_j(\theta^*)) \]

5. **Similarity Bound**:
   - The cosine similarity between gradients of clients in different clusters is bounded:
     \[ \alpha_{\text{max}}^{\text{cross}} \leq \cos\left(\frac{\pi}{k-1}\right) H_{i,j} + \sin\left(\frac{\pi}{k-1}\right) \sqrt{1 - H_{i,j}^2} \]
   - For \( H_{i,j} \) defined as:
     \[ H_{i,j} = -\gamma_i \gamma_j + \sqrt{1 - \gamma_i^2} \sqrt{1 - \gamma_j^2} \in (-1, 1] \]

6. **Intra-Cluster Similarity**:
   - The minimum similarity within a cluster is bounded:
     \[ \alpha_{\text{min}}^{\text{intra}} \geq \min_{i,j \mid I(i) = I(j)} H_{i,j} \]

7. **Special Case for Two Distributions**:
   - Simplifies to:
     \[ \alpha_{\text{max}}^{\text{cross}} \leq \max_{i \in c_1, j \in c_2} -H_{i,j} \]
     \[ \alpha_{\text{min}}^{\text{intra}} \geq \min_{i,j \mid I(i) = I(j)} H_{i,j} \]

8. **Correct Separation Rule**:
   - If the minimum intra-cluster similarity is greater than the maximum cross-cluster similarity, the partitioning is correct:
     \[ \alpha_{\text{min}}^{\text{intra}} > \alpha_{\text{max}}^{\text{cross}} \]

This theorem provides a rigorous basis for clustering clients in federated learning based on their gradient similarities, ensuring that clients with similar data distributions are grouped together.


### Explanation of the Text

**Privacy Concerns in Federated Learning**:
- **Information Leakage**: Machine learning models can unintentionally reveal information about the data they were trained on. For instance, the bias term in the last layer of a neural network can reflect the label distribution of the training data.
- **Model Inversion Attacks**: Weight updates sent from clients to the server can be exploited to infer clients’ input data through model inversion attacks.

**Mitigation**:
- **Need for Privacy**: In privacy-sensitive scenarios, it’s crucial to prevent information leakage from clients to the server.
- **Encryption Mechanism**: Clustered Federated Learning (CFL) can incorporate an encryption mechanism to prevent such leaks.

**Encryption via Orthonormal Transformations**:
- **Cosine Similarity Invariance**: The cosine similarity between two clients' weight updates and the norms of these updates remain unchanged under orthonormal transformations (like permutation of indices).
- **Formula**: 
  \[ \frac{\langle \Delta \theta_i, \Delta \theta_j \rangle}{\|\Delta \theta_i\| \|\Delta \theta_j\|} = \frac{\langle P \Delta \theta_i, P \Delta \theta_j \rangle}{\|P \Delta \theta_i\| \|P \Delta \theta_j\|} \]

**Steps to Secure Weight Updates**:
1. **Transformation Before Sending**:
   - Clients apply an orthonormal transformation \(P\) to their updates before sending them to the server.
2. **Aggregation and Broadcasting**:
   - The server averages the transformed updates and broadcasts the average back to the clients.
3. **Inverse Transformation**:
   - Clients apply the inverse transformation \(P^{-1}\) to the averaged update:
     \[ \Delta \theta = \frac{1}{n} \sum_{i=1}^n \Delta \theta_i = P^{-1} \left( \frac{1}{n} \sum_{i=1}^n P \Delta \theta_i \right) \]

**Advantages**:
- **Preservation of Privacy**: This approach ensures that the server cannot infer the original updates, maintaining client data privacy.
- **Compatibility with CFL**: This method can be seamlessly integrated into the CFL framework without altering the overall protocol.
- **Comparison with Other Methods**: Other multi-task learning approaches often require direct access to client data, making them incompatible with encryption. CFL’s compatibility with encryption is a significant advantage in privacy-sensitive applications.

### Summary
The text highlights the privacy challenges in Federated Learning and proposes using orthonormal transformations to secure weight updates against model inversion attacks. This method preserves client privacy and integrates well with CFL, making it superior to other multi-task learning approaches that require direct access to client data.


### Algorithm 4: Assigning New Clients to a Cluster

1. **Input**: 
   - New client data \(D_{\text{new}}\)
   - Parameter tree \(T = (V, E)\)

2. **Initialize at Root**:
   - \( v \leftarrow v_{\text{root}} \)
   - **Explanation**: Start at the root of the parameter tree.

3. **Traverse the Tree**:
   - \[ \text{while } | \text{Children}(v) | > 0 \text{ do} \]
     - **Explanation**: Continue until reaching a leaf node (no children).

4. **Get Children**:
   - \[ v_0, v_1 \leftarrow \text{Children}(v) \]
   - **Explanation**: Retrieve child nodes of the current node \(v\).

5. **Compute Weight Update for New Client**:
   - \[ \Delta \theta_{\text{new}} \leftarrow \text{SGD}_n(\theta_v^*, D_{\text{new}}) - \theta_v^* \]
   - **Explanation**: Compute the weight update for the new client based on the current node’s parameters.

6. **Compute Cosine Similarities**:
   - \[ \alpha_0 \leftarrow \max_{\Delta \theta \in \Delta(v \rightarrow v_1)} \alpha(\Delta \theta_{\text{new}}, \Delta \theta) \]
   - \[ \alpha_1 \leftarrow \max_{\Delta \theta \in \Delta(v \rightarrow v_2)} \alpha(\Delta \theta_{\text{new}}, \Delta \theta) \]
   - **Explanation**: Calculate the maximum cosine similarity between the new client's weight update and the updates in both child nodes.

7. **Choose Branch**:
   - \[ \text{if } \alpha_0 > \alpha_1 \text{ then} \]
     - \( v \leftarrow v_0 \)
     - **Explanation**: Move to the branch with the higher similarity.
   - \[ \text{else} \]
     - \( v \leftarrow v_1 \)

8. **Return Cluster and Parameters**:
   - \[ \text{return } c_v, \theta_v^* \]
   - **Explanation**: Once a leaf node is reached, return the cluster and corresponding parameters.

### Algorithm 5: Clustered Federated Learning with Privacy Preservation and Weight-Updates

1. **Input**:
   - Initial parameters \( \theta_0 \)
   - Branching parameters \( \epsilon_1, \epsilon_2 \)
   - Empirical risk approximation error bound \( \gamma_{\text{max}} \)
   - Number of local iterations/epochs \( n \)

2. **Output**:
   - Improved parameters on every client \( \theta_i \)

3. **Initialization**:
   - Set initial clusters \( C = \{\{1, .., m\}\} \)
   - Set initial models \( \theta_i \leftarrow \theta_0 \) for all clients
   - Set initial updates \( \Delta \theta_c \leftarrow 0 \) for all clusters
   - Exchange random seed for permutation operator \( P \) (optional, set \( P \) to identity if not used)

4. **Federated Learning Loop**:
   - \[ \text{while not converged do} \]
     - Loop until convergence criteria are met.

5. **Client Operations**:
   - \[ \text{for } i = 1, .., m \text{ in parallel do} \]
     - Each client performs:
     - \[ \theta_i \leftarrow \theta_i + P^{-1} \Delta \theta_c(i) \]
       - **Explanation**: Update local model with inverse transformed global update.
     - \[ \Delta \theta_i \leftarrow P (\text{SGD}_n(\theta_i, D_i) - \theta_i) \]
       - **Explanation**: Compute and transform weight update.

6. **Server Operations**:
   - \[ C_{\text{tmp}} \leftarrow C \]
     - **Explanation**: Initialize temporary cluster set.
   - \[ \text{for } c \in C \text{ do} \]
     - For each cluster:
     - \[ \Delta \theta_c \leftarrow \frac{1}{|c|} \sum_{i \in c} \Delta \theta_i \]
       - **Explanation**: Compute average update for the cluster.
     - \[ \text{if } \|\Delta \theta_c\| < \epsilon_1 \text{ and } \max_{i \in c} \|\Delta \theta_i\| > \epsilon_2 \text{ then} \]
       - Check splitting criteria.
     - \[ \alpha_{i,j} \leftarrow \frac{\langle \Delta \theta_i, \Delta \theta_j \rangle}{\|\Delta \theta_i\| \|\Delta \theta_j\|} \]
       - **Explanation**: Compute cosine similarities.
     - \[ c_1, c_2 \leftarrow \arg \min_{c_1 \cup c_2 = c} (\max_{i \in c_1, j \in c_2} \alpha_{i,j}) \]
       - **Explanation**: Find optimal bi-partitioning.
     - \[ \alpha_{\text{max}}^{\text{cross}} \leftarrow \max_{i \in c_1, j \in c_2} \alpha_{i,j} \]
       - **Explanation**: Compute maximum cross-cluster similarity.
     - \[ \text{if } \gamma_{\text{max}} < \sqrt{\frac{1 - \alpha_{\text{max}}^{\text{cross}}}{2}} \text{ then} \]
       - Check approximation error bound.
     - \[ C_{\text{tmp}} \leftarrow (C_{\text{tmp}} \setminus c) \cup c_1 \cup c_2 \]
       - **Explanation**: Update cluster set with new clusters.

7. **Update Cluster Set**:
   - \[ C \leftarrow C_{\text{tmp}} \]

8. **Return Parameters**:
   - \[ \text{return } \theta \]

### Summary
- **Algorithm 4**: Assigns a new client to the most appropriate cluster by traversing the parameter tree and comparing cosine similarities.
- **Algorithm 5**: Implements CFL with privacy preservation, where clients apply transformations to updates for privacy and the server handles clustering and model updates.




