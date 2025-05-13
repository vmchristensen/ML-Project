# ML-Project

Link to report: <https://docs.google.com/document/d/19rcDIslGySqNR3OBR0IxZaIJqYXKOwD6mcsEpfQuQ-A>


1. LDA (Linear Discriminant Analysis)
❌ Not Suitable

Why: Assumes features are normally distributed and classes share the same covariance matrix. Centrality scores (e.g., betweenness, PageRank) are unlikely to follow Gaussian distributions.

Class Imbalance: Poor performance on minority class (root nodes) due to reliance on class means/covariances, which are dominated by non-root examples.

2. QDA (Quadratic Discriminant Analysis)
❌ Not Suitable

Why: Relaxes LDA’s assumption by allowing class-specific covariances, but estimating separate covariance matrices for the root class (with very few samples) is unreliable. Prone to overfitting.

3. RDA (Regularized Discriminant Analysis)
⚠️ Marginally Suitable

Why: Adds regularization to QDA/LDA, which might stabilize covariance estimates. However, the core assumptions (normality, class balance) still make it suboptimal for this task.

4. Naive Bayes
⚠️ Use with Caution

Why: Assumes feature independence (unlikely for centrality scores, e.g., degree and betweenness are correlated in trees).

Pros: Handles imbalance via likelihood-based predictions.

Cons: Poor performance if feature correlations are critical.

5. k-NN (k-Nearest Neighbors)
⚠️ Use with Caution

Why: Sensitive to class imbalance (few root examples in local neighborhoods).

Mitigation: Use weighted voting or SMOTE to balance classes.

Note: Requires careful normalization (e.g., per-sentence scaling) due to varying sentence lengths.

6. Perceptron
❌ Not Suitable

Why:

No probabilistic output → Cannot rank nodes by "rootness" likelihood.

Assumes linear separability (unlikely for centrality-based features).

Fails to handle class imbalance (biased toward majority class).

7. Decision Trees
✅ Suitable

Why:

Handle non-linear relationships (useful for complex centrality interactions).

Can address imbalance via class weighting (e.g., class_weight='balanced').

Cons: Prone to overfitting; use pruning or ensemble methods.

8. Random Forest
✅ Strongly Suitable

Why:

Ensemble of trees reduces overfitting and improves generalization.

Handles imbalance via class weighting or balanced bootstrapping.

Provides feature importance (identifies key centrality measures).

Probabilistic outputs for root node ranking.

9. Logistic Regression
✅ Suitable

Why:

Provides probabilistic outputs (critical for selecting the highest-confidence root).

Handles imbalance via class weighting.

Regularization (L1/L2) prevents overfitting.

Cons: Linear decision boundaries may limit performance if relationships are non-linear.

10. Multi-layer Perceptron (MLP)
✅ Suitable (with Caveats)

Why:

Models non-linear relationships (ideal for complex feature interactions).

Handles imbalance via class weighting or oversampling.

Probabilistic outputs via sigmoid activation.

Cons: Requires hyperparameter tuning (layers, learning rate) and sufficient data.