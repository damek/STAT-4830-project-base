# development_log.md
## Project: Transformer-Based Protein Synthesis Optimization

---


### Setup
- **Objective Defined:**  
  - Optimize artificial protein synthesis by predicting masked tokens in DNA sequences.
  - Future goal: generate full sequences from functional descriptions.
- **Planning:**  
  - Drafted the problem statement with defined constraints (valid base pairs, sequence length limits) and success metrics (targeting >90% masked token prediction accuracy and biological plausibility via external tools like AlphaFold).
- **Environment Setup:**  
  - Configured the Python environment with PyTorch, BioPython, psutil, and other necessary libraries.
  - Established the project repository and basic directory structure.

### Data Pipeline Development
- **Synthetic Data Generation:**  
  - Implemented scripts to generate synthetic DNA sequences using the vocabulary `A`, `C`, `G`, `T` with a random masking probability of 0.3.
  - Inspected sample outputs to verify that the masking process and label assignments were correct.
- **Real Data Integration:**  
  - Developed a module to fetch nucleotide sequences from public genomic databases.
  - Filtered and processed sequences to extract contiguous segments meeting a minimum length requirement.
  - **Observations:**  
    - Some fetched sequences were too short, so the filtering logic was adjusted to ensure a consistent dataset.

### Initial Model Implementation
- **Transformer Model Setup:**  
  - Built a lightweight transformer model in PyTorch:
    - An embedding layer (initial embedding dimension set to 16).
    - Two transformer encoder layers with multi-head attention.
    - A final linear layer to project embeddings to the vocabulary space.
  - Verified tensor shape transformations, ensuring compatibility with PyTorch’s transformer module.
- **Training Configuration:**  
  - Set up DataLoaders with an 80/20 split for training and validation.
  - Defined the loss function as cross-entropy with an ignore index for unmasked positions.
  - Configured the Adam optimizer with a learning rate of 0.001.

### Early Experiments with Synthetic Data
- **Training Results:**  
  - Ran a preliminary 5-epoch training cycle on synthetic data.
  - Observed training loss decrease from approximately 1.60 to 1.42.
  - Masked token prediction accuracy on the validation set was in the 27–30% range.
- **Insights:**  
  - The model demonstrated learning ability, but performance was only marginally better than random guessing.
  - Decided that hyperparameter tuning and further debugging of loss computation were necessary.

---

### Integration of Real Data
- **Transition to Real Data:**  
  - Switched training from synthetic to processed real genomic sequences.
  - Adjusted model parameters by increasing the embedding dimension to 32 and setting dropout to 0.02.
  - Configured the transformer with 4 attention heads and 2 encoder layers.
- **Performance Observations:**  
  - Early training runs on real data resulted in a training loss stabilizing around 1.36.
  - Validation accuracy on masked tokens fluctuated between 28% and 31%.
  - Memory usage was monitored and remained within acceptable limits.
- **Enhancements:**  
  - Integrated detailed logging and progress bars (using `tqdm`) to monitor training dynamics.
  - Verified that the masking and label assignment were correctly applied to all sequences.

### Model Refinements and Debugging
- **Hyperparameter Tuning:**  
  - Experimented with larger batch sizes and slightly adjusted the learning rate.
  - Conducted ablation studies to assess the impact of varying the number of transformer layers and attention heads.
- **Debugging Measures:**  
  - Added logging to trace intermediate tensor outputs and gradient norms.
  - Identified and resolved issues related to non-standard nucleotide handling during data processing.
- **Future Directions Identified:**  
  - The current architecture may need further depth and additional tuning to approach the desired masked token accuracy.
  - Alternative masking strategies and possibly integrating recurrent layers for enhanced sequence context could be explored.

### Final Remarks for the Two-Day Sprint
- **Summary of Achievements:**  
  - Established robust data pipelines for both synthetic and real genomic data.
  - Developed and tested a transformer model that learns to predict masked tokens.
  - Collected preliminary performance metrics and identified key areas for improvement.
- **Lessons Learned:**  
  - Rapid iteration and detailed logging are essential for diagnosing model performance issues.
  - Even within a short timeframe, valuable insights were gained regarding architecture refinements and hyperparameter sensitivity.
- **Next Steps:**  
  - Expand the model architecture and systematically tune hyperparameters to improve accuracy.
  - Scale up experiments using additional genomic data and, if needed, leverage cloud-based resources.
  - Explore external validation methods (e.g., AlphaFold) to assess the biological plausibility of generated sequences.

---