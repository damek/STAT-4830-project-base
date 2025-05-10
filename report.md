# Problem Statement (1/2 page)
### What are you optimizing? (Be specific)
We are looking to optimize the process of artificial protein synthesis by using transformers to ideally predict function and form, and generate sequences to meet specific critereon for proteins. Initially, what we are looking at doing, is provided a partial DNA sequence (with parts blacked out / missing), we want to predict the remaining parts of the sequence. With this we want to optimize the loss (minimum loss), between the actual sequence and the predicted sequence. Further along in the development process, we want to explore predicitng the sequence itself from a functional description, and use the same/similar loss function.

### Why does this problem matter?
Currently, recent developments in protein syntehsis, such as AlphaFold, ESMFold, and others, have accelerated the process of drug development and protein synthesis significantly. The next step in protein synthesis and similar fields is being able to generate protein sequences to match a specific function, structure, and form, in order to accelerate drug development, protein development, and unlock a host of solutions to biological and medical issues whose solutions have eluded us for years.

### How will you measure success?
Currently, this is one of the biggest challenges we anticipate to see with the project. We can measure success by comparing our genearted protein sequences with existing NCBI and human genome databases, to see whether they match patterns that we can currently find in the human genome. Additionally, we can test by running AlphaFold queries to see whether the generated protein sequences would properly fold, based on AlphaFold's models to predict whether the protein's attributes align. In terms of our optimization, we want something around or above a 90% accuracy in predicting blacked out sequences.


### What are your constraints?
Our constraints are pretty biologically-based: We want to be able to develop protein sequences that fold correctly, and that ideally match the specified function. In our specific implementation, our main constraint (at least initially) will likely be sequence length. We will likely stay in 5 figures or less, and perhaps expand later down the road if we see success. Of course, there is the constraints of the sequence having valid base pairs, etc...

### What data do you need?
A lot of the data we would use for this project come from NCBI databases and human genome banks, which are mostly publicly available. The data is pretty large in size, which may lead to further issues or narrowing in the scope of the project later on. Additionally, if we can find any research or data regarding artificial protein synthesis validation methods, that would also prove helpful.

### What could go wrong?
One of our biggest concerns with carrying out this project is that validation processes are a little sketchy at the moment. While we can compare against NCBI databases and AlphaFold, we have limited ways of checking whether our generated protein sequence folds correctly or not.

# Technical Approach (1/2 page)
### Mathematical formulation (objective function, constraints)
(sorry for how the math looks, if you can paste it into overleaf or anything it looks a lot better! Couldn't figure out how to do it in markdown)

We define our objective as minimizing the discrepancy between the predicted DNA sequence and the actual sequence. For each input sequence  X  with masked positions  \mathcal{M} , the transformer predicts a probability distribution over possible base tokens  p(x_i \mid X_{\setminus \mathcal{M}})  at each masked position  i . The loss function is the cross-entropy loss computed as


\mathcal{L} = - \sum_{i \in \mathcal{M}} \log p(x_i^{\text{true}} \mid X_{\setminus \mathcal{M}})


subject to constraints ensuring that:
- The predicted sequences contain only valid base pairs.
- The sequence length remains within a defined range (initially 5-digit sequences).
- In downstream tasks, generated sequences must satisfy functional and structural constraints inferred from biological data.
### Algorithm/approach choice and justification
Our approach leverages transformer architectures, which have proven their effectiveness in handling sequential data through self-attention. We will adopt a masked language modeling strategy similar to BERT:
- Input: A partially observed DNA sequence with masked tokens.
- Process: The transformer encodes the sequence, using context from the unmasked tokens to predict the missing bases.
- Output: A probability distribution over possible nucleotides at each masked position.

The choice is justified by transformers’ ability to capture long-range dependencies in sequences, a critical aspect when predicting functionally and structurally relevant motifs in protein synthesis. This approach aligns with successful strategies in natural language processing and recent advancements in protein modeling.
### PyTorch implementation strategy
PyTorch Implementation Strategy
1.	Data Preparation:
- Tokenize DNA sequences and introduce masks in randomly selected positions.
- Create paired datasets of masked sequences (inputs) and the corresponding true tokens (labels).
2.	Model Architecture:
- Implement an embedding layer to convert tokens into dense vectors.
- Stack multiple transformer encoder layers to learn contextual representations.
- Add a final linear layer projecting to the vocabulary of base pairs, followed by a softmax activation for probability distribution.
3.	Training Loop:
- Utilize PyTorch’s standard training loop with GPU acceleration.
- Apply the cross-entropy loss function over the masked positions.
- Incorporate optimization techniques such as learning rate scheduling and gradient clipping.
- Employ early stopping and model checkpointing to prevent overfitting.
4.	Scalability Considerations:
- Use DataLoader for efficient batch processing.
- Leverage mixed precision training if necessary to manage large datasets from NCBI and other genomic repositories.
### Validation methods
Validation Methods

1. Quantitative Metrics:
- Monitor masked token prediction accuracy aiming for at least 90%.
- Evaluate loss trends on a held-out validation set to detect overfitting.

2. Biological Plausibility:
- Cross-reference generated sequences against NCBI and human genome databases to ensure adherence to known genetic patterns.
- Validate predicted protein folding using state-of-the-art tools like AlphaFold to confirm that the sequences fold into plausible three-dimensional structures.
3. Iterative Testing:
- Incorporate feedback from biological validation to refine model architecture and training strategies.
- Conduct ablation studies to determine the impact of various model components on overall performance.
### Resource requirements and constraints
Resource Requirements:
- Computing: High-performance GPUs/TPUs, ample memory and fast SSD storage to process large genomic datasets (cloud).
- Software: PyTorch and related data management tools (already have)
- Integration: Access to validation tools like AlphaFold (Open Source so should be good)

Constraints:
- Data Size: Large, complex datasets that require efficient streaming and preprocessing.
- equence Length: Starting with shorter sequences (five figures) to ensure computational feasibility.
- Time & Cost: Long training cycles and budget limits impacting computational resources and experimentation.

# Initial Results (1/2 page)
### Evidence your implementation works
- Looking at the results, we can see that our model is able to predict the masked tokens with that high of an accuracy. We can also see that the loss function isn't really able to minimize the discrepancy between the predicted and actual sequences. However the epoch's loss is still decreasing, which is a good sign that the model is learning until it converges.

### Basic performance metrics

Our initial accuracy show the model is able to predict around 33% of the masked tokens. This is honestly a bit low, but we can see that the model is learning and improving over time. However this is still not that much better than a random guess.
### Test case results
### Current limitations
- This dataset is small and may not capture the complexity of real DNA sequences, especially larger ones.
- The model is simple; larger and more complex architectures may be needed for real-world data.
### Resource usage measurements
- Memory Usage: 309.80 MB (for synthetic data)
- Additonal Memory Usage: 71 MB (for actual data)
### Unexpected challenges
- Accuracy is much lower than expected, but we can see that the model is learning and improving over time. We will need to keep an eye on this and see if we can improve it.

### Next Steps (1/2 page)
### Immediate improvements needed
- We can see the model is at a low accuracy. We will need to address either the hyperparamaters or the architecture of the model. We will try to increase dataset size (and let it run for longer) to see if we can improve the accuracy (perhaps using a cloud GPU?).
### Technical challenges to address
- Again we will need to rethink the architecture of the model. We will try to increase the size of the dataset and see if we can improve the accuracy. We will also need to look into the hyperparameter tuning.
### Questions you need help with
- How do we improve accuracy? What avenues should we explore? We will look at this ourselves, but may need help with this.
### Alternative approaches to try
- We can try using a different architecture, such as a RNN, or more deep learnign approaches if that helps. We wil also need to look into hyperparameter tuning.
### What you've learned so far
- Simply chugging in a Transformer architecture to predict the masked tokens in a DNA sequence is not
going to be enough to get a good result. We will need to look into the architecture of the model, and
see if we can improve the accuracy.