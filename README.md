# STAT 4830 Project Repository

Welcome to your project repository! This template helps you develop and implement an optimization project over the semester.

## Getting Started

1. **Finding Your Project Idea**
   - Start with our [Project Ideas Guide](docs/finding_project_ideas.md)
   - Use AI to explore and refine your ideas
   - Take time to find something you care about

   It's very important you learn to use AI tools in your work! [Noam Brown](https://x.com/polynoamial/status/1870307185961386366) (OpenAI) says that students should...
   > Practice working with AI. Human+AI will be superior to human or AI alone for the foreseeable future. Those who can work most effectively with AI will be the most highly valued.

   ![Noam tweet](figures/noam.png)

2. **Week 4 Deliverable**
  - Follow the [Week 4 Instructions](docs/assignments/week4_deliverable_instructions.md)
   - Required components:
     - Initial report draft
     - Self-critique document analyzing your report's strengths and weaknesses
     - Supporting Jupyter notebooks/code
  - Due: Friday, February 6, 2026

## Project Development Cycle

Each week follows an OODA (Observe, Orient, Decide, Act) loop that helps you improve your project systematically:

![Project Development Cycle - A diagram showing the OODA loop (Observe, Orient, Decide, Act) adapted for project development. Each phase has specific activities: Observe (Review Report, Check Results), Orient (Write Critique, Find Gaps), Decide (Plan Changes, Set Goals), and Act (Code, Run Tests). The phases are connected by arrows showing the flow of work, with a feedback loop labeled "Iterative Development" completing the cycle.](docs/figures/ooda_loop.png)

Each cycle produces specific deliverables:
- OBSERVE: Updated report draft
- ORIENT: Self-critique document
- DECIDE: Next actions plan
- ACT: Code changes & results

See the [Week 4 Instructions](docs/assignments/week4_deliverable_instructions.md) for detailed guidance on writing your first self-critique.

## Project Schedule

### Deliverables (Due Fridays)
- Week 2 (Jan 23): Email Project Team Names to Ai, Jiahao <jiahaoai@wharton.upenn.edu>
- Week 4 (Feb 6): Report Draft 1 + Code + Self Critique
- Week 5 (Feb 13): Slides Draft 1
- Week 6 (Feb 20): Report Draft 2 + Code + Self Critique
- Week 7 (Feb 27): Slides Draft 2
- Week 8: ⚡ Lightning Talks in Class (Mar 3/5) & Report Draft 3 due Friday ⚡
- Spring Break (Mar 7-15)
- Week 9 (Mar 20): Slides Draft 3
- Week 10 (Mar 27): Report Draft 4 + Code + Self Critique
- Week 11 (Apr 3): Slides Draft 4
- Week 12 (Apr 10): Report Draft 5 + Code + Self Critique
- Week 13 (Apr 17): Slides Draft 5
- Week 14 (Apr 21/23): Final Presentations in Class
- Week 15 (Apr 28): Final Report + Code + Self Critique

Note: Instructions for peer feedback will be added throughout the semester for each deliverable.

Each draft builds on the previous one, incorporating feedback and new results. You'll meet with course staff three times during the semester to discuss your progress.

## Project Grading

Each deliverable is graded on five components:
- Report (20%): Problem statement, methodology, results
- Implementation (35%): Working code, tests, experiments
- Development Process (15%): Logs, decisions, iterations
- Critiques (15%): Reflection and planning
  - Self-critiques (required)
  - Peer critiques (when assigned)
  - Response to feedback
- Repository Structure (15%): Organization, documentation, clarity

Remember:
- Quality > Quantity
- Working > Perfect

## Repository Structure

```
your-repo/
├── README.md                    # This file
├── report.md                    # Your project report
├── notebooks/                   # Jupyter notebooks
├── src/                        # Source code
├── tests/                      # Test files
└── docs/
    ├── finding_project_ideas.md    # Guide to finding your project
    ├── assignments/                # Assignment instructions
    ├── llm_exploration/           # AI conversation logs
    └── development_log.md         # Progress & decisions
```

## Development Environment

### Editor Setup
We recommend using **Cursor**. Students with a `.edu` address get **one year of Cursor Pro for free**: https://cursor.com/students. Cursor is VS Code–compatible (same shortcuts/extensions) but adds in-IDE AI assistance tuned for multi-file context and refactors.

### Required Tools
- Python 3.10+
- PyTorch
- Jupyter Notebook/Lab
- Git

## Git Setup and Workflow

### First Time Setup
1. Fork this repository
   - Click "Fork" in the top right
   - Name it `STAT-4830-[team-name]-project`
   - This creates your own copy that can receive updates

2. Set up Git (if you haven't already):
   Cursor includes Git integration and prompts you to install Git if it's missing.
   
   For detailed instructions, see the [Official Git installation guide](https://github.com/git-guides/install-git)

   After installing, set up your identity:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@upenn.edu"
   ```

3. Clone your fork:
   ```bash
   # HTTPS (easier):
   git clone https://github.com/[your-username]/STAT-4830-[team-name]-project.git

   # SSH (if you've set up SSH keys):
   git clone git@github.com:[your-username]/STAT-4830-[team-name]-project.git
   
   cd STAT-4830-[team-name]-project
   ```

4. Add upstream remote (to get updates):
   ```bash
   # HTTPS:
   git remote add upstream https://github.com/damek/STAT-4830-project-base.git

   # SSH:
   git remote add upstream git@github.com:damek/STAT-4830-project-base.git
   ```

5. Add your team members as collaborators:
   - Go to your repo on GitHub
   - Settings → Collaborators → Add people
   - Add using their GitHub usernames

### Working on Your Project
1. Create a new branch:
   ```bash
   git checkout -b exploration
   ```

2. Make changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin exploration
   ```

### Getting Updates
When the base repository is improved:
```bash
# Get updates
git fetch upstream
git checkout main
git merge upstream/main

# Update your branch
git checkout exploration
git merge main
```

### Troubleshooting
- Having Git issues? Post on Ed Discussion
- Can't push/pull? Check if you're using HTTPS or SSH
- Windows path too long? Enable long paths:
  ```bash
  git config --system core.longpaths true
  ```

## Getting Help
- Use AI tools (ChatGPT, GitHub Copilot)
- See course staff for technical issues
- Document your progress


## Spring 2025 Project Examples

Current student projects:

1. **Decentralized Recommendation for Cold-Start Personalization**  
   * **Summary:** Builds a cross-platform fashion recommender for users with little history. Synthesizes persona-level ratings, embeds ~3k products with CLIP image/text vectors, and benchmarks content-based filtering, collaborative filtering, low-rank matrix factorization, and a two-tower deep model. Evaluates RMSE/MAE and Precision/Recall@K to trade off global error vs. top-K relevance under cold-start.  
   * **Link:** [Final Report](https://github.com/kuomat/STAT-4830-vllm-project/blob/main/Final%20Report.pdf)

2. **Optimizing Attention Mechanisms in Transformer Models**  
   * **Summary:** Replaces $O(n^2)$ attention with efficient variants: learned sparse masks, Performer-style kernelized attention, and hierarchical sparsity. Trains on WikiText-2, minimizing KL-divergence to a baseline Transformer while tracking cross-entropy, coherence, and memory/latency. Shows custom masks preserve fluency with lower compute.  
   * **Link:** [Final Report](https://github.com/charisgao/STAT-4830-Optimizing-Attention-Project/blob/main/docs/report.md)

3. **Poker Zero: Risk-Aware Agents for No-Limit Hold'em**  
   * **Summary:** Designs a poker agent that blends LLM-guided reasoning with self-play reinforcement learning. Uses counterfactual regret minimization heuristics and win-rate/stack-size metrics against GTO-style opponents to study bluffing, bet sizing, and stability under incomplete information.  
   * **Link:** [Final Report](https://github.com/AC2005/STAT-4830-poker/blob/main/docs/Final%20Report.pdf)

4. **Portfolio Refinement Through Iterative Sequential Modeling (PRISM)**  
   * **Summary:** Optimizes daily portfolios with penalties on drawdown, turnover, and concentration. Formulates a multi-objective loss, applies sequential modeling to adapt weights, and benchmarks Sharpe, max drawdown, and turnover against “safe” baselines.  
   * **Link:** [Final Report](https://github.com/dhruv575/STAT-4830-project-base/blob/main/report.md)

5. **Optimization in Preference Learning**  
   * **Summary:** Predicts hotel choices using two pipelines: mixture preference models optimized via Frank–Wolfe variants, and low-rank matrix completion with bias-aware initialization and Huber loss. Expedia-derived data backtests show linear preference models outperform deeper nets under sparsity, while matrix completion boosts robustness.  
   * **Link:** [Final Report](https://github.com/Lexaun-chen/STAT-4830-Group-Project/blob/main/Final_Report.pdf)

6. **Designing Good Rewards for Reinforcement Learning on LLMs**  
   * **Summary:** Implements GRPO on Qwen-1.5B for GSM8K-style reasoning, comparing rule-based vs. hybrid perplexity rewards. Early experiments on matrix inversion validate dense rewards; hybrid absolute/relative perplexity improves stability over naive reward shaping.  
   * **Link:** [Final Report](https://github.com/JustinSQiu/STAT-4830-curriculum-learning-project/blob/main/report.md)

7. **SAT Formula Extraction via Transformer Optimization**  
   * **Summary:** Fine-tunes FLAN-T5 to emit symbolic formulas for SAT word problems, then solves them with SymPy. Uses GRPO-style training, regex parsing, and answer-level checks; reports ~81% symbolic similarity and ~72% answer accuracy with a formula-to-answer pipeline.  
   * **Link:** [Final Report](https://github.com/awu626/STAT-4830-project/blob/main/FinalReport.md)

8. **Modeling Human Behavior Without Humans – Bringing Prospect Theory to Multi-Agent RL**  
   * **Summary:** Extends MADDPG with cumulative prospect theory transforms (CPT-MADDPG) to control risk attitudes. Evaluates on Simple Tag/Spread and first-price auctions; shows risk-seeking CPT speeds early learning, loss-averse CPT enforces prudence, and shared utility aggregation preserves coordination.  
   * **Link:** [Final Report](https://github.com/sheyanlalmohammed1/STAT-4830-CPT-MARL-project/blob/main/report.pdf)

9. **Sleep is All We Need: Optimizing EEG-Based Deep Learning Models for N1 Sleep Onset Detection**  
   * **Summary:** Builds a two-stage ensemble for detecting the rare N1 sleep stage from single-channel EEG. Combines convolutional encoders, domain-specific PSD/Catch22 features, a transformer sequence model, and an N1-focused detector, improving N1 F1 from 0.38 to 0.53 while maintaining overall accuracy.  
   * **Link:** [Final Report](https://github.com/kimberlyliang/STAT-4830-GOALZ-project/blob/main/report.pdf)

10. **Optimizing Vehicle Routing with Graph-Based and Probabilistic Models**  
    * **Summary:** Compares Dijkstra/A* baselines with BERT-based trip models, reinforcement learning policies, and graph neural networks to optimize travel time and EV energy use. Uses OSMnx data plus eVED/EV trip logs; predicts routes and per-trip energy, benchmarking against historical trips and shortest-path baselines.  
    * **Link:** [Final Report](https://github.com/TheCrypted/STAT-4830-project-base/blob/main/docs/final_report.md)





