# Week 4 Deliverable Instructions

> Note: This document provides a suggested template and structure for your Week 4 deliverable. While the core requirements must be met, you have flexibility in how you organize and present your work. Feel free to adapt this structure to better suit your specific project needs, as long as you clearly communicate your problem, approach, implementation, and results.

> Important: This deliverable sets the foundation for your semester's work. While your project may evolve as you learn more, you should write this report as if this is the problem you'll be working on. Think of it as your project's initial roadmap - it should be concrete and actionable, even if you end up taking some detours later.

## 1. Report (report.md)
A 2-page markdown document in your repository containing:

### Problem Statement (1/2 page)
- What are you optimizing? (Be specific)
- Why does this problem matter?
- How will you measure success?
- What are your constraints?
- What data do you need?
- What could go wrong?

### Technical Approach (1/2 page)
- Mathematical formulation (objective function, constraints)
- Algorithm/approach choice and justification
- PyTorch implementation strategy
- Validation methods
- Resource requirements and constraints

### Initial Results (1/2 page)
- Evidence your implementation works
- Basic performance metrics
- Test case results
- Current limitations
- Resource usage measurements
- Unexpected challenges

### Next Steps (1/2 page)
- Immediate improvements needed
- Technical challenges to address
- Questions you need help with
- Alternative approaches to try
- What you've learned so far

## 2. Notebook (week4_implementation.ipynb)
A Jupyter notebook containing:

### Problem Setup
- Clear problem statement
- Mathematical formulation
- Data requirements
- Success metrics

### Implementation
- All required imports
- Objective function implementation
- Optimization algorithm
- Key parameters and choices
- Basic logging/monitoring

### Validation
- Test cases with results
- Performance measurements
- Resource monitoring
- Example outputs
- Edge case handling

### Documentation
- Code comments explaining decisions
- Markdown cells describing approach
- Known limitations
- Debug/test strategies
- Next steps

## 3. Repository Structure
```
your-repo/
├── README.md                    # Project overview
├── report.md                    # Week 4 report
├── notebooks/
│   └── week4_implementation.ipynb  # Working implementation
├── src/
│   ├── model.py                # Core optimization code
│   └── utils.py                # Helper functions
├── tests/
│   └── test_basic.py           # Basic validation tests
└── docs/
    ├── llm_exploration/        # AI conversation logs
    │   └── week4_log.md        # Week 4 conversations
    └── development_log.md      # Progress & decisions
```

## Grading Criteria

### Report 
- Clear problem definition
- Well-formulated technical approach
- Evidence of testing/validation
- Thoughtful next steps

### Implementation 
- Code runs end-to-end
- Clear objective function
- Working optimization loop
- Basic validation/testing
- Resource monitoring

### Development Process 
- AI conversations show exploration
- Failed attempts documented
- Design decisions explained
- Safety considerations
- Alternative approaches considered

### Repository Structure 
- Clean organization
- Clear documentation
- Working tests
- Complete logs

## Remember
- Start with a clear mathematical formulation
- Get a basic version working first
- Test and validate thoroughly
- Document your journey
- Use AI to explore options

## Getting Help
- Use AI to explore approaches
- Test incrementally
- Run small experiments
- See course staff for blockers

## Timeline
- Days 1-2: Problem formulation and validation
- Days 2-3: Basic implementation and testing
- Days 3-5: Refinement and documentation

Remember: A working minimal implementation beats an ambitious plan 

## Self-Critique Guidelines

Your self-critique should be a separate document that helps you plan concrete improvements for your next draft. Keep it focused and actionable - max 1 page.

### How to Write Your Self-Critique

Follow the OODA (Observe, Orient, Decide, Act) process:

1. **OBSERVE**: Read your report critically
   - Read it as if you're seeing it for the first time
   - Run your code again, checking results
   - Note initial reactions and questions

2. **ORIENT**: Analyze your work
   Write these sections:
   - **Strengths (Max 3 bullet points)**
     Example: "Clear problem statement with concrete real-world impact"
  
   - **Areas for Improvement (Max 3 bullet points)**
     Example: "Mathematical formulation needs more rigor - currently just intuitive description"
  
   - **Critical Risks/Assumptions (2-3 sentences)**
     Example: "Assuming dataset will fit in memory. Need to test with realistic data size."

3. **DECIDE**: Plan your next steps
   Write this section:
   - **Concrete Next Actions (Max 3 bullet points)**
     - Must directly address your "Areas for Improvement"
     - Be specific and achievable within a week
     Example: "Write out optimization objective function with constraints"

4. **ACT**: Prepare for execution
   Write this section:
   - **Resource Needs (2-3 sentences)**
     - What tools/knowledge/help do you need?
     - Be specific about blockers
     Example: "Need to learn PyTorch autograd for custom gradients. Will use tutorial X."

### Tips for Effective Self-Critique

1. Be brutally honest - this is for you
2. Focus on what you can change next week
3. Prioritize - don't list everything
4. Think about code AND report
5. Consider your reader's perspective

### Common Pitfalls to Avoid

1. Vague improvements ("make it better")
2. Too many actions (unfocused)
3. Just listing problems without solutions
4. Ignoring technical depth
5. Forgetting about implementation

### Using AI Help

Feel free to:
1. Ask an LLM to critique your report
2. Compare its feedback with yours
3. Use it to brainstorm improvements

But your final critique should reflect your judgment.

Remember: The goal is to have a clear plan for your next draft. Quality > Quantity.
