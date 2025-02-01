# Optimization of NYC Travel for Electric Vehicles

## Problem Statement (1/2 page)

### What are you optimizing?  
We aim to optimize the pathfinding algorithm for electric vehicle (EV) travel in NYC by incorporating multiple city-specific factors into a graph-based model. 
Initially, we will implement Dijkstra's algorithm to determine optimal paths to given locations and subsequently refine the 
algorithm using energy consumption statistics, congestion taxes, and real-time traffic data. 

### Why does this problem matter?  
With the increasing adoption of EVs and evolving urban policies such as congestion pricing, optimizing travel routes for efficiency and sustainability is crucial. 
Our project aims to provide precise insights into energy-efficient travel while factoring in urban constraints, 
distinguishing itself from standard navigation software. The goal is to incorporate an optimization in the context of specifically traveling around the city with electric vehicles.
We understand that the conditions that govern travel for traditional gas vehicles and energy consumption is slightly modified when we take into account the travel with electric vehicles.



### How will you measure success?  
- Reduction in total energy consumption per trip  
- Improved travel time efficiency under varying traffic conditions  
- Accuracy of predicted congestion and travel cost estimations  
- Comparison against existing solutions like Google Maps  

### What are your constraints?  
- Limited access to real-time traffic and congestion pricing data  
- Computational efficiency of pathfinding under large-scale NYC road networks  
- Balancing trade-offs between energy efficiency and travel time  

### What data do you need?  
- NYC road network and traffic flow data  
- Electric vehicle energy consumption models  
- Congestion tax and toll pricing details  
- Weather and road condition data  

### What could go wrong?  
- Unreliable or incomplete traffic and pricing data  
- High computational complexity leading to slow performance  
- Unexpected legal or regulatory constraints on route optimization  

## Technical Approach (1/2 page)

### Mathematical Formulation  
- **Objective Function:** Minimize a cost function incorporating travel time, energy consumption, and congestion costs  
- **Constraints:** Traffic restrictions, charging station availability, congestion pricing zones  

### Algorithm/Approach Choice and Justification  
- Start with **Dijkstra’s algorithm** for shortest path computation  
- Extend to **A* or other heuristics** incorporating real-world EV constraints  
- Use **reinforcement learning** or optimization techniques to refine predictions  

### PyTorch Implementation Strategy  
- Develop graph-based representations of NYC’s road network  
- Implement a neural network model to predict travel costs dynamically  
- Optimize pathfinding using deep reinforcement learning  

### Validation Methods  
- Compare against Google Maps routes for efficiency  
- Simulate EV travel under different constraints  
- Test on historical traffic and congestion data  

### Resource Requirements and Constraints  
- High-performance computing for large-scale graph processing  
- Access to reliable real-time traffic and congestion data  

## Initial Results (1/2 page)

### Evidence Your Implementation Works  
- Initial Dijkstra’s implementation successfully finds paths in NYC’s road graph  

### Basic Performance Metrics  
- **Execution time:** Pathfinding efficiency under different conditions  
- **Energy estimation accuracy:** Comparing predicted vs. actual EV consumption  

### Test Case Results  
- Successful test runs on small NYC road network segments  
- Identified discrepancies in congestion tax handling  

### Current Limitations  
- Lack of real-time traffic updates in initial implementation  
- Energy model requires refinement for different EV types  

### Resource Usage Measurements  
- Memory and computational load during large-scale tests  

### Unexpected Challenges  
- Integrating real-world congestion pricing data into the model  

## Next Steps (1/2 page)

### Immediate Improvements Needed  
- Incorporate real-time traffic and congestion pricing data  
- Refine the energy consumption model  

### Technical Challenges to Address  
- Handling large-scale road networks efficiently  
- Adapting the algorithm to dynamic city conditions  

### Questions You Need Help With  
- Best sources for real-time congestion data  
- Methods to validate EV energy consumption models  

### Alternative Approaches to Try  
- Experimenting with machine learning-based route prediction  
- Integrating additional urban factors (e.g., pedestrian zones, road closures)  

### What You've Learned So Far  
- Basic pathfinding can be efficiently implemented  
- Integrating real-world constraints is complex but crucial  
