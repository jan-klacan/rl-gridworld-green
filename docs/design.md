# Environment design

## Goal
Build a small GridWorld environment with core RL concepts:
- State representation
- Reward design
- Exploration vs exploitation
- Delayed reward
- Termination vs truncation
- Partial observability

## Version plan

### v0: Fully observable baseline
- Grid size: 5x5
- Agent starts at fixed or random location
- Goal is at fixed or random location
- Actions: up, down, left, right
- Observation: full state = agent position + goal position
- Reward: +1 on reaching goal, 0 otherwise
- Termination: reaching goal
- Truncation: max episode length reached

### v1: Add obstacles
- Add blocked cells
- Invalid movement leaves agent in place
- Same reward structure

### v2: Add delayed reward pressure
- Add step penalty or sparse reward comparison
- Compare:
  - sparse reward: +1 at goal, 0 otherwise
  - shaped reward: +1 at goal, small negative step cost

### v3: Partial observability
- Observation is no longer full state
- Candidate options:
  - local 3x3 patch around agent
  - noisy direction hint toward goal
- Goal: study how reduced observability changes learning

## Core questions
1. How does reward design affect learning speed?
2. How does truncation affect value estimation?
3. How does partial observability make the task harder?
4. When does shaping help, and when does it distort behavior?

## Evaluation plan
- Random policy baseline
- Tabular Q-learning on v0
- Compare sparse vs shaped rewards
- Later: DQN on larger or partially observable versions

## Risks
- Bad rewards may teach pathological behavior
- Sparse rewards may make learning very slow
- Partial observability may require memory or better state design