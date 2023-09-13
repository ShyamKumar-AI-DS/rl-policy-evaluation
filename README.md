# EX - 02 - POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT

The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States

The environment has 7 states:
* Two Terminal States: **G**: The goal state & **H**: A hole state.
* Five Transition states / Non-terminal States including  **S**: The starting state.

### Actions

The agent can take two actions:

* R: Move right.
* L: Move left.

### Transition Probabilities

The transition probabilities for each action are as follows:

* **50%** chance that the agent moves in the intended direction.
* **33.33%** chance that the agent stays in its current state.
* **16.66%** chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards

The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation
<p align="center">
<img width="600" src="https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/e7af87e7-fe73-47fa-8bea-2040b7645e44"> </p>


## POLICY EVALUATION FUNCTION

### Formula
<img width="350" src="https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/e663bd3d-fc85-41c3-9a5c-dffa57eae250">

### Program
```
Name : Shyam Kumar A
Reg No : 212221230098
```
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    while True:
      V = np.zeros(len(P))
      for s in range (len(P)):
        for prob, next_state,reward,done in P[s] [pi (s)]:
          V[s] += prob*(reward + gamma * prev_V[next_state] *(not done))
      if np.max(np.abs(prev_V - V))<theta:
        break
      prev_V = V.copy()
    return V

V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)

V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)
```

## OUTPUT:
### Policy 1
![image](https://github.com/ShyamKumar-AI-DS/rl-policy-evaluation/assets/93427182/c4f6b41e-4e70-41dd-be60-a2f5cbc03290)

![image](https://github.com/ShyamKumar-AI-DS/rl-policy-evaluation/assets/93427182/5066f05d-ee02-4482-91c4-b907e0f210ac)

![image](https://github.com/ShyamKumar-AI-DS/rl-policy-evaluation/assets/93427182/032059f7-968a-4e62-a686-d1bc5f509a9c)



### Policy 2
![image](https://github.com/ShyamKumar-AI-DS/rl-policy-evaluation/assets/93427182/99cc8455-6d8a-4702-82bb-b80d89cb73df)

![image](https://github.com/ShyamKumar-AI-DS/rl-policy-evaluation/assets/93427182/323b130f-4ade-48c4-8483-091d7c289cb2)

![image](https://github.com/ShyamKumar-AI-DS/rl-policy-evaluation/assets/93427182/0dc19665-ae0a-406a-b284-d429ceaec52c)

### Comparison
![image](https://github.com/ShyamKumar-AI-DS/rl-policy-evaluation/assets/93427182/d495d837-f98b-4d5b-9f46-c470e686cf2b)

### Conclusion
<p align="center">
![image](https://github.com/ShyamKumar-AI-DS/rl-policy-evaluation/assets/93427182/a3e945e0-683c-41ee-aa1e-a3d0ea3b39d9)



## RESULT:
Thus, a Python program is developed to evaluate the given policy.
