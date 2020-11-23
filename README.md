# Learning to Play Imperfect-Information Games by Imitating an Oracle Planner

This repository contains source code for Pommerman experiments in the paper titled "Learning to Play Imperfect-Information Games by Imitating an Oracle Planner" by Rinu Boney, Alexander Ilin, Juho Kannala and Jarno Seppänen.

Dependencies:
- pommerman package from [https://github.com/MultiAgentLearning/playground](https://github.com/MultiAgentLearning/playground)
- cpommerman (see `cython_env` for installation instructions)
- numpy 1.18.1
- pytorch 1.5.0

### Oracle Planner with Full Observability

The planning results reported in the paper can be reproduced by running:
```
python plan.py
```
Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --planner 		| fdts	 | 'fdts' or 'mcts' or 'mcs'
| --mab   		| ts | 'ts' or 'ucb'
| --n_simulations  		| 100 | number of planning rollouts at each time-step
| --horizon  		| 20	 | depth of planning rollouts in FDTS and MCS
| --n_threads 		| 5 | number of games to play in parallel
| --n_episodes 	| 100 	| total number of games to play

### Training Follower Policy with Partial Observability

The follower results reported in the paper can be reproduced by running:
```
python follow.py
```

### License

This project is licensed under the terms of the GPL-3.0 License. See LICENSE file for details.
