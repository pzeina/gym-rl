

# General comments

- I want the loss/death of higher ranks agents to be more important than lower ranks. For example, if a general dies, it should have a bigger impact on the (negative) reward than if a private dies.  

- I want to introduce a special type of agents, that, in addition to the rank, holds a "human" attribute. This attribute will be a boolean that indicates whether the agent is a human or not. The idea is that the death of a human agent should have a very high negative reward, regardless of the rank. This is to simulate the idea that human lives are more valuable than non-human lives.

- It is forbidden for a human agent to have a rank lower than any non-human agent. 

- Increase the map size, to give more freedom of movement to the agents. This will also allow for more complex strategies and tactics to be employed.

- Add a ""support" action. There is a known doctrine in French that says: "pas un pas sans appui" (not a step without support). This means that an agent should not move forward without having support from other agents. The support means fire support, that allows the moving agent to move forward without being shot at by the enemy. It should increase a bit the efficiency of the fire, when multiple coordinated agents are shooting at the same target. The support action should be a boolean that indicates whether the agent is providing support or not, and apply only to nearby agents, as in reality.


## Next steps

- Two types of adversaries: symmetric, and asymetric warfare. In symetric warfare, the ennemy has a similar structure and hierarchy and action set of orders and objectives; it mimics a conventional army. In contrast, in asymetric warfare, the ennemy is a non-conventional force acting like a guerilla force, with limited hierarchy and different operational objectives and techniques. You will have to define the action set and objectives of such an asymmetric force, which usually includes ambushes, sabotage, and hit-and-run tactics, targetting maximum casualties and disruption of the conventional force's operations, with lower importance given to the survival of the asymmetric force's own agents.

- Improve the realism of the simulation by introducing more complex terrain, with infrastructure, buildings, and natural obstacles that can affect the movement and line of sight of agents. This will require implementing a more sophisticated pathfinding algorithm and line-of-sight calculations.

- 