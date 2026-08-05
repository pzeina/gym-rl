

# General comments

- I want the loss/death of higher ranks agents to be more important than lower ranks. For example, if a general dies, it should have a bigger impact on the (negative) reward than if a private dies.  

- I want to introduce a special type of agents, that, in addition to the rank, holds a "human" attribute. This attribute will be a boolean that indicates whether the agent is a human or not. The idea is that the death of a human agent should have a very high negative reward, regardless of the rank. This is to simulate the idea that human lives are more valuable than non-human lives.

- It is forbidden for a human agent to have a rank lower than any non-human agent. 