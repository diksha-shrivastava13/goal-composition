# Goal Composition

### Research Statement
**Can we model what shards does a particular environment induce?**

### Research Plan (04.12.2025)

1. **Training agents robust to environmental interventions in variations of minigrid environments with JaxUED** ([reference](https://arxiv.org/abs/2507.03068)):

   1. On TMaze environments (via DR, Robust PLR, ReMiDi)
   2. On Keys and Chests environment (via ACCEL and DR)
   3. Standard Minigrid Maze (via ACCEL and DR)

    Almost done with this. Need to move now in the scalable oversight ([robust agents learn causal world models](https://arxiv.org/abs/2402.10877)) direction. 

2. **Theoretical work with Causal Influence Diagrams for inferring intention and instrumental goals from behaviour** ([reference](https://arxiv.org/abs/2402.07221)):

   1. Made some theoretical progress with Structural Causal Games (SCGs) for honesty and deception. The next step would be moving in the [Past-Specific Objectives (PSO)](https://arxiv.org/abs/2204.10018) algorithm direction to make sure the graphical criteria for misaligned traits aren’t met in training. 
      1. The theoretical work so far is for single goal scenarios or with complimentary, opposing goals (like honesty and deception), I’ve to expand the graphical criteria to composition of goals.
      2. Very unsure if PSO would be effective for the journey-learning bit of LLMs. At the very least, “bad behaviour getting reinforced accidentally” part of [natural emergent misalignment](https://arxiv.org/abs/2511.18397) should be mitigated. 
      3. The question of expensive post-training vs inoculation prompting needs to be tested. The first step would be to investigate failure modes of inoculation prompting. 
   2. Need to design a better eval for deception, testing on top of inoculation prompting. Basically, training an agent to be more truthful incentivizes deception & reward hacking for LMs. We already knew this from the [inoculation prompting](https://arxiv.org/abs/2510.05024) paper. 

3. **Chess Puzzles as a testbed for composition of agentic traits (shards; goals of long-term, short-term, and of varying weights)**

   1. Basically an agentic wargame where every piece is an agent contributing to the terminal goal of protecting their "kingdom", can serve as an ideal environment to analyse how collective traits like deception, honesty, fairness, sacrifice, agency, control, and goal-directedness emerge in the interactions between the agents. 
   2. Any "move" in this environment will be a composition of goals and values of different weights, allowing us to investigate how these values form, how cooperation and collusion works, and if there's a chance that the agents refuse to play since certain "powerless" agents will be necessarily sacrificed. 
   3. I'm very curious to see how these "agentic chess" sequences look, and what cooperation looks like as a complex composition of goals.

    I got this idea last Thursday, but haven’t had the time to model this yet. [Generating Creative Chess Puzzles](https://arxiv.org/abs/2510.23881) is my go-to paper for it right now, I’d really appreciate it if anyone can point me to any more resources for it. Any good papers to study on multi-agent cooperation would be amazing.
