### 1. Summary

#### a. Verdict

I have successfully solved the problem. The game has six Pure Nash Equilibria and two distinct types of Mixed Nash Equilibria.

The six Pure Nash Equilibria (PNE) are the strategy profiles where two players choose one room and the third player chooses the other room. Explicitly, these are:
$(A, A, B)$, $(A, B, A)$, $(B, A, A)$, $(A, B, B)$, $(B, A, B)$, and $(B, B, A)$.

The Mixed Nash Equilibria (MNE), where at least one player randomizes, are:
1.  A unique symmetric MNE where all three players choose Room A with probability $1/2$ and Room B with probability $1/2$.
2.  Six families of MNEs where two players choose opposite rooms with certainty, and the third player randomizes their choice between the two rooms. For example, Player 1 chooses A, Player 2 chooses B, and Player 3 chooses A with any probability $p \in (0,1)$.

#### b. Method Sketch

My method involves analyzing the conditions for Nash Equilibrium in pure and mixed strategies separately.

1.  **Pure Nash Equilibria (PNE):** I conduct an exhaustive analysis of all possible pure strategy profiles. These profiles fall into two categories based on the number of players in each room:
    *   **3-0 split:** All three players choose the same room (e.g., $(A, A, A)$). I show that any player has a profitable unilateral deviation to the empty room, so these are not PNE.
    *   **2-1 split:** Two players choose one room, and the third player chooses the other (e.g., $(A, A, B)$). I show that neither the players in the majority nor the player in the minority have a profitable unilateral deviation. This identifies all six PNEs.

2.  **Mixed Nash Equilibria (MNE):** I define a mixed strategy for each player $i$ as the probability $p_i$ of choosing Room A. A player's best response depends on the strategies of the other two players, $p_j$ and $p_k$.
    *   **Best Response Conditions:** I first derive the expected payoffs for player $i$ for choosing A or B, which are $E_i(A) = (1-p_j)(1-p_k)$ and $E_i(B) = p_j p_k$. From this, I establish the best response conditions:
        *   Player $i$ strictly prefers A if $p_j+p_k < 1$.
        *   Player $i$ strictly prefers B if $p_j+p_k > 1$.
        *   Player $i$ is indifferent between A and B if $p_j+p_k = 1$.
    *   **Case Analysis:** For a profile $(p_1, p_2, p_3)$ to be a MNE, each player's strategy must be a best response to the others'. I classify MNE based on the number of players who use a strictly mixed strategy (i.e., $p_i \in (0,1)$), which requires the indifference condition to hold for that player.
        *   **Three mixing players:** The indifference conditions for all three players ($p_1+p_2=1$, $p_2+p_3=1$, $p_1+p_3=1$) yield a unique solution: $p_1=p_2=p_3=1/2$.
        *   **Two mixing players:** The indifference conditions for the two mixing players force their probabilities to be outside the $(0,1)$ range, leading to a contradiction. No such equilibria exist.
        *   **One mixing player:** Let Player 1 mix. The indifference condition requires $p_2+p_3=1$. Since Players 2 and 3 play pure strategies, they must choose opposite rooms (e.g., $p_2=1, p_3=0$). I verify that for this configuration, the best response conditions for the pure-strategy players are satisfied for any $p_1 \in (0,1)$. This yields six families of MNEs.

### 2. Detailed Solution

Let the set of players be $N = \{1, 2, 3\}$. Each player $i \in N$ chooses an action $s_i$ from the action set $S_i = \{A, B\}$. A pure strategy profile is a tuple $s = (s_1, s_2, s_3)$. The payoff function $u_i(s)$ for player $i$ is $1$ if player $i$ is the only player to choose action $s_i$, and $0$ otherwise.

#### 1. Pure Nash Equilibria

A pure strategy profile $s^*$ is a Pure Nash Equilibrium (PNE) if no player can improve their payoff by unilaterally changing their strategy. That is, for every player $i \in N$, $u_i(s^*) \ge u_i(s'_i, s^*_{-i})$ for all $s'_i \in S_i$.

We examine the two possible types of pure strategy profiles, categorized by the distribution of players.

**Case 1: All players choose the same room (3-0 split).**
Consider the profile $s = (A, A, A)$. All three players are in Room A. No player is alone, so the payoff vector is $(u_1, u_2, u_3) = (0, 0, 0)$.
If Player 1 unilaterally deviates to B, the new profile is $s' = (B, A, A)$. In this profile, Player 1 is the only person in Room B, so their payoff is $u_1(s') = 1$. Since $u_1(s') > u_1(s)$, Player 1 has a profitable deviation.
Therefore, $(A, A, A)$ is not a PNE. By symmetry, the profile $(B, B, B)$ is also not a PNE.

**Case 2: Two players choose one room, and one player chooses the other (2-1 split).**
Consider the profile $s = (A, A, B)$. Players 1 and 2 are in Room A, and Player 3 is in Room B. Player 3 is alone, while Players 1 and 2 are not. The payoff vector is $(u_1, u_2, u_3) = (0, 0, 1)$.
Let's check for profitable deviations for each player:
*   **Player 1 (in the majority):** Current payoff is $u_1(A, A, B) = 0$. If Player 1 deviates to B, the profile becomes $(B, A, B)$. Now, Players 1 and 3 are in Room B, so Player 1's payoff is $u_1(B, A, B) = 0$. Since the payoff does not increase, Player 1 has no incentive to deviate.
*   **Player 2 (in the majority):** By symmetry with Player 1, Player 2 also has no incentive to deviate.
*   **Player 3 (in the minority):** Current payoff is $u_3(A, A, B) = 1$. If Player 3 deviates to A, the profile becomes $(A, A, A)$. All players are in Room A, so Player 3's payoff is $u_3(A, A, A) = 0$. Since the payoff would decrease, Player 3 has no incentive to deviate.

Since no player can unilaterally improve their payoff, the profile $(A, A, B)$ is a PNE.
By symmetry, all six profiles where the players are split 2-to-1 between the rooms are PNEs. These are:
*   $(A, A, B)$, $(A, B, A)$, $(B, A, A)$
*   $(B, B, A)$, $(B, A, B)$, $(A, B, B)$

#### 2. Mixed Nash Equilibria

Let $p_i \in [0, 1]$ be the probability that player $i$ chooses Room A. The probability of choosing Room B is $1-p_i$. A mixed strategy profile is $(p_1, p_2, p_3)$. We seek all equilibria where at least one player mixes, i.e., at least one $p_i \in (0,1)$.

**Best Response Conditions**
For any player $i$, let the other two players be $j$ and $k$. The expected payoff for Player $i$ for choosing A, given the strategies $p_j$ and $p_k$, is:
$E_i(A) = 1 \cdot P(\text{Pj chooses B and Pk chooses B}) = (1-p_j)(1-p_k)$.
The expected payoff for Player $i$ for choosing B is:
$E_i(B) = 1 \cdot P(\text{Pj chooses A and Pk chooses A}) = p_j p_k$.

Player $i$'s best response is:
*   To play pure A ($p_i=1$) if $E_i(A) > E_i(B)$.
*   To play pure B ($p_i=0$) if $E_i(B) > E_i(A)$.
*   To be willing to mix ($p_i \in (0,1)$) if $E_i(A) = E_i(B)$.

The condition $E_i(A) = E_i(B)$ is $(1-p_j)(1-p_k) = p_j p_k$, which simplifies to $1 - p_j - p_k + p_j p_k = p_j p_k$, or $p_j + p_k = 1$.
So, the best response conditions for player $i$ are:
1.  $p_i=1$ is a best response if $p_j+p_k \le 1$.
2.  $p_i=0$ is a best response if $p_j+p_k \ge 1$.
3.  Any $p_i \in [0,1]$ is a best response if $p_j+p_k = 1$.

For $(p_1, p_2, p_3)$ to be a Nash Equilibrium, these conditions must hold for all three players simultaneously. We analyze the cases based on the number of players who use a strictly mixed strategy.

**Case 1: All three players mix ($p_1, p_2, p_3 \in (0,1)$).**
For each player to be willing to mix, their indifference condition must hold:
1.  Player 1 mixes $\implies p_2 + p_3 = 1$
2.  Player 2 mixes $\implies p_1 + p_3 = 1$
3.  Player 3 mixes $\implies p_1 + p_2 = 1$

From (1), $p_3 = 1 - p_2$. Substituting into (2) gives $p_1 + (1 - p_2) = 1$, which implies $p_1 = p_2$.
Substituting $p_1 = p_2$ into (3) gives $2p_1 = 1$, so $p_1 = 1/2$.
This implies $p_1 = p_2 = p_3 = 1/2$.
Since these values are in $(0,1)$, this is a valid MNE. There is one such equilibrium: $(1/2, 1/2, 1/2)$.

**Case 2: Exactly two players mix.**
Let Players 1 and 2 mix ($p_1, p_2 \in (0,1)$), and Player 3 play a pure strategy ($p_3 \in \{0,1\}$).
The indifference conditions for Players 1 and 2 must hold:
*   $p_2 + p_3 = 1$
*   $p_1 + p_3 = 1$

If $p_3 = 1$, the conditions become $p_2 + 1 = 1 \implies p_2 = 0$ and $p_1 + 1 = 1 \implies p_1 = 0$. This contradicts the assumption that $p_1, p_2 \in (0,1)$.
If $p_3 = 0$, the conditions become $p_2 + 0 = 1 \implies p_2 = 1$ and $p_1 + 0 = 1 \implies p_1 = 1$. This also contradicts the assumption that $p_1, p_2 \in (0,1)$.
Thus, there are no MNE where exactly two players mix.

**Case 3: Exactly one player mixes.**
Let Player 1 mix ($p_1 \in (0,1)$), and Players 2 and 3 play pure strategies ($p_2, p_3 \in \{0,1\}$).
The equilibrium conditions are:
1.  Player 1 mixes $\implies p_2 + p_3 = 1$.
2.  Player 2 plays pure $\implies$ $p_1+p_3 \le 1$ if $p_2=1$, or $p_1+p_3 \ge 1$ if $p_2=0$.
3.  Player 3 plays pure $\implies$ $p_1+p_2 \le 1$ if $p_3=1$, or $p_1+p_2 \ge 1$ if $p_3=0$.

From condition (1), since $p_2, p_3$ are pure strategies, one must be 1 and the other must be 0.
*   **Subcase 3a: $p_2=1$ and $p_3=0$.**
    We check the conditions for Players 2 and 3.
    -   For Player 2 ($p_2=1$): We need $p_1+p_3 \le 1 \implies p_1+0 \le 1 \implies p_1 \le 1$. This is true for $p_1 \in (0,1)$.
    -   For Player 3 ($p_3=0$): We need $p_1+p_2 \ge 1 \implies p_1+1 \ge 1 \implies p_1 \ge 0$. This is true for $p_1 \in (0,1)$.
    Since all conditions are met, any profile $(p, 1, 0)$ for $p \in (0,1)$ is a MNE.

*   **Subcase 3b: $p_2=0$ and $p_3=1$.**
    We check the conditions for Players 2 and 3.
    -   For Player 2 ($p_2=0$): We need $p_1+p_3 \ge 1 \implies p_1+1 \ge 1 \implies p_1 \ge 0$. This is true for $p_1 \in (0,1)$.
    -   For Player 3 ($p_3=1$): We need $p_1+p_2 \le 1 \implies p_1+0 \le 1 \implies p_1 \le 1$. This is true for $p_1 \in (0,1)$.
    Since all conditions are met, any profile $(p, 0, 1)$ for $p \in (0,1)$ is a MNE.

By symmetry, we can have any of the three players be the one who mixes. For each choice of mixing player, the other two players must choose opposite rooms. This gives $3 \times 2 = 6$ such families of MNEs.
The six families of MNEs with exactly one mixing player are:
1.  $(p, 1, 0)$ for $p \in (0,1)$
2.  $(p, 0, 1)$ for $p \in (0,1)$
3.  $(1, p, 0)$ for $p \in (0,1)$
4.  $(0, p, 1)$ for $p \in (0,1)$
5.  $(1, 0, p)$ for $p \in (0,1)$
6.  $(0, 1, p)$ for $p \in (0,1)$
