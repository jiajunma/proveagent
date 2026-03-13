# IMO Problem Solution

*Generated: 2026-03-04 10:52:07*

---


## Problem Statement

*** Problem Statement ***
Signature of group automorphisms on extensions

# Setup 

Let $X$ be a finite set and $\sigma$ a permutation of $X$. The signature $\mathrm{sgn}(\sigma) \in \{\pm 1\}$ is defined as usual.

Let $p$ be an odd prime. Let $H = \langle g \rangle$ be a cyclic group. Let $J$ be a finite $p$-group equipped with an action of $H$ by automorphisms (i.e., $\sigma_g \in \mathrm{Aut}(J)$). Let $U \trianglelefteq J$ be a normal subgroup that is stable under the $H$-action.

Show that
\[
\mathrm{sgn}(g|_J) = \mathrm{sgn}(g|_{J/U}) \cdot \mathrm{sgn}(g|_U),
\]
where $g|_J$, $g|_U$, and $g|_{J/U}$ denote the permutations induced by $g$ on the underlying sets of $J$, $U$, and the coset space $J/U$, respectively.




## Solution

### 1. Summary

**a. Verdict**

I have successfully solved the problem. The final answer is that the identity
\[
\mathrm{sgn}(g|_J) = \mathrm{sgn}(g|_{J/U}) \cdot \mathrm{sgn}(g|_U)
\]
holds true under the given conditions.

**b. Method Sketch**

The solution proceeds by analyzing the permutation representation of the automorphism induced by $g$.

1.  **Permutation Representations:** The signature of a permutation on a finite set $X$ is the determinant of the corresponding permutation operator on the vector space $\mathbb{C}[X]$. We let $P_J$, $P_U$, and $P_{J/U}$ be the permutation operators on $\mathbb{C}[J]$, $\mathbb{C}[U]$, and $\mathbb{C}[J/U]$ corresponding to the permutations $g|_J$, $g|_U$, and $g|_{J/U}$, respectively. The problem is equivalent to proving $\det(P_J) = \det(P_{J/U}) \det(P_U)$.

2.  **Coset Decomposition Basis:** We choose a basis for $\mathbb{C}[J]$ that is adapted to the coset decomposition of $J$ with respect to $U$. Let $T$ be a set of right coset representatives for $U$ in $J$. Any element of $J$ can be uniquely written as $tu$ for $t \in T, u \in U$. We use $\{e_{tu} \mid t \in T, u \in U\}$ as a basis for $\mathbb{C}[J]$.

3.  **Block Matrix Structure and Determinant:** In this basis, the matrix $M$ of the operator $P_J$ has a block-permutation structure. The blocks are indexed by elements of $T$. The action of $g$ permutes these blocks according to the permutation $g|_{J/U}$. We prove the following general formula for the determinant of such a matrix:
    **Proposition:** Let $M=(M_{rc})$ be a $k \times k$ block matrix, where each block is $m \times m$. Suppose there is a permutation $\pi \in S_k$ such that the block $M_{rc}$ is zero unless $r=\pi(c)$. Let $A_c = M_{\pi(c), c}$. Then $\det(M) = (\mathrm{sgn}(\pi))^m \prod_{c=1}^k \det(A_c)$.
    Applying this proposition gives $\mathrm{sgn}(g|_J) = \det(M) = \mathrm{sgn}(g|_{J/U})^{|U|} \prod_{t \in T} \det(A_t)$.

4.  **Analysis of Blocks:** The linear map corresponding to each non-zero block $A_t$ is induced by a permutation on $U$ of the form $u \mapsto v_t \cdot \sigma_g(u)$ for some $v_t \in U$, where $\sigma_g$ is the automorphism on $J$ induced by $g$. The determinant is $\det(A_t) = \mathrm{sgn}(L_{v_t}) \cdot \mathrm{sgn}(g|_U)$, where $L_{v_t}$ is the left-translation permutation on $U$ by $v_t$.

5.  **Key Lemma on Left Translations:** The crucial step is proving the following lemma:
    **Lemma:** Let $G$ be a finite group of odd order. For any element $h \in G$, the signature of the left-translation permutation $L_h: x \mapsto hx$ is $\mathrm{sgn}(L_h) = 1$.
    Since $J$ is a $p$-group with $p$ an odd prime, its subgroup $U$ also has odd order. The lemma implies $\mathrm{sgn}(L_{v_t}) = 1$ for all $t \in T$.

6.  **Conclusion:** Using the lemma, we find $\det(A_t) = \mathrm{sgn}(g|_U)$ for all $t \in T$. Substituting this into the determinant formula gives $\mathrm{sgn}(g|_J) = \mathrm{sgn}(g|_{J/U})^{|U|} (\mathrm{sgn}(g|_U))^{|J/U|}$. Since $J$ is a $p$-group with $p$ odd, the orders $|U|$ and $|J/U|$ are powers of $p$ and thus are odd. This allows us to remove the exponents, yielding the desired identity.

### 2. Detailed Solution

Let $\sigma_g \in \mathrm{Aut}(J)$ be the automorphism corresponding to the action of $g \in H$. The permutations in question are $g|_J: j \mapsto \sigma_g(j)$ for $j \in J$, $g|_U: u \mapsto \sigma_g(u)$ for $u \in U$ (since $U$ is $H$-stable), and $g|_{J/U}: jU \mapsto \sigma_g(j)U$ for $jU \in J/U$ (since $U$ is normal).

The signature of a permutation $\pi$ on a finite set $X$ is the determinant of the linear operator $P_\pi$ on the complex vector space $\mathbb{C}[X]$ with basis $\{e_x\}_{x \in X}$, where $P_\pi(e_x) = e_{\pi(x)}$. Let $P_J$, $P_U$, and $P_{J/U}$ be the permutation operators on $\mathbb{C}[J]$, $\mathbb{C}[U]$, and $\mathbb{C}[J/U]$ corresponding to $g|_J$, $g|_U$, and $g|_{J/U}$. The identity we want to prove is equivalent to
\[
\det(P_J) = \det(P_{J/U}) \cdot \det(P_U).
\]

First, we establish a key lemma.

**Lemma:** Let $G$ be a finite group of odd order. For any element $h \in G$, the signature of the left-translation permutation $L_h: G \to G$ defined by $L_h(x) = hx$ is $\mathrm{sgn}(L_h) = 1$.

**Proof of Lemma:** The permutation $L_h$ decomposes into disjoint cycles. The cycles are the right cosets of the cyclic subgroup $\langle h \rangle$ in $G$. Let $|h|$ be the order of $h$. There are $|G|/|h|$ such cosets, and each has size $|h|$. Thus, $L_h$ consists of $|G|/|h|$ cycles, each of length $|h|$. The signature of a permutation on a set of size $N$ with $c$ cycles is given by $(-1)^{N-c}$. For $L_h$, we have $N = |G|$ and $c = |G|/|h|$. So,
\[
\mathrm{sgn}(L_h) = (-1)^{|G| - |G|/|h|}.
\]
By hypothesis, $|G|$ is odd. Since $|h|$ must divide $|G|$, $|h|$ is also odd. Consequently, the quotient $|G|/|h|$ is also odd. Therefore, the exponent $|G| - |G|/|h|$ is a difference of two odd integers, which is an even integer. Thus, $\mathrm{sgn}(L_h) = (-1)^{\text{even}} = 1$. This completes the proof of the lemma.

Now, we return to the main proof. Let $T$ be a set of representatives for the right cosets of $U$ in $J$. Then every element $j \in J$ has a unique representation $j = tu$ for some $t \in T$ and $u \in U$. Let $k = |T| = |J/U|$ and $m = |U|$. We choose a basis for $\mathbb{C}[J]$ adapted to this decomposition. Let $T = \{t_1, \dots, t_k\}$ and $U = \{u_1, \dots, u_m\}$. A basis for $\mathbb{C}[J]$ is given by $\{e_{t_i u_j} \mid i \in \{1, \dots, k\}, j \in \{1, \dots, m\}\}$. We order this basis by grouping elements with the same coset representative:
$B = (e_{t_1 u_1}, \dots, e_{t_1 u_m}, \quad e_{t_2 u_1}, \dots, e_{t_2 u_m}, \quad \dots, \quad e_{t_k u_1}, \dots, e_{t_k u_m})$.

The operator $P_J$ acts on a basis vector $e_{t_i u_j}$ as follows:
\[
P_J(e_{t_i u_j}) = e_{\sigma_g(t_i u_j)} = e_{\sigma_g(t_i)\sigma_g(u_j)}.
\]
The permutation $g|_{J/U}$ acts on the set of cosets $\{t_1 U, \dots, t_k U\}$. Let $\pi$ be the permutation on the indices $\{1, \dots, k\}$ such that $\sigma_g(t_i)U = t_{\pi(i)}U$. This means that for each $i \in \{1, \dots, k\}$, there exists a unique $v_i \in U$ such that $\sigma_g(t_i) = t_{\pi(i)}v_i$. Substituting this into the expression for the action of $P_J$:
\[
P_J(e_{t_i u_j}) = e_{t_{\pi(i)}v_i\sigma_g(u_j)}.
\]
Let $W_i = \mathrm{span}\{e_{t_i u_j} \mid j=1, \dots, m\}$. The space $\mathbb{C}[J]$ decomposes as a direct sum $\mathbb{C}[J] = \bigoplus_{i=1}^k W_i$. The action of $P_J$ maps the subspace $W_i$ to the subspace $W_{\pi(i)}$. The matrix of $P_J$ with respect to the basis $B$ is a block matrix $M = (M_{rc})$, where $M_{rc}$ is an $m \times m$ block representing the map from $W_c$ to $W_r$. From the above, $M_{rc}$ is the zero matrix unless $r = \pi(c)$. Let $A_c = M_{\pi(c), c}$ be the non-zero block in the $c$-th column of blocks.

We now prove a general result for the determinant of such a matrix.

**Proposition:** Let $M=(M_{rc})$ be a $k \times k$ block matrix, where each block is $m \times m$. Suppose there is a permutation $\pi \in S_k$ such that the block $M_{rc}$ is zero unless $r=\pi(c)$. Let $A_c = M_{\pi(c), c}$. Then $\det(M) = (\mathrm{sgn}(\pi))^m \prod_{c=1}^k \det(A_c)$.

**Proof of Proposition:** Let the basis of the underlying vector space be ordered in $k$ blocks of $m$ vectors each. Let $\tau$ be the permutation of the basis indices that maps the basis vector at position $(c, j)$ (j-th vector in c-th block) to position $(\pi(c), j)$. Let $P_\tau$ be the corresponding permutation matrix. The determinant of $P_\tau$ is $\mathrm{sgn}(\tau)$. The permutation $\tau$ is a composition of $m$ permutations, one for each $j \in \{1, \dots, m\}$, each of which permutes the basis vectors $\{e_{(c,j)}\}_{c=1}^k$ according to $\pi$. Thus, $\tau$ is a product of $m$ copies of $\pi$, acting on disjoint sets of basis vectors. The number of cycles in $\tau$ is $c_\tau = m \cdot c_\pi$, where $c_\pi$ is the number of cycles in $\pi$. The signature is $\mathrm{sgn}(\tau) = (-1)^{km - c_\tau} = (-1)^{m(k-c_\pi)} = ((-1)^{k-c_\pi})^m = (\mathrm{sgn}(\pi))^m$.

Consider the matrix product $M' = M P_\tau$. The columns of $M'$ are a permutation of the columns of $M$. Specifically, the block-column $c$ of $M'$ is the block-column $\pi^{-1}(c)$ of $M$. The only non-zero block in block-column $\pi^{-1}(c)$ of $M$ is $A_{\pi^{-1}(c)}$, located at row $\pi(\pi^{-1}(c))=c$. Thus, $M'$ is a block-diagonal matrix: $M' = \mathrm{diag}(A_{\pi^{-1}(1)}, \dots, A_{\pi^{-1}(k)})$.
The determinant of $M'$ is $\det(M') = \prod_{c=1}^k \det(A_{\pi^{-1}(c)}) = \prod_{i=1}^k \det(A_i)$.
Also, $\det(M') = \det(M) \det(P_\tau) = \det(M) (\mathrm{sgn}(\pi))^m$.
Equating the two expressions for $\det(M')$, we get $\det(M) (\mathrm{sgn}(\pi))^m = \prod_{i=1}^k \det(A_i)$.
Since $(\mathrm{sgn}(\pi))^m \in \{\pm 1\}$, we can multiply by it to get $\det(M) = (\mathrm{sgn}(\pi))^m \prod_{i=1}^k \det(A_i)$. This proves the proposition.

Applying the proposition to the matrix $M$ of $P_J$, we have $\pi$ as the permutation on $\{1, \dots, k\}$ induced by $g|_{J/U}$, so $\mathrm{sgn}(\pi) = \mathrm{sgn}(g|_{J/U})$. The block size is $m=|U|$. Thus,
\[
\det(P_J) = \mathrm{sgn}(g|_{J/U})^{|U|} \prod_{i=1}^k \det(A_i).
\]
Now we analyze the determinants of the blocks $A_i$. The matrix $A_i$ represents the map from $W_i$ to $W_{\pi(i)}$ given by $e_{t_i u_j} \mapsto e_{t_{\pi(i)}v_i\sigma_g(u_j)}$. We can identify $W_i$ with $\mathbb{C}[U]$ via the isomorphism $\phi_i(e_u) = e_{t_i u}$. The map represented by $A_i$ corresponds to the map $L_i = \phi_{\pi(i)}^{-1} \circ P_J \circ \phi_i: \mathbb{C}[U] \to \mathbb{C}[U]$. For any $u \in U$,
\[
L_i(e_u) = \phi_{\pi(i)}^{-1}(P_J(e_{t_i u})) = \phi_{\pi(i)}^{-1}(e_{t_{\pi(i)}v_i\sigma_g(u)}) = e_{v_i\sigma_g(u)}.
\]
This shows that $L_i$ is the permutation operator on $\mathbb{C}[U]$ corresponding to the permutation $\tau_i: u \mapsto v_i\sigma_g(u)$ on $U$. This permutation is the composition of the automorphism $g|_U: u \mapsto \sigma_g(u)$ and the left translation $L_{v_i}: u \mapsto v_i u$. So, $\tau_i = L_{v_i} \circ (g|_U)$. The determinant of its operator is the signature of the permutation:
\[
\det(A_i) = \det(L_i) = \mathrm{sgn}(\tau_i) = \mathrm{sgn}(L_{v_i} \circ (g|_U)) = \mathrm{sgn}(L_{v_i}) \mathrm{sgn}(g|_U).
\]
The group $J$ is a finite $p$-group with $p$ odd, so its order is a power of $p$. The subgroup $U$ is also a $p$-group, so its order $|U|$ is a power of $p$. Since $p$ is odd, $|U|$ is odd. By our lemma, the signature of the left translation by any element $v_i \in U$ is 1. Thus, $\mathrm{sgn}(L_{v_i}) = 1$.
This simplifies $\det(A_i)$ to $\det(A_i) = \mathrm{sgn}(g|_U)$.

Now we can compute the product of the determinants of the blocks:
\[
\prod_{i=1}^k \det(A_i) = \prod_{i=1}^k \mathrm{sgn}(g|_U) = (\mathrm{sgn}(g|_U))^k.
\]
Here $k = |T| = |J/U|$. Putting everything together:
\[
\mathrm{sgn}(g|_J) = \det(P_J) = \mathrm{sgn}(g|_{J/U})^{|U|} (\mathrm{sgn}(g|_U))^{|J/U|}.
\]
Since $J$ is a $p$-group with $p$ odd, $|U|$ and $|J/U|$ are powers of $p$. As $p$ is an odd prime, $|U|$ and $|J/U|$ must be odd integers (a power of an odd prime is always odd). For any integer $n$, if $m$ is odd, then $n^m=n$ if $n \in \{1, -1\}$. Therefore:
\begin{itemize}
    \item $\mathrm{sgn}(g|_{J/U})^{|U|} = \mathrm{sgn}(g|_{J/U})$ since $|U|$ is odd.
    \item $(\mathrm{sgn}(g|_U))^{|J/U|} = \mathrm{sgn}(g|_U)$ since $|J/U|$ is odd.
\end{itemize}
Substituting these into our equation gives:
\[
\mathrm{sgn}(g|_J) = \mathrm{sgn}(g|_{J/U}) \cdot \mathrm{sgn}(g|_U).
\]
This completes the proof.



## Verification Report

### Summary ###

**Final Verdict:** The solution's approach is viable but contains a Justification Gap.

**List of Findings:**
*   **Location:** "Proof of Proposition: ... Consider the matrix product $M' = M P_\tau$. ... Specifically, the block-column $c$ of $M'$ is the block-column $\pi^{-1}(c)$ of $M$. ... Thus, $M'$ is a block-diagonal matrix..."
    *   **Issue:** Justification Gap - The proof provided for the proposition about the determinant of a block matrix is flawed. The argument for how the matrix $M'$ is constructed and why it is block-diagonal is incorrect. While the proposition's conclusion is correct, the justification provided is not rigorous.

### Detailed Verification Log ###

**Initial Setup**
> Let $\sigma_g \in \mathrm{Aut}(J)$ be the automorphism corresponding to the action of $g \in H$. The permutations in question are $g|_J: j \mapsto \sigma_g(j)$ for $j \in J$, $g|_U: u \mapsto \sigma_g(u)$ for $u \in U$ (since $U$ is $H$-stable), and $g|_{J/U}: jU \mapsto \sigma_g(j)U$ for $jU \in J/U$ (since $U$ is normal).
>
> The signature of a permutation $\pi$ on a finite set $X$ is the determinant of the linear operator $P_\pi$ on the complex vector space $\mathbb{C}[X]$ with basis $\{e_x\}_{x \in X}$, where $P_\pi(e_x) = e_{\pi(x)}$. Let $P_J$, $P_U$, and $P_{J/U}$ be the permutation operators on $\mathbb{C}[J]$, $\mathbb{C}[U]$, and $\mathbb{C}[J/U]$ corresponding to $g|_J$, $g|_U$, and $g|_{J/U}$. The identity we want to prove is equivalent to
> \[
> \det(P_J) = \det(P_{J/U}) \cdot \det(P_U).
> \]

*   **Assessment:** Correct.
*   **Justification:** The definitions of the permutations are correct. The rephrasing of the signature as the determinant of the corresponding permutation operator is a standard and valid technique. The target equation is correctly translated into the language of determinants. Note that the problem statement is about $\mathrm{sgn}(g|_J)$, $\mathrm{sgn}(g|_{J/U})$, and $\mathrm{sgn}(g|_U)$, which correspond to $\det(P_J)$, $\det(P_{g|_{J/U}})$, and $\det(P_U)$. The solution uses $P_{J/U}$ to denote the operator for $g|_{J/U}$, which is clear from context.

**Lemma and its Proof**
> **Lemma:** Let $G$ be a finite group of odd order. For any element $h \in G$, the signature of the left-translation permutation $L_h: G \to G$ defined by $L_h(x) = hx$ is $\mathrm{sgn}(L_h) = 1$.
>
> **Proof of Lemma:** The permutation $L_h$ decomposes into disjoint cycles. The cycles are the right cosets of the cyclic subgroup $\langle h \rangle$ in $G$. ... $\mathrm{sgn}(L_h) = (-1)^{|G| - |G|/|h|}$. By hypothesis, $|G|$ is odd. Since $|h|$ must divide $|G|$, $|h|$ is also odd. Consequently, the quotient $|G|/|h|$ is also odd. Therefore, the exponent $|G| - |G|/|h|$ is a difference of two odd integers, which is an even integer. Thus, $\mathrm{sgn}(L_h) = (-1)^{\text{even}} = 1$.

*   **Assessment:** Correct.
*   **Justification:** The proof is entirely correct. The decomposition of the permutation $L_h$ into cycles corresponding to the right cosets of $\langle h \rangle$ is standard. The formula for the signature based on the number of elements and cycles is correctly applied. The argument based on the parity of the orders of the groups is also sound.

**Main Proof Setup**
> Now, we return to the main proof. Let $T$ be a set of representatives for the right cosets of $U$ in $J$. Then every element $j \in J$ has a unique representation $j = tu$ for some $t \in T$ and $u \in U$. ... The operator $P_J$ acts on a basis vector $e_{t_i u_j}$ as follows:
> \[
> P_J(e_{t_i u_j}) = e_{\sigma_g(t_i u_j)} = e_{\sigma_g(t_i)\sigma_g(u_j)}.
> \]
> The permutation $g|_{J/U}$ acts on the set of cosets $\{t_1 U, \dots, t_k U\}$. Let $\pi$ be the permutation on the indices $\{1, \dots, k\}$ such that $\sigma_g(t_i)U = t_{\pi(i)}U$. This means that for each $i \in \{1, \dots, k\}$, there exists a unique $v_i \in U$ such that $\sigma_g(t_i) = t_{\pi(i)}v_i$. ... The matrix of $P_J$ with respect to the basis $B$ is a block matrix $M = (M_{rc})$, ... $M_{rc}$ is the zero matrix unless $r = \pi(c)$.

*   **Assessment:** Correct.
*   **Justification:** The setup is logically sound. The choice of basis adapted to the coset decomposition of $J$ is appropriate. The action of the permutation operator $P_J$ is correctly described. The matrix of $P_J$ in this basis is indeed a block matrix where the non-zero blocks are determined by the permutation $\pi$ on the cosets. There is a minor imprecision: if $T$ is a set of representatives for the *right* cosets $U \backslash J$, then the unique representation is $j=ut$. If $T$ represents the *left* cosets $J/U$, the representation is $j=tu$. Since $U$ is normal, left and right cosets coincide, so this does not affect the argument.

**Proposition and its Proof**
> **Proposition:** Let $M=(M_{rc})$ be a $k \times k$ block matrix, where each block is $m \times m$. Suppose there is a permutation $\pi \in S_k$ such that the block $M_{rc}$ is zero unless $r=\pi(c)$. Let $A_c = M_{\pi(c), c}$. Then $\det(M) = (\mathrm{sgn}(\pi))^m \prod_{c=1}^k \det(A_c)$.
>
> **Proof of Proposition:** ... Let $\tau$ be the permutation of the basis indices that maps the basis vector at position $(c, j)$ ... to position $(\pi(c), j)$. ... $\mathrm{sgn}(\tau) = (\mathrm{sgn}(\pi))^m$.
> Consider the matrix product $M' = M P_\tau$. The columns of $M'$ are a permutation of the columns of $M$. Specifically, the block-column $c$ of $M'$ is the block-column $\pi^{-1}(c)$ of $M$. ... Thus, $M'$ is a block-diagonal matrix: $M' = \mathrm{diag}(A_{\pi^{-1}(1)}, \dots, A_{\pi^{-1}(k)})$.
> ... $\det(M') = \det(M) \det(P_\tau) = \det(M) (\mathrm{sgn}(\pi))^m$.
> Equating the two expressions for $\det(M')$, we get $\det(M) = (\mathrm{sgn}(\pi))^m \prod_{i=1}^k \det(A_i)$.

*   **Assessment:** Justification Gap.
*   **Justification:** The statement of the proposition is correct. However, its proof is flawed.
    1.  The matrix $P_\tau$ corresponds to the permutation $\tau$ which maps the basis vector $e_{(c,j)}$ to $e_{(\pi(c),j)}$.
    2.  The solution claims that in the product $M' = M P_\tau$, "the block-column $c$ of $M'$ is the block-column $\pi^{-1}(c)$ of $M$". This is incorrect. Multiplying $M$ on the right by a permutation matrix $P_\tau$ permutes the columns of $M$ according to $\tau^{-1}$, not its block-columns according to $\pi^{-1}$. The permutation $\tau$ is not a simple permutation of blocks.
    3.  A correct way to prove the proposition is to consider the matrix product $M P_\tau^{-1}$ (or $P_\tau M$, depending on the definition of the permutation matrix). For instance, $M P_\tau^{-1}$ can be shown to be a block-diagonal matrix (up to a permutation of the diagonal blocks), from which the determinant formula follows.
    4.  The argument presented is not rigorous and relies on an unsubstantiated, incorrect claim about the structure of $M P_\tau$. This constitutes a significant gap in the justification.
*   **Continuation:** The proposition's statement is a known result. For the purpose of verifying the rest of the solution, I will assume the proposition is true.

**Application of the Proposition and Analysis of Blocks**
> Applying the proposition to the matrix $M$ of $P_J$, we have ...
> \[
> \det(P_J) = \mathrm{sgn}(g|_{J/U})^{|U|} \prod_{i=1}^k \det(A_i).
> \]
> Now we analyze the determinants of the blocks $A_i$. ... The map represented by $A_i$ corresponds to the map $L_i = \phi_{\pi(i)}^{-1} \circ P_J \circ \phi_i: \mathbb{C}[U] \to \mathbb{C}[U]$. For any $u \in U$,
> \[
> L_i(e_u) = \phi_{\pi(i)}^{-1}(P_J(e_{t_i u})) = \phi_{\pi(i)}^{-1}(e_{t_{\pi(i)}v_i\sigma_g(u)}) = e_{v_i\sigma_g(u)}.
> \]
> This shows that $L_i$ is the permutation operator on $\mathbb{C}[U]$ corresponding to the permutation $\tau_i: u \mapsto v_i\sigma_g(u)$ on $U$. ... $\tau_i = L_{v_i} \circ (g|_U)$. ...
> \[
> \det(A_i) = \det(L_i) = \mathrm{sgn}(\tau_i) = \mathrm{sgn}(L_{v_i} \circ (g|_U)) = \mathrm{sgn}(L_{v_i}) \mathrm{sgn}(g|_U).
> \]

*   **Assessment:** Correct.
*   **Justification:** Assuming the proposition is true, its application here is correct. The identification of the block matrices $A_i$ with operators on $\mathbb{C}[U]$ is carried out correctly. The calculation of the action of this operator $L_i$ is correct, and its decomposition into a composition of left translation and the automorphism $g|_U$ is also correct. The final step using the multiplicative property of the signature is valid.

**Final Calculation**
> The group $J$ is a finite $p$-group with $p$ odd, so ... $|U|$ is odd. By our lemma, the signature of the left translation by any element $v_i \in U$ is 1. Thus, $\mathrm{sgn}(L_{v_i}) = 1$.
> This simplifies $\det(A_i)$ to $\det(A_i) = \mathrm{sgn}(g|_U)$.
> Now we can compute the product of the determinants of the blocks:
> \[
> \prod_{i=1}^k \det(A_i) = \prod_{i=1}^k \mathrm{sgn}(g|_U) = (\mathrm{sgn}(g|_U))^k.
> \]
> Here $k = |T| = |J/U|$. Putting everything together:
> \[
> \mathrm{sgn}(g|_J) = \det(P_J) = \mathrm{sgn}(g|_{J/U})^{|U|} (\mathrm{sgn}(g|_U))^{|J/U|}.
> \]
> Since $J$ is a $p$-group with $p$ odd, $|U|$ and $|J/U|$ are powers of $p$. As $p$ is an odd prime, $|U|$ and $|J/U|$ must be odd integers... Therefore:
> \begin{itemize}
>     \item $\mathrm{sgn}(g|_{J/U})^{|U|} = \mathrm{sgn}(g|_{J/U})$ since $|U|$ is odd.
>     \item $(\mathrm{sgn}(g|_U))^{|J/U|} = \mathrm{sgn}(g|_U)$ since $|J/U|$ is odd.
> \end{itemize}
> Substituting these into our equation gives:
> \[
> \mathrm{sgn}(g|_J) = \mathrm{sgn}(g|_{J/U}) \cdot \mathrm{sgn}(g|_U).
> \]

*   **Assessment:** Correct.
*   **Justification:** The application of the lemma is correct because $|U|$ is a power of an odd prime $p$, hence odd. The subsequent calculation of the product $\prod \det(A_i)$ is correct. The final formula for $\mathrm{sgn}(g|_J)$ is correctly assembled. The concluding argument that the odd exponents $|U|$ and $|J/U|$ do not change the value of the signatures (which are $\pm 1$) is correct and leads to the desired result.

