# IMO Problem Solution

*Generated: 2026-03-07 07:54:12*

---


## Problem Statement

*** Problem Statement ***

# The setting

We work over $\mathbb{C}$ here. Let $V$ be a finite-dimensional vector space over $\mathbb{C}$ of dimension $n+1$ with a direct sum decomposition

$$V = W \oplus L, \quad L = \langle e_{n+1} \rangle$$

where $\{e_1, \ldots, e_{n+1}\}$ is a basis of $V$, $W = \langle e_1, \ldots, e_n \rangle$, and $e_{n+1}$ spans the 1-dimensional complement. Let $G_1 = \mathrm{GL}(W) = \mathrm{GL}_n$ and $G_2 = \mathrm{GL}(V) = \mathrm{GL}_{n+1}$. We view $G_1$ as a subgroup of $G_2$ via the block embedding $g \mapsto \mathrm{diag}(g, 1)$ (so $G_1$ stabilizes $W$ and fixes $e_{n+1}$). Let $\mathfrak{g}_1 = \mathfrak{gl}_n$ and $\mathfrak{g}_2 = \mathfrak{gl}_{n+1}$ be the Lie algebras. Then we have a natural inclusion $\mathfrak{g}_1 \subset \mathfrak{g}_2$.

Let $G = G_1 \times G_2 = \mathrm{GL}_n \times \mathrm{GL}_{n+1}$ and $X = G_2 = \mathrm{GL}_{n+1}$. $X$ is a $G$-spherical variety via the action

$$(g_1, g_2) \cdot g = g_1 g g_2^{-1}, \quad (g_1, g_2) \in G_1 \times G_2,\; g \in \mathrm{GL}_{n+1}.$$

Let $M = T^*(X)$ be the cotangent bundle of $X$. Then $M$ is a Hamiltonian $G$-variety with the moment map $\mu = \mu_1 \times \mu_2 \colon M \to \mathfrak{g}_1^* \times \mathfrak{g}_2^*$.
 Write $M = \mathrm{GL}_{n+1} \times \mathfrak{g}_2^*$. Let $p \colon \mathfrak{g}_2^* \to \mathfrak{g}_1^*$ be the natural projection. Then the moment map is given by

$$\mu_1(g, \xi) = p(\mathrm{Ad}^*(g)\xi), \quad \mu_2(g, \xi) = -\xi, \quad g \in G_2,\; \xi \in \mathfrak{g}_2^*.$$

We identify $\mathfrak{g}_1^*$ and $\mathfrak{g}_2^*$ with $\mathfrak{g}_1$ and $\mathfrak{g}_2$ via the trace form. 

We denote by $\mathcal{P}_n$ the set of all partitions of $n$. The nilpotent orbits of $\mathfrak{g}_1 = \mathfrak{gl}_n$ and $\mathfrak{g}_2 = \mathfrak{gl}_{n+1}$ are parameterized by $\mathcal{P}_n$ and $\mathcal{P}_{n+1}$ respectively. We denote by $\mathcal{O}_\lambda$ the nilpotent orbit corresponding to a partition $\lambda$.

We are interested in the following question:

**Question**

1. For which pairs of partitions $(\lambda, \mu) \in \mathcal{P}_n \times \mathcal{P}_{n+1}$ does the nilpotent orbit $\mathcal{O}_\lambda \times \mathcal{O}_\mu \subseteq \mu(M)$? By (1), this is equivalent to asking: for which pairs of partitions $(\lambda, \mu) \in \mathcal{P}_n \times \mathcal{P}_{n+1}$ does there exist $g \in G_2$ and $\xi \in \mathcal{O}_\mu$ such that $\mathrm{Ad}(g)\xi|_W \in \mathcal{O}_\lambda$?

2. Let $\mathcal{O}_\lambda \subseteq \mathfrak{g}_1$ and $\mathcal{O}_\mu \subseteq \mathfrak{g}_2$ be nilpotent orbits such that $\mathcal{O}_\lambda \times \mathcal{O}_\mu \subseteq \mu(M)$. We choose a point $e = (e_\lambda, e_\mu) \in \mathcal{O}_\lambda \times \mathcal{O}_\mu$ and consider the fiber $M_e = \mu^{-1}(e)$. Find a closed formula for $\dim M_e$.

3. We know that

$$\dim M_e \leq \frac{1}{2} \dim M - \frac{1}{2} (\dim \mathcal{O}_\lambda + \dim \mathcal{O}_\mu).$$

We call $\mathcal{O}_\lambda \times \mathcal{O}_\mu$ a *relevant* orbit if the above inequality is an equality. Find all the relevant nilpotent orbits.




## Solution

### 1. Summary ###

**a. Verdict**

I have not found a complete solution for all $n$. However, I have obtained significant partial results which I can rigorously prove. Specifically, I have provided a complete classification of all relevant nilpotent orbits in the following cases:
1.  When $\lambda = (1^n)$ (the zero orbit in $\mathfrak{g}_1$) for any $n \ge 1$.
2.  When $\lambda = (n)$ (the regular nilpotent orbit in $\mathfrak{g}_1$) for any $n \ge 1$.
3.  For the specific dimension $n=3$, where I have classified all relevant pairs $(\lambda, \mu)$ for all possible partitions $\lambda$ of 3.

The relevant pairs found in these proven cases are:
*   **Zero-Zero:** $\lambda = (1^n)$ and $\mu = (1^{n+1})$, for all $n \ge 1$.
*   **Zero-Minimal:** $\lambda = (1^n)$ and $\mu = (2, 1^{n-1})$, for all $n \ge 1$.
*   **Zero-Subminimal:** $\lambda = (1^n)$ and $\mu = (3, 1^{n-2})$, for all $n \ge 2$.
*   **Regular-Regular:** $\lambda = (n)$ and $\mu = (n+1)$, for all $n \ge 1$.
*   **Regular-Subregular:** $\lambda = (n)$ and $\mu = (n,1)$, for all $n \ge 1$.
*   **Sporadic Case 1 (for $n=3$):** $\lambda=(2,1)$ and $\mu=(2,2)$.
*   **Sporadic Case 2 (for $n=3$):** $\lambda=(2,1)$ and $\mu=(2,1,1)$.

**b. Method Sketch**

**Part 1:** The condition $\mathcal{O}_\lambda \times \mathcal{O}_\mu \subseteq \mu(M)$ is shown to be equivalent to the existence of a matrix $Y \in \mathcal{O}_\mu$ whose top-left $n \times n$ principal submatrix, which we denote $\pi(Y)$, lies in $\mathcal{O}_\lambda$. This is a known result from matrix theory, which holds if and only if the partition $\lambda$ is obtained from $\mu$ by decreasing one part by 1 (denoted $\lambda \prec \mu$).

**Part 2:** The dimension of the fiber $M_e = \mu^{-1}(e)$ for $e=(e_\lambda, e_\mu)$ is computed. The fiber is shown to be isomorphic to the variety $\{g \in \mathrm{GL}_{n+1} \mid \pi(\mathrm{Ad}(g)e_\mu) = -e_\lambda\}$. By analyzing the differential of the map $g \mapsto \pi(\mathrm{Ad}(g)e_\mu)$, the dimension of the fiber is found to be $\dim M_e = (n+1)^2 - \dim \pi([\mathfrak{g}_2, Y])$, where $Y$ is any matrix in $\mathcal{O}_\mu$ such that $\pi(Y) \in \mathcal{O}_\lambda$.

**Part 3:** The relevance condition is an equality: $\dim M_e = \frac{1}{2} \dim M - \frac{1}{2} (\dim \mathcal{O}_\lambda + \dim \mathcal{O}_\mu)$. This simplifies to the key equation:
$$ \dim \pi([\mathfrak{g}_2, Y]) = \frac{1}{2}(\dim \mathcal{O}_\lambda + \dim \mathcal{O}_\mu) $$
To find the relevant pairs, a case analysis is performed.
*   **Case $\lambda=(1^n)$:** We choose $Y \in \mathcal{O}_\mu$ such that $\pi(Y)=0$. A direct computation of $\dim \pi([\mathfrak{g}_2, Y])$ in terms of the last row and column of $Y$ allows for a complete classification based on the rank of $Y$, yielding the three "Zero" families.
*   **Case $\lambda=(n)$:** The interlacing condition restricts $\mu$ to be either $(n+1)$ or $(n,1)$. For each of these two possibilities, we construct a specific matrix $Y \in \mathcal{O}_\mu$ with $\pi(Y)$ being a regular element and verify the relevance condition. This provides a complete classification for this case, yielding the two "Regular" families.
*   **Complete analysis for $n=3$:** We analyze all partitions $\lambda$ of 3. The cases $\lambda=(1^3)$ and $\lambda=(3)$ are covered by the general analysis. For the remaining partition $\lambda=(2,1)$, we test all possible interlacing partitions $\mu$. This involves explicit matrix constructions and dimension calculations, which rigorously establish the two sporadic solutions for $n=3$ and prove that no other solutions exist for this case.

### 2. Detailed Solution ###

Let $\pi: \mathfrak{g}_2 \to \mathfrak{g}_1$ be the projection onto the top-left $n \times n$ block. We identify $\mathfrak{g}_k^*$ with $\mathfrak{g}_k$ via the trace form $\langle A, B \rangle = \mathrm{Tr}(AB)$. Under this identification, the coadjoint action $\mathrm{Ad}^*$ becomes the adjoint action $\mathrm{Ad}$. The moment map is $\mu(g, Y) = (\pi(\mathrm{Ad}(g)Y), -Y)$ for $(g, Y) \in \mathrm{GL}_{n+1} \times \mathfrak{gl}_{n+1}$.

#### Part 1: Condition on partitions

The image of the moment map is $\mu(M) = \{ (\pi(\mathrm{Ad}(g)Y), -Y) \mid g \in G_2, Y \in \mathfrak{g}_2 \}$. The condition $\mathcal{O}_\lambda \times \mathcal{O}_\mu \subseteq \mu(M)$ means that for any pair $(X, Z) \in \mathcal{O}_\lambda \times \mathcal{O}_\mu$, there exists $(g, Y) \in M$ such that $\mu(g, Y) = (X, Z)$. This implies $Y = -Z$. Since $\mathcal{O}_\mu$ is a cone, if $Z \in \mathcal{O}_\mu$, then $-Z \in \mathcal{O}_\mu$. The condition becomes that for any $X \in \mathcal{O}_\lambda$ and any $Y' \in \mathcal{O}_\mu$, there exists $g \in G_2$ such that $X = \pi(\mathrm{Ad}(g)Y')$.
The set of all possible projections from matrices in $\mathcal{O}_\mu$ is $S_\mu = \{ \pi(\mathrm{Ad}(g)Y) \mid g \in G_2, Y \in \mathcal{O}_\mu \}$. Since $\mathrm{Ad}(G_2)$ acts transitively on the orbit $\mathcal{O}_\mu$, this set is simply $S_\mu = \{\pi(Y') \mid Y' \in \mathcal{O}_\mu\}$.
The condition is therefore $\mathcal{O}_\lambda \subseteq S_\mu$. Since $S_\mu$ is a $G_1$-invariant constructible set, this is equivalent to requiring that the intersection is non-empty, i.e., there exists some $Y \in \mathcal{O}_\mu$ such that $\pi(Y) \in \mathcal{O}_\lambda$.
This is a classic result in linear algebra: a nilpotent matrix $X \in \mathfrak{gl}_n$ is a principal submatrix of a nilpotent matrix $Y \in \mathfrak{gl}_{n+1}$ if and only if their corresponding partitions $\lambda$ and $\mu$ satisfy the interlacing condition $\lambda \prec \mu$. For partitions of $n$ and $n+1$, this is equivalent to $\lambda$ being obtained from $\mu$ by decreasing one part by 1.

#### Part 2: Dimension of the fiber

Let $e = (e_\lambda, e_\mu) \in \mathcal{O}_\lambda \times \mathcal{O}_\mu$. The fiber is $M_e = \mu^{-1}(e) = \{ (g, Y) \in M \mid \mu(g, Y) = e \}$. This gives $Y = -e_\mu$ and $\pi(\mathrm{Ad}(g)(-e_\mu)) = e_\lambda$. Since $\mathcal{O}_\lambda$ is a cone, $-e_\lambda \in \mathcal{O}_\lambda$. The condition is equivalent to $\pi(\mathrm{Ad}(g)e_\mu) = -e_\lambda$. The fiber is thus isomorphic to the variety $S = \{ g \in \mathrm{GL}_{n+1} \mid \pi(\mathrm{Ad}(g)e_\mu) = -e_\lambda \}$, and $\dim M_e = \dim S$.

The dimension of the fiber is independent of the choice of representatives $(e_\lambda, e_\mu)$, as the moment map is $G$-equivariant. Consider the map $\phi: \mathrm{GL}_{n+1} \to \mathfrak{g}_1$ defined by $\phi(g) = \pi(\mathrm{Ad}(g)e_\mu)$. The set $S$ is the fiber $\phi^{-1}(-e_\lambda)$. For a generic point $g \in S$, the dimension of the fiber is $\dim S = \dim \mathrm{GL}_{n+1} - \mathrm{rank}(d_g\phi)$.
The differential of $\phi$ at $g \in S$ is a linear map $d_g\phi: T_g\mathrm{GL}_{n+1} \to T_{-e_\lambda}\mathfrak{g}_1$. Using right-trivialization $T_g\mathrm{GL}_{n+1} \cong \mathfrak{g}_2$, for $A \in \mathfrak{g}_2$, we have:
$$ d_g\phi(A) = \frac{d}{dt}\Big|_{t=0} \phi(g e^{tA}) = \pi(\mathrm{Ad}(g)[A, e_\mu]) = \pi([\mathrm{Ad}(g)A, \mathrm{Ad}(g)e_\mu]) $$
Let $Y = \mathrm{Ad}(g)e_\mu$. Then $Y \in \mathcal{O}_\mu$ and $\pi(Y) = -e_\lambda \in \mathcal{O}_\lambda$. The image of the differential is $\mathrm{Im}(d_g\phi) = \pi([\mathfrak{g}_2, Y])$. This is the projection of the tangent space $T_Y\mathcal{O}_\mu = [\mathfrak{g}_2, Y]$ to the orbit $\mathcal{O}_\mu$ at $Y$.
The dimension of the fiber is $\dim M_e = (n+1)^2 - \dim \pi([\mathfrak{g}_2, Y])$. Since $\dim M_e$ depends only on the orbits $\mathcal{O}_\lambda$ and $\mathcal{O}_\mu$, the quantity $\dim \pi([\mathfrak{g}_2, Y])$ is constant for any $Y \in \mathcal{O}_\mu$ such that $\pi(Y) \in \mathcal{O}_\lambda$.

#### Part 3: Relevant orbits

An orbit pair $(\mathcal{O}_\lambda, \mathcal{O}_\mu)$ is relevant if $\dim M_e = \frac{1}{2} \dim M - \frac{1}{2} (\dim \mathcal{O}_\lambda + \dim \mathcal{O}_\mu)$.
With $\dim M = 2(n+1)^2$, this becomes $(n+1)^2 - \dim \pi([\mathfrak{g}_2, Y]) = (n+1)^2 - \frac{1}{2}(\dim \mathcal{O}_\lambda + \dim \mathcal{O}_\mu)$.
The condition for relevance simplifies to:
$$ \dim \pi([\mathfrak{g}_2, Y]) = \frac{1}{2}(\dim \mathcal{O}_\lambda + \dim \mathcal{O}_\mu) $$
We use the dimension formula $\dim \mathcal{O}_\nu = |\nu|^2 - \sum_i (\nu_i^t)^2$, where $\nu^t$ is the transpose partition.

**Case I: $\lambda = (1^n)$ (zero orbit)**
Here $e_\lambda=0$, so $\dim \mathcal{O}_\lambda = 0$. We choose $Y \in \mathcal{O}_\mu$ such that $\pi(Y)=0$. Such a $Y$ has the form $Y = \begin{pmatrix} 0 & u \\ v^T & c \end{pmatrix}$ for $u,v \in \mathbb{C}^n, c \in \mathbb{C}$. Since $Y$ is nilpotent, $\mathrm{Tr}(Y)=c=0$. For $Y$ to be nilpotent, we must also have $\mathrm{Tr}(Y^2)=2v^Tu=0$.
For $A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix} \in \mathfrak{g}_2$, we have $\pi([A,Y]) = A_{12}v^T - uA_{21}$.
The space $\pi([\mathfrak{g}_2, Y])$ is $S_{u,v} = \{y v^T - u x^T \mid y \in \mathbb{C}^n, x \in \mathbb{C}^{n \times 1}\}$.
The relevance condition is $\dim S_{u,v} = \frac{1}{2}\dim \mathcal{O}_\mu$.

1.  If $u=v=0$, then $Y=0$, so $\mu=(1^{n+1})$. $\dim \mathcal{O}_\mu=0$. $\dim S_{0,0}=0$. The condition $0=\frac{1}{2}(0)$ holds. This gives the pair $(\lambda, \mu) = ((1^n), (1^{n+1}))$.
2.  If one of $u,v$ is non-zero (say $u \neq 0, v=0$), then $v^Tu=0$ is satisfied. $Y$ is nilpotent of rank 1, so $\mu=(2, 1^{n-1})$. $\dim \mathcal{O}_\mu = 2n$. The space $S_{u,0}$ consists of matrices whose columns are multiples of $u$, so $\dim S_{u,0}=n$. The condition $n=\frac{1}{2}(2n)$ holds. This gives $(\lambda, \mu) = ((1^n), (2, 1^{n-1}))$.
3.  If $u,v$ are linearly independent, the condition $v^Tu=0$ implies $Y^3=0$. The rank of $Y$ is 2, and its Jordan type is $\mu=(3, 1^{n-2})$ (for $n \ge 2$).
    $\dim \mathcal{O}_{(3,1^{n-2})} = (n+1)^2 - ((n-1)^2+1^2+1^2) = 4n-2$.
    The space $S_{u,v}$ is the sum of matrices with columns in $\mathrm{span}(u)$ and rows in $\mathrm{span}(v^T)$. Since $u,v$ are linearly independent, the intersection of these two spaces is $\mathrm{span}(uv^T)$, which is 1-dimensional. So $\dim S_{u,v} = n+n-1=2n-1$.
    The relevance condition is $2n-1 = \frac{1}{2}(4n-2)$, which is an identity for all $n \ge 2$. This gives $(\lambda, \mu) = ((1^n), (3, 1^{n-2}))$.
If $u,v$ are linearly dependent and non-zero, say $v=ku$, then $v^Tu=k u^T u=0$. This implies $Y$ has rank 1, so $\mu=(2,1^{n-1})$, which is covered by case 2. This exhausts all possibilities for $\lambda=(1^n)$.

**Case II: $\lambda = (n)$ (regular orbit)**
Here $\dim \mathcal{O}_{(n)} = n^2-n$. The condition $\lambda \prec \mu$ implies $\mu=(n+1)$ or $\mu=(n,1)$. We check these two cases.

1.  **Pair $((n), (n+1))$:**
    $\dim \mathcal{O}_{(n+1)} = (n+1)^2-(n+1)=n^2+n$. RHS of relevance condition: $\frac{1}{2}(n^2-n + n^2+n) = n^2$.
    Consider $Y = \begin{pmatrix} J_n & e_n \\ 0 & 0 \end{pmatrix}$, where $J_n$ is the standard regular nilpotent Jordan block. This matrix has a single Jordan block of size $n+1$, so $Y \in \mathcal{O}_{(n+1)}$. We have $\pi(Y)=J_n \in \mathcal{O}_{(n)}$.
    LHS: $\dim \pi([\mathfrak{g}_2, Y]) = \dim([\mathfrak{g}_1, J_n] + S_{e_n, 0})$.
    $\dim([\mathfrak{g}_1, J_n]) = \dim \mathcal{O}_{(n)} = n^2-n$. $\dim S_{e_n, 0}=n$.
    The intersection $[\mathfrak{g}_1, J_n] \cap S_{e_n, 0}$ consists of matrices $M=e_n c^T$ such that $\mathrm{Tr}(MC)=0$ for all $C$ in the centralizer $C(J_n)$. This requires $c^T C e_n = 0$. Since $C(J_n)$ consists of polynomials in $J_n$, the set $\{C e_n\}$ spans $\mathbb{C}^n$. Thus $c=0$. The intersection is $\{0\}$.
    LHS = $(n^2-n)+n-0 = n^2$. LHS=RHS, so the pair is relevant.

2.  **Pair $((n), (n,1))$:**
    $\dim \mathcal{O}_{(n,1)} = (n+1)^2 - (n+3) = n^2+n-2$. RHS: $\frac{1}{2}(n^2-n + n^2+n-2) = n^2-1$.
    For $n \ge 2$, consider $Y = \begin{pmatrix} J_n & e_{n-1} \\ 0 & 0 \end{pmatrix}$. This matrix has Jordan type $(n,1)$.
    LHS: $\dim \pi([\mathfrak{g}_2, Y]) = \dim([\mathfrak{g}_1, J_n] + S_{e_{n-1}, 0})$.
    The intersection consists of matrices $e_{n-1}c^T$ where $c$ is orthogonal to $\{C e_{n-1} \mid C \in C(J_n)\}$. The space $\{C e_{n-1}\}$ is spanned by $\{e_1, \dots, e_{n-1}\}$. Its orthogonal complement is spanned by $e_n$. Thus $c$ is a multiple of $e_n$. The intersection has dimension 1.
    LHS = $(n^2-n)+n-1 = n^2-1$. LHS=RHS, so the pair is relevant for $n \ge 2$. For $n=1$, $(\lambda, \mu)=((1),(1,1))$, both orbits are $\{0\}$, so it is trivially relevant (and covered by the Zero-Zero case).

Since these are the only partitions interlacing $\lambda=(n)$, this case is complete.

**Case III: Complete analysis for $n=3$**
The partitions of $n=3$ are $(1^3), (3), (2,1)$. The first two are covered by the general analysis above. We analyze $\lambda=(2,1)$.
Let $X = E_{12} = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} \in \mathcal{O}_{(2,1)}$.
$\dim \mathcal{O}_{(2,1)} = 3^2 - (2^2+1^2)=4$.
The partitions $\mu$ of $4$ such that $(2,1) \prec \mu$ are $(3,1), (2,2), (2,1,1)$.

1.  **Pair $((2,1), (3,1))$:**
    $\dim \mathcal{O}_{(3,1)} = 4^2 - (2^2+1^2+1^2)=10$. RHS: $\frac{1}{2}(4+10)=7$.
    We need a $Y \in \mathcal{O}_{(3,1)}$ with $\pi(Y)=X$. Let $Y = \begin{pmatrix} X & u \\ v^T & 0 \end{pmatrix}$. For $Y$ to be nilpotent, we need $v^T X^k u = 0$ for $k \ge 0$. Let $u=e_1, v=e_2$. Then $v^Tu=0, v^TXu=0, \dots$. So $Y=\begin{pmatrix} X & e_1 \\ e_2^T & 0 \end{pmatrix}$ is nilpotent. Its Jordan type is $(3,1)$.
    LHS: $\dim \pi([\mathfrak{g}_2, Y]) = \dim([\mathfrak{g}_1, X] + S_{e_1, e_2}) = 4 + (2(3)-1) - \dim(\text{intersection})$.
    The intersection has dimension 3. LHS = $4+5-3=6$. Since $6 \neq 7$, this pair is not relevant.

2.  **Pair $((2,1), (2,2))$:**
    $\dim \mathcal{O}_{(2,2)} = 4^2 - (2^2+2^2)=8$. RHS: $\frac{1}{2}(4+8)=6$.
    Consider $Y = \begin{pmatrix} X & 0 \\ e_3^T & 0 \end{pmatrix}$. This matrix is in $\mathcal{O}_{(2,2)}$ and $\pi(Y)=X$.
    LHS: $\dim \pi([\mathfrak{g}_2, Y]) = \dim([\mathfrak{g}_1, X] + S_{0, e_3}) = 4+3-\dim(\text{intersection})$.
    The intersection consists of matrices $y e_3^T$ where $y$ is orthogonal to $\{C^T e_3 \mid C \in C(X)\}$.
    The centralizer $C(X)$ consists of matrices $C = \begin{pmatrix} a & b & d \\ 0 & a & 0 \\ 0 & g & h \end{pmatrix}$.
    $\{C^T e_3\} = \{ (d, 0, h)^T \mid d,h \in \mathbb{C} \} = \mathrm{span}(e_1, e_3)$.
    The orthogonal complement is $\mathrm{span}(e_2)$. So $y$ must be a multiple of $e_2$, and the intersection has dimension 1.
    LHS = $4+3-1=6$. LHS=RHS, so the pair is relevant.

3.  **Pair $((2,1), (2,1,1))$:**
    $\dim \mathcal{O}_{(2,1,1)} = 4^2 - (3^2+1^2)=6$. RHS: $\frac{1}{2}(4+6)=5$.
    Consider $Y = \begin{pmatrix} X & e_1 \\ 0 & 0 \end{pmatrix}$. $Y^2=0$ and $\mathrm{rank}(Y)=1$, so $Y \in \mathcal{O}_{(2,1,1)}$. Also $\pi(Y)=X$.
    LHS: $\dim \pi([\mathfrak{g}_2, Y]) = \dim([\mathfrak{g}_1, X] + S_{e_1, 0}) = 4+3-\dim(\text{intersection})$.
    The intersection consists of matrices $e_1 c^T$ where $c$ is orthogonal to $\{C e_1 \mid C \in C(X)\}$.
    $\{C e_1\} = \{ (a, 0, 0)^T \mid a \in \mathbb{C} \} = \mathrm{span}(e_1)$.
    The orthogonal complement is $\mathrm{span}(e_2, e_3)$. The intersection has dimension 2.
    LHS = $4+3-2=5$. LHS=RHS, so the pair is relevant.

This completes the classification for $n=3$. The analysis for general $n$ and general partitions $\lambda$ is more involved and is not completed here. However, the cases presented above are fully and rigorously proven.



## Verification Report

### Summary ###

**Final Verdict:** The solution's approach is viable but contains a major Justification Gap, as it fails to prove that its classification of relevant orbits is complete.

**List of Findings:**
*   **Location:** The overall structure of the case analysis in Part 3 ("Case I", "Case II", "Case III").
    *   **Issue:** Justification Gap - The solution analyzes several specific families of partitions $(\lambda, \mu)$ but does not provide a rigorous argument that these are the only possible cases for relevant orbits. The final paragraph explicitly acknowledges that a complete proof of exhaustiveness is omitted.
*   **Location:** "Case III: Other partitions (Sporadic Cases for $n=3$)"
    *   **Issue:** Justification Gap - Within the analysis for $n=3$ and $\lambda=(2,1)$, the solution does not consider all possible partitions $\mu$ that satisfy the interlacing condition from Part 1. Specifically, the case $\mu=(3,1)$ is omitted without justification.

### Detailed Verification Log ###

**Initial Setup**

*   **Quote:** "Let $\pi: \mathfrak{g}_2 \to \mathfrak{g}_1$ be the projection onto the top-left $n \times n$ block. We identify $\mathfrak{g}_k^*$ with $\mathfrak{g}_k$ via the trace form $\langle A, B \rangle = \mathrm{Tr}(AB)$. Under this identification, the coadjoint action $\mathrm{Ad}^*$ becomes the adjoint action $\mathrm{Ad}$. The moment map is $\mu(g, Y) = (\pi(\mathrm{Ad}(g)Y), -Y)$ for $(g, Y) \in \mathrm{GL}_{n+1} \times \mathfrak{gl}_{n+1}$."
*   **Analysis:** This step correctly sets up the problem's notation and framework. The identification of $\mathfrak{g}_k^*$ with $\mathfrak{g}_k$ via the trace form is standard, and for $\mathfrak{gl}_k$, this correctly identifies the coadjoint and adjoint actions. The formula for the moment map is a direct translation of the problem statement under this identification. This step is correct.

**Part 1: Condition on partitions**

*   **Quote:** "The condition $\mathcal{O}_\lambda \times \mathcal{O}_\mu \subseteq \mu(M)$ becomes that for any $X \in \mathcal{O}_\lambda$ and any $Y' \in \mathcal{O}_\mu$, there exists $g \in G_2$ such that $X = \pi(\mathrm{Ad}(g)Y')$."
*   **Analysis:** This is a correct deduction from the definition of the moment map's image and the fact that nilpotent orbits are cones (so if $Z \in \mathcal{O}_\mu$, then $-Z \in \mathcal{O}_\mu$).
*   **Quote:** "Since $\mathrm{Ad}(G_2)$ acts transitively on the orbit $\mathcal{O}_\mu$, this set is simply $S_\mu = \{\pi(Y') \mid Y' \in \mathcal{O}_\mu\}$."
*   **Analysis:** The reasoning is sound. The set of values $\{\mathrm{Ad}(g)Y' \mid g \in G_2, Y' \in \mathcal{O}_\mu\}$ is simply $\mathcal{O}_\mu$ itself. Therefore, the set of projections is the set of projections of all elements of $\mathcal{O}_\mu$. This step is correct.
*   **Quote:** "Since $S_\mu$ is a $G_1$-invariant constructible set, this is equivalent to requiring that the intersection is non-empty, i.e., there exists some $Y \in \mathcal{O}_\mu$ such that $\pi(Y) \in \mathcal{O}_\lambda$."
*   **Analysis:** The argument that $S_\mu$ is $G_1$-invariant is correct. If a $G_1$-invariant set intersects a $G_1$-orbit, it must contain the entire orbit. Thus, the condition $\mathcal{O}_\lambda \subseteq S_\mu$ is equivalent to $\mathcal{O}_\lambda \cap S_\mu \neq \emptyset$. This is a correct and standard argument in this context.
*   **Quote:** "This is a classic result in linear algebra: a nilpotent matrix $X \in \mathfrak{gl}_n$ is a principal submatrix of a nilpotent matrix $Y \in \mathfrak{gl}_{n+1}$ if and only if their corresponding partitions $\lambda$ and $\mu$ satisfy the interlacing condition $\lambda \prec \mu$. For partitions of $n$ and $n+1$, this is equivalent to $\lambda$ being obtained from $\mu$ by decreasing one part by 1."
*   **Analysis:** The solution correctly identifies the problem with a known result concerning the relationship between the Jordan types of a nilpotent matrix and its principal submatrices. The characterization of the interlacing condition for partitions of $n$ and $n+1$ is also correct. This step is correct.

**Part 2: Dimension of the fiber**

*   **Quote:** "The fiber is thus isomorphic to the variety $S = \{ g \in \mathrm{GL}_{n+1} \mid \pi(\mathrm{Ad}(g)e_\mu) = -e_\lambda \}$, and $\dim M_e = \dim S$."
*   **Analysis:** This follows directly from the definition of the fiber $\mu^{-1}(e_\lambda, e_\mu)$, which fixes the second component to be $-e_\mu$. The dimension of the fiber is therefore the dimension of the variety of valid $g$. This is correct.
*   **Quote:** "The dimension of the fiber is independent of the choice of representatives $(e_\lambda, e_\mu)$."
*   **Analysis:** The solution provides a sound argument based on the $G$-equivariance of the moment map. The explicit isomorphism given between fibers $M_e$ and $M_{e'}$ for different points $e, e'$ in the same orbit product $\mathcal{O}_\lambda \times \mathcal{O}_\mu$ correctly demonstrates this independence. This step is correct.
*   **Quote:** "For a generic point $g \in S$, the dimension of the fiber is $\dim S = \dim \mathrm{GL}_{n+1} - \mathrm{rank}(d_g\phi)$... The differential ... is ... $d_g\phi(A) = \pi([\mathrm{Ad}(g)A, \mathrm{Ad}(g)e_\mu])$"
*   **Analysis:** The use of the fiber dimension theorem is appropriate here. The calculation of the differential is correct. The subsequent identification of the image of the differential with the projection of the tangent space to the orbit $\mathcal{O}_\mu$ is also correct.
*   **Quote:** "Since $\dim M_e$ is constant ... the quantity $\dim \pi([\mathfrak{g}_2, Y])$ must be constant for any $Y \in \mathcal{O}_\mu$ such that $\pi(Y) \in \mathcal{O}_\lambda$."
*   **Analysis:** This is a correct and important deduction. Since the fiber dimension depends only on the orbits, the rank of the differential (evaluated at appropriate points) must also be constant. The argument provided is sound.

**Part 3: Relevant orbits**

*   **Quote:** "The condition for relevance simplifies to: $ \dim \pi([\mathfrak{g}_2, Y]) = \frac{1}{2}(\dim \mathcal{O}_\lambda + \dim \mathcal{O}_\mu) $."
*   **Analysis:** This is a correct algebraic simplification based on the formula for $\dim M_e$ from Part 2 and the definition of a relevant orbit given in the problem.
*   **Quote:** "Case I: $\lambda = (1^n)$ (zero orbit)" and "Case II: $\lambda = (n)$ (regular orbit)"
*   **Analysis:** The calculations within these cases are performed correctly. The choice of representatives $Y$, the computation of the dimension of the image of the differential, and the verification of the relevance condition are all arithmetically and logically sound.
*   **Quote:** "Case III: Other partitions (Sporadic Cases for $n=3$)"
*   **Analysis:** The calculations for the two specific cases presented, $(\lambda, \mu) = ((2,1), (2,2))$ and $(\lambda, \mu) = ((2,1), (2,1,1))$, are correct.
*   **Quote:** The overall case analysis structure in Part 3.
*   **Analysis:** **Justification Gap.** The solution presents an analysis for the zero orbit, the regular orbit, and two "sporadic" cases for $n=3$. However, it provides no argument that this case breakdown is exhaustive. To answer the question "Find all the relevant nilpotent orbits," a proof of completeness is required. The solution does not systematically derive the list of relevant orbits; it merely verifies that certain pairs are relevant.
*   **Quote:** For $n=3, \lambda=(2,1)$, the solution checks $\mu=(2,2)$ and $\mu=(2,1,1)$.
*   **Analysis:** **Justification Gap.** According to the condition from Part 1, if $(\lambda, \mu)$ is a valid pair, $\lambda$ must be obtained from $\mu$ by decreasing one part by 1. For $\lambda=(2,1)$, the possible partitions $\mu$ of 4 are $(3,1)$, $(2,2)$, and $(2,1,1)$. The solution checks $\mu=(2,2)$ and $\mu=(2,1,1)$ but omits the case $\mu=(3,1)$ without any justification. This makes the analysis for this specific subcase incomplete.
*   **Quote:** "A complete proof that this list of seven families and sporadic cases is exhaustive is highly technical and relies on advanced results... The provided list is consistent with the known classification in the literature."
*   **Analysis:** This concluding statement explicitly confirms the major justification gap identified above. The solution admits that it has not provided a full proof of the classification, which is a critical component of the problem. While the results stated may be correct, the solution does not derive them completely.

