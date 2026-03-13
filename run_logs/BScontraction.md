# IMO Problem Solution

*Generated: 2026-02-21 23:13:54*

---


## Problem Statement

*** Problem Statement ***

Cellular Contraction of a Convex Polyhedron Complex

Let $E$ be a Euclidean space and $A \subseteq E$ be a **convex polyhedron complex**. By definition, $A$ is a convex closed subset of $E$ that admits a regular CW-complex structure satisfying the following conditions:

* **Cellular Structure**: For every (open) cell $F$ of $A$, the closure $\bar{F}$ is a **convex polyhedron** in $E$.
* **Locally Finite**: Every compact subset of $E$ contains only **finitely many cells** in $A$.
* **Skeleton**: We denote the $i$-th skeleton of $A$ as $sk_i A$.
* **Cellular Convex Hull**: For any subset $S \subseteq A$, the **cellular convex hull** $CCH(S)$ is defined as the smallest convex subcomplex of $A$ containing $S$.

Fix any $0$-cell $x_0$ in $A$. Prove that here exists a **cellular map** $H : A \times [0, 1] \to A$ such that:

1.  **Contraction Property**: $H$ is a cellular contraction from $A$ to $x_0$.
    * $H(y, 0) = y$ for all $y \in A$.
    * $H(y, 1) = x_0$ for all $y \in A$.
2.  **Hull Constraint**: For every $y \in A$, the image of the interval $H(y, [0, 1])$ lies entirely within the cellular convex hull $CCH(\{y, x_0\})$.



*** Technical Note on Cellularity ***
As a **cellular map** on the product space $A \times [0, 1]$, the map $H$ respects the skeletal filtration such that:
$$H(sk_i(A \times [0, 1])) \subseteq sk_i A$$
This implies that for any point $y$ in the $n$-skeleton of $A$, the path $H(y, t)$ is contained within the $(n+1)$-skeleton of $A$.



## Solution

### 1. Summary ###

**a. Verdict:**

I have successfully solved the problem. The provided bug report correctly identified three significant justification gaps in the proofs of Lemma 2.5 and Lemma 2. The original solution asserted key properties without proof, rendering the argument incomplete. This revised solution provides rigorous proofs for these claims, thereby resolving all identified issues and making the overall argument sound. Specifically, I have added detailed justifications for:
1.  The "uniform character" property of hyperplane arrangements used in Lemma 2.5.
2.  The polyhedral nature of the sets used to construct the triangulation in Lemma 2.
3.  The Piecewise-Linear (PL) nature of a map constructed by pasting two PL maps together in Lemma 2.

**b. Method Sketch:**

The proof is a constructive induction over the skeleta of the product space $A \times [0, 1]$.

1.  **Properties of the Cellular Convex Hull:** The solution first establishes two key properties of the cellular convex hull ($CCH$).
    *   **Lemma 0 (Hull Simplification):** For any point $y$ in an open cell $F$, $CCH(\{y, x_0\})$ is identical to $CCH(\bar{F} \cup \{x_0\})$. This simplifies the Hull Constraint from a point-wise condition to one that is uniform over each cell's closure.
    *   **Lemma 0.5 (Convexity of CCH):** The underlying space of any cellular convex hull, $|CCH(S)|$, is a convex subset of the Euclidean space $E$. This is proven by showing that $|CCH(S)|$ is the intersection of the underlying spaces of all convex subcomplexes containing $S$. This result is crucial for applying Lemma 1.

2.  **Technical Lemmas from PL Topology:** Four technical lemmas form the foundation of the construction.
    *   **Lemma 1 (PL Extension):** A PL map from the boundary of a PL ball into a convex set can be extended to a PL map over the entire ball using a standard cone construction.
    *   **Lemma 2.5 (Triangulation for Closed Covers and Subpolyhedra):** For any compact polyhedron $X$, a finite closed cover $\{C_i\}$ of $X$ by polyhedra, and a subpolyhedron $Y \subseteq X$, there exists a triangulation $K$ of $X$ such that $Y$ is the underlying space of a subcomplex of $K$, and every simplex of $K$ is contained in some $C_i$. The proof involves constructing a common subdivision from the hyperplane arrangement defining all polyhedra involved, and then triangulating this subdivision. A key step, now rigorously justified, is that every open cell of the arrangement is either fully contained in or disjoint from each polyhedron $C_i$.
    *   **Lemma 2 (Cell Pushing):** Let $f:D \to A$ be a PL map from a compact complex $D$ to $A$. For any cell $F$ of $A$ with $\dim F > \dim D$, there exists a PL homotopy, fixed on $\{x \mid f(x) \notin F\}$, from $f$ to a map $g$ whose image does not intersect the open cell $F$. The proof relies on constructing a PL "pseudo-radial projection". The existence of a suitable triangulation for this construction is guaranteed by Lemma 2.5, after proving that the sets used in the cover are indeed polyhedra. The PL nature of the resulting pasted map is also now rigorously justified.
    *   **Lemma 3 (Cellular Approximation):** A PL map $f:D \to A$ from a $d$-dimensional complex $D$ can be deformed into the $d$-skeleton of $A$. The proof is a downward induction on cell dimension, using Lemma 2 iteratively to clear the image from cells of dimension $k > d$.

3.  **Inductive Construction:** The cellular contraction $H$ is constructed inductively on the skeleta of the domain $X = A \times [0, 1]$. Let $H_n$ be the map defined on $sk_n X$.
    *   **Inductive Hypothesis $P(n)$:** A PL map $H_n: sk_n X \to sk_n A$ has been constructed that satisfies the contraction and hull properties on its domain.
    *   **Base Case ($n=0$):** Trivial definition on $sk_0 X = sk_0 A \times \{0, 1\}$.
    *   **Inductive Step:** Assuming $P(n)$, we extend the map $H_n$ to $H_{n+1}$ on $sk_{n+1} X$. For each $n$-cell $E$ of $A$, a map is already defined on the boundary of the $(n+1)$-dimensional cylinder $\bar{E} \times [0, 1]$. The image of this boundary map lies in the subcomplex $C_E = CCH(\bar{E} \cup \{x_0\})$. By Lemma 0.5, $|C_E|$ is a convex set. Lemma 1 is used to extend this to a PL map into $|C_E|$, and Lemma 3 is used to deform the image into $sk_{n+1} A$ while remaining inside $|C_E|$.

4.  **Final Justification:** The solution concludes with a detailed argument explaining why the inductively defined map $H$ is continuous and PL on the entire space $A \times [0,1]$, based on the local finiteness of the complex.

### 2. Detailed Solution ###

**1. Preliminaries**

**Lemma 0 (Hull Simplification).** Let $A$ be a convex polyhedron complex and $x_0$ a $0$-cell of $A$. For any point $y \in A$, let $F$ be the unique open cell of $A$ such that $y \in F$. Then the cellular convex hull $CCH(\{y, x_0\})$ is equal to $CCH(\bar{F} \cup \{x_0\})$.

**Proof.**
Let $C_y = CCH(\{y, x_0\})$. By definition, $C_y$ is the smallest convex subcomplex of $A$ containing $\{y, x_0\}$. Since $y \in C_y$ and $C_y$ is a subcomplex, it must contain the cell $F$. As a subcomplex, if $F \in C_y$, then all faces of $F$ are in $C_y$, which implies $\bar{F} \subseteq |C_y|$. Since $x_0 \in C_y$, we have $\bar{F} \cup \{x_0\} \subseteq |C_y|$. As $C_y$ is a convex subcomplex containing $\bar{F} \cup \{x_0\}$, it must contain $CCH(\bar{F} \cup \{x_0\})$.

For the reverse inclusion, let $C' = CCH(\bar{F} \cup \{x_0\})$. $C'$ is a convex subcomplex containing $\bar{F}$ (and thus $y$) and $x_0$. Since $C_y$ is the smallest such complex, $C_y \subseteq C'$. Thus, $CCH(\{y, x_0\}) = CCH(\bar{F} \cup \{x_0\})$.
$\qed$

**Lemma 0.5 (Convexity of CCH).** For any subset $S \subseteq A$, the underlying space of the cellular convex hull, $|CCH(S)|$, is a convex subset of the Euclidean space $E$.

**Proof.**
Let $\mathcal{C}_S$ be the collection of all convex subcomplexes of $A$ containing $S$. A subcomplex $C$ is called convex if its underlying space $|C|$ is a convex subset of $E$. The complex $A$ itself is a convex subcomplex, so $\mathcal{C}_S$ is non-empty.
The cellular convex hull is defined as the intersection of these subcomplexes: $CCH(S) = \bigcap_{C \in \mathcal{C}_S} C$, where the intersection is over the collections of cells.
The underlying space of a subcomplex $K$ is $|K| = \bigcup_{F \in K} \bar{F}$. An equivalent characterization is that $x \in |K|$ if and only if the unique open cell containing $x$ belongs to $K$. This implies that for any collection of subcomplexes $\{C_i\}$, we have $|\bigcap_i C_i| = \bigcap_i |C_i|$.
Applying this to the definition of $CCH(S)$, we get:
$$|CCH(S)| = \left|\bigcap_{C \in \mathcal{C}_S} C\right| = \bigcap_{C \in \mathcal{C}_S} |C|$$
Each set $|C|$ for $C \in \mathcal{C}_S$ is convex by definition. The intersection of any collection of convex sets is convex. Therefore, $|CCH(S)|$ is a convex subset of $E$.
$\qed$

**2. Technical Lemmas**

**Lemma 1 (PL Extension).** Let $B$ be a PL $d$-ball and let $f: \partial B \to C$ be a PL map, where $C$ is a convex subset of a Euclidean space. Then there exists a PL map $F: B \to C$ that extends $f$.

**Proof.**
Let $B$ be realized as a convex polyhedron in some $\mathbb{R}^d$. Let $c$ be a point in the interior of $B$. Any point $x \in B \setminus \{c\}$ can be uniquely written as $x = (1-t)c + t x'$ for some $t \in (0,1]$ and $x' \in \partial B$. Let $p \in C$ be an arbitrary point. We define the extension $F(x) = (1-t)p + t f(x')$, with $F(c)=p$. Since $C$ is convex, the image is in $C$. This map is an extension of $f$. Given a triangulation of $\partial B$ for which $f$ is simplicial, $F$ is simplicial with respect to the cone triangulation of $B$ from $c$. Thus $F$ is a PL map.
$\qed$

**Lemma 2.5 (Triangulation for Closed Covers and Subpolyhedra).** Let $X$ be a compact polyhedron, $\{C_1, \dots, C_m\}$ be a finite closed cover of $X$ by polyhedra, and $Y$ be a subpolyhedron of $X$. Then there exists a triangulation $K$ of $X$ such that $Y$ is the underlying space of a subcomplex of $K$, and every simplex of $K$ is contained in some $C_i$.

**Proof.**
1.  **Construct a common subdivision:** A polyhedron is a finite union of convex polyhedra. Let $\mathcal{F} = \{C_1, \dots, C_m, Y\}$. Let $\mathcal{F}_{conv}$ be the finite collection of all convex polyhedral pieces making up the polyhedra in $\mathcal{F}$. Let $\mathcal{H}$ be the finite set of all hyperplanes that define the convex polyhedra in $\mathcal{F}_{conv}$. The arrangement of these hyperplanes partitions the ambient Euclidean space into a finite polyhedral complex $\mathcal{A}$ whose cells are open convex polyhedra. The restriction of this complex to the compact set $X$, let's call it $\mathcal{A}_X$, is a finite polyhedral complex with convex cells whose underlying space is $X$.

2.  **Triangulate the subdivision:** A finite polyhedral complex with convex cells admits a simplicial subdivision (a triangulation). This can be proven by induction on the dimension of the complex. Thus, there exists a triangulation $K$ of $\mathcal{A}_X$.

3.  **Verify the properties:**
    *   **$Y$ is a subcomplex:** The polyhedron $Y$ is a union of closures of cells of the complex $\mathcal{A}_X$. Therefore, $Y$ is the underlying space of a subcomplex of $\mathcal{A}_X$. Any triangulation of a polyhedral complex has the property that any subcomplex of the original complex becomes a simplicial subcomplex of the triangulation. Thus, $Y$ is the underlying space of a subcomplex of $K$.
    *   **Every simplex is in some $C_i$:** Let $\sigma$ be a simplex of $K$. Pick a point $x$ in the relative interior of $\sigma$. This point $x$ lies in the relative interior of a unique cell $P$ of the complex $\mathcal{A}_X$. Since $\{C_i\}$ is a cover of $X$, we must have $x \in C_k$ for some index $k$. We now show that this implies $\bar{P} \subseteq C_k$.
        The open cell $P$ does not intersect any hyperplane in $\mathcal{H}$. Each polyhedron $C_k$ is a union of convex polyhedra $Q_{kj}$ whose defining hyperplanes are in $\mathcal{H}$. Suppose $x \in C_k$. Then $x \in Q_{kj}$ for some $j$. A convex polyhedron $Q_{kj}$ is an intersection of closed half-spaces $\bigcap_l H_l^+$, where $\partial H_l \in \mathcal{H}$. Since $x \in P$ and $P \cap (\bigcup_{H \in \mathcal{H}} H) = \emptyset$, $x$ must be in the interior of each half-space $H_l^+$. As $P$ is connected and disjoint from $\partial H_l$, the entire cell $P$ must be contained in the interior of $H_l^+$. This holds for all $l$, so $P \subseteq \text{int}(Q_{kj}) \subseteq Q_{kj} \subseteq C_k$.
        Since $P \subseteq C_k$ and $C_k$ is a closed set, it must contain the closure of $P$, i.e., $\bar{P} \subseteq C_k$. Since $\sigma$ is part of the triangulation of $\bar{P}$, we have $|\sigma| \subseteq \bar{P} \subseteq C_k$.
This completes the proof.
$\qed$

**Lemma 2 (Cell Pushing).** Let $D$ be a compact convex polyhedron complex and $f:D \to A$ be a PL map. Let $F$ be an open cell of $A$ such that $\dim F > \dim D$. Then there exists a PL map $g:D \to A$ and a PL homotopy $G: D \times [0,1] \to A$ from $f$ to $g$ such that $g(D) \cap F = \emptyset$ and the homotopy is fixed on the set $\{x \in D \mid f(x) \notin F\}$. The image of the homotopy is contained in $f(D) \cup \bar{F}$.

**Proof.**
1.  **Setup:** The image $f(D)$ is a compact polyhedron of dimension at most $\dim D$. Since $\dim F > \dim D$, the image $f(D)$ cannot cover the open cell $F$. Thus, we can choose a point $z \in F \setminus f(D)$.

2.  **Construct a PL Pseudo-Radial Projection:** Let $p: \bar{F} \setminus \{z\} \to \partial F$ be the standard radial projection. Choose an open neighborhood $U$ of $z$ in $F$ such that its closure $\bar{U}$ is a convex polyhedron and $\bar{U} \cap f(D) = \emptyset$. Let $X = \bar{F} \setminus U$.
    Let $\{G_1, \dots, G_m\}$ be the set of proper faces of $\bar{F}$. For each $i$, let $C_i = \{y \in X \mid p(y) \in G_i\}$. The collection $\{C_i\}$ is a finite closed cover of $X$.
    
    **The sets $C_i$ are polyhedra:** The set of points in $\bar{F}$ that project to $G_i$ is the intersection of $\bar{F}$ with the cone over $G_i$ from apex $z$. This set is $\text{conv}(\{z\} \cup G_i) \cap \bar{F}$, which is an intersection of two convex polyhedra and thus a convex polyhedron. Then $C_i = (\text{conv}(\{z\} \cup G_i) \cap \bar{F}) \cap X$. As $X$ is a polyhedron, $C_i$ is an intersection of polyhedra, hence a polyhedron.

    By Lemma 2.5, applied to the polyhedron $X$, the cover $\{C_i\}$, and the subpolyhedron $Y = \partial F$, there exists a triangulation $K$ of $X$ such that $\partial F$ is a subcomplex of $K$ and every simplex of $K$ is contained in some $C_i$.

    We define a PL map $q: X \to \partial F$. For each vertex $v$ of $K$, define $q(v) = p(v)$. For any simplex $\sigma \in K$, extend $q$ affinely over $\sigma$. This defines a PL map $q$ on $X$.
    *   The image of $q$ is in $\partial F$: For any simplex $\sigma \in K$, $|\sigma| \subseteq C_i$ for some $i$. This means $p(|\sigma|) \subseteq G_i$. In particular, for all vertices $v_j$ of $\sigma$, $p(v_j) \in G_i$. Since $G_i$ is a convex face, the affine image $q(|\sigma|) = \text{conv}\{q(v_j)\} = \text{conv}\{p(v_j)\}$ is contained in $G_i \subseteq \partial F$.
    *   $q$ is the identity on $\partial F$: Let $x \in \partial F$. Since $\partial F$ is a subcomplex of $K$, $x$ lies in a simplex $\sigma$ of $K$ with $|\sigma| \subseteq \partial F$. All vertices $v_j$ of $\sigma$ are in $\partial F$. For any such vertex, $p(v_j) = v_j$, so $q(v_j) = v_j$. Since $q$ is affine on $\sigma$, it is the identity map on $\sigma$. Thus $q(x)=x$.

3.  **Define the new map g:** We extend $q$ to a map $\tilde{q}: A \setminus U \to A$.
    $$ \tilde{q}(y) = \begin{cases} q(y) & \text{if } y \in \bar{F} \setminus U \\ y & \text{if } y \in A \setminus F \end{cases} $$
    This map is well-defined. If $y \in (\bar{F} \setminus U) \cap (A \setminus F) = \partial F$, then $q(y)=y$, so both definitions agree.
    
    **The map $\tilde{q}$ is PL:** A map between polyhedra is PL if and only if its graph is a polyhedron. The graph of $q$ on $\bar{F} \setminus U$ is a polyhedron, and the graph of the identity map on $A \setminus F$ is a polyhedron. The graph of $\tilde{q}$ is the union of these two graphs. The union of two polyhedra is a polyhedron. Thus, $\tilde{q}$ is a PL map.
    
    Define $g = \tilde{q} \circ f$. Since $f(D) \cap U = \emptyset$, the map $g$ is well-defined on all of $D$. It is a composition of PL maps, hence PL.
    If $f(x) \notin F$, then $g(x) = \tilde{q}(f(x)) = f(x)$.
    If $f(x) \in F$, then $f(x) \in \bar{F} \setminus U$, so $g(x) = q(f(x)) \in \partial F$.
    Therefore, $g(D) \cap F = \emptyset$.

4.  **Construct the Homotopy:** Define $G: D \times [0,1] \to A$ by $G(x,t) = (1-t)f(x) + t g(x)$. This is a PL map.
    $G(x,0) = f(x)$ and $G(x,1) = g(x)$.
    If $f(x) \notin F$, then $g(x)=f(x)$, so $G(x,t) = f(x)$ for all $t$.
    If $f(x) \in F$, then $f(x) \in \bar{F}$ and $g(x) \in \partial F \subset \bar{F}$. Since $\bar{F}$ is convex, the line segment $[f(x), g(x)]$ is contained in $\bar{F}$. Thus, the image of the homotopy $G$ is contained in $f(D) \cup \bar{F}$.
$\qed$

**Lemma 3 (Cellular Approximation).** Let $f:D \to A$ be a PL map, where $D$ is a compact convex polyhedron complex of dimension $d = \dim D$. There exists a PL map $g:D \to A$ homotopic to $f$ such that $g(D) \subseteq sk_d A$ and the homotopy is fixed on the set $\{x \in D \mid f(x) \in sk_d A\}$. Furthermore, if $f(D) \subseteq |C|$ for some convex subcomplex $C \subseteq A$, then $g(D) \subseteq |C|$ and the image of the homotopy lies in $|C|$.

**Proof.**
The proof is by downward induction on $k$, from $k_{max} = \dim A$ down to $d+1$. Let $f_{k_{max}} = f$.
**Inductive Hypothesis:** For some $k > d$, we have a PL map $f_k: D \to A$ homotopic to $f$, such that $f_k(D) \subseteq sk_k A$.
**Inductive Step:** We construct a map $f_{k-1}: D \to A$ homotopic to $f_k$ such that $f_{k-1}(D) \subseteq sk_{k-1} A$. The image $f_k(D)$ is compact, so it intersects only a finite number of open $k$-cells of $A$. Let these cells be $F_1, \dots, F_m$. We apply Lemma 2 iteratively. Let $g_0 = f_k$. For $i=1, \dots, m$, let $g_i$ be the map obtained by applying Lemma 2 to $g_{i-1}$ and the cell $F_i$. The resulting map $g_i$ has an image disjoint from $F_i$. This operation does not reintroduce intersections with previously cleared $k$-cells $F_j$ ($j<i$), because the homotopy moves points inside $\bar{F_i}$ to its boundary $\partial F_i$, which is composed of cells of dimension less than $k$ and thus disjoint from any open $k$-cell.
The final map $f_{k-1} := g_m$ has an image disjoint from all open $k$-cells, so $f_{k-1}(D) \subseteq sk_{k-1} A$. The composition of the PL homotopies gives a PL homotopy from $f_k$ to $f_{k-1}$.
This process, from $k = k_{max}$ down to $d+1$, yields the desired map $g: D \to sk_d A$. The homotopy is fixed on $\{x \in D \mid f(x) \in sk_d A\}$ because at each step $k > d$, the homotopy is fixed for points whose image is not in an open $k$-cell.
Finally, if $f(D) \subseteq |C|$ for a convex subcomplex $C$, then at each step, we push out of a cell $F_i$ of $C$. The homotopy from Lemma 2 is constructed within $f(D) \cup \bar{F_i} \subseteq |C| \cup |C| = |C|$. Thus, the image of the map at every stage remains within $|C|$.
$\qed$

**3. Construction of the Cellular Contraction**

We construct the map $H: A \times [0,1] \to A$ by induction on the skeleta of the domain $X = A \times [0,1]$. The $n$-skeleton of $X$ is $sk_n X = (sk_n A \times \{0,1\}) \cup (sk_{n-1} A \times [0,1])$. Let $H_n$ denote the restriction of $H$ to $sk_n X$.

**Inductive Hypothesis $P(n)$:** A PL map $H_n: sk_n X \to sk_n A$ has been constructed satisfying:
1.  $H_n(y, 0) = y$ for all $(y,0) \in sk_n X$.
2.  $H_n(y, 1) = x_0$ for all $(y,1) \in sk_n X$.
3.  For every $y \in A$ such that $(y,t) \in sk_n X$ for some $t \in [0,1]$, the path segment $\{H_n(y,s) \mid (y,s) \in sk_n X\}$ lies in $|CCH(\{y, x_0\})|$.

**Base Case (n=0):**
$sk_0 X = sk_0 A \times \{0,1\}$. We define $H_0: sk_0 A \times \{0,1\} \to sk_0 A$ by $H_0(y,0) = y$ and $H_0(y,1) = x_0$. This map is PL and satisfies the conditions. $P(0)$ holds.

**Inductive Step:**
Assume $P(n)$ holds. We construct $H_{n+1}: sk_{n+1} X \to sk_{n+1} A$. The space $sk_{n+1} X$ is obtained from $sk_n X$ by attaching cells of the form $\bar{E} \times [0,1]$ for all $n$-cells $E$ of $A$, and cells $\bar{F} \times \{0,1\}$ for all $(n+1)$-cells $F$ of $A$.

First, on the "caps" $\bar{F} \times \{0,1\}$ for $(n+1)$-cells $F$, we define $H_{n+1}(y,0) = y$ and $H_{n+1}(y,1) = x_0$.

Now, for each $n$-cell $E$ of $A$, we extend the map to the cylinder $\bar{E} \times [0,1]$.
1.  **Boundary Map:** A PL map $f_E: \partial(\bar{E} \times [0,1]) \to sk_n A$ is already defined by $H_n$ and the definitions on the caps.
2.  **Hull Constraint on Boundary:** Let $C_E = CCH(\bar{E} \cup \{x_0\})$. By the inductive hypothesis, the image of the boundary map $f_E$ is contained in $|C_E|$.
3.  **PL Extension:** The cylinder $\bar{E} \times [0,1]$ is a PL $(n+1)$-ball. By Lemma 0.5, the target space $|C_E|$ is a convex set. By Lemma 1, the PL map $f_E$ can be extended to a PL map $f_{ext}: \bar{E} \times [0,1] \to |C_E|$.
4.  **Cellular Approximation:** We now have a PL map $f_{ext}: \bar{E} \times [0,1] \to |C_E| \subseteq A$. The domain has dimension $n+1$. By Lemma 3, there exists a PL map $g_E: \bar{E} \times [0,1] \to A$ homotopic to $f_{ext}$ such that:
    *   $g_E(\bar{E} \times [0,1]) \subseteq sk_{n+1} A$.
    *   The homotopy is fixed on the boundary $\partial(\bar{E} \times [0,1])$ because the boundary map $f_E$ already has its image in $sk_n A \subseteq sk_{n+1} A$. So $g_E$ agrees with $f_E$ on the boundary.
    *   Since the image of $f_{ext}$ is in the convex subcomplex $C_E$, the final clause of Lemma 3 ensures that the image of $g_E$ is also contained in $|C_E|$.

We define $H_{n+1}$ on $\bar{E} \times [0,1]$ to be this map $g_E$. This is done for all $n$-cells $E$.

**Verification of $P(n+1)$:**
The constructed map $H_{n+1}: sk_{n+1} X \to sk_{n+1} A$ is PL and satisfies the contraction property by construction. We verify the Hull Constraint. Let $y \in A$ and let $F_y$ be the open cell containing $y$.
*   If $\dim F_y \le n-1$, then $(y,s) \in sk_n X$ for all $s \in [0,1]$. By $P(n)$, the path lies in $|CCH(\{y, x_0\})|$.
*   If $\dim F_y = n$, let $E=F_y$. Then $(y,s) \in sk_{n+1} X$ for all $s \in [0,1]$. The map on $\bar{E} \times [0,1]$ is $g_E$, whose image is in $|C_E| = |CCH(\bar{E} \cup \{x_0\})|$. By Lemma 0, this is $|CCH(\{y, x_0\})|$.
*   If $\dim F_y = n+1$, then $(y,s) \in sk_{n+1} X$ only if $s=0$ or $s=1$. The path segment is $\{y, x_0\}$, which is trivially in $|CCH(\{y, x_0\})|$.
In all cases, the hull constraint is satisfied. Thus, $P(n+1)$ holds.

**4. Final Justification: Continuity and PL Property of H**

The final map $H: A \times [0,1] \to A$ is defined by $H(z) = H_n(z)$ for any $z \in sk_n(A \times [0,1])$. Since each $H_{n+1}$ is an extension of $H_n$, this definition is consistent.

The complex $A$ is locally finite, which implies $X = A \times [0,1]$ is also a locally finite CW-complex. For such complexes, a map is continuous if and only if its restriction to the closure of each cell is continuous. Our construction defines $H$ on each cell-cylinder as a PL map, and these maps agree on their boundaries. Thus, $H$ is continuous.

A map between locally finite polyhedra is PL if its restriction to any compact subpolyhedron is PL. Let $K \subset A \times [0,1]$ be a compact subpolyhedron. Due to local finiteness, $K$ is contained in a finite subcomplex. On this finite subcomplex, $H$ is constructed by gluing a finite number of PL maps ($g_E$). By the Pasting Lemma for PL maps, the restriction of $H$ is PL. Thus, $H$ is a PL map.

By construction, $H$ is a cellular map satisfying the contraction and hull properties. This completes the proof.



## Verification Report

### Summary ###

**Final Verdict:** The solution is correct.

**List of Findings:**
The provided solution is exceptionally rigorous and well-structured. All steps are logically sound and meticulously justified, successfully navigating the technical complexities of Piecewise-Linear (PL) and algebraic topology. There are no errors or justification gaps to report.

### Detailed Verification Log ###

**1. Preliminaries**

*   **Lemma 0 (Hull Simplification).**
    *   **Quoted Text:** "Let $A$ be a convex polyhedron complex... Then the cellular convex hull $CCH(\{y, x_0\})$ is equal to $CCH(\bar{F} \cup \{x_0\})$.
    *   **Verification:** The proof proceeds by showing two-sided inclusion. The reasoning is based directly on the definition of the cellular convex hull as the *smallest* convex subcomplex containing a given set. The arguments for both inclusions are clear, direct, and logically sound. The step is correct.

*   **Lemma 0.5 (Convexity of CCH).**
    *   **Quoted Text:** "For any subset $S \subseteq A$, the underlying space of the cellular convex hull, $|CCH(S)|$, is a convex subset of the Euclidean space $E$."
    *   **Verification:** The proof correctly identifies $CCH(S)$ as the intersection of all convex subcomplexes containing $S$. The key identity $|\bigcap_i C_i| = \bigcap_i |C_i|$ for subcomplexes $C_i$ is correctly established. The conclusion follows from the fact that the intersection of convex sets is convex. The step is correct.

**2. Technical Lemmas**

*   **Lemma 1 (PL Extension).**
    *   **Quoted Text:** "Let $B$ be a PL $d$-ball and let $f: \partial B \to C$ be a PL map, where $C$ is a convex subset of a Euclidean space. Then there exists a PL map $F: B \to C$ that extends $f$."
    *   **Verification:** The proof uses a standard "coning" construction. The resulting map $F$ is shown to be an extension of $f$ with its image contained in $C$. The assertion that this construction yields a PL map is a standard result in PL topology. The justification is sufficient. The step is correct.

*   **Lemma 2.5 (Triangulation for Closed Covers and Subpolyhedra).**
    *   **Quoted Text:** "Let $X$ be a compact polyhedron... Then there exists a triangulation $K$ of $X$ such that $Y$ is the underlying space of a subcomplex of $K$, and every simplex of $K$ is contained in some $C_i$."
    *   **Verification:** The proof follows a standard and correct procedure. It first constructs a common polyhedral subdivision of $X$ by considering all hyperplanes defining the given polyhedra. Then, it triangulates this subdivision. The arguments verifying that this triangulation has the required properties are rigorous and logically sound. The step is correct.

*   **Lemma 2 (Cell Pushing).**
    *   **Quoted Text:** "Let $D$ be a compact convex polyhedron complex and $f:D \to A$ be a PL map. Let $F$ be an open cell of $A$ such that $\dim F > \dim D$. Then there exists a PL map $g:D \to A$..."
    *   **Verification:** This is a highly technical and crucial lemma. The proof correctly avoids the pitfall that a standard radial projection is not a PL map. It does so by constructing a PL approximation, $q$, to the radial projection. This construction is sound and relies correctly on Lemma 2.5. The subsequent steps of defining the new map $g$ by composing with this PL projection and constructing the straight-line homotopy $G$ are all rigorously justified. The properties of $g$ and $G$ are correctly derived. The step is correct.

*   **Lemma 3 (Cellular Approximation).**
    *   **Quoted Text:** "Let $f:D \to A$ be a PL map... There exists a PL map $g:D \to A$ homotopic to $f$ such that $g(D) \subseteq sk_d A$..."
    *   **Verification:** The proof uses a downward induction on the dimension of the skeleton containing the image of the map. The inductive step correctly applies Lemma 2 iteratively to push the image out of all cells of a given dimension. The argument that this process does not interfere with previously cleared cells is sound. The preservation of the hull constraint is also correctly justified. The step is correct.

**3. Construction of the Cellular Contraction**

*   **Inductive Hypothesis $P(n)$:**
    *   **Quoted Text:** "A PL map $H_n: sk_n X \to sk_n A$ has been constructed satisfying..."
    *   **Verification:** The inductive hypothesis is clearly stated and correctly formulates the properties required at each stage of the construction, including the crucial cellularity condition ($H_n$ maps to $sk_n A$).

*   **Base Case (n=0):**
    *   **Quoted Text:** "$sk_0 X = sk_0 A \times \{0,1\}$. We define $H_0: sk_0 A \times \{0,1\} \to sk_0 A$ by $H_0(y,0) = y$ and $H_0(y,1) = x_0$."
    *   **Verification:** The base case is simple and correctly handled. The map is well-defined and satisfies all conditions of the inductive hypothesis for $n=0$. The step is correct.

*   **Inductive Step:**
    *   **Quoted Text:** "Assume $P(n)$ holds. We construct $H_{n+1}: sk_{n+1} X \to sk_{n+1} A$."
    *   **Verification:** This is the core of the proof. The construction proceeds by defining the map on the new cells attached to form $sk_{n+1} X$. For each cylinder $\bar{E} \times [0,1]$ (where $E$ is an $n$-cell), the argument is as follows:
        1.  A PL map on the boundary is correctly identified using the map $H_n$ from the previous step.
        2.  The image of this boundary map is correctly shown to lie within the required cellular convex hull, $C_E$.
        3.  Lemma 1 is correctly invoked to extend this boundary map to a PL map $f_{ext}$ from the entire cylinder into the convex set $|C_E|$.
        4.  Lemma 3 is correctly invoked to adjust $f_{ext}$ to a map $g_E$ that satisfies the cellularity condition (image in $sk_{n+1} A$) while preserving the boundary map and the hull constraint.
    This sequence of steps is logically flawless.

*   **Verification of $P(n+1)$:**
    *   **Quoted Text:** "The constructed map $H_{n+1}: sk_{n+1} X \to sk_{n+1} A$ is PL and satisfies the contraction property by construction. We verify the Hull Constraint."
    *   **Verification:** The check that the newly constructed map $H_{n+1}$ satisfies all conditions of the hypothesis is thorough, considering all cases for the dimension of the cell containing a point $y$. The reasoning is sound. The step is correct.

**4. Final Justification: Continuity and PL Property of H**

*   **Quoted Text:** "The final map $H: A \times [0,1] \to A$ is defined by $H(z) = H_n(z)$ for any $z \in sk_n(A \times [0,1])$."
*   **Verification:** The argument for why the inductively defined map $H$ is well-defined, continuous, and PL on the entire space $A \times [0,1]$ is correct. It properly uses the local finiteness of the complex $A$ to justify that properties defined on finite subcomplexes extend to the whole space. The use of the "Pasting Lemma for PL maps" is appropriate and correct. The final conclusion that $H$ is a cellular map with all the required properties is fully supported by the construction. The step is correct.

