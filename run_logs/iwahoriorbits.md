# IMO Problem Solution

*Generated: 2026-02-19 17:39:53*

---


## Problem Statement

*** Problem Statement ***


# Classification of Iwahori orbits on \(V\)

## Setup

Let \(F\) be a non-archimedean local field with ring of integers \(\mathfrak{O}\) and uniformizer \(\varpi\). Let \(V\)  be finite-dimensional vector spaces over \(F\) with dimension $n$ and standard basis \(e_1, ..., e_n\). Fix the Iwahori subgroup I$ consists of elements sending $e_i$ to the lattice $\sum_{j<= i} \mathfrak{O}e_j + \sum_{j>i}\varphi \mathfrak{O} e_j$


**Classify the orbits of the Iwahori action on \(V\).**

That is, describe a complete set of representatives for the orbit space
\[
I  \backslash V 
\]
in terms of some combinatorial data






## Solution

### 1. Summary

**a. Verdict**

I have successfully solved the problem. A complete set of representatives for the orbits of the Iwahori group $I$ on the vector space $V$ is given by the set
\[
\{0\} \cup \{ \varpi^m e_k \mid m \in \mathbb{Z}, k \in \{1, \dots, n\} \},
\]
where $0$ is the zero vector, $\varpi$ is a uniformizer of the local field $F$, $m$ ranges over all integers, and $\{e_1, \dots, e_n\}$ is the standard basis of $V$. The orbits are thus in bijection with the set $(\mathbb{Z} \times \{1, \dots, n\}) \cup \{\text{zero orbit}\}$.

**b. Method Sketch**

1.  **Problem Interpretation:** First, we interpret the definition of the Iwahori subgroup $I$. The condition that for any $g \in I$, $g(e_j) \in \sum_{i \le j} \mathfrak{O}e_i + \sum_{i>j}\varpi \mathfrak{O} e_i$ for each basis vector $e_j$ means that the matrix representation $g=(g_{ij})$ of $g$ satisfies $g_{ij} \in \mathfrak{O}$ for $i \le j$ and $g_{ij} \in \varpi\mathfrak{O}$ for $i > j$. For $g$ to be invertible with an inverse in $GL_n(\mathfrak{O})$, its diagonal entries must be units, i.e., $g_{ii} \in \mathfrak{O}^\times$.

2.  **Reduction to Primitive Vectors:** The action of $I$ on $V=F^n$ is $F$-linear and commutes with multiplication by powers of the uniformizer $\varpi$. Any non-zero vector $v \in V$ can be uniquely written as $v = \varpi^m v_0$, where $m = \min_{i} \{\text{val}(v_i)\}$ and $v_0 \in \mathfrak{O}^n$ is a primitive vector (i.e., not all its components are in $\varpi\mathfrak{O}$). The classification of $I \backslash V$ is thus determined by the classification of orbits on the set of primitive vectors, which we denote by $V_0 = \mathfrak{O}^n \setminus \varpi\mathfrak{O}^n$. The full set of orbit representatives is then $\{0\} \cup \{\varpi^m r \mid m \in \mathbb{Z}, r \in R_0\}$, where $R_0$ is a set of representatives for $I \backslash V_0$.

3.  **An Orbit Invariant:** For any primitive vector $v \in V_0$, at least one of its components must be a unit in $\mathfrak{O}$. We define an integer $k(v) = \max\{i \in \{1, \dots, n\} \mid v_i \in \mathfrak{O}^\times\}$. We prove that this integer is an invariant of the $I$-action; that is, if $v' = gv$ for some $g \in I$, then $k(v')=k(v)$. This invariant partitions the set $V_0$ into $n$ disjoint $I$-stable subsets, $V_{0,k} = \{v \in V_0 \mid k(v)=k\}$ for $k=1, \dots, n$.

4.  **Canonical Representatives:** We show that every vector in $V_{0,k}$ belongs to the orbit of the standard basis vector $e_k$. For any $v \in V_{0,k}$, we constructively find an element $g \in I$ such that $gv = e_k$. This is done in three steps, by applying matrices from $I$ that correspond to elementary row operations:
    a. Annihilate the components $v_i$ for $i < k$ by subtracting suitable multiples of the $k$-th component of the vector. The matrix for this, $g_1$, is in $I$ because $v_k$ is a unit.
    b. Annihilate the components $v_j$ for $j > k$ by subtracting suitable multiples of the $k$-th component. The matrix for this, $g_2$, is in $I$ because the components $v_j$ for $j>k$ are in $\varpi\mathfrak{O}$, a crucial property of $V_{0,k}$.
    c. Normalize the resulting vector's $k$-th component to $1$ using a diagonal matrix $g_3 \in I$.
    The total transformation is $g = g_3 g_2 g_1 \in I$.

5.  **Conclusion:** The standard basis vectors $\{e_1, \dots, e_n\}$ form a complete set of representatives for the orbits on $V_0$, as the invariant $k(e_j)=j$ ensures they lie in different orbits. Combining this with the reduction step, we obtain the final classification of orbits on $V$.

### 2. Detailed Solution

Let $V = F^n$ be the $n$-dimensional vector space over the non-archimedean local field $F$. Let $\mathfrak{O}$ be the ring of integers of $F$ and $\varpi$ be a uniformizer. The standard basis of $V$ is $\{e_1, \dots, e_n\}$.

**Step 1: The Iwahori Subgroup**

The Iwahori subgroup $I$ is defined as the set of elements $g \in GL_n(F)$ such that for each $j \in \{1, \dots, n\}$, $g(e_j) \in \sum_{i \le j} \mathfrak{O}e_i + \sum_{i>j}\varpi \mathfrak{O} e_i$.
Let $g=(g_{ij})$ be the matrix representation of an element $g \in I$ with respect to the standard basis, where $i$ is the row index and $j$ is the column index. The vector $g(e_j)$ is the $j$-th column of the matrix $g$. The condition on $g(e_j)$ translates to conditions on the entries $g_{ij}$:
*   For $i \le j$, $g_{ij} \in \mathfrak{O}$.
*   For $i > j$, $g_{ij} \in \varpi\mathfrak{O}$.

These conditions state that the matrix $g$ has entries in $\mathfrak{O}$, and its reduction modulo $\varpi$ is an upper triangular matrix. For $g$ to be an invertible element of $GL_n(\mathfrak{O})$, its determinant must be a unit in $\mathfrak{O}$, i.e., $\det(g) \in \mathfrak{O}^\times$.
Modulo $\varpi$, $\det(g) \equiv \prod_{i=1}^n g_{ii} \pmod{\varpi}$. Thus, $\det(g) \in \mathfrak{O}^\times$ if and only if $g_{ii} \in \mathfrak{O}^\times$ for all $i=1, \dots, n$.
So, $I = \{ g=(g_{ij}) \in GL_n(\mathfrak{O}) \mid g_{ij} \in \varpi\mathfrak{O} \text{ for } i>j \}$.

**Step 2: Reduction to Primitive Vectors**

The action of $I$ on $V$ is given by left matrix multiplication. The zero vector $\{0\}$ is a trivial orbit. Let $v \in V \setminus \{0\}$. Let $m = \min_{i=1, \dots, n} \{\text{val}(v_i)\}$, where $\text{val}$ is the valuation on $F$. We can write $v = \varpi^m v_0$, where $v_0 \in \mathfrak{O}^n$ and at least one component of $v_0$ is in $\mathfrak{O}^\times$. Such a vector $v_0$ is called primitive. Let $V_0 = \mathfrak{O}^n \setminus \varpi\mathfrak{O}^n$ be the set of primitive vectors.

For any $g \in I$, $g(\varpi^m v_0) = \varpi^m (g v_0)$. This means that the orbit of $v$, $I \cdot v$, is $\varpi^m$ times the orbit of $v_0$, $I \cdot v_0$. The map $v \mapsto \varpi^{-m}v$ establishes an $I$-equivariant bijection between the set of vectors with minimal valuation $m$ and the set of primitive vectors $V_0$.
Therefore, the classification of orbits on $V$ reduces to classifying the orbits on $V_0$. A full set of representatives for $I \backslash V$ will be $\{0\} \cup \{\varpi^m r \mid m \in \mathbb{Z}, r \in R_0\}$, where $R_0$ is a set of representatives for $I \backslash V_0$.

**Step 3: The Orbit Invariant $k(v)$**

Let $v = (v_1, \dots, v_n)^T \in V_0$. By definition of $V_0$, each $v_i \in \mathfrak{O}$ and there is at least one $i$ such that $v_i \in \mathfrak{O}^\times$. Let's define an integer associated to $v$:
\[ k(v) = \max\{i \in \{1, \dots, n\} \mid v_i \in \mathfrak{O}^\times \}. \]
This means $v_{k(v)} \in \mathfrak{O}^\times$ and for all $j > k(v)$, $v_j \in \varpi\mathfrak{O}$.

We now prove that $k(v)$ is an invariant of the $I$-action. Let $v \in V_0$, $g \in I$, and $v' = gv$. Let $k = k(v)$.
For any $i > k$, the $i$-th component of $v'$ is $v'_i = \sum_{j=1}^n g_{ij} v_j$. We split the sum:
\[ v'_i = \sum_{j=1}^k g_{ij} v_j + \sum_{j=k+1}^n g_{ij} v_j. \]
For the first sum, $j \le k < i$, so $i > j$, which implies $g_{ij} \in \varpi\mathfrak{O}$. Since $v_j \in \mathfrak{O}$, this sum is in $\varpi\mathfrak{O}$.
For the second sum, $j > k$, so by definition of $k$, $v_j \in \varpi\mathfrak{O}$. Since $g_{ij} \in \mathfrak{O}$, this sum is also in $\varpi\mathfrak{O}$.
Thus, for all $i > k$, $v'_i \in \varpi\mathfrak{O}$.

Now consider the $k$-th component of $v'$:
\[ v'_k = \sum_{j=1}^{k-1} g_{kj} v_j + g_{kk} v_k + \sum_{j=k+1}^n g_{kj} v_j. \]
For $j < k$, we have $k > j$, so $g_{kj} \in \varpi\mathfrak{O}$. The first sum is in $\varpi\mathfrak{O}$.
The middle term is a product of two units, $g_{kk} \in \mathfrak{O}^\times$ and $v_k \in \mathfrak{O}^\times$, so $g_{kk}v_k \in \mathfrak{O}^\times$.
For $j > k$, we have $v_j \in \varpi\mathfrak{O}$. Since $k \le j$, $g_{kj} \in \mathfrak{O}$. The third sum is in $\varpi\mathfrak{O}$.
So, $v'_k \in \varpi\mathfrak{O} + \mathfrak{O}^\times + \varpi\mathfrak{O} = \mathfrak{O}^\times$.
This shows that the largest index $i$ for which $v'_i$ is a unit is $k$. Therefore, $k(v') = k(v)$.

The invariant $k(v)$ partitions $V_0$ into $n$ disjoint $I$-stable sets $V_{0,k} = \{v \in V_0 \mid k(v)=k\}$ for $k=1, \dots, n$.

**Step 4: Canonical Representative for each Orbit**

We now show that for each $k \in \{1, \dots, n\}$, any vector $v \in V_{0,k}$ is in the orbit of the standard basis vector $e_k$. The proof is constructive. Let $v \in V_{0,k}$. By definition, $v_k \in \mathfrak{O}^\times$, $v_j \in \varpi\mathfrak{O}$ for $j>k$, and $v_i \in \mathfrak{O}$ for $i<k$. We will find an element $g \in I$ such that $gv=e_k$ by applying a sequence of transformations.

1.  **Annihilating components $i < k$:**
    For each $i \in \{1, \dots, k-1\}$, let $c_i = v_i v_k^{-1}$. Since $v_i \in \mathfrak{O}$ and $v_k^{-1} \in \mathfrak{O}^\times$, we have $c_i \in \mathfrak{O}$.
    Let $g_1 = I_n - \sum_{i=1}^{k-1} c_i E_{ik}$, where $E_{ik}$ is the elementary matrix with a 1 in position $(i,k)$. Since $i<k$, the non-zero off-diagonal entries $-c_i \in \mathfrak{O}$ are all above the main diagonal. The diagonal entries of $g_1$ are all 1. Thus, $g_1 \in I$.
    Let $v^{(1)} = g_1 v$. For $j \ge k$, the $j$-th component is $(v^{(1)})_j = v_j$. For $i < k$, the $i$-th component is $(v^{(1)})_i = v_i - c_i v_k = v_i - (v_i v_k^{-1})v_k = 0$.
    So, $v^{(1)} = (0, \dots, 0, v_k, v_{k+1}, \dots, v_n)^T$.

2.  **Annihilating components $j > k$:**
    Now we annihilate the components of $v^{(1)}$ with index greater than $k$. For each $j \in \{k+1, \dots, n\}$, let $d_j = v_j v_k^{-1}$. By hypothesis on $v$, $v_j \in \varpi\mathfrak{O}$ for $j>k$. Since $v_k^{-1} \in \mathfrak{O}^\times$, we have $d_j \in \varpi\mathfrak{O}$.
    Let $g_2 = I_n - \sum_{j=k+1}^{n} d_j E_{jk}$. Since $j>k$, the non-zero off-diagonal entries $-d_j$ are all below the main diagonal. As $d_j \in \varpi\mathfrak{O}$, and the diagonal entries are all 1, the matrix $g_2$ is in $I$.
    Let $v^{(2)} = g_2 v^{(1)}$. For $i \le k$, the $i$-th component is $(v^{(2)})_i = (v^{(1)})_i$. For $j > k$, the $j$-th component becomes $(v^{(2)})_j = (v^{(1)})_j - d_j (v^{(1)})_k = v_j - (v_j v_k^{-1})v_k = 0$.
    So, $v^{(2)} = (0, \dots, 0, v_k, 0, \dots, 0)^T = v_k e_k$.

3.  **Normalization:**
    Finally, we normalize the $k$-th component. Let $g_3$ be the diagonal matrix with $(g_3)_{kk} = v_k^{-1}$ and $(g_3)_{ii} = 1$ for $i \neq k$. Since $v_k^{-1} \in \mathfrak{O}^\times$, all diagonal entries of $g_3$ are units in $\mathfrak{O}$, so $g_3 \in I$.
    Let $v^{(3)} = g_3 v^{(2)}$. Then $v^{(3)} = g_3 (v_k e_k) = v_k (g_3 e_k) = v_k (v_k^{-1} e_k) = e_k$.

Let $g = g_3 g_2 g_1$. Since $g_1, g_2, g_3 \in I$, their product $g$ is also in $I$. We have shown that $gv = e_k$. This proves that every vector $v \in V_{0,k}$ is in the orbit of $e_k$.

**Step 5: Conclusion**

We have shown that any vector in $V_0$ is in the orbit of some $e_k$ for $k \in \{1, \dots, n\}$. To see that $\{e_1, \dots, e_n\}$ is a complete set of representatives, we must check they lie in different orbits. We use our invariant: for the vector $e_k$, its components are $(\delta_{ik})_{i=1}^n$. The last component which is a unit is the $k$-th component (which is 1). Thus, $k(e_k) = k$. Since $e_1, \dots, e_n$ all have different values for the invariant $k$, they must lie in different orbits.
So, $\{e_1, \dots, e_n\}$ is a complete set of representatives for $I \backslash V_0$.

Combining this with the reduction in Step 2, we find that a complete set of representatives for the orbits of $I$ on $V$ is:
\[ \{0\} \cup \{ \varpi^m e_k \mid m \in \mathbb{Z}, k \in \{1, \dots, n\} \}. \]
The orbits can be parameterized by a pair $(m,k) \in \mathbb{Z} \times \{1, \dots, n\}$, corresponding to the representative $\varpi^m e_k$, plus a separate orbit for the zero vector.



## Verification Report

### Summary ###

**Final Verdict:** The solution is correct.

**List of Findings:**
*   No issues were found. The solution provides a complete and rigorous classification of the Iwahori orbits on the vector space $V$.

### Detailed Verification Log ###

**Step 1: The Iwahori Subgroup**

*   **Quote:** "The Iwahori subgroup $I$ is defined as the set of elements $g \in GL_n(F)$ such that for each $j \in \{1, \dots, n\}$, $g(e_j) \in \sum_{i \le j} \mathfrak{O}e_i + \sum_{i>j}\varpi \mathfrak{O} e_i$. ... These conditions state that the matrix $g$ has entries in $\mathfrak{O}$, and its reduction modulo $\varpi$ is an upper triangular matrix. ... $\det(g) \in \mathfrak{O}^\times$ if and only if $g_{ii} \in \mathfrak{O}^\times$ for all $i=1, \dots, n$. So, $I = \{ g=(g_{ij}) \in GL_n(\mathfrak{O}) \mid g_{ij} \in \varpi\mathfrak{O} \text{ for } i>j \}$."
*   **Assessment:** Correct. The solution correctly interprets the definition of the Iwahori subgroup given in the problem statement. The translation into matrix conditions ($g_{ij} \in \mathfrak{O}$ for $i \le j$, $g_{ij} \in \varpi\mathfrak{O}$ for $i > j$) is accurate. The deduction that this corresponds to matrices in $GL_n(\mathfrak{O})$ that are upper triangular modulo $\varpi$ is also correct. The condition for invertibility, namely that the diagonal entries must be units, is correctly derived from the determinant condition.

**Step 2: Reduction to Primitive Vectors**

*   **Quote:** "Let $v \in V \setminus \{0\}$. Let $m = \min_{i=1, \dots, n} \{\text{val}(v_i)\}$. We can write $v = \varpi^m v_0$, where $v_0 \in \mathfrak{O}^n$ and at least one component of $v_0$ is in $\mathfrak{O}^\times$. ... For any $g \in I$, $g(\varpi^m v_0) = \varpi^m (g v_0)$. ... Therefore, the classification of orbits on $V$ reduces to classifying the orbits on $V_0$. A full set of representatives for $I \backslash V$ will be $\{0\} \cup \{\varpi^m r \mid m \in \mathbb{Z}, r \in R_0\}$, where $R_0$ is a set of representatives for $I \backslash V_0$."
*   **Assessment:** Correct. The argument for reducing the problem to the classification of orbits on the set of primitive vectors $V_0 = \mathfrak{O}^n \setminus \varpi\mathfrak{O}^n$ is sound. The action of $I$ commutes with scalar multiplication, so the orbit of $v = \varpi^m v_0$ is determined by the orbit of $v_0$. The argument implicitly relies on the fact that the action of $I$ preserves the set of primitive vectors (i.e., if $v_0$ is primitive, so is $gv_0$ for $g \in I$). This fact is a direct consequence of the calculations in Step 3, which show that if $v_0$ has a unit component, so does $gv_0$, and that $gv_0$ remains in $\mathfrak{O}^n$. The logic is therefore complete.

**Step 3: The Orbit Invariant $k(v)$**

*   **Quote:** "Let $v = (v_1, \dots, v_n)^T \in V_0$. ... Let's define an integer associated to $v$: $k(v) = \max\{i \in \{1, \dots, n\} \mid v_i \in \mathfrak{O}^\times \}$. ... We now prove that $k(v)$ is an invariant of the $I$-action."
*   **Assessment:** Correct. The function $k(v)$ is well-defined for any primitive vector $v \in V_0$. The proof of its invariance under the action of $I$ is rigorous and detailed. The analysis of the components of $v' = gv$ is broken down correctly.
    *   For $i > k(v)$, the proof correctly shows $v'_i \in \varpi\mathfrak{O}$ by analyzing the terms in the sum $v'_i = \sum_j g_{ij}v_j$ and using the properties of $g \in I$ and the definition of $k(v)$.
    *   For $i = k(v)$, the proof correctly shows $v'_k \in \mathfrak{O}^\times$ by demonstrating that $v'_k$ is the sum of a unit ($g_{kk}v_k$) and terms in $\varpi\mathfrak{O}$.
    These two facts together establish that $k(gv) = k(v)$, so $k(v)$ is indeed an orbit invariant.

**Step 4: Canonical Representative for each Orbit**

*   **Quote:** "We now show that for each $k \in \{1, \dots, n\}$, any vector $v \in V_{0,k}$ is in the orbit of the standard basis vector $e_k$. The proof is constructive."
*   **Assessment:** Correct. The solution provides a clear, constructive proof that any vector $v$ in the set $V_{0,k} = \{v \in V_0 \mid k(v)=k\}$ can be mapped to $e_k$ by an element of $I$. This proves that all vectors in $V_{0,k}$ belong to the same orbit, which must be the orbit of $e_k$ (since $k(e_k)=k$). The construction is performed in three steps, each using a transformation matrix that is correctly shown to be in $I$.
    1.  The matrix $g_1$ correctly annihilates the components $v_i$ for $i<k$. It is shown to be in $I$ because its off-diagonal entries are above the main diagonal and its diagonal entries are units.
    2.  The matrix $g_2$ correctly annihilates the components $v_j$ for $j>k$. It is shown to be in $I$ because its off-diagonal entries are below the main diagonal and belong to $\varpi\mathfrak{O}$, and its diagonal entries are units.
    3.  The matrix $g_3$ correctly normalizes the $k$-th component to 1. It is a diagonal matrix with units on the diagonal and is therefore in $I$.
The product $g = g_3 g_2 g_1$ is in $I$ because $I$ is a group, and $gv = e_k$. This implies $v = g^{-1}e_k$, so $v$ is in the orbit of $e_k$. The entire construction is flawless.

**Step 5: Conclusion**

*   **Quote:** "We have shown that any vector in $V_0$ is in the orbit of some $e_k$ for $k \in \{1, \dots, n\}$. To see that $\{e_1, \dots, e_n\}$ is a complete set of representatives, we must check they lie in different orbits. We use our invariant: ... $k(e_k) = k$. Since $e_1, \dots, e_n$ all have different values for the invariant $k$, they must lie in different orbits. ... Combining this with the reduction in Step 2, we find that a complete set of representatives for the orbits of $I$ on $V$ is: $\{0\} \cup \{ \varpi^m e_k \mid m \in \mathbb{Z}, k \in \{1, \dots, n\} \}$."
*   **Assessment:** Correct. The conclusion logically follows from the previous steps. The invariant $k(v)$ is correctly used to demonstrate that the orbits represented by $e_1, \dots, e_n$ are distinct. This establishes that $\{e_1, \dots, e_n\}$ is a complete set of representatives for the orbits on $V_0$. The final step of scaling these representatives by powers of the uniformizer $\varpi^m$ and including the zero orbit provides the correct and complete classification for the orbits on all of $V$.

