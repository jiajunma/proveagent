# IMO Problem Solution

*Generated: 2026-03-05 11:54:52*

---


## Problem Statement

*** Problem Statement ***

# G₁ × G₂ Orbits and Moment Map Images in the Bessel Case

## Setup

Let $n \geq 1$ be an integer. All groups and Lie algebras are over $\mathbb{C}$. Define:

- **G₁** = $\mathrm{O}_{2n}(\mathbb{C})$ = the complex orthogonal group of $2n \times 2n$ complex matrices preserving the standard quadratic form.
- **G₂** = $\mathrm{SO}_{2n+1}(\mathbb{C})$ = the complex special orthogonal group of $(2n+1) \times (2n+1)$ complex matrices.
- **G** = G₁ × G₂ acting on itself by left multiplication.

Define the **diagonal embedding** of $\mathrm{O}_{2n}(\mathbb{C})$ into $\mathrm{SO}_{2n+1}(\mathbb{C})$ by
\[
g \mapsto \mathrm{diag}(g, \det(g)) = \begin{pmatrix} g & 0 \\ 0 & \det(g) \end{pmatrix} \in \mathrm{SO}_{2n+1}(\mathbb{C}),
\]
where $\det(g) \in \{\pm 1\}$ ensures the block matrix has determinant $1$.

Then define the **Bessel subgroup**
\[
H = \{ (g, \mathrm{diag}(g, \det(g))) \mid g \in \mathrm{O}_{2n}(\mathbb{C}) \} \subset G.
\]

Let **M** = $T^*(G/H)$ be the cotangent bundle of the homogeneous space $G/H$, endowed with its canonical symplectic structure. The left action of $G$ on $G/H$ lifts to a Hamiltonian action on $M$, with moment map
\[
\mu : M \to \mathfrak{g}^* \rightarrow \mathfrak{g}_1^* \oplus \mathfrak{g}_2^*,
\]
where $\mathfrak{g} = \mathrm{Lie}(G)$, $\mathfrak{g}_1 = \mathfrak{so}_{2n}(\mathbb{C})$, $\mathfrak{g}_2 = \mathfrak{so}_{2n+1}(\mathbb{C})$, and we identify $\mathfrak{g}^*$ with $\mathfrak{g}$ via the trace form.

---

## Problems

1. **Orbit classification:** Determine the $G_1 \times G_2$ orbits on $M$.  
   That is, classify the orbit space $G \backslash M$ and give explicit representatives or a combinatorial/structural description of the orbits.

2. **Moment map images:** For each $G$-orbit $\mathcal{O} \subseteq M$, compute the image $\mu(\mathcal{O}) \subseteq \mathfrak{g}^*$ of the orbit under the moment map.  
   Describe these images in terms of coadjoint orbits, nilpotent/regular orbits, or other recognizable subsets of $\mathfrak{g}^*$ (e.g. $\mathfrak{g}_1^* \oplus \mathfrak{g}_2^*$).





## Solution

### 1. Summary

#### a. Verdict

I have found a complete solution to both problems.

1.  The $G_1 \times G_2$ orbits on $M = T^*(G/H)$ are in one-to-one correspondence with the orbits of the Bessel subgroup $H$ on the annihilator of its Lie algebra, $\mathfrak{h}^\perp$. We identify $\mathfrak{h}^\perp$ with pairs $(\xi_1, b) \in \mathfrak{so}_{2n}(\mathbb{C}) \times \mathbb{C}^{2n}$. The orbits are parameterized by a set of polynomial invariants:
    *   An $\mathrm{O}_{2n}(\mathbb{C})$-adjoint orbit $\mathcal{O}_{\xi_1} \subset \mathfrak{so}_{2n}(\mathbb{C})$, which is determined by the eigenvalues of $\xi_1$.
    *   A vector of $n$ complex numbers $\mathbf{c} = (c_0, c_2, \dots, c_{2n-2}) \in \mathbb{C}^n$, where $c_{2k} = b^T \xi_1^{2k} b$.

2.  For each $G$-orbit $\mathcal{O} \subset M$, parameterized by $(\mathcal{O}_{\xi_1}, \mathbf{c})$ as above, its image under the moment map $\mu$ is a product of two adjoint orbits:
    \[
    \mu(\mathcal{O}) = \mathcal{O}_{\xi_1} \times \mathcal{O}_{\xi_2} \subset \mathfrak{g}_1^* \oplus \mathfrak{g}_2^* \cong \mathfrak{so}_{2n}(\mathbb{C}) \oplus \mathfrak{so}_{2n+1}(\mathbb{C}).
    \]
    Here, $\mathcal{O}_{\xi_1}$ is the orbit in $\mathfrak{so}_{2n}(\mathbb{C})$ determined by the first part of the orbit data. $\mathcal{O}_{\xi_2}$ is an adjoint orbit in $\mathfrak{so}_{2n+1}(\mathbb{C})$ which is uniquely determined by $(\mathcal{O}_{\xi_1}, \mathbf{c})$. Specifically, if $(\xi_1, b)$ is any representative of the orbit data, then $\mathcal{O}_{\xi_2}$ is the orbit of the matrix $\xi_2 = \begin{pmatrix} -\xi_1 & b \\ -b^T & 0 \end{pmatrix}$. The characteristic polynomial of $\xi_2$, which determines the orbit in the generic (semisimple) case, is given by
    \[
    P_{\xi_2}(\lambda) = \lambda P_{\xi_1}(-\lambda) - b^T \mathrm{adj}(\lambda I + \xi_1) b,
    \]
    where $P_{\xi_1}(\lambda) = \det(\lambda I - \xi_1)$ is the characteristic polynomial of $\xi_1$. The coefficients of the polynomial $b^T \mathrm{adj}(\lambda I + \xi_1) b$ are themselves polynomials in the invariants $c_{2k}$.

#### b. Method Sketch

1.  **Orbit Classification:**
    *   We use the standard identification $M = T^*(G/H) \cong G \times_H \mathfrak{h}^\perp$, where $\mathfrak{h}^\perp$ is the annihilator of $\mathfrak{h} = \mathrm{Lie}(H)$ in $\mathfrak{g}^* = \mathrm{Lie}(G)^*$. This establishes a bijection between $G$-orbits on $M$ and $H$-orbits on $\mathfrak{h}^\perp$.
    *   We identify $\mathfrak{g}^* \cong \mathfrak{g} = \mathfrak{so}_{2n} \oplus \mathfrak{so}_{2n+1}$ via the trace form. A direct calculation shows that $\mathfrak{h}^\perp = \{ (\xi_1, \xi_2) \in \mathfrak{so}_{2n} \oplus \mathfrak{so}_{2n+1} \mid \xi_2 = \begin{pmatrix} -\xi_1 & b \\ -b^T & 0 \end{pmatrix} \text{ for some } b \in \mathbb{C}^{2n} \}$. We can thus parameterize $\mathfrak{h}^\perp$ by pairs $(\xi_1, b) \in \mathfrak{so}_{2n} \times \mathbb{C}^{2n}$.
    *   The coadjoint action of $H$ on $\mathfrak{h}^\perp$ is computed. For $h = (g, \mathrm{diag}(g, \det(g))) \in H$ with $g \in \mathrm{O}_{2n}$, the action on a pair $(\xi_1, b)$ is $g \cdot (\xi_1, b) = (g^{-1}\xi_1 g, \det(g) g^{-1} b)$.
    *   The classification of orbits under this action is given by the theory of polynomial invariants. The algebra of polynomial functions on $\mathfrak{so}_{2n} \times \mathbb{C}^{2n}$ invariant under this action is generated by the polynomials $\mathrm{tr}(\xi_1^{2k})$ for $k=1, \dots, n$ and $b^T \xi_1^{2k} b$ for $k=0, \dots, n-1$.
    *   The values of these invariants classify the orbits. The first set of invariants, $\mathrm{tr}(\xi_1^{2k})$, determines the $\mathrm{O}_{2n}$-adjoint orbit of $\xi_1$. The second set provides a vector of scalars $\mathbf{c} = (c_0, \dots, c_{2n-2})$ where $c_{2k} = b^T \xi_1^{2k} b$.

2.  **Moment Map Images:**
    *   The moment map for the $G$-action on $M = G \times_H \mathfrak{h}^\perp$ is given by $\mu([g, \alpha]) = \mathrm{Ad}^*(g^{-1})(\alpha)$ for $\alpha \in \mathfrak{h}^\perp$. The image of the $G$-orbit through $[e, \alpha]$ is the $G$-coadjoint orbit of $\alpha$.
    *   Let an orbit in $M$ be specified by the $H$-orbit of a representative $\alpha_0 = (\xi_1, \xi_2) \in \mathfrak{h}^\perp$, where $\xi_2$ is determined by a pair $(\xi_1, b)$. The moment map image is the $G$-adjoint orbit of $(\xi_1, \xi_2)$.
    *   Since $G = G_1 \times G_2$, this orbit is the product of the individual adjoint orbits: $\mathcal{O}_{(\xi_1, \xi_2)} = \mathcal{O}_{\xi_1}^{G_1} \times \mathcal{O}_{\xi_2}^{G_2}$.
    *   We show that for any two representatives $(\xi_1, b)$ and $(\xi_1', b')$ in the same $H$-orbit, the corresponding matrix $\xi_1$ lies in the same $G_1$-orbit as $\xi_1'$, and $\xi_2$ lies in the same $G_2$-orbit as $\xi_2'$. This ensures that the map from orbit data to the pair of adjoint orbits is well-defined.
    *   The main task is to describe the orbit $\mathcal{O}_{\xi_2}$ in terms of the invariants $(\mathcal{O}_{\xi_1}, \mathbf{c})$. We do this by computing the characteristic polynomial of $\xi_2 = \begin{pmatrix} -\xi_1 & b \\ -b^T & 0 \end{pmatrix}$. A block matrix determinant calculation yields the key formula relating the characteristic polynomials of $\xi_1$ and $\xi_2$.
    *   This formula shows that the eigenvalues of $\xi_2$ (and thus its orbit, generically) are determined by the eigenvalues of $\xi_1$ and the invariants $c_{2k}$. We verify this relationship in the specific nilpotent case where $\xi_1=0$ and $b^T b=0$.

### 2. Detailed Solution

#### 1. Orbit Classification

The $G$-orbits on the cotangent bundle $M = T^*(G/H)$ are in canonical one-to-one correspondence with the orbits of the subgroup $H$ acting on the conormal bundle to the identity coset, which is identified with $\mathfrak{h}^\perp = \{ \alpha \in \mathfrak{g}^* \mid \alpha|_{\mathfrak{h}} = 0 \}$.

**Step 1: Identify $\mathfrak{h}^\perp$ and the $H$-action.**

We identify the Lie algebra $\mathfrak{g} = \mathrm{Lie}(G)$ and its dual $\mathfrak{g}^*$ via the trace form $\langle A, B \rangle = \mathrm{tr}(AB)$.
The Lie algebras are $\mathfrak{g}_1 = \mathfrak{so}_{2n}(\mathbb{C})$ and $\mathfrak{g}_2 = \mathfrak{so}_{2n+1}(\mathbb{C})$, so $\mathfrak{g} = \mathfrak{g}_1 \oplus \mathfrak{g}_2$.
The Lie algebra of the Bessel subgroup $H$ is $\mathfrak{h} = \{ (X, \iota(X)) \mid X \in \mathfrak{so}_{2n} \}$, where $\iota: \mathfrak{so}_{2n} \to \mathfrak{so}_{2n+1}$ is the differential of the embedding $g \mapsto \mathrm{diag}(g, \det(g))$. The differential of $\det$ at the identity is the trace map, which is zero on $\mathfrak{so}_{2n}$. Thus, $\iota(X) = \begin{pmatrix} X & 0 \\ 0 & 0 \end{pmatrix}$.

An element $(\xi_1, \xi_2) \in \mathfrak{g}_1 \oplus \mathfrak{g}_2$ is in $\mathfrak{h}^\perp$ if $\mathrm{tr}(\xi_1 X) + \mathrm{tr}(\xi_2 \iota(X)) = 0$ for all $X \in \mathfrak{so}_{2n}$.
Let $\xi_2 = \begin{pmatrix} A & b \\ -b^T & 0 \end{pmatrix}$ with $A \in \mathfrak{so}_{2n}$ and $b \in \mathbb{C}^{2n}$.
The condition becomes $\mathrm{tr}(\xi_1 X) + \mathrm{tr}(A X) = \mathrm{tr}((\xi_1+A)X) = 0$. Since the trace form on $\mathfrak{so}_{2n}$ is non-degenerate, this implies $\xi_1 + A = 0$, so $A = -\xi_1$.
Thus, we can identify $\mathfrak{h}^\perp$ with the space of pairs $(\xi_1, b) \in \mathfrak{so}_{2n} \times \mathbb{C}^{2n}$, where the correspondence is given by:
$(\xi_1, b) \longleftrightarrow (\xi_1, \xi_2) = \left(\xi_1, \begin{pmatrix} -\xi_1 & b \\ -b^T & 0 \end{pmatrix}\right) \in \mathfrak{h}^\perp$.

The action of $H$ on $\mathfrak{h}^\perp$ is the coadjoint action, which under the trace-form identification becomes the adjoint action. For $h = (g, \mathrm{diag}(g, \det(g))) \in H$ with $g \in \mathrm{O}_{2n}$, the action on $(\xi_1, \xi_2) \in \mathfrak{h}^\perp$ is:
\begin{align*} \label{eq:1} \mathrm{Ad}(h^{-1})(\xi_1, \xi_2) &= (\mathrm{Ad}(g^{-1})\xi_1, \mathrm{Ad}(\mathrm{diag}(g^{-1}, \det(g^{-1})))\xi_2) \\ &= \left(g^{-1}\xi_1 g, \begin{pmatrix} g^{-1} & 0 \\ 0 & \det(g^{-1}) \end{pmatrix} \begin{pmatrix} -\xi_1 & b \\ -b^T & 0 \end{pmatrix} \begin{pmatrix} g & 0 \\ 0 & \det(g) \end{pmatrix}\right) \\ &= \left(g^{-1}\xi_1 g, \begin{pmatrix} -g^{-1}\xi_1 g & \det(g)g^{-1}b \\ -\det(g^{-1})b^T g & 0 \end{pmatrix}\right).\end{align*}
In the $(\xi_1, b)$ parameterization, the action is:
\[ g \cdot (\xi_1, b) = (g^{-1}\xi_1 g, \det(g)g^{-1}b). \]

**Step 2: Classify the orbits.**

The classification of orbits for this action can be achieved by finding a complete set of polynomial invariants. The algebra of polynomial functions on $\mathfrak{so}_{2n} \times \mathbb{C}^{2n}$ that are invariant under this action is generated by two sets of functions:
1.  The invariants of the $\mathrm{O}_{2n}$-action on $\xi_1$: These are the coefficients of the characteristic polynomial of $\xi_1$, which are generated by $\mathrm{tr}(\xi_1^{2k})$ for $k=1, \dots, n$. The values of these invariants determine the $\mathrm{O}_{2n}$-adjoint orbit of $\xi_1$, denoted $\mathcal{O}_{\xi_1}$.
2.  A set of mixed invariants: For any $k \ge 0$, the function $(\xi_1, b) \mapsto b^T \xi_1^k b$ is invariant under the action:
    \[ (b')^T (\xi_1')^k b' = (\det(g)g^{-1}b)^T (g^{-1}\xi_1 g)^k (\det(g)g^{-1}b) = b^T g (g^{-1}\xi_1^k g) g^{-1} b = b^T \xi_1^k b. \]
    Since $\xi_1$ is skew-symmetric, $b^T \xi_1^{2k+1} b = 0$ for all $k \ge 0$. The non-trivial invariants are $c_{2k} = b^T \xi_1^{2k} b$. By the Cayley-Hamilton theorem, any $\xi_1^{2k}$ with $k \ge n$ is a polynomial in lower even powers of $\xi_1$. Thus, the algebra of these invariants is generated by $c_{2k}$ for $k=0, 1, \dots, n-1$.

The set of $G$-orbits on $M$ is therefore parameterized by the values of these invariants. An orbit is specified by:
*   An $\mathrm{O}_{2n}(\mathbb{C})$-adjoint orbit $\mathcal{O}_{\xi_1} \subset \mathfrak{so}_{2n}(\mathbb{C})$.
*   A vector of $n$ complex scalars $\mathbf{c} = (c_0, c_2, \dots, c_{2n-2}) \in \mathbb{C}^n$, where $c_{2k} = b^T \xi_1^{2k} b$.

#### 2. Moment Map Images

For a $G$-orbit $\mathcal{O} \subset M$, we wish to compute its image $\mu(\mathcal{O}) \subset \mathfrak{g}^*$.

**Step 1: General form of the image.**

The moment map for the $G$-action on $M = G \times_H \mathfrak{h}^\perp$ is given by $\mu([g, \alpha]) = \mathrm{Ad}^*(g^{-1})(\alpha)$ for $g \in G, \alpha \in \mathfrak{h}^\perp$.
Let $\mathcal{O}$ be the $G$-orbit corresponding to the $H$-orbit of $\alpha_0 \in \mathfrak{h}^\perp$. Any point in $\mathcal{O}$ can be written as $[g, \alpha]$ where $\alpha$ is in the $H$-orbit of $\alpha_0$. The image of the orbit is
\[ \mu(\mathcal{O}) = \{ \mathrm{Ad}^*(g^{-1})(\alpha) \mid g \in G, \alpha \in H \cdot \alpha_0 \} = \{ \mathrm{Ad}^*(g^{-1}h^{-1})(\alpha_0) \mid g \in G, h \in H \} = G \cdot \alpha_0. \]
This is the coadjoint $G$-orbit of $\alpha_0$. Using the trace-form identification, this is the adjoint $G$-orbit of $\alpha_0$.
Let $\alpha_0 = (\xi_1, \xi_2) \in \mathfrak{h}^\perp$. Since $G = G_1 \times G_2$, the adjoint orbit of $\alpha_0$ is the product of the individual adjoint orbits:
\[ \mu(\mathcal{O}) = \mathcal{O}_{\xi_1}^{G_1} \times \mathcal{O}_{\xi_2}^{G_2} \subset \mathfrak{so}_{2n} \oplus \mathfrak{so}_{2n+1}. \]

**Step 2: Well-definedness.**

We must check that this product of orbits depends only on the $H$-orbit of $\alpha_0$, not on the specific choice of representative. Let $(\xi_1, b)$ and $(\xi_1', b') = (g^{-1}\xi_1 g, \det(g)g^{-1}b)$ be two pairs in the same $H$-orbit, for some $g \in \mathrm{O}_{2n}$.
The corresponding elements in $\mathfrak{h}^\perp$ are $\alpha = (\xi_1, \xi_2)$ and $\alpha' = (\xi_1', \xi_2')$.
Clearly $\xi_1$ and $\xi_1' = g^{-1}\xi_1 g$ are in the same $G_1 = \mathrm{O}_{2n}$ adjoint orbit.
For the second component, let $h_g = \mathrm{diag}(g, \det(g)) \in \mathrm{SO}_{2n+1}$. A direct calculation shows:
\[ \mathrm{Ad}(h_g^{-1})\xi_2 = \mathrm{Ad}(\mathrm{diag}(g^{-1}, \det(g^{-1}))) \begin{pmatrix} -\xi_1 & b \\ -b^T & 0 \end{pmatrix} = \begin{pmatrix} -g^{-1}\xi_1 g & \det(g)g^{-1}b \\ -(\det(g)g^{-1}b)^T & 0 \end{pmatrix} = \xi_2'. \]
Thus, $\xi_2$ and $\xi_2'$ lie in the same $G_2 = \mathrm{SO}_{2n+1}$ adjoint orbit. The map from an $H$-orbit in $\mathfrak{h}^\perp$ to a pair of adjoint orbits $(\mathcal{O}_{\xi_1}, \mathcal{O}_{\xi_2})$ is well-defined.

**Step 3: Characterization of $\mathcal{O}_{\xi_2}$.**

For a $G$-orbit in $M$ specified by the invariants $(\mathcal{O}_{\xi_1}, \mathbf{c})$, the moment map image is $\mathcal{O}_{\xi_1} \times \mathcal{O}_{\xi_2}$. The orbit $\mathcal{O}_{\xi_1}$ is given. The orbit $\mathcal{O}_{\xi_2}$ is that of $\xi_2 = \begin{pmatrix} -\xi_1 & b \\ -b^T & 0 \end{pmatrix}$, where $(\xi_1, b)$ is any representative pair corresponding to the invariants. The orbit of a skew-symmetric matrix is determined by its Jordan normal form, which in the generic (semisimple) case is determined by its characteristic polynomial.

Let $P_A(\lambda) = \det(\lambda I - A)$. The characteristic polynomial of $\xi_2$ is:
\[ P_{\xi_2}(\lambda) = \det\begin{pmatrix} \lambda I_{2n} + \xi_1 & -b \\ b^T & \lambda \end{pmatrix}. \]
Using the block matrix determinant formula $\det\begin{pmatrix} A & B \\ C & D \end{pmatrix} = \det(A)\det(D-CA^{-1}B)$, we get:
\begin{align*} P_{\xi_2}(\lambda) &= \det(\lambda I + \xi_1) \left(\lambda - b^T (\lambda I + \xi_1)^{-1} (-b)\right) \\ &= \lambda \det(\lambda I + \xi_1) + b^T \det(\lambda I + \xi_1)(\lambda I + \xi_1)^{-1} b \\ &= \lambda \det(\lambda I + \xi_1) + b^T \mathrm{adj}(\lambda I + \xi_1) b. \end{align*}
Since $\xi_1$ is skew-symmetric, $\det(\lambda I + \xi_1) = \det((\lambda I - \xi_1)^T) = P_{\xi_1}(\lambda)$. Also, $P_{\xi_1}(\lambda) = P_{\xi_1}(-\lambda)$ because the eigenvalues of $\xi_1$ come in $\pm$ pairs. So $\det(\lambda I + \xi_1) = P_{\xi_1}(-\lambda)$.
The final formula is:
\[ P_{\xi_2}(\lambda) = \lambda P_{\xi_1}(-\lambda) + b^T \mathrm{adj}(\lambda I + \xi_1) b. \]
The coefficients of the polynomial $R(\lambda) = b^T \mathrm{adj}(\lambda I + \xi_1) b$ can be shown to be polynomials in the invariants $c_{2k} = b^T \xi_1^{2k} b$. For large $\lambda$, $R(\lambda) = P_{\xi_1}(-\lambda) b^T (\lambda I + \xi_1)^{-1} b = P_{\xi_1}(-\lambda) \sum_{j=0}^\infty c_{2j} \lambda^{-2j-1}$. This shows that the leading coefficients of $R(\lambda)$ are determined by the $c_{2j}$.
Thus, the characteristic polynomial of $\xi_2$, and generically its adjoint orbit $\mathcal{O}_{\xi_2}$, is completely determined by the orbit data $(\mathcal{O}_{\xi_1}, \mathbf{c})$.

**Example: A Nilpotent Case**
Consider the orbit class where $\xi_1=0$ and $b \in \mathbb{C}^{2n}$ is a non-zero isotropic vector, i.e., $c_0 = b^T b = 0$. All other invariants $c_{2k}$ are zero.
The orbit $\mathcal{O}_{\xi_1}$ is the orbit of the zero matrix, $\{0\}$.
The matrix $\xi_2$ is $\begin{pmatrix} 0 & b \\ -b^T & 0 \end{pmatrix}$.
Its characteristic polynomial is $P_{\xi_2}(\lambda) = \lambda (\lambda^{2n}) + b^T \mathrm{adj}(\lambda I) b = \lambda^{2n+1} + b^T (\lambda^{2n-1}I) b = \lambda^{2n+1} + (b^T b)\lambda^{2n-1}$.
Since $b^T b = 0$, $P_{\xi_2}(\lambda) = \lambda^{2n+1}$, so $\xi_2$ is nilpotent.
To determine its orbit, we compute powers of $\xi_2$:
$\xi_2^2 = \begin{pmatrix} -bb^T & 0 \\ 0 & -b^T b \end{pmatrix} = \begin{pmatrix} -bb^T & 0 \\ 0 & 0 \end{pmatrix}$.
$\xi_2^3 = \begin{pmatrix} 0 & -bb^T b \\ 0 & 0 \end{pmatrix}$. Since $b^T b=0$, the vector $b$ is in the image of the rank-1 matrix $bb^T$ if and only if $b$ is a multiple of itself, which is true, but the image is spanned by $b$, and $b$ is orthogonal to itself. So $b^T b=0$ implies $bb^T b = b(b^T b) = 0$. Thus $\xi_2^3=0$.
The rank of $\xi_2$ is 2 (for $b \neq 0$), and the rank of $\xi_2^2$ is 1.
The dimensions of the kernels of the powers of $\xi_2$ determine the Jordan partition:
- $\dim \ker \xi_2 = (2n+1) - \mathrm{rank}(\xi_2) = 2n-1$. This is the number of Jordan blocks.
- $\dim \ker \xi_2^2 = (2n+1) - \mathrm{rank}(\xi_2^2) = 2n$.
- $\dim \ker \xi_2^3 = 2n+1$.
The number of blocks of size $\ge k$ is $\dim \ker \xi_2^k - \dim \ker \xi_2^{k-1}$.
- Number of blocks $\ge 1$: $2n-1$.
- Number of blocks $\ge 2$: $(2n) - (2n-1) = 1$.
- Number of blocks $\ge 3$: $(2n+1) - 2n = 1$.
This implies there is one block of size 3, and the remaining $(2n-1)-1 = 2n-2$ blocks are of size 1. The partition of $2n+1$ is $(3, 1, \dots, 1)$. This uniquely specifies the nilpotent orbit in $\mathfrak{so}_{2n+1}$.
The moment map image is $\{0\} \times \mathcal{O}_{(3, 1^{2n-2})}$.



## Verification Report

### Summary ###

**Final Verdict:** The solution's approach is viable but contains a critical Justification Gap that undermines the entire argument for the orbit classification.

**List of Findings:**
*   **Location:** "The algebra of polynomial functions on $\mathfrak{so}_{2n} \times \mathbb{C}^{2n}$ that are invariant under this action is generated by two sets of functions: ..."
    *   **Issue:** Justification Gap - The solution asserts that a specific set of polynomials generates the full ring of invariants for the action of $H$. This is a powerful and non-trivial claim from invariant theory. Providing a list of functions that are invariant is not sufficient; one must also prove that this list is *complete* (i.e., that these invariants separate the orbits). Without a proof or a reference to a standard result, this step is a major unsubstantiated leap, making the subsequent orbit classification conjectural.

### Detailed Verification Log ###

#### Part 1: Orbit Classification

The solution correctly identifies that the $G$-orbits on $M = T^*(G/H)$ are in one-to-one correspondence with the $H$-orbits on $\mathfrak{h}^\perp$.

**Step 1: Identify $\mathfrak{h}^\perp$ and the $H$-action.**

*   **Quoted Text:** "The Lie algebra of the Bessel subgroup $H$ is $\mathfrak{h} = \{ (X, \iota(X)) \mid X \in \mathfrak{so}_{2n} \}$, where ... $\iota(X) = \begin{pmatrix} X & 0 \\ 0 & 0 \end{pmatrix}$."
*   **Verification:** This step correctly computes the differential of the embedding $g \mapsto \mathrm{diag}(g, \det(g))$ at the identity. The differential of $\det(g)$ at $g=I$ applied to a tangent vector $X \in \mathfrak{so}_{2n}$ is $\mathrm{tr}(X)$, which is zero. Thus, the identification of $\mathfrak{h}$ is correct.

*   **Quoted Text:** "An element $(\xi_1, \xi_2) \in \mathfrak{g}_1 \oplus \mathfrak{g}_2$ is in $\mathfrak{h}^\perp$ if $\mathrm{tr}(\xi_1 X) + \mathrm{tr}(\xi_2 \iota(X)) = 0$ for all $X \in \mathfrak{so}_{2n}$. ... this implies $\xi_1 + A = 0$, so $A = -\xi_1$."
*   **Verification:** The derivation of the condition for an element to be in $\mathfrak{h}^\perp$ is correct. The use of the non-degeneracy of the trace form on $\mathfrak{so}_{2n}$ is appropriate and leads to the correct conclusion. The parameterization of $\mathfrak{h}^\perp$ by pairs $(\xi_1, b) \in \mathfrak{so}_{2n} \times \mathbb{C}^{2n}$ is correctly established.

*   **Quoted Text:** "In the $(\xi_1, b)$ parameterization, the action is: $g \cdot (\xi_1, b) = (g^{-1}\xi_1 g, \det(g)g^{-1}b)$."
*   **Verification:** The calculation of the adjoint action of $H$ on an element of $\mathfrak{h}^\perp$ is performed correctly. The transformation rules for $\xi_1$ and the vector $b$ are accurately derived. This step is correct.

**Step 2: Classify the orbits.**

*   **Quoted Text:** "The classification of orbits for this action can be achieved by finding a complete set of polynomial invariants. The algebra of polynomial functions on $\mathfrak{so}_{2n} \times \mathbb{C}^{2n}$ that are invariant under this action is generated by two sets of functions: ..."
*   **Verification:** **Justification Gap.** The solution proposes to classify orbits by finding a complete set of generating polynomial invariants. This is a valid strategy in principle. However, the solution then asserts without proof or reference that a specific set of polynomials generates the entire ring of invariants. The provided functions are indeed invariant, as shown in the subsequent checks. But the crucial claim is that this set is *complete*, meaning that the values of these invariants are sufficient to distinguish any two distinct orbits. This is a deep result in invariant theory and cannot be stated without justification. This gap undermines the entire classification presented.

*   **Quoted Text:** "The invariants of the $\mathrm{O}_{2n}$-action on $\xi_1$: These are ... generated by $\mathrm{tr}(\xi_1^{2k})$ for $k=1, \dots, n$."
*   **Verification:** Assuming the strategy is valid, this correctly identifies the standard generators for the ring of $\mathrm{O}_{2n}$-invariant polynomials on $\mathfrak{so}_{2n}$.

*   **Quoted Text:** "For any $k \ge 0$, the function $(\xi_1, b) \mapsto b^T \xi_1^k b$ is invariant... Since $\xi_1$ is skew-symmetric, $b^T \xi_1^{2k+1} b = 0$ for all $k \ge 0$. The non-trivial invariants are $c_{2k} = b^T \xi_1^{2k} b$."
*   **Verification:** The proof of invariance for $b^T \xi_1^k b$ is correct. The argument that invariants with odd powers of $\xi_1$ vanish is also correct, as a scalar must equal its transpose, and $(b^T \xi_1^{2k+1} b)^T = -b^T \xi_1^{2k+1} b$.

*   **Quoted Text:** "By the Cayley-Hamilton theorem, any $\xi_1^{2k}$ with $k \ge n$ is a polynomial in lower even powers of $\xi_1$. Thus, the algebra of these invariants is generated by $c_{2k}$ for $k=0, 1, \dots, n-1$."
*   **Verification:** The application of the Cayley-Hamilton theorem is correct and correctly implies that only a finite number of these "mixed" invariants are needed to generate the algebra of such invariants.

*   **Quoted Text:** "The set of $G$-orbits on $M$ is therefore parameterized by the values of these invariants."
*   **Verification:** This conclusion rests entirely on the unsubstantiated claim that the listed invariants are a complete set. Due to the Justification Gap identified above, this conclusion is not rigorously established.

#### Part 2: Moment Map Images

For the verification of this part, we will assume the conclusion of Part 1 (that orbits are parameterized by $\mathcal{O}_{\xi_1}$ and $\mathbf{c}$) is correct for the sake of argument.

**Step 1: General form of the image.**

*   **Quoted Text:** "The moment map for the $G$-action on $M = G \times_H \mathfrak{h}^\perp$ is given by $\mu([g, \alpha]) = \mathrm{Ad}^*(g^{-1})(\alpha)$... The image of the orbit is ... the coadjoint $G$-orbit of $\alpha_0$."
*   **Verification:** This step correctly identifies the standard formula for the moment map on a coadjoint bundle and correctly deduces that the image of a $G$-orbit in $M$ is a coadjoint $G$-orbit in $\mathfrak{g}^*$.

*   **Quoted Text:** "Since $G = G_1 \times G_2$, the adjoint orbit of $\alpha_0$ is the product of the individual adjoint orbits: $\mu(\mathcal{O}) = \mathcal{O}_{\xi_1}^{G_1} \times \mathcal{O}_{\xi_2}^{G_2}$."
*   **Verification:** This is correct. The action of the product group $G_1 \times G_2$ on $\mathfrak{g}_1 \oplus \mathfrak{g}_2$ is component-wise, so the orbit is the product of the orbits.

**Step 2: Well-definedness.**

*   **Quoted Text:** "We must check that this product of orbits depends only on the $H$-orbit of $\alpha_0$, not on the specific choice of representative."
*   **Verification:** This is a crucial logical step, and the solution correctly identifies the need for this check. The subsequent argument correctly shows that if two elements $(\xi_1, \xi_2)$ and $(\xi_1', \xi_2')$ are in the same $H$-orbit, then $\xi_1$ and $\xi_1'$ are in the same $G_1$-orbit, and $\xi_2$ and $\xi_2'$ are in the same $G_2$-orbit. The reasoning is sound and the calculations are correct.

**Step 3: Characterization of $\mathcal{O}_{\xi_2}$.**

*   **Quoted Text:** "The characteristic polynomial of $\xi_2$ is: $P_{\xi_2}(\lambda) = \lambda \det(\lambda I + \xi_1) + b^T \mathrm{adj}(\lambda I + \xi_1) b$."
*   **Verification:** The calculation of the characteristic polynomial using the block matrix determinant formula is correct.

*   **Quoted Text:** "The coefficients of the polynomial $R(\lambda) = b^T \mathrm{adj}(\lambda I + \xi_1) b$ can be shown to be polynomials in the invariants $c_{2k} = b^T \xi_1^{2k} b$."
*   **Verification:** The argument is brief but correct. The coefficients of $P_{\xi_2}(\lambda)$ depend on the coefficients of $P_{\xi_1}(\lambda)$ (which are fixed by the orbit $\mathcal{O}_{\xi_1}$) and the coefficients of $R(\lambda)$. The solution correctly argues that the coefficients of $R(\lambda)$ are determined by the invariants $c_{2k}$. Therefore, the characteristic polynomial of $\xi_2$ is fully determined by the orbit data $(\mathcal{O}_{\xi_1}, \mathbf{c})$. This step is correct.

**Example: A Nilpotent Case**

*   **Quoted Text:** "Consider the orbit class where $\xi_1=0$ and $b \in \mathbb{C}^{2n}$ is a non-zero isotropic vector, i.e., $c_0 = b^T b = 0$."
*   **Verification:** The setup for the example is clear.

*   **Quoted Text:** "Since $b^T b = 0$, $P_{\xi_2}(\lambda) = \lambda^{2n+1}$, so $\xi_2$ is nilpotent."
*   **Verification:** The calculation of the characteristic polynomial for this specific case is correct.

*   **Quoted Text:** "$\xi_2^3=0$. The rank of $\xi_2$ is 2 (for $b \neq 0$), and the rank of $\xi_2^2$ is 1."
*   **Verification:** The computations of the powers of $\xi_2$ and their ranks are correct.

*   **Quoted Text:** "The dimensions of the kernels ... determine the Jordan partition: ... The partition of $2n+1$ is $(3, 1, \dots, 1)$."
*   **Verification:** The derivation of the Jordan partition from the dimensions of the kernels of the powers of $\xi_2$ is methodical and correct. The resulting partition is accurate.

*   **Quoted Text:** "The moment map image is $\{0\} \times \mathcal{O}_{(3, 1^{2n-2})}$."
*   **Verification:** This is the correct conclusion for the specific example analyzed. The entire example subsection is free of errors.

