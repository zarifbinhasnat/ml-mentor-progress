# 🎓 SPECIAL CLASS — ML QUIZ: Neural Networks From Scratch
**Source material:** *Lecture 6 — Neural Networks* (CSE 4621, Ishmam Tashdeed) | **Bonus session** — outside the daily roadmap, full deep dive

---

### How this session is organized

Today we go *wide* instead of deep-one-concept-per-day. We start at the very beginning — the **perceptron**, the single artificial neuron that started it all in 1957 — code it from scratch, watch it fail at something trivial, and use that failure as the motivating reason the entire rest of this lecture (and most of your roadmap) exists. Then we build a full **neural network from scratch** in NumPy, deriving every line of the forward and backward pass symbol-by-symbol exactly as the slides do.

Several topics in this lecture (activations, forward pass, cost functions, backprop, gradient descent, learning rate) **overlap with concepts you've already done** — wherever that happens you'll see a 🔗 link back to that day's lesson so you can cross-reference rather than re-derive. Several other topics (vanishing/exploding gradients, weight init, momentum, LR scheduling, dropout, batch norm) are **previews of concepts still ahead on your roadmap** — flagged so you know "this is coming, here's the trailer."

At the very end: a **Gauntlet of tricky questions** — conceptual and mathematical, pulled straight from the slide material, with a locked answer key at the bottom.

---

## 1. 🧩 The Perceptron — the atom of every neural net

### Intuition

In 1957, Frank Rosenblatt built a machine called the Mark I Perceptron — a single artificial neuron that takes a vector of inputs, multiplies each by a learned weight, sums them up (plus a bias), and outputs a decision. In modern notation:

$$\hat{y} = w^\top x + b$$

(or, for a *classification* perceptron, $\hat{y} = \text{sign}(w^\top x + b)$ — output $+1$ or $-1$/$0$ depending on which side of a hyperplane $x$ falls on).

Geometrically, $w^\top x + b = 0$ defines a **hyperplane** (a line, in 2D) that splits the input space into two halves. The perceptron can learn *any* such line — but only a line. This is the single most important limitation in the history of neural networks, and the slide makes it vivid with two truth tables:

| | OR | XOR |
|---|---|---|
| **(0,0)** | 0 | 0 |
| **(0,1)** | 1 | 1 |
| **(1,0)** | 1 | 1 |
| **(1,1)** | 1 | 0 |

Plot these four points on a 2D grid and color them by output (0 = red, 1 = green). For **OR**, you can draw a single straight line that separates the reds from the greens — **linearly separable**. For **XOR**, the greens are at $(0,1)$ and $(1,0)$ — diagonally opposite corners — and the reds are at $(0,0)$ and $(1,1)$, the *other* diagonal. **No single straight line can separate them.** This is exactly slide 3's picture.

> 🔗 **Already covered:** the single neuron's forward computation $z = w^\top x + b,\ a=\sigma(z)$ is Concept 1, [Day 1 — Artificial Neuron & Forward Pass](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_02_DAY_1.md).

### The Math — Perceptron Learning Rule

The classic Rosenblatt update, run one example at a time:

$$w \leftarrow w + \eta\,(y - \hat{y})\,x, \qquad b \leftarrow b + \eta\,(y-\hat{y})$$

If the prediction is correct ($y=\hat{y}$), nothing changes. If wrong, the weight vector is nudged toward the misclassified point (or away from it, depending on the sign of the error) — a primitive, pre-gradient-descent form of the update rule you now know from Concept 10/11.

**Convergence guarantee (Perceptron Convergence Theorem):** if the data *is* linearly separable, this update rule is guaranteed to find a separating hyperplane in a finite number of steps. If the data is **not** linearly separable (like XOR), the perceptron will *never converge* — it will oscillate forever, endlessly correcting one point at the expense of another.

---

## 2. 🐍 CODE — Perceptron From Scratch

```python
import numpy as np

class Perceptron:
    def __init__(self, n_inputs, lr=0.1):
        self.w = np.zeros(n_inputs)
        self.b = 0.0
        self.lr = lr

    def predict(self, x):
        z = self.w @ x + self.b
        return 1 if z >= 0 else 0

    def fit(self, X, y, epochs=20, verbose=True):
        history = []
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                y_hat = self.predict(xi)
                err = yi - y_hat
                self.w += self.lr * err * xi      # w <- w + eta*(y - yhat)*x
                self.b += self.lr * err
                errors += int(err != 0)
            history.append(errors)
            if verbose:
                print(f"epoch {epoch:2d}  misclassified={errors}  w={self.w}  b={self.b:.2f}")
        return history


X = np.array([[0,0], [0,1], [1,0], [1,1]])

# --- OR: linearly separable -> perceptron converges ---
y_or = np.array([0, 1, 1, 1])
print("=== Training on OR ===")
p_or = Perceptron(n_inputs=2)
p_or.fit(X, y_or, epochs=10)

# --- XOR: NOT linearly separable -> perceptron never converges ---
y_xor = np.array([0, 1, 1, 0])
print("\n=== Training on XOR ===")
p_xor = Perceptron(n_inputs=2)
hist = p_xor.fit(X, y_xor, epochs=10)
print(f"\nMisclassifications per epoch: {hist}")
print("Notice: it never reaches 0 -- it oscillates forever.")
```

**Expected output (shapes, not exact numbers):** the OR run drives `misclassified` to `0` within a handful of epochs and *stays* there. The XOR run's `misclassified` count **never hits 0** — it bounces between 1 and 3 indefinitely, no matter how long you train or how small `lr` is. That's not a bug. That's the Perceptron Convergence Theorem telling you, empirically, "this problem is not linearly separable — give up on a single neuron."

---

## 3. 🏗️ From Perceptron to MLP — Escaping Linear Separability

### Intuition

XOR can be written as a combination of two *simpler* linearly-separable functions:

$$\text{XOR}(x_1,x_2) = \text{OR}(x_1,x_2) \ \text{AND}\ \text{NAND}(x_1,x_2)$$

Each of OR and NAND is linearly separable on its own (each is just a half-plane). XOR is the **intersection** of two half-planes — a region with a *bent* boundary, not a straight one. A single perceptron draws one straight line; **two perceptrons in a hidden layer, feeding into a third**, can draw two lines and combine their outputs — and a bent boundary is exactly what you get. This is the entire motivation for the **Multi-Layer Perceptron (MLP)**: stack perceptrons so that each layer's *output* becomes the next layer's *input*, and non-linear activations between layers let the composition draw curves, not just lines.

### The Math — MLP notation (slide 4)

For each layer $l = 1, \dots, L$:

$$h^{[l]} = \sigma\big(W^{[l]} h^{[l-1]} + b^{[l]}\big)$$

with **parameters** $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$, $b^{[l]} \in \mathbb{R}^{n_l}$. Layer $0$ is the input, layer $L$ is the output. The **total parameter count**:

$$\text{Total} = \sum_{l=1}^{L} \big(n_l \cdot n_{l-1} + n_l\big)$$

— each term is (weight matrix size) + (bias vector size). "**Depth**" = number of layers $L$; "**Width**" = number of units $n_l$ per layer. These two numbers are *the* two knobs of model capacity, and almost every architectural decision in deep learning (ResNet depth, Transformer width, etc.) is a more sophisticated version of choosing depth vs. width.

> 🔗 **Already covered:** the single-neuron building block $z=w^\top x+b,\ a=\sigma(z)$ is Concept 1, [Day 1](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_02_DAY_1.md) — an MLP is literally that block, copy-pasted $n_l$ times per layer and stacked $L$ times.

---

## 4. ⚡ Activation Functions — Why Depth Without Non-Linearity Is Pointless

### Intuition

Slide 5 asks the killer question: **"What if we don't use any activation function?"** Suppose every layer is just $h^{[l]} = W^{[l]} h^{[l-1]} + b^{[l]}$, with no $\sigma$. Then composing two layers gives:

$$h^{[2]} = W^{[2]}\big(W^{[1]}x+b^{[1]}\big) + b^{[2]} = \underbrace{(W^{[2]}W^{[1]})}_{W_{\text{eff}}}x + \underbrace{(W^{[2]}b^{[1]}+b^{[2]})}_{b_{\text{eff}}}$$

That's **still just a linear function of $x$** — a 100-layer network with no activations collapses, mathematically, to a single linear layer. Depth buys you *nothing* without non-linearity. Activation functions are what let each layer carve out a genuinely new decision boundary rather than just relabeling the same hyperplane.

### Requirements (slide 6) and the lineup

A good activation should be **differentiable** (so gradients flow — Concept 7/8/9), **fast to compute**, and ideally **zero-centered output** (so gradients don't all push weights in the same direction, which slows convergence — this becomes important again in weight init, Section 9).

| Activation | Formula | Derivative | Range | Notes |
|---|---|---|---|---|
| **Sigmoid** | $\sigma(z)=\dfrac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ | $(0,1)$ | Saturates at both ends → max derivative $0.25$ (Section 8) |
| **Tanh** | $\dfrac{e^z-e^{-z}}{e^z+e^{-z}}$ | $1-\tanh^2(z)$ | $(-1,1)$ | Zero-centered, but still saturates |
| **ReLU** | $\max(0,z)$ | $1$ if $z>0$, else $0$ | $[0,\infty)$ | Cheap, no saturation for $z>0$, but "dying ReLU" for $z\le0$ |
| **Leaky ReLU** | $z$ if $z>0$, else $az$ ($a{=}0.2$ on the slide) | $1$ or $a$ | $(-\infty,\infty)$ | Fixes dying ReLU with a small negative slope |
| **GELU / SiLU** | $z\cdot\Phi(z)$ (smooth, S-shaped near 0) | smooth everywhere | $\approx(-0.17,\infty)$ | Used in Transformers; smooth version of ReLU, slight negative dip near 0 |

> 🔗 **Already covered:** Sigmoid & tanh — Concept 2, [Day 2](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_03_DAY_2.md). ReLU, LeakyReLU, GELU, SiLU — Concept 3, [Day 3](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_04_DAY_3.md). The slide's "Nonlinearities" plot (ReLU vs GELU, GELU dipping slightly below 0 near $z=0$) is exactly the curve discussed there.

---

## 5. ➡️ Forward Pass — Full Vectorized Derivation (slide notation)

### Intuition

A forward pass is nothing more than: *multiply, add bias, squash, repeat.* The slides build this up symbol-by-symbol, which is worth following exactly once because every later derivation (cost, backprop) reuses this notation.

### The Math

**Single neuron** (slide 9) — the atom:

$$z = w^\top x + b, \qquad a = \sigma(z)$$

**Single hidden unit, layer 1, unit 1** (slide 10) — same thing, with layer/unit superscripts. Note the slide's convention: the bias entering layer $1$ is written $b_1^{[0]}$ (i.e., indexed by the *source* layer):

$$z_1^{[1]} = w_1^{[1]\top} x + b_1^{[0]}, \qquad a_1^{[1]} = \sigma\big(z_1^{[1]}\big)$$

**Two hidden units** (slide 11) — repeat for unit 2:

$$z_2^{[1]} = w_2^{[1]\top} x + b_2^{[0]}, \qquad a_2^{[1]} = \sigma\big(z_2^{[1]}\big)$$

**Stack all hidden units into matrices** (slide 12) — this is the "aha": stacking $w_i^{[1]\top}$ as *rows* of a matrix $W^{[1]}$ turns four separate dot products into **one matrix-vector product**:

$$z^{[1]} = \begin{bmatrix} w_1^{[1]\top}\\ w_2^{[1]\top}\\ w_3^{[1]\top}\\ w_4^{[1]\top}\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix} + \begin{bmatrix}b_1^{[0]}\\b_2^{[0]}\\b_3^{[0]}\\b_4^{[0]}\end{bmatrix} = W^{[1]}x + b^{[0]}, \qquad a^{[1]} = \sigma\big(z^{[1]}\big) = \begin{bmatrix}\sigma(z_1^{[1]})\\\sigma(z_2^{[1]})\\\sigma(z_3^{[1]})\\\sigma(z_4^{[1]})\end{bmatrix}$$

**Output layer** (slide 13) — the hidden activations $a^{[1]}$ become the *input* to layer 2, exactly the same pattern one level up:

$$z^{[2]} = W^{[2]}a^{[1]} + b^{[1]}, \qquad a^{[2]} = \sigma\big(z^{[2]}\big) = \hat{y}$$

**Vectorize across $m$ examples** (slide 14) — stack examples as *columns* of a matrix $X \in \mathbb{R}^{n_x \times m}$. Every equation above gains a capital letter (matrix instead of vector) and the bias broadcasts across columns:

$$Z^{[1]} = W^{[1]}X + b^{[1]}, \quad A^{[1]} = \sigma(Z^{[1]}), \quad Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}, \quad A^{[2]} = \sigma(Z^{[2]})$$

(Note: from here on the slides relabel the bias entering layer $l$ as $b^{[l]}$ rather than $b^{[l-1]}$ — both conventions appear across the deck; what matters is "the bias that's added when *computing* $z^{[l]}$".)

> 🔗 **Already covered:** Concept 1, [Day 1 — Artificial Neuron & Forward Pass](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_02_DAY_1.md) derived exactly this single-neuron-to-vectorized-layer progression.

---

## 6. 💰 Cost Function — Binary, Multi-Class, and Regularized

### Intuition

The cost function $J(W,b)$ is the single scalar number that gradient descent is trying to minimize. Its *shape* is determined by the task:

- **Binary classification** ($y\in\{0,1\}$, one output neuron): **binary cross-entropy**.
- **Multi-class classification** ($K$ classes, $K$ output neurons, softmax): each $y^{(i)}$ becomes a **one-hot vector** in $\{0,1\}^K$ (slide 16's grid of MNIST-style digit images is illustrating exactly this — each image's label is a one-hot vector over 10 digit classes).

### The Math

Binary cross-entropy **with L2 regularization** (logistic regression, slide 17):

$$J(W) = -\frac{1}{m}\left[\sum_{i=1}^{m} y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right] + \frac{\lambda}{2m}\sum_{j=1}^{n}W_j^2$$

For a full **neural network**, this generalizes to a sum over all $K$ output units *and* a regularization term over **every weight in every layer**:

$$J(W,b) = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}\Big[y_k^{(i)}\log\big(\hat{y}_k^{(i)}\big) + \big(1-y_k^{(i)}\big)\log\big(1-\hat{y}_k^{(i)}\big)\Big] \;+\; \frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{n^{[l]}}\sum_{j=1}^{n^{[l-1]}}\big(W_{ij}^{[l]}\big)^2$$

Read the regularization term carefully: it's a triple sum — over **layers**, then over every **row** $i$ and **column** $j$ of each layer's weight matrix. In plain English: *every single weight in the network gets squared and added up* (biases are conventionally excluded).

> 🔗 **Already covered:** MSE loss — Concept 4, [Day 4](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_05_DAY_4.md). Softmax + cross-entropy (the $K$-class case above) — Concept 5, [Day 5](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_06_DAY_5.md).
>
> **One subtle connection worth flagging now:** why cross-entropy and not MSE for classification? With a sigmoid output and MSE loss, $\frac{\partial \mathcal{L}}{\partial z} = (a-y)\sigma'(z)$ — if $\sigma(z)$ is saturated (near 0 or 1), $\sigma'(z)\approx0$ and the gradient vanishes *even when the prediction is badly wrong*. Cross-entropy's $\frac{\partial \mathcal{L}}{\partial z}=a-y$ (derived in Section 7) has **no $\sigma'(z)$ term at all** — it cancels out algebraically. This is your first real glimpse of Section 8 (vanishing gradients) and it's why cross-entropy is the default for classification.

---

## 7. 📉 Gradient Descent — Recap

### Intuition

Slide 18's loop is the master algorithm everything else in this lecture exists to improve:

```
repeat until convergence {
    for l = 1 ... L:
        W[l] = W[l] - alpha * dJ/dW[l]
        b[l] = b[l] - alpha * dJ/db[l]
}
```

The 3D loss-surface plot on the slide (a bumpy landscape with "epochs: 0" marked at a high point, descending toward a blue valley) is a direct visualization of what $\theta_{t+1}=\theta_t-\eta g_t$ traces out in parameter space. Crucially, the slide notes that **neural network cost surfaces are non-convex** — unlike logistic regression's nice bowl, an MLP's $J(W,b)$ has many local minima, flat plateaus, and **saddle points** (points that are a minimum along some directions and a maximum along others). Vanilla gradient descent, looking only at the *current* gradient, is famously bad at all three: it crawls across flat regions, zig-zags across steep ones, and can get completely stuck at a saddle point where the gradient is exactly zero but it's not a minimum.

> 🔗 **Already covered:** Concept 10, [Day 10 — Gradient Descent (batch/SGD/mini-batch)](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_10_DAY_10.md), and Concept 11, [Day 11 — The Learning Rate](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_13_DAY_11.md) — the eigenvalue/condition-number analysis there ($\eta^\star=2/(\lambda_{\max}+\lambda_{\min})$, $\kappa=\lambda_{\max}/\lambda_{\min}$) is the precise mathematical reason "oscillates in steep directions, slow in flat directions" happens — it's literally a high-$\kappa$ loss surface. The saddle-point problem is the new piece, and it's the headline reason Section 10 (momentum) exists.

---

## 8. 🔄 Backpropagation — Full Derivation (slide notation)

### Intuition

Backprop is the chain rule, applied systematically, right-to-left through the computational graph. The slides derive it for the simplest possible case — **logistic regression**, i.e. a "network" with zero hidden layers — because every multi-layer case is just this pattern, copy-pasted once per layer, propagating *backward*.

### The Math

**The forward graph** (slide 19):

$$x,w,b \;\longrightarrow\; z=w^\top x+b \;\longrightarrow\; a=\sigma(z) \;\longrightarrow\; \mathcal{L}(a,y)=-y\log(a)-(1-y)\log(1-a)$$

**Backward, one node at a time** — this is the chain rule made completely explicit:

$$\frac{\partial \mathcal{L}}{\partial a} = -\frac{y}{a}+\frac{1-y}{1-a} \qquad\text{(derivative of the loss w.r.t. its own input)}$$

$$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial a}\cdot\frac{\partial a}{\partial z} = \frac{\partial \mathcal{L}}{\partial a}\cdot \sigma'(z) \qquad\text{(chain rule across the }\sigma\text{ node)}$$

And then, beautifully, for the final two:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial z}\cdot\frac{\partial z}{\partial w} = \frac{\partial \mathcal{L}}{\partial z}\cdot x, \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial z}\cdot\frac{\partial z}{\partial b} = \frac{\partial \mathcal{L}}{\partial z}\cdot 1 = \frac{\partial \mathcal{L}}{\partial z}$$

**Why this matters for the full network:** the quantity $\frac{\partial\mathcal{L}}{\partial z}$ — call it $dz$ — is the *only* thing that needs to be propagated backward from one layer to the previous one. Once you have $dz^{[l]}$ for a layer, computing $dW^{[l]}$ and $db^{[l]}$ for *that layer* is "free" (just multiply by $x$ or $1$), and computing $dz^{[l-1]}$ (to keep propagating) is one more chain-rule step through $W^{[l]}$ and $\sigma'$.

### Generalizing to a full network — the vectorized backward pass

For a 2-layer network with $A^{[2]}=\hat{Y}$ (sigmoid output, cross-entropy loss — recall their combination cancels $\sigma'(z)$ algebraically, Section 6):

$$dZ^{[2]} = A^{[2]} - Y$$

$$dW^{[2]} = \frac{1}{m}\,dZ^{[2]} A^{[1]\top}, \qquad db^{[2]} = \frac{1}{m}\sum dZ^{[2]}$$

$$dZ^{[1]} = \big(W^{[2]\top} dZ^{[2]}\big) \odot \sigma'(Z^{[1]})$$

$$dW^{[1]} = \frac{1}{m}\,dZ^{[1]} X^\top, \qquad db^{[1]} = \frac{1}{m}\sum dZ^{[1]}$$

Three patterns to internalize (each one is a chain-rule fact, just applied to matrices):

1. **$dZ^{[2]}=A^{[2]}-Y$** is the cross-entropy + sigmoid simplification from Section 6 — no $\sigma'$ needed at the output layer.
2. **The transpose flip**: $dW^{[l]}$ involves $A^{[l-1]\top}$ (not $A^{[l-1]}$) because $z=Wa+b$ means $\partial z_i/\partial W_{ij}=a_j$ — to get a matrix of the right shape ($n^{[l]}\times n^{[l-1]}$, same as $W^{[l]}$), you need $dZ^{[l]}\,(n^{[l]}\times m)$ times $A^{[l-1]\top}\,(m\times n^{[l-1]})$. Similarly, propagating *backward* through $W^{[l]}$ uses $W^{[l]\top}$ — the transpose of the forward weight matrix carries the error signal back to the previous layer's units.
3. **$\odot\,\sigma'(Z^{[1]})$** (Hadamard/element-wise product) is the chain rule through the activation function — every hidden unit's incoming error gets scaled by *that unit's own* local slope.

> 🔗 **Already covered:** Concept 6, [Day 6 — Computational Graphs](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_07_DAY_6.md); Concept 7, [Day 7 — Chain Rule Refresher](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_08_DAY_7.md); Concept 8, [Day 8 — Backprop I (credit-assignment intuition)](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_09_DAY_8.md); Concept 9, [Day 9 — Backprop II (full math, one hidden layer)](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_09_DAY_9.md). **Day 9's derivation and this slide's $dZ^{[2]}, dW^{[2]}, dZ^{[1]}, \dots$ formulas are the exact same algorithm** — if anything looked unfamiliar above, Day 9 is where it was built up term-by-term.

---

## 9. 🕳️ Vanishing & Exploding Gradients — *(preview: Concepts 20 & 21, ~9 days out)*

### Intuition

Backprop is a **chain of multiplications**. $dZ^{[1]}$ depends on $dZ^{[2]}$ through a multiplication by $\sigma'(Z^{[1]})$ and $W^{[2]\top}$; $dZ^{[0]}$ would depend on $dZ^{[1]}$ the same way; and so on. Multiply $L$ numbers, each a bit less than 1, and the product shrinks **exponentially** with depth. Multiply $L$ numbers each a bit more than 1, and the product **explodes** exponentially. Depth, which is supposed to be the source of a neural net's power, is also the source of its most notorious training pathology — and it's pure exponential arithmetic.

### The Math

Sigmoid's derivative is $\sigma'(z)=\sigma(z)(1-\sigma(z))$. As a function of $\sigma(z)\in(0,1)$, $p(1-p)$ is a downward parabola maximized at $p=0.5$, where $p(1-p)=0.25$. So:

$$\max_z \sigma'(z) = 0.25$$

If every layer's activation derivative is *at best* $0.25$ (and typically much less, since most activations aren't sitting exactly at $z=0$), then in a 20-layer sigmoid network, the gradient signal reaching the first layer is at best:

$$0.25^{20} = 9.094947 \times 10^{-13}$$

That's a gradient roughly **one trillion times smaller** than the one at the output layer. In floating point, updates of that magnitude are functionally zero — the early layers simply **stop learning**. This single number, $0.25^{20}\approx10^{-12}$, is the entire reason sigmoid/tanh fell out of favor for deep networks and ReLU-family activations (derivative exactly $1$ for $z>0$, no shrinkage at all) took over.

**Exploding gradients** are the mirror image: if the chain of multiplied terms (weights × activation derivatives) is consistently $>1$, the gradient — and the resulting weight updates — grow exponentially with depth, producing huge, destabilizing steps (often manifesting as `NaN` losses).

**Mitigations (slide 26):**
- **Redesign the network** — residual/skip connections (Concept 41) give the gradient an *additive* path around the multiplicative chain, so it doesn't have to survive 20 multiplications to get back to layer 1.
- **Gradient clipping** — cap $\|g_t\|$ directly (you derived *why* this works mathematically back in [Day 11](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_13_DAY_11.md)'s concept-answer: it bounds the step regardless of instantaneous curvature).
- **Proper initialization** — Section 10, next.
- **Normalization layers** — Section 12.

> This is your roadmap's **Concept 20 (Vanishing gradients)** and **Concept 21 (Exploding gradients & clipping)**, arriving in about 9 days at the current pace. Consider this the trailer.

---

## 10. 🎲 Weight Initialization — Xavier & He *(preview: Concepts 18 & 19, ~7 days out)*

### Intuition

Slide 27 asks: what if we initialize **all weights to zero**? Every unit in a layer then computes the *exact same* $z=0\cdot x+b$, gets the *exact same* gradient during backprop, and gets updated *identically* forever. With $W=0$, every hidden unit in a layer is a perfect clone of every other — the network behaves as if that layer had **width 1**, no matter how many units it "actually" has. This is the **symmetry-breaking problem**, and it's why weights must be initialized *randomly* — but randomly *how*?

### The Math

Slide 28's derivation: suppose, after some random init, weights have mean $0$ and variance $1$, and inputs also have mean $0$ and variance $1$. For $z=\sum_{i=1}^{n} w_i x_i$, since $w_i, x_i$ are independent with mean 0:

$$\text{Var}(z) = \sum_{i=1}^{n}\text{Var}(w_i x_i) = \sum_{i=1}^{n}\text{Var}(w_i)\text{Var}(x_i) = n\cdot 1\cdot 1 = n$$

So if you draw weights from a unit-variance distribution, the variance of $z$ **grows by a factor of $n$ at every layer** (where $n$ is the layer's fan-in). Stack 10 layers with $n=100$ and your activations' variance has been multiplied by $100^{10}$ — pure explosion (or, if $n<1$ somehow, vanishing). To keep the signal's scale **constant across layers**, you need:

$$\text{Var}(z) = n\cdot\text{Var}(w)\cdot\text{Var}(x) \stackrel{!}{=} \text{Var}(x) \quad\Longrightarrow\quad \text{Var}(w) = \frac{1}{n}$$

This is the core idea behind both schemes — they differ in exactly *which* $n$ (fan-in only, or fan-in and fan-out) and by a constant factor that accounts for the activation function:

- **Xavier/Glorot** (for sigmoid, tanh — symmetric, non-zeroing activations): $\text{Var}(w) = \dfrac{2}{n_{\text{in}}+n_{\text{out}}}$ — balances the forward-pass variance argument above *and* the equivalent backward-pass argument for gradients, by averaging fan-in and fan-out.
- **He** (for ReLU, Leaky ReLU, GELU): $\text{Var}(w) = \dfrac{2}{n_{\text{in}}}$ — the extra factor of $2$ compensates for ReLU zeroing out (on average) **half** of its inputs, which would otherwise halve the variance at every layer.

> This is your roadmap's **Concept 18 (Xavier/Glorot)** and **Concept 19 (He init)**, arriving in about 7 days. The "zeros fail" symmetry argument above *is* Concept 18's opening hook.

---

## 11. 🚀 SGD with Momentum *(preview: Concept 13, ~2 days out)*

### Intuition

Picture a heavy ball rolling down the bumpy, ravine-shaped loss surface from Section 7. Vanilla gradient descent is like a ball with **no mass** — at every instant it moves exactly in the direction of the local slope, so on the ravine walls it bounces back and forth (zig-zag), making slow net progress along the ravine floor. A ball *with* mass and momentum, by contrast, builds up speed in directions it's been consistently pushed — the floor direction — while the side-to-side wall-bouncing forces partially **cancel out** (today pushed left, a moment ago pushed right → net sideways velocity stays small).

### The Math (slide 31)

Maintain a velocity $v_t$ (initialized to 0):

$$v_{t+1} = \beta v_t - \alpha\,\frac{\partial \mathcal{L}(w_t)}{\partial w_t}$$

$$w_{t+1} = w_t + v_{t+1}$$

$\beta\in[0,1)$ (commonly $0.9$) controls "how much memory" — $\beta=0$ recovers vanilla GD exactly; $\beta\to1$ means velocity barely decays, so the optimizer keeps moving in roughly its recent average direction even if the *current* gradient briefly points elsewhere. This is precisely the fix for the saddle-point and high-$\kappa$-zig-zag problems flagged in Section 7 — the accumulated velocity can carry the optimizer *through* a near-zero-gradient saddle region using momentum built up beforehand.

> This is your roadmap's **Concept 13 (Momentum)**, arriving in ~2 days. RMSProp, Adam, and AdamW (mentioned on the slide) are Concepts 15–16, a few days after that — they take the *same* "remember recent direction" idea and additionally adapt the step size *per parameter*.

---

## 12. 📅 Learning Rate Scheduling *(preview: Concept 17, ~6 days out)*

### Intuition

Slide 32, in three bullets, states a conclusion you can now *derive* yourself from Day 11's math: early in training, $\theta$ is far from any minimum and the loss surface is often sharper ($\lambda_{\max}$ larger) — a **large LR** makes fast progress but risks the instability bound $\eta<2/\lambda_{\max}$. Late in training, $\theta$ is near a minimum where the surface is typically flatter — a **small LR** allows fine-grained convergence without overshooting. A single constant $\eta$ is a compromise that's suboptimal at *both* ends. **LR schedules** (step decay, cosine decay, warmup) are simply $\eta$ becoming a function of $t$ rather than a constant — directly tracking how $\lambda_{\max}(H_t)$ changes over the course of training.

> 🔗 Builds directly on Concept 11, [Day 11 — The Learning Rate](https://github.com/zarifbinhasnat/ml-mentor-progress/blob/main/LESSON_2026_06_13_DAY_11.md). This is your roadmap's **Concept 17**, arriving in ~6 days.

---

## 13. 🛡️ Regularization, Dropout & Batch Normalization *(preview: Concepts 24, 26, 27, ~13–16 days out)*

### Co-adaptation and Dropout (slides 33–34)

L1/L2 regularization (Concepts 24–25, soon) penalizes large weights globally, but doesn't directly address a subtler failure mode: a network can become **co-adapted** — many downstream neurons learn to rely on one specific upstream neuron detecting one specific (possibly fragile/spurious) feature. If that feature is noise or doesn't generalize, the whole dependent chain fails on new data.

**Dropout** fixes this by randomly **deleting** neurons during training: each neuron is kept with probability $p$ (independently, per neuron, per forward/backward pass); a dropped neuron outputs (and back-propagates) exactly $0$. Every mini-batch effectively trains a *different random sub-network*. No neuron can "depend" on any specific other neuron always being present — so the network is forced to learn **redundant, distributed** representations. (At test time, dropout is turned off, and outputs are scaled to match the expected magnitude seen during training — the "inverted dropout" trick.)

### Batch Normalization (slides 35–39)

**The problem — internal covariate shift:** in a deep network, the distribution of layer-2's *inputs* depends on layer-1's weights, which are themselves changing during training. So every layer is constantly trying to learn a mapping for a "moving target" input distribution — training is like trying to hit a target that keeps shifting underneath you.

**The fix:** for each mini-batch, compute the batch mean $\mu_B$ and variance $\sigma_B^2$ of each feature/channel, and normalize:

$$\hat{x}_i = \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$$

This forces every layer's input to have mean $0$, variance $1$ — *but* what if the optimal distribution for learning *isn't* mean-0/variance-1? BatchNorm adds two **learned** parameters $\gamma,\beta$ per channel:

$$y_i = \gamma\hat{x}_i + \beta$$

so the network can learn to recover *any* mean and variance it needs — $\gamma,\beta$ give it back the freedom that raw normalization took away, while the normalization step itself keeps the loss surface smoother (easier optimization) throughout training. **At inference**, there's no "batch" to compute statistics from (you might predict on a single example) — so BatchNorm uses a **running average** of $\mu_B,\sigma_B^2$ accumulated during training, applied as fixed constants.

**The normalization family** (slide 39) — all four do "subtract mean, divide by std, then $\gamma,\beta$"; they differ only in *which axis* they average over:

| Variant | Normalizes across | Typically used in |
|---|---|---|
| **Batch Norm** | the mini-batch (per channel) | CNNs |
| **Layer Norm** | the features of *one* sample | Transformers |
| **Instance Norm** | one channel of one sample | Style transfer |
| **Group Norm** | groups of channels, per sample | CV, when batch size / GPU memory is small |

> This is your roadmap's **Concept 24/25 (L1/L2)**, **Concept 26 (Dropout)**, and **Concept 27/28 (BatchNorm/LayerNorm)**, arriving in roughly 11–16 days.

---

## 14. 🩺 Debugging Neural Networks — A Practical Checklist (slides 40–41)

This is the section nobody slides through in lecture but everybody needs at 2am the night before a deadline. Four symptoms, four checklists:

**Loss not decreasing at all:**
- Check the learning rate (too small → no visible progress; too large → see "NaN" below).
- Check data normalization (features on wildly different scales distort the loss surface — directly affects $\kappa$ from Day 11).
- Check that labels are actually correct (a shuffled/misaligned label column is shockingly common).
- **Sanity check:** try to overfit on a *single batch* (or even a single sample) first. If the network can't drive loss to ~0 on one example, the bug is in your code, not your hyperparameters.

**Loss becomes `NaN`:**
- Exploding gradients → add gradient clipping (Section 9).
- Bad weight initialization → switch to He/Xavier (Section 10).

**Validation loss increases while training loss keeps decreasing** (classic overfitting):
- Add dropout, L2 regularization, or reduce model capacity (fewer/narrower layers).
- Check for **data leakage** (e.g., validation examples that are near-duplicates of training examples, or features that encode the label).

**Training is very slow:**
- Vanishing gradients → switch to ReLU-family activations, add batch normalization.
- Learning rate too small.
- Bad weight initialization → He/Xavier.

Notice how every fix here is a forward-reference to a concept you've now previewed in Sections 9–13 — this checklist is, in a sense, the "index" of the entire second half of your roadmap.

---

## 15. 🐍 CODE — Neural Network From Scratch (the thing the perceptron couldn't do)

This is a 2-layer network (one hidden layer) that matches **every symbol** from Section 5 (forward pass) and Section 8 (backprop) exactly — $Z^{[1]},A^{[1]},Z^{[2]},A^{[2]}=\hat{Y}$, $dZ^{[2]}=A^{[2]}-Y$, etc. — with He/Xavier init (Section 10) and momentum (Section 11) built in. We train it on **XOR** — the exact dataset the perceptron in Section 2 oscillated on forever.

```python
import numpy as np

class TwoLayerNN:
    """
    Forward:  Z1 = W1 @ X + b1 ;  A1 = sigma(Z1)
              Z2 = W2 @ A1 + b2 ;  A2 = sigma(Z2) = Yhat
    Backward: dZ2 = A2 - Y                       (cross-entropy + sigmoid output)
              dW2 = (1/m) dZ2 @ A1.T ; db2 = mean(dZ2)
              dZ1 = (W2.T @ dZ2) * sigma'(Z1)    (Hadamard product, chain rule)
              dW1 = (1/m) dZ1 @ X.T  ; db1 = mean(dZ1)
    """
    def __init__(self, n_x, n_h, n_y, init="he", seed=1):
        rng = np.random.default_rng(seed)
        if init == "he":
            s1, s2 = np.sqrt(2.0/n_x), np.sqrt(2.0/n_h)
        elif init == "xavier":
            s1, s2 = np.sqrt(2.0/(n_x+n_h)), np.sqrt(2.0/(n_h+n_y))
        else:  # "zeros" -- to reproduce the symmetry-breaking failure (Section 10)
            s1, s2 = 0.0, 0.0
        self.W1 = rng.standard_normal((n_h, n_x)) * s1
        self.b1 = np.zeros((n_h, 1))
        self.W2 = rng.standard_normal((n_y, n_h)) * s2
        self.b2 = np.zeros((n_y, 1))
        # momentum velocities (Section 11)
        self.vW1 = np.zeros_like(self.W1); self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2); self.vb2 = np.zeros_like(self.b2)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = self.sigmoid(Z2)
        return A2, (X, Z1, A1, Z2, A2)

    def backward(self, cache, Y):
        X, Z1, A1, Z2, A2 = cache
        m = X.shape[1]
        dZ2 = A2 - Y
        dW2 = (1/m) * dZ2 @ A1.T
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = (self.W2.T @ dZ2) * (A1 * (1 - A1))   # sigma'(Z1) = A1*(1-A1)
        dW1 = (1/m) * dZ1 @ X.T
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1, dW2, db2

    def step(self, grads, lr=0.5, beta=0.9):
        dW1, db1, dW2, db2 = grads
        self.vW1 = beta*self.vW1 - lr*dW1; self.W1 += self.vW1
        self.vb1 = beta*self.vb1 - lr*db1; self.b1 += self.vb1
        self.vW2 = beta*self.vW2 - lr*dW2; self.W2 += self.vW2
        self.vb2 = beta*self.vb2 - lr*db2; self.b2 += self.vb2

    def loss(self, A2, Y, eps=1e-9):
        return -np.mean(Y*np.log(A2+eps) + (1-Y)*np.log(1-A2+eps))


# XOR -- the exact dataset the perceptron in Section 2 could never converge on
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]], dtype=float)        # shape (n_x=2, m=4)
Y = np.array([[0, 1, 1, 0]], dtype=float)        # shape (n_y=1, m=4)

net = TwoLayerNN(n_x=2, n_h=4, n_y=1, init="he", seed=1)

for epoch in range(5001):
    A2, cache = net.forward(X)
    grads = net.backward(cache, Y)
    net.step(grads, lr=0.5, beta=0.9)
    if epoch % 1000 == 0:
        print(f"epoch {epoch:5d}  loss={net.loss(A2, Y):.4f}  preds={A2.round(3)}")

print("\nFinal predictions (target = [0, 1, 1, 0]):")
print(net.forward(X)[0].round(3))
```

**Expected behavior:** loss starts near $\ln 2 \approx 0.693$ (random guessing) and drops toward ~0 within a couple thousand epochs; the final predictions converge close to `[0, 1, 1, 0]` — XOR, **solved**, by exactly one hidden layer of 4 nonlinear units.

**Two experiments worth running yourself:**
1. Change `init="he"` to `init="zeros"`. Loss gets **stuck exactly at $\ln 2$** and predictions converge to four identical numbers — Section 10's symmetry-breaking failure, live.
2. Change `beta=0.9` to `beta=0.0` (no momentum) and compare how many epochs it takes to reach the same loss — a direct, hands-on measurement of Section 11's claim about momentum and saddle/plateau regions (XOR's loss surface has a notoriously flat plateau early in training).

---

## 16. ⚔️ THE GAUNTLET — Tricky Questions From the Slides

No hints this time — these are interview-screen style, mixing conceptual and "compute it" mathematical questions, straight from today's material. Answers are locked at the very bottom; resist scrolling.

**Q1 (Conceptual).** A perceptron, given infinite training time and an arbitrarily small learning rate, will *never* converge on XOR. Explain precisely *why*, in terms of the geometry of the decision boundary it's capable of representing — and explain why "just train longer" cannot fix it.

**Q2 (Mathematical — parameter counting).** A network has an input of size $10$, two hidden layers of widths $16$ and $8$, and an output layer of size $3$. Using the formula from Section 3, compute the **total number of trainable parameters** (weights + biases).

**Q3 (Conceptual).** Suppose you initialize *all* weights **and** all biases to zero in a network with a hidden layer of width 8. After one step of gradient descent, are the 8 hidden units still identical to each other? Now suppose instead only the *weights* are zero but the *biases* are initialized to 8 different random values — does that fix the symmetry problem? Why or why not?

**Q4 (Mathematical — vanishing gradients, beyond the slide).** The slide computes $0.25^{20}\approx9.09\times10^{-13}$, assuming every layer's pre-activation is exactly $z=0$ (where $\sigma'(z)$ is maximized at $0.25$). Now suppose instead every layer's pre-activation happens to be $z=2$ for a 10-layer sigmoid network. Given $\sigma(2)\approx0.881$, compute $\sigma'(2)$, then compute the 10-layer attenuation factor $\sigma'(2)^{10}$. Is it bigger or smaller than $0.25^{10}$ — and does that match your intuition about "$z=0$ is the best case for gradient flow"?

**Q5 (Conceptual).** Section 6 claims that for a sigmoid output with cross-entropy loss, $\frac{\partial\mathcal{L}}{\partial z}=a-y$ — with **no** $\sigma'(z)$ term. Starting from $\frac{\partial\mathcal{L}}{\partial a}=-\frac{y}{a}+\frac{1-y}{1-a}$ and $\sigma'(z)=\sigma(z)(1-\sigma(z))=a(1-a)$, multiply these two and show algebraically that the $\sigma'(z)$ term cancels.

**Q6 (Mathematical — He init).** A layer has fan-in $n_{\text{in}}=512$. Using He initialization, what is $\text{Var}(w)$, and what is the corresponding standard deviation?

**Q7 (Conceptual, tricky).** Why is the **bias** term conventionally excluded from L2 regularization (look back at Section 6's triple-sum formula — note it sums over $W_{ij}^{[l]}$, never $b^{[l]}$)?

**Q8 (Conceptual, tricky — dropout).** During training, a layer uses dropout with keep-probability $p=0.5$. At test time, a careless engineer forgets to disable dropout *and* forgets the inverted-dropout rescaling. What happens to the **expected magnitude** of that layer's output at test time compared to what the next layer was trained to expect — and in which direction does it shift?

**Q9 (Mathematical — momentum steady state).** Using $v_{t+1}=\beta v_t - \alpha g$ with a **constant** gradient $g$ every step (and $v_0=0$), derive the steady-state velocity $v_\infty = \lim_{t\to\infty} v_t$ in terms of $\alpha,\beta,g$. For $\beta=0.9$, by what factor is the eventual per-step displacement larger than a single vanilla-GD step of size $-\alpha g$?

**Q10 (Conceptual, tricky — saddle points).** At a saddle point, $\nabla\mathcal{L}=0$ — the same condition as at a local minimum. (a) Why doesn't vanilla gradient descent "know the difference" and just stop correctly at a saddle the way it would at a minimum? (b) Give one reason momentum *can* help escape a saddle, and one reason it might *not* always.

**Q11 (Conceptual — normalization variants).** You're training a Transformer where each training batch contains sequences of very different lengths (heavy padding) and the batch size is small (e.g., 2, due to GPU memory). Explain why **Layer Normalization** is the standard choice here rather than **Batch Normalization** — refer to *what each one averages over* (Section 13's table).

**Q12 (Conceptual — evaluation metrics, ties to lecture outline).** A binary classifier outputs $P(y=1\mid x)\in[0,1]$ for every example. You lower the decision threshold from $0.5$ to $0.3$. (a) What happens to **recall**, and why? (b) What happens to **precision**, in general, and why is it not guaranteed to move monotonically the way recall does? (c) Does the model's **ROC-AUC** change when you do this? Why or why not?

---

## 💬 Marching Orders

You now have the full picture this lecture paints: **perceptron → MLP → forward pass → cost → backprop → why training breaks (vanishing/exploding gradients, bad init, saddle points) → how we fix it (momentum, schedules, dropout, batchnorm)**. Notice that roughly two-thirds of this lecture is concepts you've *already* built from first principles (Days 1–11, linked throughout) — the rest (Sections 9–13) is your roadmap for the next ~2.5 weeks, now previewed end-to-end. Run the Section 15 code, do both experiments (zero-init and no-momentum), and *then* attempt the Gauntlet without looking anything up.

---
---

## 🔒 ANSWER KEY

**A1.** A single perceptron's decision boundary is the set $\{x : w^\top x + b = 0\}$ — a single hyperplane (a straight line in 2D), which always splits the plane into exactly two convex half-planes. XOR's positive class $\{(0,1),(1,0)\}$ and negative class $\{(0,0),(1,1)\}$ are arranged on opposite diagonals of the unit square — any straight line that separates one diagonal pair from the other would have to simultaneously be on "both sides" of itself, which is impossible. This is a **representational** limitation — the hypothesis class (all possible hyperplanes) simply does not contain a function that solves XOR. "Training longer" only searches *within* that hypothesis class more thoroughly; it cannot expand the class. The Perceptron Convergence Theorem guarantees convergence *if and only if* a separating hyperplane exists — for XOR, none does, so the algorithm provably oscillates forever (you can prove no fixed point of the update rule exists).

**A2.** Using $\text{Total}=\sum_{l=1}^{L}(n_l n_{l-1}+n_l)$ with $n_0{=}10, n_1{=}16, n_2{=}8, n_3{=}3$:
- Layer 1: $16\times10+16 = 176$
- Layer 2: $8\times16+8 = 136$
- Layer 3: $3\times8+3 = 27$
- **Total = $176+136+27 = 339$ parameters.**

**A3.** With *everything* (weights and biases) at zero: $z=0$ for every unit, $a=\sigma(0)=0.5$ for every unit, and every hidden unit receives the *identical* gradient signal during backprop (since $dW^{[1]}_{ij}\propto dz_i^{[1]}\cdot x_j$, and every $dz_i^{[1]}$ is identical across $i$ when $W^{[2]}$ rows are also identical/zero). So yes — after one step, all 8 hidden units are still identical, and remain identical forever (an inductive argument: identical inputs + identical weights + identical update rule ⟹ identical outputs, every step). Giving the **biases** 8 different random values *does* break the symmetry of the *activations* ($a_i=\sigma(b_i)$ now differ), but the **weights** $W^{[1]}_{i,:}$ are still all zero and receive updates that depend on $dz_i^{[1]}\cdot x_j$ — since $x_j$ is the same for all units and $dz_i^{[1]}$ now *does* differ slightly (because $a_i$ differ), the weights *will* start to diverge from each other — so random bias init **partially** breaks symmetry, but it's a weak, slow effect compared to randomizing the weights directly (which is why standard practice randomizes weights, not just biases).

**A4.** $\sigma(2)\approx0.8808$, so $\sigma'(2)=\sigma(2)(1-\sigma(2))\approx0.8808\times0.1192\approx0.1050$. Then $\sigma'(2)^{10}\approx0.1050^{10}\approx 1.6\times10^{-10}$. Compare to $0.25^{10}=(0.25^2)^5=0.0625^5\approx9.5\times10^{-7}$. Since $0.1050 < 0.25$, we get $0.1050^{10} < 0.25^{10}$ — i.e. $1.6\times10^{-10} < 9.5\times10^{-7}$, so the $z=2$ case attenuates the gradient **more** than the $z=0$ case, **despite** $z=0$ only being the *theoretical worst case bound* in the other direction — wait, careful: $z=0$ gives the *maximum possible* $\sigma'$ ($0.25$), so it's the **best case** for gradient flow, and indeed $0.25^{10} > 0.1050^{10}$ confirms $z=0$ vanishes *less* than $z=2$ at equal depth. This matches intuition: $z=0$ (max derivative = $0.25$) is the *most favorable* scenario for gradients, and any $z\ne0$ (like $z=2$, where the sigmoid is starting to saturate) makes vanishing **strictly worse**. The slide's $0.25^{20}$ is already the *best-case* bound — real networks, with $z\ne0$ almost everywhere, vanish **even faster** than that number suggests.

**A5.** $\frac{\partial\mathcal{L}}{\partial z}=\frac{\partial\mathcal{L}}{\partial a}\cdot\sigma'(z) = \left(-\frac{y}{a}+\frac{1-y}{1-a}\right)\cdot a(1-a)$. Distribute: $-\frac{y}{a}\cdot a(1-a) + \frac{1-y}{1-a}\cdot a(1-a) = -y(1-a) + a(1-y) = -y+ya+a-ay = a-y$. The $a$ and $(1-a)$ factors from $\sigma'(z)$ exactly cancel the $a$ and $(1-a)$ denominators from $\frac{\partial\mathcal{L}}{\partial a}$, leaving the remarkably simple $a-y$ — independent of how saturated $\sigma$ is.

**A6.** He initialization: $\text{Var}(w)=\dfrac{2}{n_{\text{in}}}=\dfrac{2}{512}=\dfrac{1}{256}$. Standard deviation $=\sqrt{1/256}=1/16=0.0625$.

**A7.** Regularization exists to penalize **complexity/sensitivity of the decision boundary to the inputs** — large *weights* mean the output changes a lot for small changes in input (high sensitivity, prone to overfitting noise). The **bias** is a pure offset — it shifts the decision boundary without changing how sensitive the output is to $x$. Penalizing $b$ wouldn't reduce overfitting in the same sense, and could actively *hurt* the model by discouraging it from learning the correct baseline/offset for the data (e.g., a dataset where 90% of labels are 1 needs a large positive bias — penalizing that is actively counterproductive).

**A8.** With keep-probability $p=0.5$, during training each unit's output is zero half the time, so the *expected* sum reaching the next layer is half of what it would be with no dropout. The standard fix ("inverted dropout") divides the kept activations by $p$ during training so the expected magnitude matches the no-dropout case. If at test time dropout is left **on** (random zeroing still happening) **and** the $1/p$ rescaling is also missing, the expected output magnitude of that layer is reduced by a factor of $p$ (here, **halved**) compared to what downstream layers were calibrated for during training — predictions become systematically biased/shifted toward whatever the network outputs for "weaker than expected" inputs (e.g., outputs pushed toward 0 for a sigmoid/ReLU stack), and the model becomes nondeterministic at test time (different forward passes on the same input give different outputs) on top of that.

**A9.** $v_{t+1}=\beta v_t-\alpha g$. At steady state $v_\infty=\beta v_\infty - \alpha g \Rightarrow v_\infty(1-\beta)=-\alpha g \Rightarrow v_\infty = \dfrac{-\alpha g}{1-\beta}$. For $\beta=0.9$: $v_\infty=\dfrac{-\alpha g}{0.1}=-10\alpha g$ — **10x** the displacement of a single vanilla-GD step $-\alpha g$. (This is also why, when switching from vanilla GD to momentum, people sometimes reduce $\alpha$ — the *effective* long-run step size is amplified by $\frac{1}{1-\beta}$.)

**A10.** (a) Vanilla GD's update rule only looks at $\nabla\mathcal{L}(\theta_t)$ — and $\nabla\mathcal{L}=0$ at *both* a saddle point and a local minimum, so the update $\theta_{t+1}=\theta_t-\eta\cdot 0=\theta_t$ is identical in both cases. The algorithm has no mechanism to distinguish "I'm at a minimum, stop" from "I'm at a saddle, the gradient just happens to be zero *here* but there's a descent direction infinitesimally nearby." (b) Momentum can help because the **velocity accumulated approaching the saddle is nonzero** even when the instantaneous gradient *at* the saddle is zero — so $w_{t+1}=w_t+v_{t+1}$ still moves, carrying the optimizer past/through the saddle using "inertia" from before. It might **not** always help because if the approach to the saddle was itself slow/decelerating (velocity already decayed toward zero by the time the saddle is reached, e.g. a long flat plateau leading into it), there may be little accumulated velocity left to carry the optimizer through — momentum helps most when the saddle is approached "with speed," not when the whole region is uniformly flat.

**A11.** Batch Normalization computes statistics **across the batch dimension**, per feature/channel — with batch size 2, $\mu_B,\sigma_B^2$ are estimated from only 2 samples, an extremely noisy estimate, and heavy padding means many "positions" in the batch are meaningless pad tokens that would corrupt those statistics further. Layer Normalization instead computes statistics **across the feature dimension, for one sample (one token) at a time** — its statistics don't depend on batch size at all (works fine even with batch size 1) and aren't affected by *other* sequences' padding in the batch, since each token is normalized independently using only its own feature vector. This batch-size/padding independence is why LayerNorm is the standard in Transformers.

**A12.** (a) **Recall increases (or stays the same), never decreases**, when you lower the threshold from 0.5 to 0.3 — every example that was already classified positive (score $\ge0.5$) is still classified positive (score $\ge0.3$), *plus* some additional examples with scores in $[0.3,0.5)$ now also become positive. More true positives are captured (true positive rate up), so recall $=\frac{TP}{TP+FN}$ can only go up. (b) **Precision is not monotonic** — it depends on the *ratio* of newly-captured true positives to newly-captured false positives among the examples with scores in $[0.3,0.5)$. If most of those borderline examples are actually negatives, precision $=\frac{TP}{TP+FP}$ drops (more FPs added than TPs); if most are actually positives, precision could even rise slightly. There's no guarantee either way — it depends entirely on the data in that score band. (c) **ROC-AUC does not change.** ROC-AUC is computed by sweeping the threshold across *all* possible values and measuring the area under the resulting TPR-vs-FPR curve — it's a summary of the model's ranking quality across **every** threshold simultaneously. Picking one particular threshold (0.5 vs 0.3) just selects *one point* on a curve whose overall shape — and therefore area — is entirely determined by the model's score *ordering* of examples, which hasn't changed at all.

