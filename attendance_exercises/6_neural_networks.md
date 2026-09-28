---
title: "Exercise: Adaptive Moment Estimation (Adam)"
published: "29 September 2026"
---

# Learning objectives

Apply Adam by hand to a scalar optimization problem. Calculate the moving averages of gradients and squared gradients, apply bias correction, perform parameter updates, and explain the difference from ordinary gradient descent.

# The problem and update rules

Minimize the function

::: numbered
$$f(x)=(x-3)^2\qquad f'(x)=2(x-3)$$
:::

Its minimum is at $x^*=3$. Start from $x_0=0$, with $m_0=v_0=0$, and use

::: numbered
$$\alpha=0.1\qquad\beta_1=0.9\qquad\beta_2=0.999\qquad\epsilon=10^{-8}$$
:::

At iteration $t$, calculate the following quantities in this order:

::: numbered
$$\begin{split}
g_t&=f'(x_{t-1})\\
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2\\
\hat m_t&=\frac{m_t}{1-\beta_1^t}\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}\\
x_t&=x_{t-1}-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\end{split}$$
:::

Here $m_t$ tracks gradients and $v_t$ tracks squared gradients. The hats denote bias-corrected values. Keep at least six decimal places in intermediate calculations; round the reported results at the end. The small stabilizer $\epsilon$ is added **outside** the square root.

```{=latex}
\newpage
```
# Exercise 1 - Iteration 1: guided calculation

Starting from $x_0=m_0=v_0=0$, calculate:

1.  **Gradient.** Evaluate $g_1=2(x_0-3)$.
2.  **First moment.** Use $m_1=0.9m_0+0.1g_1$.
3.  **Second moment.** Use $v_1=0.999v_0+0.001g_1^2$.
4.  **Bias correction.** Calculate $\hat m_1=m_1/(1-0.9)$ and $\hat v_1=v_1/(1-0.999)$.
5.  **Parameter update.** Calculate
    $$x_1=x_0-0.1\frac{\hat m_1}{\sqrt{\hat v_1}+10^{-8}}$$

Record your answers:

::: center
::: {.course-table columns="@{}p{0.328\\linewidth}p{0.638\\linewidth}@{}" font-size="normal" tabcolsep="3pt"}
| Quantity   | Iteration 1 |
|:-----------|:------------|
| $g_1$      |             |
| $m_1$      |             |
| $v_1$      |             |
| $\hat m_1$ |             |
| $\hat v_1$ |             |
| $x_1$      |             |
:::
:::

# Exercise 2 - Iteration 2

Use your previous values for $x_1,m_1,v_1$. Repeat the five steps:

1.  Calculate $g_2=2(x_1-3)$.
2.  Calculate $m_2=0.9m_1+0.1g_2$.
3.  Calculate $v_2=0.999v_1+0.001g_2^2$.
4.  Correct the bias using $1-0.9^2$ and $1-0.999^2$.
5.  Calculate $x_2=x_1-0.1\hat m_2/(\sqrt{\hat v_2}+10^{-8})$.

::: center
::: {.course-table columns="@{}p{0.328\\linewidth}p{0.638\\linewidth}@{}" font-size="normal" tabcolsep="3pt"}
| Quantity   | Iteration 2 |
|:-----------|:------------|
| $g_2$      |             |
| $m_2$      |             |
| $v_2$      |             |
| $\hat m_2$ |             |
| $\hat v_2$ |             |
| $x_2$      |             |
:::
:::

# Exercise 3 - Iteration 3 {#exercise-3---iteration-3 needspace="16"}

Perform the third iteration using the same update rules, now with the bias-correction denominators $1-0.9^3$ and $1-0.999^3$. Complete the table.

::: center
::: {.course-table columns="@{}p{0.328\\linewidth}p{0.638\\linewidth}@{}" font-size="normal" tabcolsep="3pt"}
| Quantity   | Iteration 3 |
|:-----------|:------------|
| $x_2$      |             |
| $g_3$      |             |
| $m_3$      |             |
| $v_3$      |             |
| $\hat m_3$ |             |
| $\hat v_3$ |             |
| $x_3$      |             |
:::
:::

# Short conceptual questions

1.  What does $m_t$ represent? Choose one answer and explain it in one or two sentences.
    1.  The average of the parameters seen so far.
    2.  An exponentially weighted moving average of the gradients.
    3.  An exponentially weighted moving average of squared parameters.
    4.  The learning rate.
2.  What does $v_t$ represent? Choose one answer.
    1.  An exponentially weighted moving average of the gradients.
    2.  An exponentially weighted moving average of the squared gradients.
    3.  The current objective value.
    4.  The accumulated parameter updates.
3.  Why does Adam use $g_t^2$ in $v_t$? Explain how a large gradient magnitude affects the second moment and the denominator of the parameter update.
4.  Why are the bias-correction factors $1-\beta_1^t$ and $1-\beta_2^t$ needed? Use the initialization $m_0=v_0=0$ in your explanation.

# Key concepts

::: center
::: {.course-table columns="@{}p{0.328\\linewidth}p{0.638\\linewidth}@{}" font-size="normal" tabcolsep="3pt"}
| Concept                 | Meaning                                             |
|:-----------------------|:---------------------------------------------|
| Gradient $g_t$          | Derivative at the previous parameter value          |
| First moment $m_t$      | Exponentially weighted average of gradients         |
| Second raw moment $v_t$ | Exponentially weighted average of squared gradients |
| Bias correction         | Compensates for initialization at zero              |
| Learning rate $\alpha$  | Overall scale of the update                         |
| Stabilizer $\epsilon$   | Small positive denominator term                     |
:::
:::
