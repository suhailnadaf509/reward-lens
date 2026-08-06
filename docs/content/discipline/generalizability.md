# Generalizability theory

**How much of that score is the response, and how much is which judge you drew on which day?**

A grader is a measurement device and a score is a measurement, so that question has had a worked answer since Cronbach, Gleser, Nanda and Rajaratnam published it in 1972. Generalizability theory decomposes an observed score into the object being measured and every facet of the measurement procedure, then tells you what a differently sized procedure would cost and buy.

It is standard in educational measurement. It has essentially never been pointed at a reward model.

## There is no package for it, so this is one

`reward_lens.stats.gtheory` is a self-contained implementation, and it exists because there was nothing to depend on. A full index scan of PyPI's 449,089 projects returns no generalizability-theory package. R's `gtheory` package was removed from CRAN on 2025-03-24, because email to its maintainer was undeliverable.

That makes this module a standalone contribution rather than a wrapper, and it is small enough to read in one sitting.

## What is implemented

Two fully crossed designs with one observation per cell. The two-facet \(p \times r\) and the three-facet \(p \times r \times o\), where \(p\) is the object of measurement, the thing whose differences you want to resolve, and \(r\) and \(o\) are facets of the measurement procedure: which grader, which repeat call, which rubric, which response style.

The sums of squares are the textbook ones and the expected mean squares are inverted in closed form. The seven-component inversion for the three-facet design is:

```text
sigma2(pro,e) = MS_pro
sigma2(pr)    = (MS_pr - MS_pro) / n_o
sigma2(po)    = (MS_po - MS_pro) / n_r
sigma2(ro)    = (MS_ro - MS_pro) / n_p
sigma2(p)     = (MS_p - MS_pr - MS_po + MS_pro) / (n_r * n_o)
sigma2(r)     = (MS_r - MS_pr - MS_ro + MS_pro) / (n_p * n_o)
sigma2(o)     = (MS_o - MS_po - MS_ro + MS_pro) / (n_p * n_r)
```

The last component is named `pro,e` and not `pro`. With one observation per cell the three-way interaction and the residual are the same term and nothing in the design can separate them, so writing `pro` alone would claim an interaction estimate the design does not contain.

```python
from reward_lens.stats.gtheory import crossed_pro

# scores is (responses, graders, passes), one score per cell
g = crossed_pro(scores, object_label="response", facet_labels=("grader", "pass"))
print(g.render())

d = g.d_study(r=4, o=2)          # what four graders on two passes would buy
print(d.generalizability, d.dependability)
```

```text
G-study, p x r x o, n_p = 20, n_r = 3, n_o = 2
  p                 1.12822   49.3%
  r                       0    0.0%  TRUNCATED
  o                       0    0.0%  TRUNCATED
  pr               0.300469   13.1%
  po               0.186618    8.2%
  ro               0.180625    7.9%
  pro,e            0.491828   21.5%
  total             2.28776
  2 component(s) estimated below zero and truncated: r, o. Their true values are near zero and
  this design could not resolve them; do not read a truncated component as an established zero.
```

That transcript is on twenty synthetic responses with a known signal, so the components are worth nothing except as a shape. The `TRUNCATED` markers and the sentence under the table are the point of the next section.

## Two things this module is careful about

**Negative variance estimates are truncated at zero and the truncation is recorded.** The method of moments can return a negative component, and silently clamping it misrepresents which facet dominates. The same discipline is in `stats.variance` and for the same reason.

**An unbalanced design is refused rather than approximated.** The inversion above assumes every cell is filled exactly once. Running it on a design with holes gives a biased answer that looks identical to an unbiased one, which is the failure mode that most deserves a refusal. `check_balance` reports the gaps and `fit_unbalanced` names what to install instead, which is `statsmodels`, and says plainly that it is in no declared extra of this package.

## The finite-universe correction is not a detail

This is the part worth carrying away even if you never call the module.

A facet has a universe of \(N_i\) levels and you sample \(n_i\) of them. Declaring \(N_i = n_i\), that is, declaring the facet fixed, moves the object-by-facet interaction out of error and into universe-score variance. Brennan showed in 1992 that this raises reliability from `0.74` to `0.88` on the same data, while destroying any claim to generalise to new levels of that facet.

That is the mathematics of benchmark overfitting, published thirty-four years ago. A leaderboard number computed over one fixed rubric on one fixed item set is a fixed-facet reliability quoted as if it were a random-facet one.

`GStudy.declare_fixed` computes both numbers so the trade is visible in one call. It takes the facet key rather than its label, and a `GStudy` is immutable, so it returns a new study and both numbers stay available:

```python
free  = g.d_study(r=3, o=2).generalizability            # generalises to new graders
fixed = g.declare_fixed("r").d_study(r=3, o=2).generalizability   # these graders only
```

The general form the D-study uses, for a term \(a\) over a set of facets, is

```text
error share    = [1 - prod_{i in a} (n_i / N_i)] * sigma2(a) / prod_{i in a} n_i
universe share =      prod_{i in a} (n_i / N_i)  * sigma2(a) / prod_{i in a} n_i
```

which reduces to the fully random model when every universe is infinite and to Brennan's mixed model when a facet is exhausted. The two shares sum to the term's total contribution, so declaring a facet fixed moves variance between the numerator and the denominator of the coefficient and never creates or destroys any.

The default universe is infinite for every facet, which is the conservative direction: declaring a facet fixed always raises the reliability, so a library that defaulted to fixed would flatter every design it was handed. And "finite" rather than "exhausted" is the right word, because the correction is continuous. A facet with a universe of a hundred levels sampled at twenty gets four fifths of the random-model error and one fifth folded into universe score; exhausting the universe is the endpoint of that scale rather than a separate mode.

Report both, side by side. A reliability quoted without saying which universe it generalises over is not a reliability.

## Where it is used

The G-study is what makes [effective group size](../how-to/effective-group-size.md) more than a rename of \(K\): the reliability of a single observed score is what \(K\) gets multiplied by, and the G-study is where that reliability comes from with an interval attached. `measure/metrology/` is the series that consumes it.

References: Brennan, *Generalizability Theory* (Springer, 2001), chapters 3 and 5; Shavelson and Webb, *Generalizability Theory: A Primer* (Sage, 1991), chapter 4.
