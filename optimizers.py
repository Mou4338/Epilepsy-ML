"""
Metaheuristic optimizers used to train the neural network.

  * PSO    - Particle Swarm Optimization (Kennedy & Eberhart, 1995; inertia-weight form)
  * APO    - Artificial Protozoa Optimizer (Wang et al., 2024)
  * SBO    - Satin Bowerbird Optimizer (Moosavi & Bardsiri, 2017)
  * QPSBO  - Quantum Parity Satin Bowerbird Optimizer (proposed)

Every optimizer minimises f(x) inside the box [lb, ub] and stops when the shared
budget of function evaluations (FEs) is used up, so all four get exactly the
same number of fitness calls. The Problem wrapper counts FEs and records the
best-so-far value after every evaluation, which gives a fair convergence curve.
"""
import numpy as np


class Problem:
    def __init__(self, f, dim, lb, ub, max_fe):
        self.f, self.dim, self.lb, self.ub, self.max_fe = f, dim, lb, ub, max_fe
        self.fe = 0
        self.best_f = np.inf
        self.best_x = None
        self.history = []          # best-so-far after each FE

    @property
    def done(self):
        return self.fe >= self.max_fe

    def clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def evaluate(self, X):
        """Evaluate a population (rows), never exceeding the FE budget.
        Rows beyond the budget get +inf so they are never selected."""
        X = np.atleast_2d(X)
        out = np.full(len(X), np.inf)
        for i, x in enumerate(X):
            if self.fe >= self.max_fe:
                break
            v = self.f(x)
            self.fe += 1
            out[i] = v
            if v < self.best_f:
                self.best_f, self.best_x = v, x.copy()
            self.history.append(self.best_f)
        return out

    def curve(self, n_points):
        h = np.asarray(self.history)
        idx = np.linspace(0, len(h) - 1, n_points).astype(int)
        return h[idx]


# --------------------------------------------------------------------------- PSO
def pso(prob, rng, n_pop=30, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0):
    d, lb, ub = prob.dim, prob.lb, prob.ub
    vmax = 0.2 * (ub - lb)
    X = rng.uniform(lb, ub, (n_pop, d))
    V = rng.uniform(-vmax, vmax, (n_pop, d))
    F = prob.evaluate(X)
    P, PF = X.copy(), F.copy()
    g = P[np.argmin(PF)].copy()
    T = prob.max_fe // n_pop
    t = 0
    while not prob.done:
        w = w_max - (w_max - w_min) * t / T
        r1, r2 = rng.random((n_pop, d)), rng.random((n_pop, d))
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X)
        V = np.clip(V, -vmax, vmax)
        X = prob.clip(X + V)
        F = prob.evaluate(X)
        imp = F < PF
        P[imp], PF[imp] = X[imp], F[imp]
        g = P[np.argmin(PF)].copy()
        t += 1
    return prob


# --------------------------------------------------------------------------- APO
def apo(prob, rng, n_pop=30, n_pairs=1, pf_max=0.1):
    """Artificial Protozoa Optimizer: autotrophic / heterotrophic foraging,
    dormancy and reproduction (Wang, Snasel, Mirjalili, Pan, Kong & Shehadeh, KBS 2024)."""
    d, lb, ub = prob.dim, prob.lb, prob.ub
    eps = 1e-12
    X = rng.uniform(lb, ub, (n_pop, d))
    F = prob.evaluate(X)
    T = prob.max_fe // n_pop
    t = 0
    while not prob.done:
        order = np.argsort(F)
        X, F = X[order], F[order]
        pf = pf_max * rng.random()
        dr = set(rng.permutation(n_pop)[: int(np.ceil(n_pop * pf))])
        Xnew = X.copy()
        for i in range(n_pop):
            if i in dr:                                    # dormancy / reproduction
                pdr = 0.5 * (1 + np.cos((1 - (i + 1) / n_pop) * np.pi))
                if rng.random() < pdr:
                    Xnew[i] = rng.uniform(lb, ub, d)       # dormancy
                else:                                      # reproduction
                    flag = 1 if rng.random() < 0.5 else -1
                    Mr = np.zeros(d)
                    Mr[rng.permutation(d)[: int(np.ceil(rng.random() * d))]] = 1
                    Xnew[i] = X[i] + flag * rng.random() * (lb + rng.random(d) * (ub - lb)) * Mr
            else:                                          # foraging
                fr = rng.random() * (1 + np.cos(t / T * np.pi))
                Mf = np.zeros(d)
                Mf[rng.permutation(d)[: int(np.ceil(d * (i + 1) / n_pop))]] = 1
                pah = 0.5 * (1 + np.cos(t / T * np.pi))
                if rng.random() < pah:                     # autotrophic
                    j = rng.integers(n_pop)
                    epn = np.zeros(d)
                    for _ in range(n_pairs):
                        km = rng.integers(0, i + 1) if i > 0 else 0
                        kp = rng.integers(i, n_pop)
                        wa = np.exp(-abs(F[km] / (F[kp] + eps)))
                        epn += wa * (X[km] - X[kp])
                    Xnew[i] = X[i] + fr * (X[j] - X[i] + epn / n_pairs) * Mf
                else:                                      # heterotrophic
                    sgn = 1 if rng.random() < 0.5 else -1
                    Xnear = (1 + sgn * rng.random(d) * (1 - t / T)) * X[i]
                    epn = np.zeros(d)
                    for k in range(1, n_pairs + 1):
                        im, ip = max(i - k, 0), min(i + k, n_pop - 1)
                        wh = np.exp(-abs(F[im] / (F[ip] + eps)))
                        epn += wh * (X[im] - X[ip])
                    Xnew[i] = X[i] + fr * (Xnear - X[i] + epn / n_pairs) * Mf
        Xnew = prob.clip(Xnew)
        Fnew = prob.evaluate(Xnew)
        imp = Fnew < F
        X[imp], F[imp] = Xnew[imp], Fnew[imp]
        t += 1
    return prob


# --------------------------------------------------------------------------- SBO
def _sbo_probs(F):
    fit = np.where(F >= 0, 1.0 / (1.0 + F), 1.0 + np.abs(F))
    return fit / fit.sum()


def sbo(prob, rng, n_pop=30, alpha=0.94, p_mut=0.05, z=0.02):
    """Satin Bowerbird Optimizer (Moosavi & Bardsiri, 2017)."""
    d, lb, ub = prob.dim, prob.lb, prob.ub
    sigma = z * (ub - lb)
    X = rng.uniform(lb, ub, (n_pop, d))
    F = prob.evaluate(X)
    while not prob.done:
        P = _sbo_probs(F)
        elite = X[np.argmin(F)]
        # roulette-wheel choice of a target bower for every (bird, dimension)
        tgt = rng.choice(n_pop, size=(n_pop, d), p=P)
        lam = alpha / (1.0 + P[tgt])
        Xt = X[tgt, np.arange(d)]
        Xnew = X + lam * ((Xt + elite) / 2.0 - X)
        mut = rng.random((n_pop, d)) < p_mut
        Xnew[mut] += sigma * rng.standard_normal(mut.sum())
        Xnew = prob.clip(Xnew)
        Fnew = prob.evaluate(Xnew)
        # merge old and new bowers, keep the best n_pop
        Xa, Fa = np.vstack([X, Xnew]), np.concatenate([F, Fnew])
        keep = np.argsort(Fa)[:n_pop]
        X, F = Xa[keep], Fa[keep]
    return prob


# ------------------------------------------------------------------------- QPSBO
def qpsbo(prob, rng, n_pop=30, alpha=0.94, p_mut=0.05, z=0.02,
          beta_max=1.0, beta_min=0.5, d_theta=0.05 * np.pi, p_parity=0.75):
    """Quantum Parity Satin Bowerbird Optimizer (proposed).

    Three changes to SBO:
    1. Qubit register: each bird carries a qubit angle theta per dimension.
       Measuring it (P(|1>) = sin^2 theta) decides, dimension by dimension,
       whether the bird uses the classic SBO bower move (|0>, exploitation)
       or a quantum delta-potential-well jump (|1>, exploration):
           x = p +/- beta * |mbest - x| * ln(1/u)
       where p mixes the bird's own (retained) bower with the elite bower and
       mbest is the mean bower of the colony.
    2. Quantum rotation gate: after evaluation, birds that improved rotate
       theta towards |0> (exploit more); birds that stagnated rotate towards
       |1> (explore more). Theta stays inside [0.05, pi/2 - 0.05].
    3. Parity operator: the worse half of the colony is reflected through
       the elite bower (spatial-parity inversion with a random contraction),
           x_par = elite + r * (elite - x),
       and the better of x / x_par survives. This gives every weak bird an
       "opposite" look at the search space around the best solution.
    The extra parity evaluations are charged to the same FE budget.
    """
    d, lb, ub = prob.dim, prob.lb, prob.ub
    sigma = z * (ub - lb)
    th_lo, th_hi = 0.05, np.pi / 2 - 0.05
    X = rng.uniform(lb, ub, (n_pop, d))
    F = prob.evaluate(X)
    theta = np.full((n_pop, d), np.pi / 4)           # equal superposition
    T = max(1, prob.max_fe // int(n_pop * (1 + p_parity / 2)))
    t = 0
    while not prob.done:
        beta = beta_max - (beta_max - beta_min) * min(t / T, 1.0)
        P = _sbo_probs(F)
        e = np.argmin(F)
        elite = X[e].copy()
        mbest = X.mean(axis=0)                     # mean best bower (colony keeps only best bowers)

        # --- 1. measure qubits and build the hybrid move
        bits = rng.random((n_pop, d)) < np.sin(theta) ** 2
        tgt = rng.choice(n_pop, size=(n_pop, d), p=P)
        lam = alpha / (1.0 + P[tgt])
        Xt = X[tgt, np.arange(d)]
        classic = X + lam * ((Xt + elite) / 2.0 - X)
        phi = rng.random((n_pop, d))
        attractor = phi * X + (1 - phi) * elite
        u = rng.random((n_pop, d)) + 1e-12
        sgn = np.where(rng.random((n_pop, d)) < 0.5, -1.0, 1.0)
        quantum = attractor + sgn * beta * np.abs(mbest - X) * np.log(1.0 / u)
        Xnew = np.where(bits, quantum, classic)
        mut = rng.random((n_pop, d)) < p_mut
        Xnew[mut] += sigma * rng.standard_normal(mut.sum())
        Xnew = prob.clip(Xnew)
        Fnew = prob.evaluate(Xnew)

        # --- 2. quantum rotation gate
        # Keep the quantum state attached to the corresponding bird.
        theta_old = theta.copy()
        theta_new = theta_old.copy()

        improved = Fnew < F
        theta_new[improved] -= d_theta
        theta_new[~improved] += d_theta
        theta_new = np.clip(theta_new, th_lo, th_hi)

        # merge (elitist, as in SBO)
        Xa = np.vstack([X, Xnew])
        Fa = np.concatenate([F, Fnew])
        Ta = np.vstack([theta_old, theta_new])

        keep = np.argsort(Fa)[:n_pop]
        X, F, theta = Xa[keep], Fa[keep], Ta[keep]

        # --- 3. parity operator on the worse half
        if not prob.done:
            elite = X[0]
            worst = np.arange(n_pop // 2, n_pop)
            sel = worst[rng.random(len(worst)) < p_parity]
            if len(sel):
                r = rng.random((len(sel), d))
                Xp = prob.clip(elite + r * (elite - X[sel]))
                Fp = prob.evaluate(Xp)
                better = Fp < F[sel]
                X[sel[better]], F[sel[better]] = Xp[better], Fp[better]
        t += 1
    return prob


ALGORITHMS = {"PSO": pso, "APO": apo, "SBO": sbo, "QP-SBO": qpsbo}
