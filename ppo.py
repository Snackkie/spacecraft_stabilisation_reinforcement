import argparse
import json
import time
import numpy as np

import attitude as A

OBS_DIM, ACT_DIM = 7, 3
W_SCALE = 1.0
FRAME_SKIP = 5


class VecAttitudeEnv:
    def __init__(self, n_envs, domain_rand=False, w_max=1.0, seed=0,
                 episode_len=A.N // FRAME_SKIP):
        self.n = n_envs
        self.dr = domain_rand
        self.w_max = w_max
        self.T = episode_len
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((n_envs, 4)); self.w = np.zeros((n_envs, 3))
        self.I = np.repeat(A.I_NOM[None], n_envs, 0)
        self.I_inv = np.linalg.inv(self.I)
        self.dist = np.zeros((n_envs, 3))
        self.t = np.zeros(n_envs, int)
        self.reset_idx(np.arange(n_envs))

    def reset_idx(self, idx):
        k = len(idx)
        q = self.rng.normal(size=(k, 4))
        self.q[idx] = q / np.linalg.norm(q, axis=1, keepdims=True)

        u = self.rng.uniform(0, 1, size=(k, 1))
        self.w[idx] = u * self.rng.uniform(-self.w_max, self.w_max, size=(k, 3))
        self.t[idx] = 0
        if self.dr:
            I = np.zeros((k, 3, 3))
            d = A.I_NOM.diagonal() * self.rng.uniform(0.8, 1.2, size=(k, 3))
            I[:, [0, 1, 2], [0, 1, 2]] = d
            off = self.rng.uniform(-0.5, 0.5, size=(k, 3))
            I[:, 0, 1] = I[:, 1, 0] = off[:, 0]
            I[:, 0, 2] = I[:, 2, 0] = off[:, 1]
            I[:, 1, 2] = I[:, 2, 1] = off[:, 2]
            self.I[idx] = I
            self.I_inv[idx] = np.linalg.inv(I)
            self.dist[idx] = self.rng.uniform(-0.003, 0.003, size=(k, 3))

    def reset(self):
        self.reset_idx(np.arange(self.n))
        return self.observe()

    def observe(self):
        q, w = self.q, self.w
        if self.dr:
            q = q + self.rng.normal(0, 0.002, size=q.shape)
            q = q / np.linalg.norm(q, axis=1, keepdims=True)
            w = w + self.rng.normal(0, 0.001, size=w.shape)
        return np.concatenate([A.error_quat(q), w / W_SCALE], axis=1)

    @staticmethod
    def reward(q, w, a):
        theta = A.angle_error(q)
        wn = np.linalg.norm(w, axis=1)
        r = -theta / np.pi - 0.5 * wn - 0.01 * np.sum(a**2, axis=1) / 3
        r += 1.0 * ((theta < np.deg2rad(2.0)) & (wn < 0.01))
        return r

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        tau = a * A.TAU_MAX + self.dist
        for _ in range(FRAME_SKIP):
            self.q, self.w = A.step(self.q, self.w, tau, self.I, self.I_inv)
        self.t += 1
        r = self.reward(self.q, self.w, a)
        trunc = self.t >= self.T

        obs_final = self.observe()
        if trunc.any():
            self.reset_idx(np.where(trunc)[0])
        obs = self.observe() if trunc.any() else obs_final
        return obs, r, trunc, obs_final


class MLP:
    def __init__(self, sizes, rng, out_gain=1.0):
        self.params = []
        for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = out_gain if i == len(sizes) - 2 else np.sqrt(2)
            W = rng.normal(size=(m, n))

            u, _, vt = np.linalg.svd(W, full_matrices=False)
            W = gain * (u if u.shape == (m, n) else vt)
            self.params += [W, np.zeros(n)]

    def forward(self, x):
        cache = [x]
        n_layers = len(self.params) // 2
        for i in range(n_layers):
            W, b = self.params[2*i], self.params[2*i + 1]
            x = x @ W + b
            if i < n_layers - 1:
                x = np.tanh(x)
            cache.append(x)
        return x, cache

    def backward(self, cache, grad_out):
        grads = [None] * len(self.params)
        g = grad_out
        n_layers = len(self.params) // 2
        for i in reversed(range(n_layers)):
            x_in = cache[i]
            grads[2*i] = x_in.T @ g
            grads[2*i + 1] = g.sum(0)
            if i > 0:
                g = (g @ self.params[2*i].T) * (1 - cache[i]**2)
        return grads


class Adam:
    def __init__(self, params, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1**self.t); vh = v / (1 - self.b2**self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_grads(grads, max_norm):
    norm = np.sqrt(sum(np.sum(g**2) for g in grads))
    if norm > max_norm:
        grads = [g * (max_norm / norm) for g in grads]
    return grads


class PPOAgent:
    def __init__(self, seed=0, hidden=64, log_std_init=-0.5):
        rng = np.random.default_rng(seed)
        self.actor = MLP([OBS_DIM, hidden, hidden, ACT_DIM], rng, out_gain=0.01)
        self.critic = MLP([OBS_DIM, hidden, hidden, 1], rng, out_gain=1.0)
        self.log_std = np.full(ACT_DIM, log_std_init)
        self.rng = rng

    def act(self, obs, deterministic=False):
        mu, _ = self.actor.forward(obs)
        if deterministic:
            return mu
        return mu + np.exp(self.log_std) * self.rng.normal(size=mu.shape)

    def log_prob(self, mu, a):
        std = np.exp(self.log_std)
        return (-0.5 * (((a - mu) / std)**2).sum(1) - self.log_std.sum()
                - 0.5 * ACT_DIM * np.log(2 * np.pi))

    def value(self, obs):
        return self.critic.forward(obs)[0][:, 0]

    def save(self, path):
        np.savez(path, *self.actor.params, *self.critic.params, self.log_std)

    @classmethod
    def load(cls, path):
        ag = cls()
        d = np.load(path)
        arrs = [d[f'arr_{i}'] for i in range(len(d.files))]
        na = len(ag.actor.params)
        ag.actor.params = arrs[:na]
        ag.critic.params = arrs[na:2*na]
        ag.log_std = arrs[-1]
        return ag

    def controller(self):
        held = {}

        def ctrl(k, q, w):
            if k % FRAME_SKIP == 0:
                obs = np.concatenate([A.error_quat(q), w / W_SCALE])[None]
                a = np.clip(self.act(obs, deterministic=True)[0], -1, 1)
                held['tau'] = a * A.TAU_MAX
            return held['tau']
        return ctrl


def train(name, total_steps=12_000_000, n_envs=32, n_steps=256, epochs=10,
          n_minibatches=8, gamma=0.99, lam=0.95, clip=0.2, lr=3e-4,
          vf_coef=0.5, ent_coef=0.0, max_grad=0.5, reward_scale=0.1,
          domain_rand=False, seed=0, log_every=20, resume=None):
    env = VecAttitudeEnv(n_envs, domain_rand=domain_rand, seed=seed)
    agent = PPOAgent(seed=seed)
    if resume:
        agent = PPOAgent.load(resume)
        agent.rng = np.random.default_rng(seed + 1)
        env.rng = np.random.default_rng(seed + 1)
    opt_a = Adam(agent.actor.params + [agent.log_std], lr=lr)
    opt_c = Adam(agent.critic.params, lr=lr)

    obs = env.reset()
    n_iters = total_steps // (n_envs * n_steps)
    batch = n_envs * n_steps
    mb = batch // n_minibatches
    log = json.load(open(f'{name}_log.json')) if resume else []
    step_offset = log[-1]['steps'] if log else 0
    ep_ret = np.zeros(n_envs); finished = []
    t_start = time.time()

    for it in range(n_iters):
        frac = 1.0 - it / n_iters
        opt_a.lr = opt_c.lr = lr * frac

        B_obs = np.zeros((n_steps, n_envs, OBS_DIM))
        B_act = np.zeros((n_steps, n_envs, ACT_DIM))
        B_logp = np.zeros((n_steps, n_envs))
        B_rew = np.zeros((n_steps, n_envs))
        B_val = np.zeros((n_steps, n_envs))
        B_trunc = np.zeros((n_steps, n_envs), bool)
        B_vfinal = np.zeros((n_steps, n_envs))
        for t in range(n_steps):
            mu, _ = agent.actor.forward(obs)
            a = mu + np.exp(agent.log_std) * agent.rng.normal(size=mu.shape)
            B_obs[t], B_act[t] = obs, a
            B_logp[t] = agent.log_prob(mu, a)
            B_val[t] = agent.value(obs)
            obs, r, trunc, obs_final = env.step(a)
            B_rew[t] = r * reward_scale
            B_trunc[t] = trunc
            if trunc.any():
                B_vfinal[t, trunc] = agent.value(obs_final[trunc])
            ep_ret += r
            if trunc.any():
                finished += list(ep_ret[trunc]); ep_ret[trunc] = 0

        last_val = agent.value(obs)
        adv = np.zeros((n_steps, n_envs)); gae = 0
        for t in reversed(range(n_steps)):
            next_val = last_val if t == n_steps - 1 else B_val[t + 1]
            next_val = np.where(B_trunc[t], B_vfinal[t], next_val)
            delta = B_rew[t] + gamma * next_val - B_val[t]
            gae = delta + gamma * lam * np.where(B_trunc[t], 0.0, gae)
            adv[t] = gae
        ret = adv + B_val

        f_obs = B_obs.reshape(batch, OBS_DIM); f_act = B_act.reshape(batch, ACT_DIM)
        f_logp = B_logp.ravel(); f_adv = adv.ravel(); f_ret = ret.ravel()
        clipfracs = []
        for _ in range(epochs):
            perm = agent.rng.permutation(batch)
            for s in range(0, batch, mb):
                i = perm[s:s + mb]
                o, a, lp_old = f_obs[i], f_act[i], f_logp[i]
                ad = f_adv[i]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)

                mu, cache = agent.actor.forward(o)
                std = np.exp(agent.log_std)
                lp = agent.log_prob(mu, a)
                ratio = np.exp(lp - lp_old)
                unclipped = np.where(ad > 0, ratio < 1 + clip, ratio > 1 - clip)
                clipfracs.append(1 - unclipped.mean())
                dlp = -(ad * ratio * unclipped) / len(i)
                z = (a - mu) / std
                g_mu = dlp[:, None] * z / std
                g_logstd = (dlp[:, None] * (z**2 - 1)).sum(0) - ent_coef
                g_actor = agent.actor.backward(cache, g_mu)
                g_all = clip_grads(g_actor + [g_logstd], max_grad)
                opt_a.step(g_all)
                agent.log_std = np.maximum(agent.log_std, -3.0)
                opt_a.params[-1] = agent.log_std

                v, cache_c = agent.critic.forward(o)
                g_v = vf_coef * 2 * (v[:, 0] - f_ret[i])[:, None] / len(i)
                opt_c.step(clip_grads(agent.critic.backward(cache_c, g_v), max_grad))

        if it % log_every == 0 or it == n_iters - 1:
            mr = float(np.mean(finished[-64:])) if finished else float('nan')
            entry = dict(iter=it, steps=step_offset + (it + 1) * batch, mean_ep_return=mr,
                         std=float(np.exp(agent.log_std).mean()),
                         clipfrac=float(np.mean(clipfracs)),
                         minutes=(time.time() - t_start) / 60)
            log.append(entry)
            print(json.dumps(entry), flush=True)
            agent.save(f'{name}.npz')
            json.dump(log, open(f'{name}_log.json', 'w'))
    agent.save(f'{name}.npz')
    json.dump(log, open(f'{name}_log.json', 'w'))
    return agent


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='ppo_dr')
    p.add_argument('--dr', action='store_true', help='рандомизация параметров')
    p.add_argument('--steps', type=int, default=12_000_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--resume', default=None, help='файл весов для дообучения')
    args = p.parse_args()
    train(args.name, total_steps=args.steps, domain_rand=args.dr, seed=args.seed,
          lr=args.lr, resume=args.resume)
