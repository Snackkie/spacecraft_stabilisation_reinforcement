"""
Сравнение регуляторов:
  * CasADi (постановка предыдущей работы: метод Эйлера, холодный старт);
  * CasADi (улучшенная постановка: RK4 + тёплый старт от ПД-регулятора);
  * ПД-регулятор;
  * PPO без рандомизации и PPO с рандомизацией параметров.

Все регуляторы проверяются на одной и той же «истинной» модели
(attitude.step: RK4 + нормировка кватерниона).
Результаты: results/*.png и results/results.json.
"""
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import attitude as A
from ppo import PPOAgent

OUT = 'results'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})

COLORS = {
    'CasADi (Эйлер)': '#9e9e9e',
    'CasADi (RK4)': '#1f77b4',
    'ПД-регулятор': '#2ca02c',
    'PPO': '#ff7f0e',
    'PPO + рандомизация': '#d62728',
}
PD_GAINS = (0.05, 0.5)
t_axis = np.arange(A.N + 1) * A.DT


def pd_warm_start(q0, w0, I=A.I_NOM):
    qs, ws, ts, _ = A.simulate(A.make_pd(*PD_GAINS), q0, w0, I)
    return qs, ws, ts


def run_all_nominal(agents):
    """Номинальный сценарий из предыдущей работы."""
    res, traj = {}, {}

    print('CasADi, постановка предыдущей работы...', flush=True)
    U, X, J, t = A.solve_casadi()
    qs, ws, ts, _ = A.simulate(A.casadi_open_loop(U))
    res['CasADi (Эйлер)'] = dict(A.metrics(qs, ws, ts), solve_time=t, J_solver=J,
                                 q_norm_max_in_model=float(np.linalg.norm(X[0:4], axis=0).max()))
    traj['CasADi (Эйлер)'] = (qs, ws, ts)

    print('CasADi, RK4 + тёплый старт...', flush=True)
    U, X, J, t = A.solve_casadi_rk4(init=pd_warm_start(A.Q0, A.W0))
    qs, ws, ts, _ = A.simulate(A.casadi_open_loop(U))
    res['CasADi (RK4)'] = dict(A.metrics(qs, ws, ts), solve_time=t, J_solver=J)
    traj['CasADi (RK4)'] = (qs, ws, ts)
    U_rk4 = U

    qs, ws, ts, tc = A.simulate(A.make_pd(*PD_GAINS))
    res['ПД-регулятор'] = dict(A.metrics(qs, ws, ts), ctrl_time_per_step_ms=tc / A.N * 1e3)
    traj['ПД-регулятор'] = (qs, ws, ts)

    for name, ag in agents.items():
        qs, ws, ts, tc = A.simulate(ag.controller())
        res[name] = dict(A.metrics(qs, ws, ts), ctrl_time_per_step_ms=tc / A.N * 1e3)
        traj[name] = (qs, ws, ts)
    return res, traj, U_rk4


def plot_nominal(traj):
    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for name, (qs, ws, ts) in traj.items():
        c = COLORS[name]
        ax[0].plot(t_axis, np.rad2deg(A.angle_error(qs)), color=c, label=name)
        ax[1].plot(t_axis, np.linalg.norm(ws, axis=1), color=c, label=name)
        ax[2].plot(t_axis[1:], np.linalg.norm(ts, axis=1), color=c, label=name, lw=0.8)
    ax[0].set_ylabel('Ошибка ориентации, °')
    ax[1].set_ylabel('|ω|, рад/с')
    ax[2].set_ylabel('|τ|, Н·м')
    ax[2].set_xlabel('Время, с')
    ax[0].legend(fontsize=9)
    ax[1].set_yscale('log'); ax[1].set_ylim(1e-4, 2)
    ax[0].set_title('Номинальный сценарий: q0 = [0.707, 0.707, 0, 0], ω0 = [0.9, 0.5, 0.5]')
    fig.tight_layout(); fig.savefig(f'{OUT}/nominal_comparison.png', dpi=150); plt.close(fig)

    # Детальный вид последних 40 секунд
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, (qs, ws, ts) in traj.items():
        if name == 'CasADi (Эйлер)':
            continue
        ax.plot(t_axis, np.rad2deg(A.angle_error(qs)), color=COLORS[name], label=name)
    ax.set_xlim(80, 120); ax.set_ylim(0, 30)
    ax.axhline(2, color='k', ls=':', lw=1)
    ax.set_xlabel('Время, с'); ax.set_ylabel('Ошибка ориентации, °')
    ax.set_title('Окончание манёвра (пунктир — допуск 2°)'); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f'{OUT}/nominal_zoom.png', dpi=150); plt.close(fig)


def cos_angle_curve(qs):
    """Тот же показатель, что на графике предыдущей работы."""
    lin_v = np.array([0.01, 0.01, 0.005])
    corner = np.array([-0.05, -0.05, -0.05]); offset = np.array([0.05, 0.05, 0])
    pos_c, pos_k = [], []
    for k in range(A.N):
        q = qs[k] / np.linalg.norm(qs[k]); q0, q1, q2, q3 = q
        R = np.array([[1-2*(q2*q2+q3*q3), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
                      [2*(q1*q2+q0*q3), 1-2*(q1*q1+q3*q3), 2*(q2*q3-q0*q1)],
                      [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1*q1+q2*q2)]])
        p = lin_v * k * A.DT
        pos_c.append(p); pos_k.append(corner @ R.T + offset + p)
    v1 = np.diff(pos_c, axis=0); v2 = np.diff(pos_k, axis=0)
    return np.sum(v1*v2, 1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))


def plot_cos(traj):
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in ['CasADi (RK4)', 'PPO + рандомизация']:
        if name in traj:
            ax.plot(cos_angle_curve(traj[name][0]), color=COLORS[name], label=name)
    ax.set_ylim(-1, 1); ax.set_xlabel('Шаг'); ax.set_ylabel('Косинус угла')
    ax.set_title('Сонаправленность векторов движения'); ax.legend()
    fig.tight_layout(); fig.savefig(f'{OUT}/cos_angle.png', dpi=150); plt.close(fig)


def random_initial_conditions(n, seed=123, w_max=1.0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w = rng.uniform(-w_max, w_max, size=(n, 3))
    return q, w


def summarize(ms):
    ok = [m for m in ms if np.isfinite(m['t_settle'])]
    return dict(
        n=len(ms),
        success_rate=len(ok) / len(ms),
        mean_settle_time=float(np.mean([m['t_settle'] for m in ok])) if ok else None,
        median_final_angle_deg=float(np.median([m['final_angle_deg'] for m in ms])),
        mean_J=float(np.mean([m['J'] for m in ms])),
        mean_energy=float(np.mean([m['energy'] for m in ms])),
    )


def run_random(agents, n_fast=200, n_casadi=20):
    """Случайные начальные условия. CasADi решается заново для каждого."""
    qs0, ws0 = random_initial_conditions(n_fast)
    out = {}
    pd = A.make_pd(*PD_GAINS)
    out['ПД-регулятор'] = [A.metrics(*A.simulate(pd, q, w)[:3]) for q, w in zip(qs0, ws0)]
    for name, ag in agents.items():
        out[name] = [A.metrics(*A.simulate(ag.controller(), q, w)[:3]) for q, w in zip(qs0, ws0)]
        print(name, 'random done', flush=True)
    cas, times = [], []
    for i in range(n_casadi):
        try:
            U, X, J, t = A.solve_casadi_rk4(qs0[i], ws0[i], init=pd_warm_start(qs0[i], ws0[i]))
            cas.append(A.metrics(*A.simulate(A.casadi_open_loop(U), qs0[i], ws0[i])[:3]))
            times.append(t)
        except RuntimeError as e:          # IPOPT не сошёлся
            print('IPOPT failed on IC', i, e, flush=True)
            cas.append(dict(J=np.nan, t_settle=np.nan, final_angle_deg=np.nan,
                            final_rate=np.nan, energy=np.nan, mean_angle_deg=np.nan))
        print('casadi IC', i, round(times[-1] if times else 0, 1), 's', flush=True)
    out['CasADi (RK4)'] = cas
    summary = {k: summarize(v) for k, v in out.items()}
    # на тех же 20 начальных условиях, что и CasADi
    summary_20 = {k: summarize(v[:n_casadi]) for k, v in out.items()}
    summary['CasADi (RK4)']['mean_solve_time'] = float(np.mean(times))
    return summary, summary_20, out


def perturbed_inertia(n, seed=7):
    rng = np.random.default_rng(seed)
    Is = []
    for _ in range(n):
        d = 10 * rng.uniform(0.8, 1.2, 3)
        off = rng.uniform(-0.5, 0.5, 3)
        Is.append(np.array([[d[0], off[0], off[1]],
                            [off[0], d[1], off[2]],
                            [off[1], off[2], d[2]]]))
    return Is


def run_robustness(agents, U_rk4, n=50):
    """
    Номинальные начальные условия, но реальный тензор инерции отличается
    от номинального. Программа CasADi рассчитана по номинальной модели.
    """
    out = {}
    Is = perturbed_inertia(n)
    out['CasADi (RK4)'] = [A.metrics(*A.simulate(A.casadi_open_loop(U_rk4), I=I)[:3]) for I in Is]
    out['ПД-регулятор'] = [A.metrics(*A.simulate(A.make_pd(*PD_GAINS), I=I)[:3]) for I in Is]
    for name, ag in agents.items():
        out[name] = [A.metrics(*A.simulate(ag.controller(), I=I)[:3]) for I in Is]

    # зависимость от масштаба инерции (все моменты умножены на k)
    ks = np.linspace(0.7, 1.3, 13)
    scale = {name: [] for name in out}
    for k in ks:
        I = A.I_NOM * k
        scale['CasADi (RK4)'].append(A.metrics(*A.simulate(A.casadi_open_loop(U_rk4), I=I)[:3])['final_angle_deg'])
        scale['ПД-регулятор'].append(A.metrics(*A.simulate(A.make_pd(*PD_GAINS), I=I)[:3])['final_angle_deg'])
        for name, ag in agents.items():
            scale[name].append(A.metrics(*A.simulate(ag.controller(), I=I)[:3])['final_angle_deg'])

    fig, ax = plt.subplots(figsize=(9, 4))
    for name, v in scale.items():
        ax.plot(ks, v, 'o-', color=COLORS[name], label=name)
    ax.set_yscale('log'); ax.axhline(2, color='k', ls=':', lw=1)
    ax.set_xlabel('Отношение реального тензора инерции к номинальному')
    ax.set_ylabel('Ошибка ориентации в конце, °')
    ax.set_title('Устойчивость к ошибке в тензоре инерции'); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f'{OUT}/robustness_inertia.png', dpi=150); plt.close(fig)

    return {k: summarize(v) for k, v in out.items()}, dict(k=ks.tolist(), final_angle_deg=scale)


def plot_learning_curves(names):
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, label in names.items():
        if not os.path.exists(f'{name}_log.json'):
            continue
        log = json.load(open(f'{name}_log.json'))
        ax.plot([e['steps'] / 1e6 for e in log], [e['mean_ep_return'] for e in log],
                color=COLORS[label], label=label)
    ax.set_xlabel('Шаги взаимодействия со средой, млн')
    ax.set_ylabel('Суммарное вознаграждение за эпизод')
    ax.set_title('Кривые обучения PPO'); ax.legend()
    fig.tight_layout(); fig.savefig(f'{OUT}/learning_curves.png', dpi=150); plt.close(fig)


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    return x


if __name__ == '__main__':
    agents = {'PPO': PPOAgent.load('ppo_nominal.npz'),
              'PPO + рандомизация': PPOAgent.load('ppo_dr.npz')}
    plot_learning_curves({'ppo_nominal': 'PPO', 'ppo_dr': 'PPO + рандомизация'})

    nominal, traj, U_rk4 = run_all_nominal(agents)
    plot_nominal(traj); plot_cos(traj)
    print(json.dumps(to_jsonable(nominal), ensure_ascii=False, indent=1), flush=True)

    robust, scale = run_robustness(agents, U_rk4)
    print(json.dumps(to_jsonable(robust), ensure_ascii=False, indent=1), flush=True)

    rand, rand20, _ = run_random(agents)
    print(json.dumps(to_jsonable(rand), ensure_ascii=False, indent=1), flush=True)

    json.dump(to_jsonable(dict(nominal=nominal, random_ic=rand, random_ic_same20=rand20,
                               robustness=robust, inertia_scale=scale)),
              open(f'{OUT}/results.json', 'w'), ensure_ascii=False, indent=1)
    print('done')
