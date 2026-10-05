import time
import numpy as np


I_NOM = np.diag([10.0, 10.0, 10.0])
DT = 0.1
N = 1200
TAU_MAX = 0.1
Q_TARGET = np.array([1.0, 0.0, 0.0, 0.0])
Q0 = np.array([0.707, 0.707, 0.0, 0.0])
W0 = np.array([0.9, 0.5, 0.5])


ANGLE_TOL = np.deg2rad(2.0)
RATE_TOL = 0.01


def quat_mul(p, q):
    p0, p1, p2, p3 = np.moveaxis(p, -1, 0)
    q0, q1, q2, q3 = np.moveaxis(q, -1, 0)
    return np.stack([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0,
    ], axis=-1)


def quat_conj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def error_quat(q, q_target=Q_TARGET):
    qe = quat_mul(quat_conj(q_target), q)
    sign = np.where(qe[..., :1] < 0, -1.0, 1.0)
    return qe * sign


def angle_error(q, q_target=Q_TARGET):
    qe = quat_mul(quat_conj(q_target), q)
    qn = np.linalg.norm(q, axis=-1)
    return 2.0 * np.arccos(np.clip(np.abs(qe[..., 0]) / qn, 0.0, 1.0))


def derivatives(q, w, tau, I, I_inv):
    w_quat = np.concatenate([np.zeros(w.shape[:-1] + (1,)), w], axis=-1)
    dq = 0.5 * quat_mul(q, w_quat)
    Iw = np.einsum('...ij,...j->...i', I, w)
    dw = np.einsum('...ij,...j->...i', I_inv, tau - np.cross(w, Iw))
    return dq, dw


def step(q, w, tau, I=I_NOM, I_inv=None, dt=DT, substeps=4):
    if I_inv is None:
        I_inv = np.linalg.inv(I)
    h = dt / substeps
    for _ in range(substeps):
        k1q, k1w = derivatives(q, w, tau, I, I_inv)
        k2q, k2w = derivatives(q + 0.5*h*k1q, w + 0.5*h*k1w, tau, I, I_inv)
        k3q, k3w = derivatives(q + 0.5*h*k2q, w + 0.5*h*k2w, tau, I, I_inv)
        k4q, k4w = derivatives(q + h*k3q, w + h*k3w, tau, I, I_inv)
        q = q + h/6.0 * (k1q + 2*k2q + 2*k3q + k4q)
        w = w + h/6.0 * (k1w + 2*k2w + 2*k3w + k4w)
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q, w


def simulate(controller, q0=Q0, w0=W0, I=I_NOM, n_steps=N):
    I_inv = np.linalg.inv(I)
    q = np.asarray(q0, float) / np.linalg.norm(q0)
    w = np.asarray(w0, float)
    qs, ws, taus = [q], [w], []
    t_ctrl = 0.0
    for k in range(n_steps):
        t0 = time.perf_counter()
        tau = np.clip(controller(k, q, w), -TAU_MAX, TAU_MAX)
        t_ctrl += time.perf_counter() - t0
        q, w = step(q, w, tau, I, I_inv)
        qs.append(q); ws.append(w); taus.append(tau)
    return np.array(qs), np.array(ws), np.array(taus), t_ctrl


def metrics(qs, ws, taus):
    theta = angle_error(qs)
    wn = np.linalg.norm(ws, axis=1)
    ok = (theta < ANGLE_TOL) & (wn < RATE_TOL)

    if ok[-1]:
        bad = np.where(~ok)[0]
        t_settle = (bad[-1] + 1) * DT if len(bad) else 0.0
    else:
        t_settle = np.nan

    dq = np.minimum(np.sum((qs[1:] - Q_TARGET)**2, axis=1),
                    np.sum((qs[1:] + Q_TARGET)**2, axis=1))
    J = np.sum(taus**2) + np.sum(dq) + np.sum(ws[-1]**2)
    return dict(
        J=J,
        t_settle=t_settle,
        final_angle_deg=np.rad2deg(theta[-1]),
        final_rate=wn[-1],
        energy=np.sum(taus**2) * DT,
        mean_angle_deg=np.rad2deg(theta.mean()),
    )


def solve_casadi(q0=Q0, w0=W0, I=I_NOM, n_steps=N, verbose=False):
    import casadi as ca
    I_inv = np.linalg.inv(I)
    opti = ca.Opti()
    U = opti.variable(3, n_steps)
    X = opti.variable(7, n_steps + 1)
    opti.subject_to(X[:, 0] == ca.vertcat(np.asarray(q0, float), np.asarray(w0, float)))
    J = 0
    for k in range(n_steps):
        q, w, u = X[0:4, k], X[4:7, k], U[:, k]
        w_dot = I_inv @ (u - ca.cross(w, I @ w))
        dq = 0.5 * ca.vertcat(
            -q[1]*w[0] - q[2]*w[1] - q[3]*w[2],
            q[0]*w[0] + q[2]*w[2] - q[3]*w[1],
            q[0]*w[1] - q[1]*w[2] + q[3]*w[0],
            q[0]*w[2] + q[1]*w[1] - q[2]*w[0])
        opti.subject_to(X[:, k+1] == ca.vertcat(q + DT*dq, w + DT*w_dot))
        J += ca.sumsqr(u) + ca.sumsqr(X[0:4, k+1] - Q_TARGET)
    J += ca.sumsqr(X[4:7, n_steps])
    opti.minimize(J)
    opti.subject_to(opti.bounded(-TAU_MAX, ca.vec(U), TAU_MAX))
    opts = {} if verbose else {'print_time': 0}
    sopts = {} if verbose else {'print_level': 0, 'sb': 'yes'}
    opti.solver('ipopt', opts, sopts)
    t0 = time.perf_counter()
    sol = opti.solve()
    t_solve = time.perf_counter() - t0
    return np.array(sol.value(U)).T, np.array(sol.value(X)), float(sol.value(J)), t_solve


def solve_casadi_rk4(q0=Q0, w0=W0, I=I_NOM, n_steps=N, init=None, verbose=False):
    import casadi as ca
    I_inv = np.linalg.inv(I)
    x = ca.MX.sym('x', 7); u = ca.MX.sym('u', 3)

    def f(x, u):
        q, w = x[0:4], x[4:7]
        dq = 0.5 * ca.vertcat(
            -q[1]*w[0] - q[2]*w[1] - q[3]*w[2],
            q[0]*w[0] + q[2]*w[2] - q[3]*w[1],
            q[0]*w[1] - q[1]*w[2] + q[3]*w[0],
            q[0]*w[2] + q[1]*w[1] - q[2]*w[0])
        return ca.vertcat(dq, I_inv @ (u - ca.cross(w, I @ w)))

    k1 = f(x, u); k2 = f(x + DT/2*k1, u); k3 = f(x + DT/2*k2, u); k4 = f(x + DT*k3, u)
    F = ca.Function('F', [x, u], [x + DT/6*(k1 + 2*k2 + 2*k3 + k4)])
    Fmap = F.map(n_steps)

    opti = ca.Opti()
    U = opti.variable(3, n_steps)
    X = opti.variable(7, n_steps + 1)
    q0n = np.asarray(q0, float) / np.linalg.norm(q0)
    opti.subject_to(X[:, 0] == ca.vertcat(q0n, np.asarray(w0, float)))
    opti.subject_to(X[:, 1:] == Fmap(X[:, :-1], U))
    J = ca.sumsqr(U) + ca.sumsqr(X[0:4, 1:] - ca.repmat(Q_TARGET, 1, n_steps)) \
        + ca.sumsqr(X[4:7, n_steps])
    opti.minimize(J)
    opti.subject_to(opti.bounded(-TAU_MAX, ca.vec(U), TAU_MAX))
    if init is not None:
        qi, wi, ui = init
        opti.set_initial(X, np.vstack([qi.T, wi.T]))
        opti.set_initial(U, ui.T)
    opts = {} if verbose else {'print_time': 0}
    sopts = {'max_iter': 3000}
    if not verbose:
        sopts.update({'print_level': 0, 'sb': 'yes'})
    opti.solver('ipopt', opts, sopts)
    t0 = time.perf_counter()
    sol = opti.solve()
    t_solve = time.perf_counter() - t0
    return np.array(sol.value(U)).T, np.array(sol.value(X)), float(sol.value(J)), t_solve


def casadi_open_loop(U):
    return lambda k, q, w: U[k]


def make_pd(kp=0.1, kd=1.0):
    def ctrl(k, q, w):
        qe = error_quat(q)
        return -kp * qe[1:] - kd * w
    return ctrl
