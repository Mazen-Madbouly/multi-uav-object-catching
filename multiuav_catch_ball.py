import casadi as ca
import rockit as r
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from fontTools.ttLib.tables import T_S_I__2

g = 9.81

# Parameters
m_d = 2.0          # drone mass (kg)
m_b = 0.2          # basket mass (kg)
m_o = 0.05         # ball mass (kg)

L = 1.5            # cable length (m)

# Hyperparameters
dt1guess = 1
dt2guess = 3
t1_guess = dt1guess
t2_guess = t1_guess + dt2guess
w_u = 1e-3         # weight on thrust magnitude in cost function
w_v = 10           # weight on relative velocity magnitude in cost function

# Simulation parameters
N1 = 30     # phase 1 sampled points
N2 = 15     # phase 2 sampled points

# Initial & terminal conditions
h0 = 0.5                   # initial altitude of basket
alpha0 = np.deg2rad(20)    # initial cable angle from vertical

p1_0 = ca.vertcat(0,  L*ca.sin(alpha0), -(h0+L*ca.cos(alpha0)))
p2_0 = ca.vertcat(0, -L*ca.sin(alpha0), -(h0+L*ca.cos(alpha0)))
pb_0 = ca.vertcat(0, 0, -h0)
po_0 = ca.vertcat(-5, 0, -3)

v1_0 = ca.vertcat(0, 0, 0)
v2_0 = ca.vertcat(0, 0, 0)
vb_0 = ca.vertcat(0, 0, 0)
vo_0 = ca.vertcat(4, 0, -5)

a1_0 = ca.vertcat(0, 0, 0)
a2_0 = ca.vertcat(0, 0, 0)

tau_0 = m_b*g / (2*ca.cos(alpha0))
u1_0 = ca.vertcat(0,  tau_0*ca.sin(alpha0), -(m_d*g + tau_0*ca.cos(alpha0)))
u2_0 = ca.vertcat(0, -tau_0*ca.sin(alpha0), -(m_d*g + tau_0*ca.cos(alpha0)))

v1_f = ca.vertcat(0, 0, 0)
v2_f = ca.vertcat(0, 0, 0)
vb_f = ca.vertcat(0, 0, 0)

a1_f = ca.vertcat(0, 0, 0)
a2_f = ca.vertcat(0, 0, 0)

# Control constraints parameters
theta_max = np.deg2rad(60)             # thrust tilt from -e_z
rho_min = 5                            # min drone thrust magnitude [N]
rho_max = 52                           # max drone thrust magnitude [N]
rhodot_min = -200                      # min rate of change of thrust magnitude [N/s]
rhodot_max =  300                      # max rate of change of thrust magnitude [N/s]

# Path constraints parameters
beta_max = np.deg2rad(37)  # cable tilt from +e_z
d_min = 0.5                # minimum UAV-UAV distance [m]
h_min = 1.0                # minimum UAV altitude [m]
h_max = 2.9                # max height of flight space [m]
tau_min = 0.1              # minimum tension [N]
eps_catch = 0.01           # catch tolerance [m^2]

v_poly = np.array([
    [-0.7911, -2.031],
    [-1.378,  -0.838],
    [-1.532,   0.574],
    [-1.284,   1.881],
    [-0.184,   2.429],
    [ 1.021,   2.035],
    [ 1.470,   0.834],
    [ 1.379,  -0.805],
    [ 0.952,  -2.026],
])

hull = sp.ConvexHull(v_poly)    # feasible flight space: intersection of halfspaces A
A = hull.equations[:, :2]
b = -hull.equations[:, 2]

def create_stage(ocp, t0, tf, m_d, m_l, N, A, b, catch =False):
    '''
    Create a rockit.stage object with system dynamics, control constraints and
    path constraints for this specific stage.
    '''
    # TODO: After successful solving of the problem, do a parametric study for the problem
    stage = ocp.stage(t0=t0, T=tf)

    # States
    p1 = stage.state(3)
    v1 = stage.state(3)
    p2 = stage.state(3)
    v2 = stage.state(3)
    pb = stage.state(3)
    vb = stage.state(3)
    # Control inputs
    u1 = stage.control(3)           # drone 1 thrust vector
    u2 = stage.control(3)           # drone 2 thrust vector
    # Parameters
    md = stage.parameter()
    ml = stage.parameter()
    # Algebraic variables
    tau1 = stage.algebraic()       # tension magnitude in cable 1
    tau2 = stage.algebraic()       # tension magnitude in cable 2

    # Dynamics
    stage.set_value(md,m_d)
    stage.set_value(ml,m_l)
    ez = ca.DM([0, 0, 1])
    e1 = (pb - p1) / ca.norm_2(pb - p1)
    e2 = (pb - p2) / ca.norm_2(pb - p2)
    T1 = tau1 * e1
    T2 = tau2 * e2
    a1 = (md*g*ez + T1 +u1)/md
    a2 = (md*g*ez + T2 +u2)/md
    ab = (ml*g*ez - T1 - T2)/ml

    stage.set_der(p1, v1)
    stage.set_der(p2, v2)
    stage.set_der(v1, a1)
    stage.set_der(v2, a2)
    stage.set_der(pb, vb)
    stage.set_der(vb, ab)

    stage.add_alg(ca.dot(pb - p1 , ab - a1) + ca.dot(vb - v1 , vb - v1))
    stage.add_alg(ca.dot(pb - p2 , ab - a2) + ca.dot(vb - v2 , vb - v2))

    # Control constraints
    dt = stage.T / N
    stage.subject_to(rho_min <= (ca.norm_2(u1) <= rho_max))                # thrust magnitude constraint
    stage.subject_to(rho_min <= (ca.norm_2(u2) <= rho_max))                # thrust magnitude constraint
    stage.subject_to(ca.dot(-ez,u1) >= ca.norm_2(u1) * ca.cos(theta_max))  # thrust tilt angle constraint
    stage.subject_to(ca.dot(-ez,u2) >= ca.norm_2(u2) * ca.cos(theta_max))  # thrust tilt angle constraint
    # thrust magnitude rate of change constraint
    stage.subject_to(rhodot_min * dt <= ((ca.norm_2(stage.next(u1)) - ca.norm_2(u1)) <= rhodot_max * dt))
    stage.subject_to(rhodot_min * dt <= ((ca.norm_2(stage.next(u2)) - ca.norm_2(u2)) <= rhodot_max * dt))

    # Path constraints
    stage.subject_to(ca.mtimes(A,p1[:2]) <= b)                              # feasible flight space
    stage.subject_to(ca.mtimes(A,p2[:2]) <= b)                              # feasible flight space
    stage.subject_to(ca.mtimes(A,pb[:2]) <= b)                              # feasible flight space
    stage.subject_to(-p1[2] <= h_max)                                       # max height of flight space
    stage.subject_to(-p2[2] <= h_max)                                       # max height of flight space
    stage.subject_to(-pb[2] <= h_max)                                       # max height of flight space
    stage.subject_to(-p1[2] >= h_min)                                       # min altitude
    stage.subject_to(-p2[2] >= h_min)                                       # min altitude
    stage.subject_to(-pb[2] >= 0)                                           # physical constraint ( floor )
    stage.subject_to(ca.norm_2(p1[:2] - p2[:2]) >= d_min)                   # min distance between drones
    stage.subject_to(tau1 > tau_min)                                        # positive tension
    stage.subject_to(tau2 > tau_min)                                        # positive tension
    stage.subject_to(ca.dot(ez,e1) >= ca.norm_2(T1) * ca.cos(beta_max))     # tension tilt angle constraint
    stage.subject_to(ca.dot(ez,e2) >= ca.norm_2(T2) * ca.cos(beta_max))     # tension tilt angle constraint

    po = vo = None
    if catch:
        # object states
        po = stage.state(3)
        vo = stage.state(3)

        # object dynamics
        ao = g*ez
        stage.set_der(po, vo)
        stage.set_der(vo, ao)

    # Method
    #stage.method(r.MultipleShooting(N=N, M=1, intg='idas'))
    stage.method(r.DirectCollocation(N=N, M=1, degree=3, scheme='radau'))

    return dict(stage=stage, p1=p1, p2=p2, pb=pb, po=po, v1=v1, v2=v2, vb=vb,
                vo=vo, a1=a1, a2=a2, ab=ab, u1=u1, u2=u2, tau1=tau1, tau2=tau2)

def link_stages(ocp, stage1, stage2):
    '''Define the continuity constraints to end time and states of stage1 and the
    initial time and state of stage2 to link them together.
    '''

    ocp.subject_to(stage1.tf == stage2.t0)
    # TODO: Check whether its needed to add an explict constraint for mass jump between stages
    for i in range(len(stage2.states)):
        ocp.subject_to(stage2.at_t0(stage2.states[i]) == stage1.at_tf(stage1.states[i]))

# Main - setting up the OCP
ocp = r.Ocp()

# Stage 1 - Approach and Catch
s1 = create_stage(ocp, t0=0, tf=r.FreeTime(t1_guess),m_d=m_d, m_l=m_b, N=N1, A=A, b=b, catch=True)
stage1 = s1['stage']

# Stage 1 - Initial condition constraints
ocp.subject_to(stage1.at_t0(s1['p1']) == p1_0)
ocp.subject_to(stage1.at_t0(s1['p2']) == p2_0)
ocp.subject_to(stage1.at_t0(s1['pb']) == pb_0)
ocp.subject_to(stage1.at_t0(s1['po']) == po_0)
ocp.subject_to(stage1.at_t0(s1['v1']) == v1_0)
ocp.subject_to(stage1.at_t0(s1['v2']) == v2_0)
ocp.subject_to(stage1.at_t0(s1['vb']) == vb_0)
ocp.subject_to(stage1.at_t0(s1['vo']) == vo_0)
ocp.subject_to(stage1.at_t0(s1['a1']) == a1_0)
ocp.subject_to(stage1.at_t0(s1['a2']) == a2_0)

# Stage 1 - Terminal condition
ocp.subject_to(stage1.at_tf(ca.sumsqr(s1['pb']-s1['po'])) <= eps_catch )

# Stage 1 - Cost function
stage1.add_objective(stage1.T)
stage1.add_objective(stage1.integral(w_u * (ca.sumsqr(s1['u1']) + ca.sumsqr(s1['u2']))
                                   + w_v * (ca.sumsqr(s1['vb'] - s1['vo']))))

# Stage 2 - Stopping
s2 = create_stage(ocp, t0=r.FreeTime(t1_guess), tf=r.FreeTime(t2_guess),m_d=m_d, m_l=m_b+m_o, N=N2, A=A, b=b)
stage2 = s2['stage']

link_stages(ocp, stage1, stage2)

# Terminal condition constraints
n = stage2.at_tf(ca.cross(s2['pb']-s2['p1'] , s2['pb']-s2['p2']))
ocp.subject_to(stage2.at_tf(s2['v1']) == v1_f)
ocp.subject_to(stage2.at_tf(s2['v2']) == v2_f)
ocp.subject_to(stage2.at_tf(s2['a1']) == a1_f)
ocp.subject_to(stage2.at_tf(s2['a2']) == a2_f)
ocp.subject_to(ca.dot(n , stage2.at_tf(s2['vb'])) == 0)
ocp.subject_to(ca.dot(n , stage2.at_tf(s2['ab'])) == 0)

# Stage 2 - Cost function
stage2.add_objective(stage2.T)
stage2.add_objective(stage2.integral(w_u * (ca.sumsqr(s2['u1']) + ca.sumsqr(s2['u2']))))

ocp.subject_to(stage1.T > 0.1)
ocp.subject_to(stage2.T > 0.5)

# Initial guess
stage1.set_initial(s1['p1'], p1_0)
stage1.set_initial(s1['p2'], p2_0)
stage1.set_initial(s1['pb'], pb_0)
stage1.set_initial(s1['u1'], u1_0)
stage1.set_initial(s1['u2'], u2_0)
stage1.set_initial(s1['tau1'], tau_0)
stage1.set_initial(s1['tau2'], tau_0)
stage1.set_initial(s1['po'], po_0)
stage1.set_initial(s1['vo'], vo_0)

stage2.set_initial(s2['p1'], p1_0)
stage2.set_initial(s2['p2'], p2_0)
stage2.set_initial(s2['pb'], pb_0)
stage2.set_initial(s2['u1'], u1_0)
stage2.set_initial(s2['u2'], u2_0)
stage2.set_initial(s2['tau1'], tau_0)
stage2.set_initial(s2['tau2'], tau_0)

# Solver options
opts = {"expand": True,
        "verbose": False,
        "print_time": True,
        "error_on_fail": False,
        "ipopt": {"linear_solver": "mumps",  # "ma57" is faster!
                  "max_iter": 2000,
                  'print_level': 5,
                  'sb': 'yes',  # Suppress IPOPT banner
                  'tol': 1e-6,
                  }}
ocp.solver("ipopt", opts)

try:
    sol = ocp.solve()
    T1 = sol.value(stage1.T)
    T2 = sol.value(stage2.T)
    print(f"T1 = {T1}", f" T2 ={T2}")
except RecursionError as e:
    sol = ocp.non_converged_solution
    ocp.show_infeasibilities(1e-6)  # constraints violated by more than 1e-6
    print(e)

stats = sol.stats
print(f"\n{stats['return_status']} after {stats['iter_count']} iterations")

if not stats['success']:
    print("*** Not converged: plots show the last iterate, not an optimum ***")

ocp.spy()
ocp.jacobian()
ocp.hessian()
ocp.is_parameter_appearing_in_sys()
ocp.gist()
ocp.debug()

# Structural rank of the equality-constraint Jacobian: rank < rows means some
# equalities are redundant (e.g. rows that are identically 0 = 0)
opti = ocp._method.opti
eq = np.where(ca.DM(sol.value(opti.lbg)).full().ravel()
              == ca.DM(sol.value(opti.ubg)).full().ravel())[0]
J_eq = ca.DM.ones(J_sp)[eq.tolist(), :]
print(f"equality Jacobian: {len(eq)} rows, structural rank {ca.sprank(J_eq.sparsity())}")

# =============================================================================
# Post-processing
# =============================================================================
def sample(key, grid='integrator', stages=(s1, s2)):
    '''Sample a dict entry ('p1', 'u1', ...) or a function of the stage dict
    (lambda s: ...) over the given stages and join them in time.'''
    ts, xs = [], []
    for s in stages:
        expr = s[key] if isinstance(key, str) else key(s)
        kw = dict(refine=10) if grid == 'integrator' else {}
        t, x = sol(s['stage']).sample(expr, grid=grid, **kw)
        ts.append(t); xs.append(x)
    return np.concatenate(ts), np.concatenate(xs)

T1 = float(sol.value(stage1.T))
T2 = float(sol.value(stage2.T))

def finish(axs, title):
    '''Catch marker, grid and labels for a column of time plots.'''
    for ax in np.atleast_1d(axs):
        ax.axvline(T1, color='k', ls='--', lw=0.8)
        ax.grid(True)
        ax.legend(loc='best', fontsize=8)
    np.atleast_1d(axs)[-1].set_xlabel('t [s]')
    plt.suptitle(title)
    plt.tight_layout()

# Catch summary (last sample of stage 1)
_, pb_1 = sample('pb', stages=(s1,));  _, po_1 = sample('po', stages=(s1,))
_, vb_1 = sample('vb', stages=(s1,));  _, vo_1 = sample('vo', stages=(s1,))
print(f"T1 = {T1:.3f} s, T2 = {T2:.3f} s")
print(f"catch: |pb-po| = {np.linalg.norm(pb_1[-1]-po_1[-1]):.3f} m, "
      f"|vb-vo| = {np.linalg.norm(vb_1[-1]-vo_1[-1]):.3f} m/s, "
      f"basket altitude = {-pb_1[-1, 2]:.2f} m")

# 1) 3D positions (altitude = -z in NED)
ax = plt.figure(figsize=(8, 6)).add_subplot(projection='3d')
for key in ('p1', 'p2', 'pb'):
    _, p = sample(key)
    ax.plot(p[:, 0], p[:, 1], -p[:, 2], label=key)
    ax.scatter(p[0, 0], p[0, 1], -p[0, 2], marker='o')
ax.plot(po_1[:, 0], po_1[:, 1], -po_1[:, 2], 'k:', label='ball')
ax.scatter(pb_1[-1, 0], pb_1[-1, 1], -pb_1[-1, 2], color='r', s=40, label='catch')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('altitude [m]')
ax.legend(); plt.title('Positions (o = start)')

# 2) Velocities, component by component
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
for j, c in enumerate('xyz'):
    for key in ('v1', 'v2', 'vb'):
        t, v = sample(key)
        axs[j].plot(t, v[:, j], label=key)
    t, v = sample('vo', stages=(s1,))
    axs[j].plot(t, v[:, j], 'k:', label='vo')
    axs[j].set_ylabel(f'v{c} [m/s]')
finish(axs, 'Velocities (NED)')

# 3) Cable tension (algebraic variables live on the collocation points)
fig, ax = plt.subplots(figsize=(9, 3.5))
for key in ('tau1', 'tau2'):
    t, tau = sample(key, grid='integrator_roots')
    ax.plot(t, tau, '.-', ms=3, label=key)
ax.axhline(tau_min, color='r', lw=0.8, label='tau_min')
ax.set_ylabel('tension [N]')
finish(ax, 'Cable tension')

# 4) Control actions: magnitude, tilt and rate of change of magnitude
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
for key in ('u1', 'u2'):
    t, un = sample(lambda s: ca.norm_2(s[key]), grid='control')
    t, tilt = sample(lambda s: ca.acos(-s[key][2] / ca.norm_2(s[key])), grid='control')
    axs[0].step(t, un, where='post', label=key)
    axs[1].step(t, np.rad2deg(tilt), where='post', label=key)
    dt_, dun = np.diff(t), np.diff(un)
    ok = dt_ > 0                     # skip the zero-length step at the stage boundary
    axs[2].plot(t[:-1][ok], dun[ok] / dt_[ok], '.-', label=key)
for y in (rho_min, rho_max):        axs[0].axhline(y, color='r', lw=0.8)
axs[1].axhline(np.rad2deg(theta_max), color='r', lw=0.8)
for y in (rhodot_min, rhodot_max):  axs[2].axhline(y, color='r', lw=0.8)
axs[0].set_ylabel('||u|| [N]'); axs[1].set_ylabel('tilt [deg]'); axs[2].set_ylabel('d||u||/dt [N/s]')
finish(axs, 'Control actions')

# 5) Relative velocity basket - ball (stage 1)
t, dv = sample(lambda s: s['vb'] - s['vo'], stages=(s1,))
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
for j, c in enumerate('xyz'):
    axs[0].plot(t, dv[:, j], label=c)
axs[1].plot(t, np.linalg.norm(dv, axis=1), label='|vb - vo|')
axs[0].set_ylabel('vb - vo [m/s]'); axs[1].set_ylabel('[m/s]')
finish(axs, 'Relative velocity basket - ball')

# 6) Relative position basket - ball (stage 1)
t, dp = sample(lambda s: s['pb'] - s['po'], stages=(s1,))
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
for j, c in enumerate('xyz'):
    axs[0].plot(t, dp[:, j], label=c)
axs[1].semilogy(t, np.linalg.norm(dp, axis=1), label='|pb - po|')
axs[1].axhline(np.sqrt(eps_catch), color='r', lw=0.8, label='catch radius')
axs[0].set_ylabel('pb - po [m]'); axs[1].set_ylabel('[m]')
finish(axs, 'Relative position basket - ball')

# 7) Constraint check: altitudes, drone separation, cable tilt, cable length
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(9, 10))
for key in ('p1', 'p2', 'pb'):
    t, p = sample(key)
    axs[0].plot(t, -p[:, 2], label=key)
for y in (0, h_min, h_max):  axs[0].axhline(y, color='r', lw=0.8)
t, d = sample(lambda s: ca.norm_2(s['p1'][:2] - s['p2'][:2]))
axs[1].plot(t, d, label='|p1 - p2| (xy)')
axs[1].axhline(d_min, color='r', lw=0.8)
for p in ('p1', 'p2'):
    t, ang = sample(lambda s: ca.acos((s['pb'][2] - s[p][2]) / ca.norm_2(s['pb'] - s[p])))
    axs[2].plot(t, np.rad2deg(ang), label=f'cable {p[-1]}')
    t, ln = sample(lambda s: ca.norm_2(s['pb'] - s[p]))
    axs[3].plot(t, ln - L, label=f'cable {p[-1]}')
axs[2].axhline(np.rad2deg(beta_max), color='r', lw=0.8)
axs[0].set_ylabel('altitude [m]'); axs[1].set_ylabel('distance [m]')
axs[2].set_ylabel('cable tilt [deg]'); axs[3].set_ylabel('length - L [m]')
finish(axs, 'Constraint check')

plt.show()