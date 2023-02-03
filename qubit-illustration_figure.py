import numpy as np
import matplotlib.pyplot as plt

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

omega_0 = 100

def draw_pulse(S, scale_type='time'):
    """
    Draw field pulse.

    .. image:: https://placekitten.com/200/139

    :param S:
    :param frame_id:
    :param scale_type:
    :return:
    """
    # def delta(t):
    #     delta = np.ones_like(t)
    #     if scale_type == 'time':
    #         delta[t/S < 0.4] = 0
    #         delta[t/S > 0.6] = 2
    #         return delta * 120
    #     elif scale_type == 'value':
    #         delta[t < 0.4] = 0
    #         delta[t > 0.6] = 2
    #         return S * delta * 120

    def phaseshift(t):
        pt = np.zeros_like(t)
        slope = 70
        if scale_type == 'time':
            mask = np.logical_and(t/S > 0, t/S < 0.4)
            pt[mask] = t[mask]/S*slope
            mask = np.logical_and(t/S >= 0.4, t/S < 0.6)
            pt[mask] = 0.4*slope
            mask = np.logical_and(t/S >= 0.6, t/S < 1)
            pt[mask] = 0.4*slope - (t[mask]/S - 0.6)*slope
        else:
            mask = np.logical_and(t > 0, t < 0.4)
            pt[mask] = t[mask]*slope
            mask = np.logical_and(t >= 0.4, t < 0.6)
            pt[mask] = 0.4*slope
            mask = np.logical_and(t >= 0.6, t < 1)
            pt[mask] = 0.4*slope - (t[mask] - 0.6)*slope
        return pt

    # def E(t):
    #     return 1 + 0.1*t

    def E(t):
        if scale_type == 'time':
            return gaussian(t/S, 0.5, 0.15)
        elif scale_type == 'value':
            return S * gaussian(t, 0.5, 0.15)

    ts = np.linspace(0, 1.5, 5000)
    dt = ts[1] - ts[0]
    phase = 0
    phases = []
    # phaseshifts = []
    # for i in range(ts.shape[0]):
    #     phase = omega_0*dt + phaseshift(ts[i])
    #     phases.append(phase)
    #     # phaseshifts.append(phase - omega)
    # phases = np.array(phases)

    phases = omega_0*ts + phaseshift(ts)
    signal = E(ts) * np.sin(phases)
    return ts, signal, E(ts), phaseshift(ts)
    # plt.show()

figsizefactor = 0.8
fig, axarr = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True, figsize=(11*figsizefactor,4.7*figsizefactor))

def add_plots(S = 1.5, color = 'C0', scale_type='time', label=None):
    ts, signal, Es, phaseshifts = draw_pulse(S, scale_type=scale_type)
    ax = axarr[0]
    # ax.set_title(f'Scale factor: {S:.3f}')

    ax.plot(ts, signal, color=color, alpha=0.5, label=label)
    ax.plot(ts, Es, '--', color=color)
    ax.set_ylabel('Field strength\n, a.u.')
    ax.set_ylim(-1.6, 1.6)
    # ax.legend()
    ax = axarr[1]
    if scale_type == 'time':
        ax.plot(ts, phaseshifts, color=color, linewidth=2 if color == 'black' else 2,
                zorder=10 if color == 'black' else 5,
                alpha=1 if color == 'black' else 0.7)
    else:
        ax.plot(ts, phaseshifts, color=color, linewidth=2 if color == 'black' else 6,
                zorder=10 if color == 'black' else 5,
                alpha=1 if color == 'black' else 0.7)

add_plots(S=1, color='black', scale_type='time', label='Input pulse')
add_plots(S=1.5, color='C0', scale_type='time')
add_plots(S=1.5, color='C2', scale_type='value')


ax = axarr[1]
ax.set_xlabel('Time, a.u.')
ax.set_ylabel('Phase shift,\n rad.')
ax.set_ylim(0, 33)
ax.set_xlim(0, 1.55)

for ax in axarr:
    simpleaxis(ax)
axarr[0].spines['bottom'].set_visible(False)
axarr[0].tick_params(bottom=False)
# axarr[0].legend()
plt.tight_layout()
fig.savefig('tests/qubits/figures/for_si_figure_noborders.png', dpi=300)
plt.show()
# fig.savefig(f'tests/qubits/figures/frames_{scale_type}/{frame_id:08d}.png', dpi=300)
# plt.close(fig)

# # def delta(t):
# #     return 600*t**2
# for frame, S in enumerate(np.linspace(1,1.5, 100)):
#     print(f'Rendering frame {frame}')
#     draw_pulse(S, frame, scale_type='time')

