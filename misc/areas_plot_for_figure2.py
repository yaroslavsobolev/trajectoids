import matplotlib.pyplot as plt
import numpy as np

from compute_trajectoid import *

def plot_gb_areas(ax, sweeped_scales, gb_areas, mark_one_scale, scale_to_mark):

    ## This code injects the sampled point at the solution_scale. Otherwise the root can be not at one of sampled points
    # ii = np.searchsorted(sweeped_scales, solution_scale)
    # sweeped_scales = np.insert(sweeped_scales, ii, solution_scale)
    # mismatch_angles = np.insert(mismatch_angles, ii, np.pi * np.sign(interp1d(sweeped_scales, gb_areas)(solution_scale)))

    ax.plot(sweeped_scales, gb_areas)
    ax.axhline(y=np.pi, color='black', alpha=0.5)
    ax.axhline(y=0, color='black', alpha=0.3)
    ax.axhline(y=-1 * np.pi, color='black', alpha=0.5)
    if mark_one_scale:
        # ax.scatter([solution_scale], [np.pi * np.sign(interp1d(sweeped_scales, gb_areas)(solution_scale))], s=20, color='red')
        value_at_scale_to_mark = interpolate.interp1d(sweeped_scales, gb_areas)(scale_to_mark)
        ax.scatter([scale_to_mark], [value_at_scale_to_mark], s=20, color='red')
        scale_to_mark = 0.37
        value_at_scale_to_mark = interpolate.interp1d(sweeped_scales, gb_areas)(scale_to_mark)
        ax.scatter([scale_to_mark], [value_at_scale_to_mark], s=20, color='C0')
        scale_to_mark = 0.649
        value_at_scale_to_mark = interpolate.interp1d(sweeped_scales, gb_areas)(scale_to_mark)
        ax.scatter([scale_to_mark], [value_at_scale_to_mark], s=20, color='C0')
    ax.set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
    ax.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])
    ax.set_ylim(-np.pi * 2 * 1.01, np.pi * 2 * 1.01)
    ax.set_ylabel('Norm. spherical\n area $S(r)/r^2$')
    ax.set_xlabel('$1/r$')

# Second attempt:
plotscalefac = 0.65
fig, ax = plt.subplots(figsize=(7.3*plotscalefac,2.5*plotscalefac))
length_of_path = np.load('examples/penannular_smooth/length_of_path.npy')
xfactor = length_of_path/(2 * np.pi)
sweeped_scales = np.load('examples/penannular_smooth/sweeped_scales.npy')
gb_areas = np.load('examples/penannular_smooth/gb_areas.npy')

mark_one_scale=True
scale_to_mark=1
ax.plot(sweeped_scales * xfactor, gb_areas, color='C1')
ax.axhline(y=np.pi, color='black', alpha=0.5)
ax.axhline(y=0, color='black', alpha=0.3)
ax.axhline(y=-1 * np.pi, color='black', alpha=0.5)
if mark_one_scale:
    # ax.scatter([solution_scale], [np.pi * np.sign(interp1d(sweeped_scales, gb_areas)(solution_scale))], s=20, color='red')
    value_at_scale_to_mark = interpolate.interp1d(sweeped_scales, gb_areas)(scale_to_mark)
    ax.scatter([scale_to_mark * xfactor], [value_at_scale_to_mark], s=20, color='red', zorder=20)
    scale_to_mark = 0.37
    value_at_scale_to_mark = interpolate.interp1d(sweeped_scales, gb_areas)(scale_to_mark)
    ax.scatter([scale_to_mark * xfactor], [value_at_scale_to_mark], s=20, color='black', zorder=20)
    scale_to_mark = 0.649
    value_at_scale_to_mark = interpolate.interp1d(sweeped_scales, gb_areas)(scale_to_mark)
    ax.scatter([scale_to_mark * xfactor], [value_at_scale_to_mark], s=20, color='black', zorder=20)
# ax.set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
# ax.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])
# ax.set_ylim(-np.pi * 2 * 1.01, np.pi * 2 * 1.01)
# ax.set_ylabel('Area $S(\sigma)$')
# ax.set_xlabel('Path\'s scale factor $\sigma$')
ax.set_xlim(-0.01, 5)

ax2 = ax.twiny()
sweeped_scales = np.load('examples/brownian_path_2/figures/sweeped_scales.npy')
gb_areas = np.load('examples/brownian_path_2/figures/gb_areas.npy')
length_of_path = np.load('examples/brownian_path_2/figures/length_of_path.npy')
xfactor = length_of_path/(2 * np.pi)

renormalize_the_scale_by = 1
sweeped_scales = np.array(sweeped_scales)/ renormalize_the_scale_by

mark_one_scale=True
scale_to_mark=27.136794417872355 / renormalize_the_scale_by
ax2.plot(sweeped_scales * xfactor, gb_areas, color='C4')
ax2.axhline(y=np.pi, color='black', alpha=0.5)
ax2.axhline(y=0, color='black', alpha=0.3)
ax2.axhline(y=-1 * np.pi, color='black', alpha=0.5)
if mark_one_scale:
    # ax.scatter([solution_scale], [np.pi * np.sign(interp1d(sweeped_scales, gb_areas)(solution_scale))], s=20, color='red')
    value_at_scale_to_mark = interpolate.interp1d(sweeped_scales, gb_areas)(scale_to_mark)
    ax2.scatter([scale_to_mark * xfactor], [value_at_scale_to_mark], s=20, color='red', zorder=20)
ax2.set_yticks([-np.pi/2, 0, np.pi])
ax2.set_yticklabels(['$-$π/2', '0', 'π'])
ax2.set_ylim(-np.pi/2, 1.01, np.pi * 1 * 1.3)
ax.set_ylabel('Norm. spherical\n area $S(r)/r^2$')
ax.set_xlabel('Scale $\sigma = L/(2 \pi r)$')
ax2.set_xlim(-0.1, 78)


plt.ylim(-np.pi/2, 4)
plt.tight_layout()
fig.savefig('misc/figures/areas.png', dpi=300)
plt.show()