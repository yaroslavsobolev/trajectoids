from compute_trajectoid import *

def replot_path_from_numpy(target_folder, plot_midpoints=False):
    input_path = np.load(target_folder + 'folder_for_path/path_data.npy')
    plot_three_path_periods(input_path, plot_midpoints=plot_midpoints, savetofile=target_folder + '/input_path')

two_period_trajectoid_folders = \
    ['random_doubled_1',
    'random_doubled_2',
    'random_doubled_3',
    'random_doubled_4',
    'random_doubled_5',
    'little-prince-2']
#
# for folder_name in two_period_trajectoid_folders:
#     replot_path_from_numpy(f'examples/{folder_name}/', plot_midpoints=True)

one_period_trajectoid_folders = \
    ['random_unclosed_1',
    'swirl_1',
    'zago_v4']

for folder_name in one_period_trajectoid_folders:
    replot_path_from_numpy(f'examples/{folder_name}/', plot_midpoints=False)