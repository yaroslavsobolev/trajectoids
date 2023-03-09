import numpy as np
from numpy.linalg import norm as lnorm
import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import convex_hull_image, remove_small_objects
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import label
from scipy import interpolate
from scipy.optimize import curve_fit
from compute_trajectoid import rotate_2d
import os
from tqdm import tqdm

def makedir_if_needed(path):
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

def number_of_files(target_dir):
    _, _, files = next(os.walk(target_dir))
    file_count = len(files)
    return file_count

def convert_to_signal(raw_frame, two_colors=False):
    if not two_colors:
        return raw_frame[:,:,2].astype(np.float) - raw_frame[:,:,1].astype(np.float) + raw_frame[:,:,0].astype(np.float)
    else:
        return raw_frame[:, :, 1].astype(np.float) + 1.5*raw_frame[:, :, 0].astype(np.float)
        # return -raw_frame[:, :, 2].astype(np.float) + raw_frame[:, :, 1].astype(np.float) + raw_frame[:, :, 0].astype(
        # np.float)

def get_median_frame(min_frame, target_folder, nframes, step=10, two_colors=False):
    list_of_frames = []
    for frame_id in range(0, nframes, step):
        print(f'Loading frame {frame_id} for background.')
        if frame_id < min_frame:
            continue
        frame_file = target_folder + '/frames/frame{0:03d}.jpg'.format(frame_id)
        raw_frame = io.imread(frame_file)
        list_of_frames.append(convert_to_signal(raw_frame, two_colors=two_colors))
    return np.median(np.array(list_of_frames), axis=0)

def trace_trajectory_from_video_frames(target_folder, threshold=25, min_frame=0, nframes=False, do_debug_plots=False,
                                       two_colors=False, bkg_nframes=False, bkg_minframe=False, bkg_step=10):
    if not nframes:
        nframes = number_of_files(target_folder + '/frames/')
    makedir_if_needed(target_folder + '/processed_frames')

    if not bkg_nframes:
        bkg_nframes = nframes
    if not bkg_minframe:
        bkg_minframe = min_frame

    #get background
    background_frame = get_median_frame(bkg_minframe, target_folder, nframes=bkg_nframes, two_colors=two_colors, step=bkg_step)
    if do_debug_plots:
        plt.imshow(background_frame)
        plt.show()

    cmass_xs = []
    cmass_ys = []
    for frame_id in tqdm(range(nframes)):
        if frame_id < min_frame:
            continue
        frame_file = target_folder + '/frames/frame{0:03d}.jpg'.format(frame_id)
        raw_frame = io.imread(frame_file)
        fig, ax = plt.subplots(figsize=(8, 8 * raw_frame.shape[0] / raw_frame.shape[1]))
        if frame_id >= min_frame:
            channel_diff = convert_to_signal(raw_frame, two_colors=two_colors) - background_frame
            # if two_colors:
            #     channel_diff = np.abs(channel_diff)
            if do_debug_plots:
                plt.imshow(channel_diff)
                plt.show()
            frame = channel_diff > threshold
            if do_debug_plots:
                plt.imshow(frame)
                plt.show()

            def getLargestCC(segmentation):
                labels = label(segmentation)
                assert( labels.max() != 0 ) # assume at least 1 CC
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                return largestCC
            largest = getLargestCC(frame)
            largest_comp = np.zeros_like(frame)
            largest_comp[largest] = 1
            # plt.imshow(largest_comp)
            # frame = remove_small_objects(frame, min_size=15500)
            # plt.imshow(frame)
            frame = largest_comp

            chull = convex_hull_image(frame)
            if do_debug_plots:
                plt.imshow(chull)
                plt.show()
            plt.imshow(raw_frame)
            cmass = center_of_mass(chull)
            cmass_xs.append(cmass[1])
            cmass_ys.append(cmass[0])
            ### fancy_coloring_of_trajectory
            if len(cmass_xs) > 1:
                for i in range(len(cmass_xs)-1):
                    plt.plot([cmass_xs[i], cmass_xs[i+1]], [cmass_ys[i], cmass_ys[i+1]], color='white', linewidth=2, alpha=0.4)
            # plt.plot(cmass_xs, cmass_ys, color='greenyellow', linewidth=2, alpha=0.6)
            if not two_colors:
                plt.scatter(cmass[1], cmass[0], s=100, c='limegreen', alpha=0.5)
            else:
                plt.scatter(cmass[1], cmass[0], s=100, c='white', alpha=0.5)

        else:
            plt.imshow(raw_frame)
        ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        # plt.show()
        fig.savefig(target_folder + '/processed_frames/frames{0:03d}.png'.format(frame_id), dpi=200)
        plt.close(fig)

    np.savetxt(target_folder + '/trajectory_x.txt', cmass_xs)
    np.savetxt(target_folder + '/trajectory_y.txt', cmass_ys)

def plot_experimental_trajectory(target_folder):
    xs = np.loadtxt(target_folder + '/trajectory_x.txt')
    ys = np.loadtxt(target_folder + '/trajectory_y.txt')
    f1 = plt.figure(1, figsize=(10,3))
    plt.plot(xs, -1*ys, alpha=1)
    plt.axis('equal')
    f1.savefig(target_folder + '/trajectory_plot.png', dpi=300, transparent=True)
    plt.show()

def match_scale_and_angle(target_folder = 'examples/random_doubled_3', video_folder = 'examples/random_doubled_3/video2',
                          cropfrom=100, cropto = -50, x0 = 6.5, # - 1.1
                          y0 = 0.1, # + 1
                          initial_scale=1.1e-2,
                          initial_angle=0,
                          do_plot=True
                          ):
    input_path = np.load(target_folder + '/folder_for_path/path_data.npy')
    # make interpolator for the true path
    dataxlen = np.max(input_path[:, 0])
    true_path = np.vstack( (input_path[:-1, :] + np.array([dataxlen * i, 0]) for i in range(4)) )
    if do_plot:
        plt.plot(true_path[:, 0], true_path[:, 1], '-', alpha=0.4)
    # plt.show()
    true_path_interp = interpolate.interp1d(true_path[:, 0], true_path[:, 1])

    # experimental trajectory
    xs = np.loadtxt(video_folder + '/trajectory_x.txt')[cropfrom:cropto]
    ys = -1*np.loadtxt(video_folder + '/trajectory_y.txt')[cropfrom:cropto]
    ys = ys - ys[0]
    xs = xs - xs[0]

    if do_plot:
        plt.scatter(x0 + xs * initial_scale, y0 + ys * initial_scale, alpha=0.5, color='C1')
        plt.axis('equal')
        plt.show()

    # match scale, rotation and shift
    def func(x, scale, angle, x0, y0):
        data_rotated = np.copy(true_path)
        for i in range(data_rotated.shape[0]):
            data_rotated[i, :] = rotate_2d(data_rotated[i, :], angle)
        true_path_interp = interpolate.interp1d(data_rotated[:, 0], data_rotated[:, 1], fill_value='extrapolate')
        y_here = (true_path_interp(x*scale + x0) - y0) / scale
        return y_here

    if do_plot:
        plt.scatter(xs, ys, alpha=0.5, color='C1')
        plt.plot(xs, func(xs, initial_scale, 0, x0, y0), color='C0')
        plt.axis('equal')
        plt.show()

    popt, pcov = curve_fit(func, xs, ys, p0=(initial_scale, initial_angle, x0, y0),
                           bounds = [[0, -np.pi/4, -np.inf, -np.inf], [np.inf, np.pi/4, np.inf, np.inf]])
    print(popt)
    if do_plot:
        plt.plot(xs, ys, alpha=0.5, color='C0')
        plt.plot(xs, func(xs, *popt), 'g--',
                 label=f'{popt}', alpha=0.5)
        plt.legend()
        plt.axis('equal')
        plt.show()

    scale, angle, x0, y0 = popt
    print(f'Scale is: {scale}')
    if do_plot:
        plt.plot(true_path[:, 0], true_path[:, 1], '-', color='black', alpha=0.5)
    traj_vectors = np.vstack((x0 + xs * scale, y0 + ys * scale)).T
    for i in range(traj_vectors.shape[0]):
        traj_vectors[i, :] = rotate_2d(traj_vectors[i, :], -angle)
    # plt.plot(, y0 + ys * scale, alpha=0.5, color='C1')
    if do_plot:
        plt.plot(traj_vectors[:, 0], traj_vectors[:, 1], color='C0', alpha=0.5)
        plt.axis('equal')
        plt.show()
    return true_path, traj_vectors


if __name__ == '__main__':
    # target_folder = 'examples/little-prince-2/video'
    # trace_trajectory_from_video_frames(target_folder, threshold = 25, min_frame = 0, nframes = False, do_debug_plots = False)
    # plot_experimental_trajectory(target_folder)

    # target_folder = 'examples/random_bridged_1/video'
    # # trace_trajectory_from_video_frames(target_folder, threshold = 25, min_frame = 0, nframes = False, do_debug_plots = False)
    # plot_experimental_trajectory(target_folder)

    # target_folder = 'examples/random_doubled_1/video'
    # trace_trajectory_from_video_frames(target_folder, threshold = 25, min_frame = 0, nframes = False, do_debug_plots = False)
    # plot_experimental_trajectory(target_folder)

    ### this is for custom coloring of bouncing parts
    # bouncy_regions = [[20, 30], [40, 70]]
    # xs = np.loadtxt(target_folder + '/trajectory_x.txt')
    # ys = np.loadtxt(target_folder + '/trajectory_y.txt')
    # f1 = plt.figure(1, figsize=(10,3))
    # plt.plot(xs, -1*ys, alpha=1)
    # for bounce_region in bouncy_regions:
    #     plt.plot(xs[bounce_region[0]:bounce_region[1]],
    #              -1 * ys[bounce_region[0]:bounce_region[1]],
    #              alpha=1, color='C2')
    # plt.axis('equal')
    # f1.savefig(target_folder + '/trajectory_plot.png', dpi=300, transparent=True)
    # plt.show()

    # target_folder = 'examples/random_doubled_3/video'
    # trace_trajectory_from_video_frames(target_folder, threshold = 25, min_frame = 0, nframes = False, do_debug_plots = False)
    # plot_experimental_trajectory(target_folder)

    # target_folder = 'examples/random_doubled_4/video'
    # trace_trajectory_from_video_frames(target_folder, threshold = 25, min_frame = 0, nframes = False, do_debug_plots = False)
    # plot_experimental_trajectory(target_folder)

    # target_folder = 'examples/random_doubled_5/video'
    # trace_trajectory_from_video_frames(target_folder, threshold = 25, min_frame = 0, nframes = False, do_debug_plots = False)
    # plot_experimental_trajectory(target_folder)

    # target_folder = 'examples/random_doubled_3/video2'
    # trace_trajectory_from_video_frames(target_folder, threshold = 25, min_frame = 0, nframes = False, do_debug_plots = False)
    # plot_experimental_trajectory(target_folder)

    # target_folder = 'examples/little-prince-2/video_2color'
    # trace_trajectory_from_video_frames(target_folder, threshold = 25, min_frame = 0, nframes = False, do_debug_plots=False,
    #                                    two_colors=True)
    # plot_experimental_trajectory(target_folder)

    target_folder = 'examples/random_unclosed_1/video'
    # trace_trajectory_from_video_frames(target_folder, threshold = 25, min_frame = 0, nframes = False, do_debug_plots = False)
    trace_trajectory_from_video_frames(target_folder, threshold=25, min_frame=70, nframes=False, do_debug_plots=False, bkg_nframes=60, bkg_minframe=1, bkg_step=1)
    plot_experimental_trajectory(target_folder)

    # # COMPARING 6D POSE TRACKING TO CENTROID-OF-SHADOW METHOD
    # target_folder = 'examples/random_doubled_3/video2'
    # do_plot = False
    # true_path, traj_vectors_centroids = match_scale_and_angle(target_folder = 'examples/random_doubled_3',
    #                                                           video_folder = 'examples/random_doubled_3/video2',
    #                                                           cropfrom=120, cropto = -50,
    #                                                           x0 = 6.8, # - 1.1
    #                                                           y0 = 0.2, # + 1
    #                                                           initial_scale=1.1e-2,
    #                                                           initial_angle=0,
    #                                                           do_plot=do_plot
    #                                                           )
    # true_path, traj_vectors_fulltracking = match_scale_and_angle(target_folder = 'examples/random_doubled_3',
    #                                                           video_folder = 'examples/random_doubled_3/video2/tracking',
    #                                                           cropfrom=120, cropto = -50,
    #                                                           x0 = 6.8,
    #                                                           y0 = 0.2, # + 1
    #                                                           initial_scale=55,
    #                                                           initial_angle=0,
    #                                                           do_plot=do_plot
    #                                                           )
    # # convert everything to mmillimeters
    # units_to_mm = 1/0.010923799436747648 / 1920 * 335
    # x0 = 105
    # for points in [true_path, traj_vectors_centroids, traj_vectors_fulltracking]:
    #     points[:, :] = points[:, :] * units_to_mm
    #     points[:, 0] = points[:, 0] - x0
    # figscale_factor = 0.85
    # fig, axarr = plt.subplots(2,1, sharex=True, figsize=(14*figscale_factor, 2*3.4*figscale_factor))
    # true_from = 300
    # true_to = -250
    # ax = axarr[0]
    # linewidth = 0.75
    # alpha = 1
    # ax.plot(true_path[true_from:true_to, 0], true_path[true_from:true_to, 1], '-', color='black', alpha=alpha,
    #         linewidth=linewidth, label='Theoretical (intended) in-plane path of the center of mass')
    # ax.plot(traj_vectors_centroids[:, 0], traj_vectors_centroids[:, 1], color='C0', alpha=alpha,
    #         linewidth=linewidth, label='Experimental path of the centroid of visible shape')
    # ax.plot(traj_vectors_fulltracking[:, 0], traj_vectors_fulltracking[:, 1], color='C1', alpha=alpha,
    #         linewidth=linewidth,
    #         label='Experimental path of the center of mass from full tracking of position and orientation (6D pose tracking)')
    # ax.set_xlabel('X coordinate, mm')
    # ax.set_ylabel('Y coordinate, mm')
    # ax.legend()
    # ax.axis('equal')
    # ax.set_ylim(-5, 30)
    # ax.set_xlim(105-x0, 330-x0)
    # ax.xaxis.set_tick_params(labelbottom=True)
    #
    # true_path_interp = interpolate.interp1d(true_path[:, 0], true_path[:, 1])
    # error_centroids = traj_vectors_centroids[:, 1] - true_path_interp(traj_vectors_centroids[:, 0])
    # error_fulltracking = traj_vectors_fulltracking[:, 1] - true_path_interp(traj_vectors_fulltracking[:, 0])
    # print(f'RMS error of centroids: {np.std(error_centroids)}')
    # print(f'RMS error of full tracking: {np.std(error_fulltracking)}')
    #
    # ax = axarr[1]
    # ax.axhline(y=0, color='black')
    # ax.fill_between(x=traj_vectors_centroids[:, 0], y1=0, y2=error_centroids, color='C0', alpha=0.5,
    #                 label='By centroid of visible shape')
    # ax.plot(traj_vectors_centroids[:, 0], error_centroids, color='C0', alpha=0.9)
    # ax.fill_between(x=traj_vectors_fulltracking[:, 0], y1=0, y2=error_fulltracking, color='C1', alpha=0.5,
    #                 label='By 6D pose tracking')
    # ax.plot(traj_vectors_fulltracking[:, 0], error_fulltracking, color='C1', alpha=0.9)
    # ax.set_xlabel('X coordinate, mm')
    # ax.set_ylabel('Difference in Y coordinate\nbetween the experimenta path\nand the intended path, mm')
    # ax.legend(loc='upper center', title='Method of estimating the center of mass location:')
    # plt.tight_layout()
    # fig.savefig('examples/random_doubled_3/video2/comparison_of_cetroid_to_fulltracking.png', dpi=300)
    # plt.show()