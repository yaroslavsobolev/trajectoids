import numpy as np
from numpy.linalg import norm as lnorm
import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import convex_hull_image, remove_small_objects
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import label
import os

def makedir_if_needed(path):
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

def number_of_files(target_dir):
    _, _, files = next(os.walk(target_dir))
    file_count = len(files)
    return file_count

def convert_to_signal(raw_frame):
    return raw_frame[:,:,2].astype(np.float) - raw_frame[:,:,1].astype(np.float) + raw_frame[:,:,0].astype(np.float)

def get_median_frame(min_frame, target_folder, nframes, step=10):
    list_of_frames = []
    for frame_id in range(0, nframes, step):
        print(f'Loading frame {frame_id} for background.')
        if frame_id < min_frame:
            continue
        frame_file = target_folder + '/frames/frame{0:03d}.jpg'.format(frame_id)
        raw_frame = io.imread(frame_file)
        list_of_frames.append(convert_to_signal(raw_frame))
    return np.median(np.array(list_of_frames), axis=0)

def trace_trajectory_from_video_frames(target_folder, threshold=25, min_frame=0, nframes=False, do_debug_plots=False):
    if not nframes:
        nframes = number_of_files(target_folder + '/frames/')
    makedir_if_needed(target_folder + '/processed_frames')

    #get background
    background_frame = get_median_frame(min_frame, target_folder, nframes=nframes)

    cmass_xs = []
    cmass_ys = []
    for frame_id in range(nframes):
        if frame_id < min_frame:
            continue
        print(frame_id)
        frame_file = target_folder + '/frames/frame{0:03d}.jpg'.format(frame_id)
        raw_frame = io.imread(frame_file)
        fig, ax = plt.subplots(figsize=(8, 8 * raw_frame.shape[0] / raw_frame.shape[1]))
        if frame_id >= min_frame:
            channel_diff = convert_to_signal(raw_frame) - background_frame
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
            plt.scatter(cmass[1], cmass[0], s=100, c='limegreen', alpha=0.5)
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


if __name__ == '__main__':
    # target_folder = 'examples/little-prince-2/video'
    # trace_trajectory_from_video_frames(target_folder, threshold = 25, min_frame = 0, nframes = False, do_debug_plots = False)
    # plot_experimental_trajectory(target_folder)

    # target_folder = 'examples/random_bridged_1/video'
    # # trace_trajectory_from_video_frames(target_folder, threshold = 25, min_frame = 0, nframes = False, do_debug_plots = False)
    # plot_experimental_trajectory(target_folder)

    target_folder = 'examples/random_doubled_1/video'
    # trace_trajectory_from_video_frames(target_folder, threshold = 25, min_frame = 0, nframes = False, do_debug_plots = False)
    plot_experimental_trajectory(target_folder)
    # this is for custong coloring of bouncing parts
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