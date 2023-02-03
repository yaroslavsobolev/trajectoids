import numpy as np
from numpy.linalg import norm as lnorm
import trimesh
import matplotlib.pyplot as plt

def get_projection_center_path(target_folder, shape_filename, out_name = 'projection_trajectory.npy'):
    data = np.load(target_folder + 'folder_for_path/path_data.npy')

    def rotation_from_point_to_point(pointA, pointB):
        vector_to_next_point = pointB - pointA
        axis_of_rotation = [vector_to_next_point[1], -vector_to_next_point[0], 0]
        theta = np.linalg.norm(vector_to_next_point)
        rotation_matrix = trimesh.transformations.rotation_matrix(angle=-1*theta,
                                                direction=axis_of_rotation,
                                                point=[0, 0, 0])
        return rotation_matrix, theta

    def get_mesh_projection_center(mesh):
        mesh3 = mesh.copy()
        for i in range(mesh3.vertices.shape[0]):
            mesh3.vertices[i,2] = 0
        return mesh3.centroid[:-1]

    trimesh.util.attach_to_log()
    mesh1 = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                           faces=[[0, 1, 2]])
    mesh1 = trimesh.load(target_folder + shape_filename, force='mesh')
    scene1 = mesh1.scene()
    mesh2 = mesh1.copy()
    N = data.shape[0]-1
    projection_center_delta = [get_mesh_projection_center(mesh2)]
    for i in range(N):
        rotation_matrix, theta = rotation_from_point_to_point(data[i], data[i+1])
        mesh2.apply_transform(rotation_matrix)
        projection_center_delta.append(get_mesh_projection_center(mesh2))

    projection_center_delta = np.array(projection_center_delta)
    projection_center = data + projection_center_delta

    np.save(target_folder + out_name, projection_center)
    plt.plot(data[:,0], data[:,1])
    plt.plot(projection_center[:, 0], projection_center[:, 1])
    plt.scatter(data[N,0], data[N,1])
    plt.show()

    cut_size = 2
    base_box = trimesh.creation.box(extents=[cut_size,cut_size,cut_size],
                                    transform=trimesh.transformations.translation_matrix([0,0,-2]))
    scene1.add_geometry(base_box)
    base_box2 = trimesh.creation.box(extents=[cut_size,cut_size,cut_size],
                                    transform=trimesh.transformations.translation_matrix([3,0,-2]))
    scene1.add_geometry(base_box2)
    base_box3 = trimesh.creation.box(extents=[cut_size,cut_size,cut_size],
                                    transform=trimesh.transformations.translation_matrix([0,6,-2]))
    scene1.add_geometry(base_box3)
    scene1.show(flags={'wireframe': True})


def plot_centroid_comparison(target_folder):
    data = np.load(target_folder + 'folder_for_path/path_data.npy')
    figtraj = plt.figure(10, figsize=(10, 5))
    dataxlen = np.max(data[:, 0])
    projection_centers = np.load(target_folder + 'projection_trajectory.npy')

    def plot_periods(data, linestyle, linewidth, color='black', secondaryalpha=0.3):
        plt.plot(data[:, 0], data[:, 1], color=color, alpha=secondaryalpha, linestyle=linestyle, linewidth=linewidth)
        plt.plot(dataxlen + data[:, 0], data[:, 1], color=color, alpha=1, linestyle=linestyle, linewidth=linewidth)
        plt.plot(2 * dataxlen + data[:, 0], data[:, 1], color=color, alpha=secondaryalpha, linestyle=linestyle,
                 linewidth=linewidth)
        plt.plot(3 * dataxlen + data[:, 0], data[:, 1], color=color, alpha=secondaryalpha, linestyle=linestyle,
                 linewidth=linewidth)

    plot_periods(data, '-', linewidth=0.5, secondaryalpha=1)
    plot_periods(projection_centers, '-', linewidth=0.5, color='C1', secondaryalpha=1)

    plt.scatter(data[-1, 0], data[-1, 1], s=35, color='black')
    plt.scatter(dataxlen + data[-1, 0], data[-1, 1], s=35, color='black')
    plt.axis('equal')
    figtraj.savefig(target_folder + 'centroid_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # get_projection_center_path(target_folder='trajectory_projects/normal_path_2/', shape_filename='normal_v2.obj',
    #                            out_name='normal_2_projection_trajectory.npy')

    # get_projection_center_path(target_folder='trajectory_projects/swirl_1/', shape_filename='swirl_v1.obj',
    #                            out_name='projection_trajectory.npy')

    # get_projection_center_path(target_folder='trajectory_projects/zago_v4/', shape_filename='zago_v4.obj',
    #                            out_name='projection_trajectory.npy')

    # get_projection_center_path(target_folder='examples/random_doubled_4/', shape_filename='3dsmax_files/random_doubled_4_sel.obj',
    #                            out_name='projection_trajectory.npy')
    plot_centroid_comparison(target_folder='examples/random_doubled_4/')

