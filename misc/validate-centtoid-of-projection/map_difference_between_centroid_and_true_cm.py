import numpy as np
from numpy.linalg import norm as lnorm
import trimesh
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

target_folder = 'examples/random_doubled_4/'
shape_filename = '3dsmax_files/random_doubled_4_sel.obj'

# target_folder = 'trajectory_projects/swirl_1/'
# shape_filename='swirl_v1.obj'

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


mesh1 = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                       faces=[[0, 1, 2]])
mesh1 = trimesh.load(target_folder + shape_filename, force='mesh')
scene1 = mesh1.scene()

# plot for all possible orientations. Two rotations. First longitude. Then latitude.
def rotate_to_lonlat_and_get_centroid_dislacement(mesh, longitude, latitude):
    mesh_here = mesh.copy()
    rotation_matrix_longitude = trimesh.transformations.rotation_matrix(angle=longitude,
                                            direction=[0, 0, 1],
                                            point=[0, 0, 0])
    mesh_here.apply_transform(rotation_matrix_longitude)
    rotation_matrix_latitude = trimesh.transformations.rotation_matrix(angle=latitude,
                                            direction=[1, 0, 0],
                                            point=[0, 0, 0])
    mesh_here.apply_transform(rotation_matrix_latitude)
    return np.linalg.norm(get_mesh_projection_center(mesh_here))

N = 100
x = np.linspace(0, 2*np.pi, N)
y = np.linspace(0, np.pi, N)

X, Y = np.meshgrid(x,y)
Z = np.zeros_like(X)

t0 = time.time()
for i in tqdm(range(Z.shape[0]), desc='1st loop'):
   for j in range(Z.shape[1]):
       Z[i,j] = rotate_to_lonlat_and_get_centroid_dislacement(mesh1, longitude=X[i,j], latitude=Y[i,j])
print(f'Time: {time.time() - t0}')
print(f'Max distance = {np.max(Z)}')

fig = plt.figure(figsize=(4.6,4))
plt.pcolor(X/np.pi*180, (Y-np.pi/2)/np.pi*180, Z, vmin=0)
plt.xlabel('Longitude, degrees')
plt.ylabel('Latitude, degrees')
clb = plt.colorbar()
clb.set_label('Distance between the centroid of shape projection and\nthe projection of center of mass for trajectoid'
                 ' with r=1', fontsize=9)
clb_ticks = clb.get_ticks()
clb_ticks = np.insert(clb_ticks, 0, 0)
clb_ticks = np.append(clb_ticks, np.around(np.max(Z), decimals=3))
clb.set_ticks(clb_ticks)
plt.tight_layout()
fig.savefig(target_folder + 'centroid_distance_map.png', dpi=300)
plt.show()
