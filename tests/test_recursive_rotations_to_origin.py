from compute_trajectoid import *

input_path = make_random_path(seed=0, make_ends_horizontal=False, start_from_zero=True, end_with_zero=True, amplitude=3)

i = input_path.shape[0]-3
assert np.isclose(rotation_to_origin(i, input_path, use_cache=False, recursive=True),
           rotation_to_origin(i, input_path, use_cache=False, recursive=False)).all()
print('Assertion ok')

t0 = time.time()
for i in range(input_path.shape[0]):
    rotation_to_origin(i, input_path, use_cache=False, recursive=True)
print(f'time: {time.time()-t0}')

t0 = time.time()
for i in range(input_path.shape[0]):
    rotation_to_origin(i, input_path, use_cache=True, recursive=True)
print(f'cached time: {time.time()-t0}')