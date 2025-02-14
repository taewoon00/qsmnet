import scipy.io
import numpy as np
import h5py
import time
import os
import sys
import math
import shutil

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''
File Path
'''
FILE_PATH_INPUT = './Data/'
FILE_NAME = 'train_data.mat'
FILE_PATH_OUTPUT = './Data/'

# 불필요한 체크포인트 폴더 삭제
if os.path.isdir(FILE_PATH_INPUT + "/.ipynb_checkpoints"):
    shutil.rmtree(FILE_PATH_INPUT + "/.ipynb_checkpoints")

'''
Constant Variables
    PS: Patch size
    patch_num: Order of Dimension: [x, y, z]
'''
PS = 64
patch_num = [6, 8, 7]

gyro = 42.5775e6
delta_TE = 0.005
CF = 123177385
Dr = 114

print("*** Patching start !!! ***")
start_time = time.time()

# .mat 파일 로드
m = scipy.io.loadmat(FILE_PATH_INPUT + FILE_NAME)
cosmos_sus = m['chi_cosmos']
field = m['phs_tissue']
mask = m['mask']

### 뇌 영역을 타이트하게 crop ###
y_max = np.max(np.where(mask[:, :, :, 0] != 0)[0])
x_max = np.max(np.where(mask[:, :, :, 0] != 0)[1])
z_max = np.max(np.where(mask[:, :, :, 0] != 0)[2])
y_min = np.min(np.where(mask[:, :, :, 0] != 0)[0])
x_min = np.min(np.where(mask[:, :, :, 0] != 0)[1])
z_min = np.min(np.where(mask[:, :, :, 0] != 0)[2])

new_y_min = int(y_min - PS/2)
new_y_max = int(y_max + PS/2)
new_x_min = int(x_min - PS/2)
new_x_max = int(x_max + PS/2)
new_z_min = int(z_min - PS/2)
new_z_max = int(z_max + PS/2)

# 경계값 조정
new_y_min = max(new_y_min, 0)
new_x_min = max(new_x_min, 0)
new_z_min = max(new_z_min, 0)
new_y_max = min(new_y_max, mask.shape[0] - 1)
new_x_max = min(new_x_max, mask.shape[1] - 1)
new_z_max = min(new_z_max, mask.shape[2] - 1)

print('Cropped coordinates:')
print('y:', new_y_min, new_y_max)
print('x:', new_x_min, new_x_max)
print('z:', new_z_min, new_z_max)

origin_mask = mask.copy()
print('Origin size:', np.shape(origin_mask))

# 관심 영역으로 crop
cosmos_sus = cosmos_sus[new_y_min:new_y_max, new_x_min:new_x_max, new_z_min:new_z_max, :]
field = field[new_y_min:new_y_max, new_x_min:new_x_max, new_z_min:new_z_max, :]
mask = mask[new_y_min:new_y_max, new_x_min:new_x_max, new_z_min:new_z_max, :]

### Hz를 ppm으로 변환 (여기서는 그대로 사용) ###
field_in_ppm = field

matrix_size = np.shape(mask)
strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)]
print('Matrix size:', matrix_size)
print('Strides:', strides)

# 총 패치 수 계산 (방향 수 * patch_num[0]*patch_num[1]*patch_num[2])
num_patches = matrix_size[-1] * patch_num[0] * patch_num[1] * patch_num[2]
print("Total number of patches:", num_patches)

# HDF5 파일을 열고 미리 데이터셋을 생성
result_file = h5py.File(FILE_PATH_OUTPUT + 'train_patch.hdf5', 'w')
dset_cosmos = result_file.create_dataset('chi_cosmos', shape=(num_patches, PS, PS, PS), dtype='float32')
dset_field  = result_file.create_dataset('phs_tissue', shape=(num_patches, PS, PS, PS), dtype='float32')
dset_mask   = result_file.create_dataset('mask', shape=(num_patches, PS, PS, PS), dtype='float32')

# 정규화 인자 계산을 위한 변수 초기화
cosmos_sus_sum   = 0.0
cosmos_sus_sumsq = 0.0
field_sum   = 0.0
field_sumsq = 0.0
n_element   = 0

patch_index = 0
# 각 패치를 순차적으로 생성하여 HDF5에 저장
for direction in range(matrix_size[-1]):
    for i in range(patch_num[0]):
        for j in range(patch_num[1]):
            for k in range(patch_num[2]):
                y_start = i * strides[0]
                x_start = j * strides[1]
                z_start = k * strides[2]
                
                patch_cosmos = cosmos_sus[y_start:y_start+PS, x_start:x_start+PS, z_start:z_start+PS, direction]
                patch_field  = field_in_ppm[y_start:y_start+PS, x_start:x_start+PS, z_start:z_start+PS, direction]
                patch_mask   = mask[y_start:y_start+PS, x_start:x_start+PS, z_start:z_start+PS, direction]
                
                # HDF5 데이터셋에 저장
                dset_cosmos[patch_index, ...] = patch_cosmos
                dset_field[patch_index, ...]  = patch_field
                dset_mask[patch_index, ...]   = patch_mask
                patch_index += 1

                # 유효(voxel > 0)한 영역에 대해 정규화 값 계산
                valid_voxels = (patch_mask > 0)
                sus_vals = patch_cosmos[valid_voxels]
                fld_vals = patch_field[valid_voxels]

                cosmos_sus_sum   += np.sum(sus_vals)
                cosmos_sus_sumsq += np.sum(sus_vals ** 2)
                field_sum   += np.sum(fld_vals)
                field_sumsq += np.sum(fld_vals ** 2)
                n_element   += np.count_nonzero(valid_voxels)

# 전역 정규화 인자 계산
cosmos_sus_mean = cosmos_sus_sum / n_element
cosmos_sus_var  = (cosmos_sus_sumsq / n_element) - (cosmos_sus_mean ** 2)
cosmos_sus_std  = np.sqrt(cosmos_sus_var)
field_mean = field_sum / n_element
field_var  = (field_sumsq / n_element) - (field_mean ** 2)
field_std  = np.sqrt(field_var)

# 정규화 인자를 .mat 파일로 저장
scipy.io.savemat(FILE_PATH_OUTPUT + 'train_patch_norm_factor.mat',
                 mdict={'cosmos_sus_mean': cosmos_sus_mean, 'cosmos_sus_std': cosmos_sus_std,
                        'field_mean': field_mean, 'field_std': field_std,
                        'n_element': n_element})
print("Final input data size : ", dset_mask.shape)

result_file.close()
print("*** Saving Done !!! ***")
