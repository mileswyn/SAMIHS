import SimpleITK as sitk
import numpy as np
import os

scans_path = '.\BCIHM\ct_scans_gz'
masks_path = '.\BCIHM\masks_gz'

scans_path_2d = '.\BCIHM\ct_2d'
masks_path_2d = '.\BCIHM\mask_2d'

ground_truths = os.listdir(masks_path)

for pa in os.listdir(scans_path):
    scan_img = sitk.ReadImage(os.path.join(scans_path, pa))
    assert pa in ground_truths
    mask_img = sitk.ReadImage(os.path.join(masks_path, pa))
    scan_arr = sitk.GetArrayFromImage(scan_img)
    label_arr = sitk.GetArrayFromImage(mask_img)
    for i in range(scan_arr.shape[0]):
        slice_name = 'BCIHM_' + pa.split('.')[0] + '_' + str(i).zfill(3) + '.npy'
        mask_name = 'BCIHM_' + pa.split('.')[0] + '_' + str(i).zfill(3) + '_seg' + '.npy'
        np.save(os.path.join(scans_path_2d, slice_name), scan_arr[i])
        np.save(os.path.join(masks_path_2d, mask_name), label_arr[i])
    print(pa, np.sum(scan_arr))