import os

aria_path = '../data/FOVCrop-padding/ARIA-FOVCrop-padding/images'
refuge_path = '../data/FOVCrop-padding/REFUGE-FOVCrop-padding/train/images/all'
ddr_path = '../data/FOVCrop-padding/DDR-FOVCrop-padding/train/images'
idrid_path = '../data/FOVCrop-padding/IDRiD-FOVCrop-padding/train/images'

task_id = 1

with open('task_info.txt', 'w') as f:
    for path in [aria_path, refuge_path, ddr_path, idrid_path]:
        for file in os.listdir(path):
            f.write(file + ' ' + str(task_id) + '\n')
        if task_id != 3:
            task_id += 1