from skimage.transform import resize
from glob import glob
from os.path import join
import numpy as np

dataset_root = "C:/data/RAVEN-10000"
size = 80

file_names = [f for f in glob(join(dataset_root, "*", "*.npz"))]

for fn in file_names:
    data = np.load(fn)
    image = data["image"]
    resized = np.stack([resize(image[idx], (size, size)) for idx in range(image.shape[0])], axis = 0)
    np.savez(fn.replace("npz", "small.npz"), image = resized,
             target = data["target"],
             predict = data["predict"],
             meta_matrix = data["meta_matrix"],
             meta_target = data["meta_target"],
             structure = data["structure"],
             meta_structure = data["meta_structure"])


