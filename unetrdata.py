import numpy as np
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from scipy.ndimage import gaussian_filter
import torchvision.utils as vutils
import torchvision
from torchvision import transforms
import pickle
import math

def undiscretize(frame):
    frame = np.array(frame)
    
    if len(frame.shape) == 2:
        frame = np.expand_dims(frame, 0)

    def normal_distribution_at(x, y, array_size=None):
        if array_size is None:
            print("Sorry, you have to pass array_size explicitly")
            import sys
            sys.exit()
        
        std_dev = 10   # Standard deviation
        X, Y = np.meshgrid(np.arange(array_size[0]), np.arange(array_size[1]))

        normal_distribution = np.exp(-((X - x)**2 + (Y - y)**2) / (2 * std_dev**2))

        return normal_distribution
    
    array_size = frame.shape[1:]
    normals = []
    dummy = normal_distribution_at(1,1,array_size)
    dummy[:, :] = 0
    #print(dummy.shape)
    normals.append(dummy)
    for c,x,y in zip(*np.nonzero(frame)):
        value = frame[c,x,y]
        normal_dist = normal_distribution_at(x,y,array_size)
        
        normal_dist_scaled = normal_dist * value
        
        normals.append(normal_dist_scaled)
    #print(normals)
    normals_summed = np.sum(normals, axis=0)
    #print(len(normals_summed), normals_summed.shape)
    return torch.Tensor(normals_summed)

import torchvision.transforms.functional as TF

class FixedTransform():
    def __init__(self, min_angle, max_angle, crop_height, crop_width):
        self.angle = torch.FloatTensor(1).uniform_(min_angle, max_angle).item()
        self.position = None
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, img, heatmap=None):
        # Check if the input is a PIL Image or a torch Tensor
        if isinstance(img, torch.Tensor):
            # For torch Tensors, the channel dimension is typically first
            _, h, w = img.size()
        else:
            # For PIL images, use the .size attribute
            w, h = img.size
        
        if self.position is None:
            th, tw = self.crop_height, self.crop_width
            if w == tw and h == th:
                self.position = (0, 0)
            else:
                i = torch.randint(0, h - th + 1, size=(1,)).item()
                j = torch.randint(0, w - tw + 1, size=(1,)).item()
                self.position = (i, j)

        img = TF.rotate(img, self.angle)
        img = TF.crop(img, *self.position, self.crop_height, self.crop_width)
        img = TF.resize(img, (224, 224))
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)

        return img

def get_centroids_vectors_from_burst(BASE_DIR, burst="Burst4_A4_1_VesselID-29_1-0"):
    directory = f"{BASE_DIR}/{burst}/img1/"
    
    data = pd.read_csv(
        f"{BASE_DIR}/{burst}/gt/gt.txt",
        names=["img_id", "cell_id", "bba", "bbb", "bbc", "bbd", "a", "b", "c", "d"],
        dtype={"img_id": int, "cell_id": int}
    )
    
    CENTROID_MAPS = []
    for idx, _img in enumerate([x for x in sorted(os.listdir(directory)) if x.endswith(".tiff")]):
        img = directory + _img
        idx = idx + 1
        
        current_img = Image.open(directory + [x for x in sorted(os.listdir(directory)) if x.endswith(".tiff")][0])
        centroid_map = np.zeros(current_img.size)
        for _, row in data[data["img_id"] == idx].iterrows():
            bba = int(row.bba)
            bbb = int(row.bbb)
            bbc = int(row.bbc)
            bbd = int(row.bbd)

            centroid_x = int(bba + bbc / 2)
            centroid_y = int(bbb + bbd / 2)
            
            try:
                centroid_map[centroid_x, centroid_y] = 1
            except:
                print(f"Error at centroid_map {centroid_x}, {centroid_y}")
        CENTROID_MAPS.append(centroid_map.T)
        
    return CENTROID_MAPS

#CENTROID_MAPS = get_centroids_vectors_from_burst("Burst4_A4_1_VesselID-29_1-0")

def get_vectors(BASE_DIR, burst="Burst4_A4_1_VesselID-29_1-0"):
    directory = f"{BASE_DIR}{burst}/img1/"
    
    data = pd.read_csv(
        f"{BASE_DIR}/{burst}/gt/gt.txt",
        names=["img_id", "cell_id", "bba", "bbb", "bbc", "bbd", "a", "b", "c", "d"],
        dtype={"img_id": int, "cell_id": int}
    )
    
    images = [x for x in sorted(os.listdir(f"{BASE_DIR}/{burst}/img1/")) if x.endswith(".tiff")]
    dir_vectors = defaultdict(list)
    pos_vectors = defaultdict(list)
    for cell_id in set(data.cell_id):
        _data = data[data["cell_id"] == cell_id]
        dir_vectors[int(cell_id)] = [np.array([0., 0.])]*len(images)
        pos_vectors[int(cell_id)] = [np.array([0., 0.])]*len(images)

        for _, row in _data.iterrows():
            this_img = row.img_id
            this_img_id = int(this_img)

            # get current centroid
            bba = int(row.bba)
            bbb = int(row.bbb)
            bbc = int(row.bbc)
            bbd = int(row.bbd)

            centroid_x = bba + bbc / 2
            centroid_y = bbb + bbd / 2
            pos_vectors[int(cell_id)][this_img_id - 1] = [int(centroid_x), int(centroid_y)]
            
            # get last images centroid
            lastdata = data[data["cell_id"] == cell_id]
            lastdata = lastdata[lastdata["img_id"] == (this_img_id - 1)]

            if len(lastdata) >= 1:
                _row = lastdata.iloc(0)[0]
                _bba = int(_row.bba)
                _bbb = int(_row.bbb)
                _bbc = int(_row.bbc)
                _bbd = int(_row.bbd)
                _centroid_x = _bba + _bbc / 2
                _centroid_y = _bbb + _bbd / 2
                #print(cell_id, this_img, centroid_x, _centroid_x)
                dir_vectors[int(cell_id)][this_img_id - 1] = [_centroid_x - centroid_x, _centroid_y - centroid_y]
            else:
                #print(cell_id, this_img, "CELL NOT FOUND IN PREVIOUS FRAME")
                pass
    return dir_vectors, pos_vectors

def get_vector_maps(BASE_DIR, burst="Burst4_A4_1_VesselID-29_1-0"):
    
    data = pd.read_csv(
        f"{BASE_DIR}/{burst}/gt/gt.txt",
        names=["img_id", "cell_id", "bba", "bbb", "bbc", "bbd", "a", "b", "c", "d"],
        dtype={"img_id": int, "cell_id": int}
    )
    
    directory = f"{BASE_DIR}/{burst}/img1/"
    images = [x for x in sorted(os.listdir(f"{BASE_DIR}/{burst}/img1/")) if x.endswith(".tiff")]
    
    dir_vectors, pos_vectors = get_vectors(BASE_DIR, burst)
    current_img = Image.open(directory + [x for x in os.listdir(directory) if x.endswith(".tiff")][0])
    vector_maps = []
    
    for i, img in enumerate(images):
        vector_map = np.stack([np.zeros(current_img.size), np.zeros(current_img.size)], axis=0)
        for cell_id in set(data.cell_id):
            pos_x, pos_y = pos_vectors[cell_id][i]
            dir_x, dir_y = dir_vectors[cell_id][i]
            try:
                vector_map[:, int(pos_x), int(pos_y)] = np.array([dir_x, dir_y])
            except:
                print(f"Error in vector_map at {pos_x} {pos_y}")
        
        vector_map = np.swapaxes(vector_map,2,1)
        vector_maps.append(vector_map)
        
    return vector_maps

#VECTOR_MAPS = get_vector_maps()

class CentroidVectorData:
    
    def __init__(self, data_path, sequence_length=4):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.sequences = []
        self.sequence_start_indices = []
        self.current_start_index = 0

        #sequences_cache_path = os.path.join(self.data_path, 'sequences_cache.pkl')
        #penalty_sequences_cache = os.path.join(self.data_path, 'penalty_sequences_cache.pkl')
        #start_indices_cache_path = os.path.join(self.data_path, 'start_indices_cache.pkl')

        
        for burst in os.listdir(data_path):

            if burst.endswith(".py") or burst.endswith((".pkl", ".cache.p")) or burst.startswith(".ipynb_checkpoints"):
                continue

            cache_path = os.path.join(data_path, burst + ".cache.p")
            # Check if cache exists
            if os.path.exists(cache_path):
                print("loading cached", burst)
                with open(cache_path, 'rb') as cache_file:
                    loaded_frames, CENTROID_MAPS, VECTOR_MAPS, penalty_maps = pickle.load(cache_file)
            else:
                print("loading", burst)
                # Preprocessing steps
                CENTROID_MAPS = get_centroids_vectors_from_burst(self.data_path, burst)
                VECTOR_MAPS = get_vector_maps(self.data_path, burst)

                CENTROID_MAPS = [undiscretize(x) for x in CENTROID_MAPS]

                _vector_maps = []
                for vm in VECTOR_MAPS:
                    a = undiscretize(torch.Tensor(vm[0]))
                    b = undiscretize(torch.Tensor(vm[1]))
                    child = torch.stack([a, b], axis=0)
                    _vector_maps.append(child)
                VECTOR_MAPS = _vector_maps

                frames = sorted([x for x in os.listdir(os.path.join(self.data_path, burst, "img1")) if x.endswith(".tiff")])
                penalty_maps = [np.load(os.path.join(self.data_path, burst, "img1", x + ".penaltymap.npy")) for x in frames]
                frames = [os.path.join(self.data_path, burst, "img1", x) for x in frames if x.endswith(('.tiff', '.tif'))]

                # Open and load the images
                loaded_frames = []
                for frame_path in frames:
                    with Image.open(frame_path) as img:
                        loaded_frames.append(img.copy())

                # Save to cache
                with open(cache_path, 'wb') as cache_file:
                    cache_data = (loaded_frames, CENTROID_MAPS, VECTOR_MAPS, penalty_maps)
                    pickle.dump(cache_data, cache_file)

            maps = (CENTROID_MAPS, VECTOR_MAPS)
            assert len(maps[0]) == len(maps[1])

            self.sequences.append((loaded_frames, maps, penalty_maps))
            self.sequence_start_indices.append(self.current_start_index)
            self.current_start_index += len(maps[0]) - self.sequence_length

    def __getitem__(self, idx):

        start_item = max([x for x in self.sequence_start_indices if x <= idx])
        start_index = self.sequence_start_indices.index(start_item)
        
        local_index = idx - start_item
        centroid_maps = self.sequences[start_index][1][0][local_index:local_index+self.sequence_length]
        vector_maps = self.sequences[start_index][1][1][local_index:local_index+self.sequence_length]
        frames = self.sequences[start_index][0][local_index:local_index+self.sequence_length]
        penalty_maps = self.sequences[start_index][2][local_index:local_index+self.sequence_length]
        
        frames = [torch.Tensor(np.asarray(x)) for x in frames]
        centroid_maps = [torch.Tensor(x) for x in centroid_maps]
        vector_maps = [torch.Tensor(x) for x in vector_maps]
        penalty_maps = [torch.Tensor(x) for x in penalty_maps]
        
        # Do the random-ish crop
        out_of_bounds = True
        while out_of_bounds:
            fixed_transformation = FixedTransform(min_angle=0, max_angle=359, crop_height=64, crop_width=64)
            
            
            _frames = [fixed_transformation(x.permute(2,1,0)) for x in frames]
            _centroid_maps = [fixed_transformation(x.unsqueeze(0)) for x in centroid_maps]
            _vector_maps = [fixed_transformation(x) for x in vector_maps]
            _penalty_maps = [fixed_transformation(x.unsqueeze(0)) for x in penalty_maps]
            
            ex = _frames[0]
            zero = torch.Tensor([0,0,0])
            out_of_bounds = torch.allclose(ex[:, 0, 0], zero) or \
                torch.allclose(ex[:, 0, -1], zero) or \
                torch.allclose(ex[:, -1, 0], zero) or \
                torch.allclose(ex[:, -1, -1], zero)
        
        # Combine the crops into one huge picture
        frames_batch = torch.stack(_frames)
        centroid_maps_batch = torch.stack(_centroid_maps)
        vector_maps_batch = torch.stack(_vector_maps)
        penalty_maps_batch = torch.stack(_penalty_maps)
        
        nrow = int(math.sqrt(self.sequence_length))
        frames_single = vutils.make_grid(frames_batch, nrow=nrow, padding=0, normalize=False)
        centroid_maps_single = vutils.make_grid(centroid_maps_batch, nrow=nrow, padding=0, normalize=False)
        vector_maps_single = vutils.make_grid(vector_maps_batch, nrow=nrow, padding=0, normalize=False)
        penalty_maps_single = vutils.make_grid(penalty_maps_batch, nrow=nrow, padding=0, normalize=False)

        return frames_single / 255, centroid_maps_single[0], vector_maps_single, penalty_maps_single

    def __len__(self):
        return len(self.sequence_start_indices) - self.sequence_length        

