import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.transform import resize
from scipy.stats import zscore

NUM_WORKERS = 8 #less than 256

def mask_postprocessing(mask, scale_factor=0.95, threshold=0.05, iterations=8):
    #post processing to smooth out the mask
    for _ in range(iterations):
        
        # Downsample
        original_shape = mask.shape
        scaled_shape = np.array(mask.shape)
        scaled_shape[1] = int(np.round(original_shape[1] * scale_factor))
        scaled_shape[2] = int(np.round(original_shape[2] * scale_factor))
        downsampled_mask = resize(mask, scaled_shape, anti_aliasing=True)
    
        # Upsample the mask back to the original size
        upsampled_mask = resize(downsampled_mask, original_shape, anti_aliasing=True)

        # Threshold the upsampled mask
        upsampled_mask[upsampled_mask < threshold] = 0.
        upsampled_mask[upsampled_mask >= threshold] = 1.

        mask = upsampled_mask
        
    return upsampled_mask

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    train_transform: transforms.Compose, 
    test_transform: transforms.Compose, 
    batch_size: int, 
    num_frames: int=16,
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  train_data = DICOM_Dataset(train_dir, transform=train_transform, normalize=True, denoising=None, mask_pp=None, num_frames=num_frames)
  test_data = DICOM_Dataset(test_dir, transform=test_transform, normalize=True, denoising=None, mask_pp=None, num_frames=num_frames)
  
  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader

class DICOM_Dataset(Dataset):
    def __init__(self, data_folder, transform=None, normalize=True, denoising=None, mask_pp=None, num_frames=16):
        self.data_folder = data_folder
        self.image_folder = os.path.join(data_folder, 'image')
        self.mask_folder = os.path.join(data_folder, 'mask')
        self.transform = transform
        self.normalize = normalize
        self.denoising = denoising
        self.mask_pp = mask_pp

        self.file_list = sorted(set(os.listdir(self.image_folder)))

        #we need to make a dictionary with index and a list of the total indices
        self.num_frames = num_frames
        self.index_mapping, self.frame_mapping, self.total_length = self.build_mapping()

    def extract_overlapping_slices(self, data, slice_size, check_empty=False):
        h, w, num_frames = data.shape
        slices = []
        for start_frame in range(0, num_frames - slice_size + 1):
            slice_data = data[:, :, start_frame:start_frame + slice_size]
            #check here if slice_data for the mask is empty
            if not check_empty:
                slices.append(slice_data)
            elif np.any(slice_data == 1):
                slices.append(slice_data)
        return slices

    def build_mapping(self):
        index_mapping = {}
        frame_mapping = {}
        total_index = 0

        for filename in self.file_list:
            if filename.endswith(".npy"):
                index = filename.split(".")[0]  # Extract the index from the filename
                image_path = os.path.join(self.image_folder, filename)

                # Load the 3D numpy array
                image_data = np.load(image_path).astype(np.float32)

                # Check for bad images that cannot be normalized and skip
                check_std = np.std(np.expand_dims(image_data, axis=2), axis=(0,1,2), keepdims=True)
                if (np.any(check_std == 0)):
                    continue
                
                # Extract overlapping 3D arrays of size (h, w, 16)
                image_slices = self.extract_overlapping_slices(image_data, slice_size=self.num_frames)

                # Update the mapping with the original index and total index
                for i in range(len(image_slices)):
                    index_mapping[total_index] = int(index)
                    frame_mapping[total_index] = int(i)
                    total_index += 1
        
        return index_mapping, frame_mapping, total_index

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):

        image_index = self.index_mapping.get(idx)
        frame_index = self.frame_mapping.get(idx)

        # Load 3D numpy arrays
        image_filename = f"{image_index}.npy"
        mask_filename = f"{image_index}_mask.npy"
        
        image_path = os.path.join(self.image_folder, image_filename)
        mask_path = os.path.join(self.mask_folder, mask_filename)
        
        image_data = np.load(image_path).astype(np.float32)
        mask_data = np.load(mask_path).astype(np.float32)
        
        # Extract the specific subarray based on the frame index
        image = image_data[:, :, frame_index:frame_index + self.num_frames]
        mask = mask_data[:, :, frame_index:frame_index + self.num_frames]

        # Add mono channel
        image = np.expand_dims(image, axis=2)
        mask = np.expand_dims(mask, axis=2)

        # Apply denoising
        if self.denoising:
            image = self.denoising(image)
            
        # Z-score normalization
        if self.normalize:
            mean = np.mean(image, axis=(0,1,2), keepdims=True)
            std = np.std(image, axis=(0,1,2), keepdims=True)
            image = (image - mean) / (std)
        
        ## anomalies check for the inputs
        # if (np.isnan(image).any()):
        #     print("NaN found in NumPy arrays after normalization!")
        # if (np.isinf(image).any()):
        #     print("Inf found in NumPy arrays after normalization!")
        # if (np.all(image == 0)):
        #     print("All zeros found in NumPy arrays after normalization!")

        mask = np.where(mask < 0.5, 0.0, 1.0).astype(np.float32)
        # Apply transform to each frame
        if self.transform:  
            
            transformed_frames = []
            transformed_frames_mask = []
            
            state = torch.get_rng_state()
            
            for frame in range(image.shape[-1]):
                #print(image[:, :, :, frame].shape)
                torch.set_rng_state(state)
                transformed_frame = self.transform(image[:, :, :, frame])
                transformed_frames.append(transformed_frame)
                
                torch.set_rng_state(state)               
                transformed_frame_mask = self.transform(mask[:, :, :, frame])
                transformed_frames_mask.append(transformed_frame_mask)

            image = torch.stack(transformed_frames, dim=-1)
            mask = torch.stack(transformed_frames_mask, dim=-1)

        if torch.isnan(image).any():
            print("NaN values found in PyTorch tensors!")
            print(image_filename)
        if torch.isinf(image).any():
            print("Inf values found in PyTorch tensors!")
            print(image_filename)

        # ensure binary masks
        mask = torch.where(mask < 0.5, torch.tensor(0.), torch.tensor(1.))
        
        if self.mask_pp:
            mask = torch.tensor(self.mask_pp(mask, scale_factor=0.95, threshold=0.10, iterations=3))
        
        # torch wants dimensions (batch_size, c, h, w, frames)
        # transpose here if needed

        #assert image.shape == mask.shape
        #assert torch.all(torch.logical_or(mask == 0., mask == 1.))

        return image, mask
        
## For data AUG, not used for now
# class AddGaussianNoise(object):
    # def __init__(self, mean=0, std=0.3, max_abs_value=0.49):
        # self.std = std
        # self.mean = mean
        # self.max_abs_value = max_abs_value
        
    # def __call__(self, tensor):
        # noise = torch.randn(tensor.size()) * self.std + self.mean
        # noise = torch.clamp(noise, -self.max_abs_value, self.max_abs_value)
        # return tensor + noise    
    # def __repr__(self):
        # return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
