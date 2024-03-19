# VNET PYTORCH DOCKER - README #

This application builds and trains a Vnet pytorch model for image segmentantion, including definition of the dataset classes, data augmentation, training workflow and evaluation metrics. Originally designed for organ and lesion prostate MRI segmentation.

### Model details ###

*vnet_model.py* is based on this [repo by mattmacy](https://github.com/mattmacy/vnet.pytorch). In turn the model was based on the paper [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797) by Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. Several modifications and additions were made, check the script for more details.

Different from the original paper and repo the number of input and split channels is variable and can be defined in the model config file. This will directly modify the size and complexity of the model so choose with care. The standard value of input and split channels for single MRI segmentation would be 1 and 16 respectively.

Several losses can be used for training. By default the model can be trained using the NLL loss (T) or the BCE + DICE + JACCARD loss (F) defined in the nll config parameter, other losses can alse be configured in the train and metrics scripts.

### Requirements ###

Your input segmentation data should follow this structure:

* data
    * train
        * image
        * mask  
    * test
        * image
        * mask
    
Each image folder should contain numpy files indexed by an integer and the mask folders should contain the corresponding mask pair with a suffix (e.g. 23.npy and 23_mask.npy).

The numpy arrays must use the shape (Height, Width, Depth). The images can have variable depth and the dataset classes extract overlapping frame collections of a shared fixed depth defined by the frame depth config parameter.

By default, intensity Z-Normalization is performed, and other preprocessing is not done and it is left for the user. For the binary masks there is preprocessing available to attempt to fix abnormalities not enabled by default.

This model was originally thought for prostate MRIs so a center cropping is applied in the dataset transformations, change them if needed.

### Preeliminary results and segmentation examples

An example for a segmentation for the SV prostate region after training:

<img src="https://github.com//fabibombo/vnet_segmentation_docker/blob/main/pictures/example1.png?raw=true" alt="Segmentation Preview" width="400">

And following is an example of the obtained loss, dice and recall for the combined CZ and TZ region after a training episode:

<img src="https://github.com//fabibombo/vnet_segmentation_docker/blob/main/pictures/results1.png?raw=true" alt="Results Preview" width="500">

<img src="https://github.com//fabibombo/vnet_segmentation_docker/blob/main/pictures/results2.png?raw=true" alt="Results Preview" width="500">

<img src="https://github.com//fabibombo/vnet_segmentation_docker/blob/main/pictures/results3.png?raw=true" alt="Results Preview" width="500">

<!--- ![Segmentation Preview](https://github.com//fabibombo/vnet_segmentation_docker/blob/main/pictures/example1.png?raw=true) 

![Results Preview](https://github.com//fabibombo/vnet_segmentation_docker/blob/main/pictures/results1.png?raw=true)

![Results Preview](https://github.com//fabibombo/vnet_segmentation_docker/blob/main/pictures/results2.png?raw=true)

![Results Preview](https://github.com//fabibombo/vnet_segmentation_docker/blob/main/pictures/results3.png?raw=true) -->

### Commands ###

The docker image can be built with:

```docker build --no-cache -f vnet_pytorch.dockerfile -t vnet-pytorch-docker .```

Example command to run the docker image interactively:

```docker run -it --rm --user=$USER_ID --gpus '"device=$DEVICE_ID"' -v "$DATA_PATH":/data/ -v "$CODE_PATH":/vnet/ vnet-pytorch-docker```

### Train the model ###

When running or already inside the container one can train the model by running the training script alongside the config path parameter:

```python train_vnet.py --config /vnet/config/vnet_config.yaml```

To run the command detached from the terminal run:

```docker run -d --rm --user=$USER_ID --gpus '"device=$DEVICE_ID"' -v "$DATA_PATH":/data/ -v "$CODE_PATH":/vnet/ vnet-pytorch-docker bash -c "python train_vnet.py --config /vnet/config/vnet_config.yaml"```


By: David Vallmanya Poch

Contact: davidvp12@gmail.com
