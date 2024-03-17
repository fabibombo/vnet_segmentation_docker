import os
import torch
import data_setup, engine, utils
import vnet_model
from torchvision import transforms
import metrics
import yaml
import argparse
import vnet_model

def parse_arguments():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Training script for VNET model.')

    # Add an argument for the config file path
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config file')

    # Parse the command-line arguments and return the parsed arguments
    return parser.parse_args()

def main():
    
    args = parse_arguments()
    config_path = args.config
    
    print("Creating working directories.")
    models_dir = "./models/"
    results_dir = "./results/"
    directories = [models_dir, results_dir]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    print("Reading config file.")
    with open(config_path, "r") as ymlfile:
        vnet_cfg = yaml.safe_load(ymlfile)

    TRAIN_DIR = vnet_cfg["data"]["train_dir"]
    TEST_DIR = vnet_cfg["data"]["test_dir"]
    DATA_AUG = vnet_cfg["data"]["data_aug"]
    CENTER_CROP_SIZE = vnet_cfg["data"]["center_crop_size"]
    AUG_CENTER_CROP_SIZE = vnet_cfg["data"]["aug_center_crop_size"]

    DEVICE = vnet_cfg["vnet"]["device"]
    MODEL_NAME = vnet_cfg["vnet"]["model_name"]
    LEARNING_RATE = vnet_cfg["vnet"]["learning_rate"]
    BATCH_SIZE = vnet_cfg["vnet"]["batch_size"]
    NUM_EPOCHS = vnet_cfg["vnet"]["num_epochs"]
    FRAME_DEPTH = vnet_cfg["vnet"]["frame_depth"]
    INPUT_CHANNELS = vnet_cfg["vnet"]["input_channels"]
    SPLIT_CHANNELS = vnet_cfg["vnet"]["split_channels"]
    ELU = vnet_cfg["vnet"]["elu"]
    SE = vnet_cfg["vnet"]["se"]
    NLL = vnet_cfg["vnet"]["nll"]
    SCHEDULE_LR = vnet_cfg["vnet"]["lr_schedule"]

    # Setup target device
    if DEVICE:
        device = DEVICE
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    ## TRANSFORMS ##
    # Transforms (better to define elsewhere, left for future work) 
    # designed to do centercropping for Prostate, not suitable for some datasets/organs
    simple_transform = transforms.Compose([
        transforms.ToPILImage(mode='F'),
        transforms.CenterCrop((CENTER_CROP_SIZE, CENTER_CROP_SIZE)),
        transforms.ToTensor(), 
    ])

    if DATA_AUG:
        AUG_CENTER_CROP_SIZE = vnet_cfg["data"]["aug_center_crop_size"]
        aug_transform = transforms.Compose([
            transforms.ToPILImage(mode='F'),
            transforms.CenterCrop((AUG_CENTER_CROP_SIZE, AUG_CENTER_CROP_SIZE)),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=(CENTER_CROP_SIZE, CENTER_CROP_SIZE), scale=(0.3,1)),
            transforms.ToTensor(),
        ])
        train_transform = aug_transform
    else:
        train_transform = simple_transform

    test_transform = simple_transform
    ## END TRANSFORMS ##

    print("Loading the images.")
    train_dataloader, test_dataloader = data_setup.create_dataloaders(train_dir=TRAIN_DIR,
                                                                      test_dir=TEST_DIR,
                                                                      train_transform=train_transform, 
                                                                      test_transform=test_transform, 
                                                                      batch_size=BATCH_SIZE,
                                                                      num_frames=FRAME_DEPTH
                                                                      )

    # Create model
    print("Creating model.")
    model = vnet_model.VNet(elu=ELU, se=SE, nll=NLL, input_ch=INPUT_CHANNELS, split_ch=SPLIT_CHANNELS).to(device)

    model.name = MODEL_NAME

    # Set loss and optimizer
    if NLL:
        criterion = torch.nn.NLLLoss()
    else:
        criterion = metrics.DiceJacBCELoss()
        
    optimizer = torch.optim.Adam(model.parameters(),
                                 betas = (0.9, 0.999),
                                 lr=LEARNING_RATE)
    
    if SCHEDULE_LR == True:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, epochs=NUM_EPOCHS, steps_per_epoch=len(train_dataloader))
    else:
        lr_scheduler = None
    
    # Start training
    results = engine.train(model=model,
                             train_dataloader=train_dataloader,
                             test_dataloader=test_dataloader,
                             optimizer=optimizer,
                             lr_scheduler=lr_scheduler,
                             criterion=criterion,
                             epochs=NUM_EPOCHS,
                             results_path=results_dir,
                             models_path=models_dir,
                             device=device,
                             nll=NLL)

    # Save the model
    utils.save_model(model=model,
                   target_dir=models_dir,
                   model_name=model_name+"_weights.pth")
    print("Trained model saved in: " + models_dir + model_name + "_weights.pth")

if __name__ == "__main__":
    main()
