#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import yaml
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import sys
import glob

# Load configuration from a YAML file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Define the main function for encoding images
def encode_images(config_path):
    # Load configuration
    config = load_config(config_path)
    dataset_dir = config['dataset_dir']
    encodings_dir = config['encodings_dir']
    batch_size = config.get('batch_size', 32)
    excluded_models = config.get('excluded_models', [])

    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Expanded list of pre-trained models to 70 models
    model_names = config['available_models']

    # Exclude specified models
    excluded_models = config.get('excluded_models', [])
    model_names = [model for model in model_names if model not in excluded_models]

    # Define the local directory for storing models and temporary files
    local_model_dir = config.get('local_model_dir', "pretrained_models")
    os.makedirs(local_model_dir, exist_ok=True)
    torch.hub.set_dir(local_model_dir)  # Set PyTorch to use the specified local directory for model storage

    # Define a transform to resize images and convert them to tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the 'encodings' directory if it does not exist
    os.makedirs(encodings_dir, exist_ok=True)

    # Custom loader function to handle loading errors
    def pil_loader_with_error_handling(path: str):
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                return img.convert("RGB")
        except UnidentifiedImageError:
            print(f"Skipping file {path}: cannot identify image file.")
            return None

    # Load the dataset using ImageFolder with the custom loader
    dataset = ImageFolder(root=dataset_dir, transform=transform, loader=pil_loader_with_error_handling)

    # Filter out invalid images
    valid_indices = [i for i, (img, _) in enumerate(dataset.imgs) if dataset.loader(dataset.imgs[i][0]) is not None]
    valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
    dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Display the number of images to be encoded
    num_images = len(valid_dataset)
    print(f"Number of images to be encoded: {num_images}")

    # Loop through each model and encode images
    latest_model = sorted(model_names, key=lambda model: os.path.getctime(os.path.join(encodings_dir, model)) if os.path.exists(os.path.join(encodings_dir, model)) else 0, reverse=True)[0]
    for model_name in model_names:
        if model_name == latest_model:
            print(f"Resuming with most recently incomplete model: {model_name}")
        else:
            print(f"Processing with model: {model_name}")

        model_encodings_dir = os.path.join(encodings_dir, model_name)
        os.makedirs(model_encodings_dir, exist_ok=True)

        # Check if the encodings for this model already exist or if there are incomplete encodings
        if encodings_exist(model_encodings_dir) and not os.path.exists(os.path.join(model_encodings_dir, 'features_temp.npy')):
            print(f"Encodings for model {model_name} already exist and are complete. Skipping to next model.")
            continue

        # Load or download model
        model = load_or_download_model(model_name, local_model_dir, device)

        # Modify the model to use only the feature extractor part
        model = modify_model_for_feature_extraction(model, model_name)

        # Set the model to evaluation mode
        model.eval()

        all_features = []
        all_labels = []

        # Try to load saved batches if any exist
        loaded_features, loaded_labels, last_batch_index = load_saved_batches(model_encodings_dir)
        # Initialize last_batch_index to 0 if no saved batches are available
        if last_batch_index is None:
            last_batch_index = 0

        # If there are saved batches, resume from where it left off        
        if loaded_features is not None and loaded_labels is not None:
            all_features = list(loaded_features)
            all_labels = list(loaded_labels)
            print(f"Resuming from saved batch for model: {model_name}")

        for batch_index, (images, labels) in enumerate(tqdm(dataloader)):
            # Skip batches that have already been processed
            if batch_index < last_batch_index:
                continue

            images = images.to(device)

            with torch.no_grad():
                output = model(images)
                output = process_output(output, model_name)

            features = output.cpu().numpy()
            all_features.append(features)
            all_labels.extend(labels.numpy())

            # Save temporary batches after each iteration
            last_batch_index = batch_index + 1
            save_temp_batches(model_encodings_dir, np.array(all_features, dtype=object), np.array(all_labels, dtype=object), last_batch_index)

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)

        # Display the dimensions of all_features and all_labels
        print(f"Dimensions of all_features: {all_features.shape}")
        print(f"Dimensions of all_labels: {all_labels.shape}")

        # Save the encoded images and labels as .npy files
        np.save(os.path.join(model_encodings_dir, 'encoded_images.npy'), all_features)
        np.save(os.path.join(model_encodings_dir, 'labels.npy'), all_labels)

        # Delete temporary batches after completion if encoding is successful
        delete_temp_batches(model_encodings_dir)
        # Remove last batch index file if encoding is successful
        last_batch_index_path = os.path.join(model_encodings_dir, 'last_batch_index.npy')
        if os.path.exists(last_batch_index_path):
            os.remove(last_batch_index_path)

        print(f"Encoding and saving complete for model: {model_name}")

    print("All models processed!")

# Helper functions for loading, processing, and saving data
def load_or_download_model(model_name, local_model_dir, device):
    """
    Load a pre-trained model from the local directory if available; otherwise, download it.
    If a corrupt model is detected, delete it and download it again.
    """
    checkpoints_dir = os.path.join(local_model_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoints_dir, f"{model_name}.pth")

    # Attempt to find existing model checkpoint by pattern matching
    existing_checkpoints = glob.glob(os.path.join(checkpoints_dir, f"{model_name}*.pth"))

    try:
        if existing_checkpoints:
            print(f"Found existing model weights for {model_name}. Loading from local directory.")
            model = getattr(models, model_name)(weights=None)
            model.load_state_dict(torch.load(existing_checkpoints[0]))
        else:
            print(f"Model weights for {model_name} not found. Downloading...")
            model = getattr(models, model_name)(weights='DEFAULT')
            torch.save(model.state_dict(), checkpoint_path)
    except (RuntimeError, ValueError, OSError) as e:
        print(f"Error loading model weights for {model_name}: {e}. Deleting and re-downloading...")
        # Delete potentially corrupt model checkpoint
        for checkpoint in existing_checkpoints:
            try:
                os.remove(checkpoint)
            except OSError:
                print(f"Failed to delete corrupt checkpoint: {checkpoint}")
        # Redownload the model
        model = getattr(models, model_name)(weights='DEFAULT')
        torch.save(model.state_dict(), checkpoint_path)

    return model.to(device)

#def load_or_download_model(model_name, local_model_dir, device):
#    checkpoints_dir = os.path.join(local_model_dir, 'checkpoints')
#    os.makedirs(checkpoints_dir, exist_ok=True)
#    checkpoint_path = os.path.join(checkpoints_dir, f"{model_name}.pth")

    # Attempt to find existing model checkpoint by pattern matching
#    existing_checkpoints = glob.glob(os.path.join(checkpoints_dir, f"{model_name}*.pth"))

#    if existing_checkpoints:
#        print(f"Found existing model weights for {model_name}. Loading from local directory.")
#        model = getattr(models, model_name)(weights=None)
#        model.load_state_dict(torch.load(existing_checkpoints[0]))
#    else:
#        print(f"Model weights for {model_name} not found. Downloading...")
#        model = getattr(models, model_name)(weights='DEFAULT')
#        torch.save(model.state_dict(), checkpoint_path)
#    return model.to(device)

def modify_model_for_feature_extraction(model, model_name):
    # Modify the model to use only the feature extractor part
    if 'resnet' in model_name or 'resnext' in model_name or 'wide_resnet' in model_name:
        model = torch.nn.Sequential(*list(model.children())[:-2])
    elif 'mobilenet' in model_name or 'mnasnet' in model_name:
        model.classifier = torch.nn.Identity()
    elif 'vgg' in model_name or 'alexnet' in model_name:
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
    elif 'squeezenet' in model_name:
        model.classifier = torch.nn.Identity()
    elif 'googlenet' in model_name or 'inception' in model_name:
        model.fc = torch.nn.Identity()
    elif 'efficientnet' in model_name or 'convnext' in model_name:
        model.classifier = torch.nn.Identity()
    elif 'shufflenet' in model_name or 'regnet' in model_name:
        model.fc = torch.nn.Identity()  # Removing the classifier head for feature extraction
    elif 'densenet' in model_name:
        model.classifier = torch.nn.Identity()
    else:
        raise ValueError(f"Model {model_name} is not supported for feature extraction.")
    return model

def model_weights_exist(model_name, local_model_dir):
    checkpoints_dir = os.path.join(local_model_dir, 'checkpoints')
    return len(glob.glob(os.path.join(checkpoints_dir, f"{model_name}*.pth"))) > 0

def encodings_exist(model_encodings_dir):
    features_path = os.path.join(model_encodings_dir, 'encoded_images.npy')
    labels_path = os.path.join(model_encodings_dir, 'labels.npy')
    return os.path.exists(features_path) and os.path.exists(labels_path)

def load_saved_batches(model_encodings_dir):
    temp_features_path = os.path.join(model_encodings_dir, 'features_temp.npy')
    temp_labels_path = os.path.join(model_encodings_dir, 'labels_temp.npy')
    last_batch_index_path = os.path.join(model_encodings_dir, 'last_batch_index.npy')
    if os.path.exists(temp_features_path) and os.path.exists(temp_labels_path) and os.path.exists(last_batch_index_path):
        try:
            features = np.load(temp_features_path, allow_pickle=True)
            labels = np.load(temp_labels_path, allow_pickle=True)
            last_batch_index = np.load(last_batch_index_path, allow_pickle=True)
            return features, labels, last_batch_index
        except (EOFError, ValueError):
            print(f"Error loading temporary batches for {model_encodings_dir}. Deleting corrupted files and restarting from last known good state.")
            # Delete corrupted files to allow fresh restart
            if os.path.exists(temp_features_path):
                os.remove(temp_features_path)
            if os.path.exists(temp_labels_path):
                os.remove(temp_labels_path)
            if os.path.exists(last_batch_index_path):
                os.remove(last_batch_index_path)
    return None, None, None

def save_temp_batches(model_encodings_dir, features, labels, last_batch_index):
    temp_features_path = os.path.join(model_encodings_dir, 'features_temp.npy')
    temp_labels_path = os.path.join(model_encodings_dir, 'labels_temp.npy')
    last_batch_index_path = os.path.join(model_encodings_dir, 'last_batch_index.npy')
    np.save(temp_features_path, features, allow_pickle=True)
    np.save(temp_labels_path, labels, allow_pickle=True)
    np.save(last_batch_index_path, last_batch_index, allow_pickle=True)

def delete_temp_batches(model_encodings_dir):
    temp_features_path = os.path.join(model_encodings_dir, 'features_temp.npy')
    temp_labels_path = os.path.join(model_encodings_dir, 'labels_temp.npy')
    last_batch_index_path = os.path.join(model_encodings_dir, 'last_batch_index.npy')
    if os.path.exists(temp_features_path):
        os.remove(temp_features_path)
    if os.path.exists(temp_labels_path):
        os.remove(temp_labels_path)
    if os.path.exists(last_batch_index_path):
        os.remove(last_batch_index_path)

def process_output(output, model_name):
    global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
    if len(output.size()) == 4:
        output = global_avg_pool(output)
        output = output.view(output.size(0), -1)
    elif len(output.size()) != 2:
        raise ValueError(f"Unexpected output size from model {model_name}: {output.size()}")
    return output

# Command-line interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python encode_images.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    encode_images(config_path)

