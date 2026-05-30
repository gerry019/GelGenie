import torch
import sys

#  SET PATHS
sys.path.append("/content/drive/MyDrive/my_projects/GelGenie/python-gelgenie")

#import to be able to build the model as in gelgenie
from gelgenie.segmentation.networks import model_configure

# Path to the checkpoint file
checkpoint_path = "/content/drive/MyDrive/my_projects/checkpoint_epoch_600(1).pth"

# Load the checkpoint, and from it, print out the keys and get the network
# The checkpoint was built using key and values
saved_dict = torch.load(checkpoint_path, map_location="cpu")
saved_state_dict = saved_dict["network"]

print("Loaded checkpoint keys:", saved_dict.keys())


#  Build a model as in gel genie (no summary or docstring needed for now)
model, _, _ = model_configure(
    device="cpu",
    model_name="smp_unet",
    encoder_name="resnet18",
    classes=3,
    in_channels=1 # greyscale
)
# Get all learnable parameters and shapes
current_state_dict = model.state_dict()

# Create a new dictionary
filtered_state_dict = {}
for k, v in saved_state_dict.items(): # For the layers and tensors
    # i the layer are the tensor shape are the same
    if k in current_state_dict and v.shape == current_state_dict[k].shape:
        # Copy that key value pair into the filtered dictionary
        filtered_state_dict[k] = v

# To update the model to that and ignore the missing and unexpected keys which will be also returned
missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False) # False to avoid errors

# Print
print("\n Matching layers loaded")
for k, v in filtered_state_dict.items():
    print(f"{k} {v.shape}")

print("\n Missing layers (not loaded, randomly initialized) ")
for m in missing:
    print(f"{m} expected shape {current_state_dict[m].shape}")

print("\n Unexpected layers (in checkpoint but not in model) ")
for u in unexpected:
    print(u)

