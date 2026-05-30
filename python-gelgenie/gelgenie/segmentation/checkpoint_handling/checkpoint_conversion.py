import torch
import sys

#  SET PATHS
sys.path.append("/content/drive/MyDrive/my_projects/GelGenie/python-gelgenie")

# import to be able to build the model as in gelgenie
from gelgenie.segmentation.networks import model_configure
from gelgenie.segmentation.training.training_setup import core_setup

# Path to the checkpoint file (old model with 2 classes)
checkpoint_path = "/content/drive/MyDrive/my_projects/checkpoint_epoch_600(1).pth"

# Load the checkpoint, and from it, print out the keys and get the network
saved_dict = torch.load(checkpoint_path, map_location="cpu")
saved_state_dict = saved_dict["network"]
print("Loaded checkpoint keys:", saved_dict.keys())

# New model = 3 classes instead of 2
model, _, _ = model_configure(
    model_name="smp_unet",
    encoder_name="resnet18",
    classes=3,
    in_channels=1   # grayscale
)

# Get its state dict (random init at this point)
current_state_dict = model.state_dict()

#  Load old weights into new model
try:
    msg = model.load_state_dict(saved_state_dict, strict=False)
    # f string type format to check that everything was loaded by going through the lists of the incompatible keys
    print(f"Loaded with {len(msg.missing_keys)} missing and {len(msg.unexpected_keys)} unexpected keys")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    raise

# Print matching/missing/unexpected
print("\n Matching layers loaded:")
for k, v in saved_state_dict.items():
    if k in current_state_dict and v.shape == current_state_dict[k].shape:
        print(f"{k} {v.shape}")

print("\n Missing layers (not loaded, randomly initialized):")
for m in msg.missing_keys:
    print(f"{m} expected shape {current_state_dict[m].shape}")

print("\n Unexpected layers (in checkpoint but not in model):")
for u in msg.unexpected_keys:
    print(u)

#  Reinitialize optimizer + scheduler as with gelgenie code
# We don't load optimizer/scheduler from old checkpoint, since class count changed.
optimizer, scheduler = core_setup(
    network=model,
    lr=1e-4,                       # matching config
    optimizer_type="adam",
    scheduler_type="CosineAnnealingWarmRestarts",
    scheduler_specs={"restart_period": 100}
)

# Build new checkpoint dict
# Epoch to 579 as before.
new_checkpoint = {
    "network": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler else None,
    "epoch": 600
}

#  Save checkpoint
save_path = "/content/drive/MyDrive/my_projects/sharp_band_checpoint_600"
torch.save(new_checkpoint, save_path)
print(f"\n New checkpoint saved at: {save_path}")