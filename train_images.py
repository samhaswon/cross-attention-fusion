"""
Train the model
"""

import os
import sys
from typing import Union, Callable, List
import warnings

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.v2.functional as tf
from torch.utils.tensorboard import SummaryWriter
# pylint: disable=import-error
from get_device import get_device
from models.mhsa_2 import MHSAViT


ROOT_DIRECTORY = "features/train"
EVAL_DIRECTORY = "features/eval"
NUM_EPOCHS = 100
TENSOR_SIZE = 512
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 96
USE_AMP = True


class MalwareDataset(Dataset):
    """
    Custom Dataset class for loading malware/benign files
    """
    def __init__(
            self,
            root_dir: str,
            tensor_size: int,
            transform: Union[None, Callable] = None
    ):
        """
        Args:
        :param root_dir: Directory with all the subdirectories (benign/malware).
        :param transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.tensor_size = tensor_size
        self.transform = transform
        self.data = []
        self.labels = []

        # Walk through the root directory to get all files and their labels
        for label_dir in ['benign', 'malicious']:
            label = 0 if label_dir == 'benign' else 1
            dir_path = os.path.join(root_dir, label_dir)
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if os.path.isfile(file_path) and os.path.getsize(file_path) > 1000:
                    self.data.append(file_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]

        # Load the image
        image = Image.open(file_path)

        image = tf.to_image(image)
        image = tf.to_dtype(image, torch.float32, scale=True)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


def load_checkpoint(model_in, optimizer_in, file_path="checkpoint.pth") -> None:
    """
    Load a saved checkpoint.
    :param model_in: The model (variable) to load into.
    :param optimizer_in: The optimizer (variable) to load into.
    :param file_path: The path to the checkpoint.
    :return: None
    """
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path)
        model_in.load_state_dict(checkpoint["state"]["state_dict"])
        optimizer_in.load_state_dict(checkpoint["state"]["optimizer"])
    else:
        print(f"Checkpoint not found at `{file_path}`. Starting from scratch")


def save_model(model_in: nn.Module, optimizer_in, device: str, t_size: int) -> None:
    """
    Saves the model in ONNX format.

    :param model_in: The trained model.
    :param optimizer_in: The model's optimizer (for the checkpoint).
    :param device: The device where the model is located.
    :param t_size: The size of the feature portion of the tensor used.
    """
    input_tensor_size = (1, 3, t_size, t_size)
    x = torch.randn(*input_tensor_size, requires_grad=True)
    x = x.to(device)

    onnx_file_name = "model.onnx"
    torch.onnx.export(
        model_in,
        x,
        onnx_file_name,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model saved to: {onnx_file_name}\t", end="")
    torch.save(
        {"state":
            {
                "state_dict": model_in.state_dict(),
                "optimizer": optimizer_in.state_dict()
            }
        },
        "checkpoint.pth")
    print("Checkpoint saved to: checkpoint.pth\n")


if __name__ == '__main__':
    # For tensorboard
    writer = SummaryWriter()

    # PyTorch sometimes sends a warning, so ignore that since what we're doing is fine
    warnings.filterwarnings("ignore")

    DEVICE = str(get_device())
    if USE_AMP:
        print("Using Automatic Mixed Precision (AMP) for the forward pass")
    else:
        print("Not using Automatic Mixed Precision (AMP) for the forward pass")
    # If your loss is doing nothing, this is likely the reason
    if "cuda" in DEVICE:
        torch.backends.cuda.matmul.allow_tf32 = True
        print("Using TF32 for some calculations")
    # Model instantiation
    print("Instantiating model")
    model = MHSAViT(
        image_size=512,
        patch_size=32,
        num_layers=6,
        num_heads=6,
        hidden_dim=768,
        mlp_dim=1536,
        dropout=0.01,
        attention_dropout=0
    )

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for binary classification
    if USE_AMP:
        # Increased eps parameter due to collapsing loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1E-7)
    else:
        # Default eps
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1E-8)

    model.to(DEVICE)

    # Training dataset and data loader
    dataset = MalwareDataset(ROOT_DIRECTORY, tensor_size=TENSOR_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Eval dataset and data loader
    eval_dataset = MalwareDataset(EVAL_DIRECTORY, tensor_size=TENSOR_SIZE)
    eval_dataloader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE, num_workers=4)

    scheduler = CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=1E-6)
    grad_scaler = torch.cuda.amp.GradScaler()

    load_checkpoint(model, optimizer)

    print("Starting training")
    step = 1

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        # A variable, not a constant, so ignore the warning
        # pylint: disable=invalid-name
        running_loss = 0.0
        count = 1
        print(" ")

        # Lists for training loop accuracy
        result_list: List[int] = []
        result_list_np: np.array = np.array([])

        for data, labels in dataloader:
            sys.stdout.write("\033[F")
            print(f"\tIteration {count}/{len(dataloader)} ", end="")
            count += 1
            # For BCELoss, labels need to be of shape (batch_size, 1)
            labels = labels.unsqueeze(1)

            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Autocast for the forward pass in FP16 (NVIDIA Turing architecture and newer)
            with torch.autocast(device_type=str(DEVICE), dtype=torch.float16, enabled=USE_AMP):
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, labels)
                writer.add_scalar("Loss/step", loss, step)
            step += 1
            # Send the results to the cpu
            result_np = outputs.cpu().data.numpy()
            labels = labels.cpu().data.numpy()

            # Backward pass and optimize
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # Clip gradients if their norm exceeds 1
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            # Figure out the accuracy
            result_list_np = np.append(result_list_np, result_np)
            tmp_results = (
                    np.where(result_np < 0, 0, 1) == labels.astype(np.float32)
            ).astype(np.float32)
            result_list.extend([int(x) for x in tmp_results])

            running_loss += loss.item()
            print(f"Loss: {loss.item()}")
        save_model(model, optimizer, DEVICE, t_size=TENSOR_SIZE)
        print(f"Training result mean: {np.mean(result_list_np)}")
        print(f"Training accuracy: {sum(result_list) / len(result_list) * 100:.4f}%\n")

        # Do a model validation pass
        model.eval()
        result_list: List[int] = []
        result_list_np: np.array = np.array([])
        # pylint: disable=invalid-name
        count = 1

        for data, labels in eval_dataloader:
            sys.stdout.write("\033[F")
            print(f"\tEval iteration {count}/{len(eval_dataloader)}")
            labels = labels.unsqueeze(1)

            data = data.to(DEVICE)
            labels = np.array(labels)  # labels.to(DEVICE)

            # Gradient accumulation in the evaluation step costs 2GB of VRAM, so don't do that
            with torch.no_grad(), torch.autocast(
                    device_type=str(DEVICE),
                    dtype=torch.float16,
                    enabled=USE_AMP):
                result = model(data)
            result_np = result.cpu().data.numpy()

            result_list_np = np.append(result_list_np, result_np)
            tmp_results = (
                    np.where(result_np < 0, 0, 1) == labels.astype(np.float32)
            ).astype(np.float32)
            result_list.extend([int(x) for x in tmp_results])
            count += 1
        print(f"Evaluation result mean: {np.mean(result_list_np)}")
        print(f"Evaluation accuracy: {sum(result_list) / len(result_list) * 100:.4f}%\n")
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {running_loss / len(dataloader)}")
        writer.add_scalar("Loss/epoch", running_loss / len(dataloader), epoch)
        writer.flush()
    writer.close()
    print("Finished Training.")
