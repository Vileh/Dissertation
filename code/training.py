# Standard library imports
import json
import random

# Third-party imports
import numpy as np
from sklearn.neighbors import NearestNeighbors

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Torchvision imports
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Local imports
import models

def get_transforms(training_augments=[]):
    transform_clean = [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.3717, 0.4016, 0.3666), (0.1553, 0.1433, 0.1397))
    ]
    transform_train = [transforms.Resize(224)] + training_augments + [transforms.ToTensor(), transforms.Normalize((0.3717, 0.4016, 0.3666), (0.1553, 0.1433, 0.1397))]
    
    return transforms.Compose(transform_train), transforms.Compose(transform_clean)

def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            batch_embeddings = model(images, None)
            embeddings.append(batch_embeddings.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    return embeddings, labels

def find_nearest_neighbors(train_embeddings, test_embeddings, train_labels, k=5):
    neigh = NearestNeighbors(n_neighbors=k, metric='cosine')
    neigh.fit(train_embeddings)

    distances, indices = neigh.kneighbors(test_embeddings)

    predicted_labels = []
    for idx in indices:
        labels = train_labels[idx]
        predicted_label = np.bincount(labels).argmax()
        predicted_labels.append(predicted_label)

    return np.array(predicted_labels)

def arcface_train_loop(model_name, training_augments, label_map, batch_size=64, num_epochs=100, lr=1e-3, weight_decay=5e-4, momentum=0.9, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {device}")

    transform_train, transform_clean = get_transforms(training_augments)

    for split in ['time_aware_split', 'encounter_split', 'random_split']:
        # Load the datasets
        train_dataset = models.CustomImageFolder(f'../data/SeaTurtleIDHeads/splits/{split}/train', label_map=label_map[split], transform=transform_train)
        clean_train_dataset = models.CustomImageFolder(f'../data/SeaTurtleIDHeads/splits/{split}/train', label_map=label_map[split], transform=transform_clean)
        test_dataset = models.CustomImageFolder(f'../data/SeaTurtleIDHeads/splits/{split}/test', label_map=label_map[split], transform=transform_clean)
        val_dataset = models.CustomImageFolder(f'../data/SeaTurtleIDHeads/splits/{split}/val', label_map=label_map[split], transform=transform_clean)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        clean_train_loader = torch.utils.data.DataLoader(clean_train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        # Get number of classes
        num_classes = len(set(train_dataset.classes))

        model = models.ArcFaceEN(num_classes)
        model.to(device);

        # Initialize lists to store the losses and accuracies
        losses = {'train': [], 'val': [], 'test': []}
        accuracies = {'train': [], 'val': [], 'test': [], 'val_embed': [], 'test_embed': []}

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()

        # Define the optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

        # Define the CosineAnnealingLR scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}; Split: {split}")

            model.train()
            total_samples = 0
            running_loss = 0.0
            running_accuracy = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                total_samples += images.size(0)  # Increment total samples

                # Forward pass
                outputs = model(images, labels)

                # Compute the cross-entropy loss
                loss = criterion(outputs, labels)

                accuracy = (outputs.argmax(1) == labels).float().mean()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)  # Multiply by batch size
                running_accuracy += accuracy.item() * images.size(0)  # Multiply by batch size

            epoch_loss = running_loss / total_samples
            epoch_accuracy = running_accuracy / total_samples

            losses['train'].append(epoch_loss)
            accuracies['train'].append(epoch_accuracy)
            if verbose:
                print(f"--> Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")

            # Evaluation on the validation set
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_accuracy = 0.0
                val_total_samples = 0

                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    val_total_samples += images.size(0)  # Increment total samples

                    outputs = model(images, labels)

                    loss = criterion(outputs, labels)
                    accuracy = (outputs.argmax(1) == labels).float().mean()

                    val_loss += loss.item() * images.size(0)  # Multiply by batch size
                    val_accuracy += accuracy.item() * images.size(0)  # Multiply by batch size

                val_epoch_loss = val_loss / val_total_samples
                val_epoch_accuracy = val_accuracy / val_total_samples

                losses['val'].append(val_epoch_loss)
                accuracies['val'].append(val_epoch_accuracy)

                if verbose:
                    print(f"--> Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}")

                test_loss = 0.0
                test_accuracy = 0.0
                test_total_samples = 0

                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    test_total_samples += images.size(0)  # Increment total samples
                    outputs = model(images, labels)

                    loss = criterion(outputs, labels)
                    accuracy = (outputs.argmax(1) == labels).float().mean()

                    test_loss += loss.item() * images.size(0)  # Multiply by batch size
                    test_accuracy += accuracy.item() * images.size(0)  # Multiply by batch size

                test_loss /= test_total_samples
                test_accuracy /= test_total_samples

                losses['test'].append(test_loss)
                accuracies['test'].append(test_accuracy)

                if verbose:
                    print(f"--> Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

                if epoch % 5 == 0:
                    # Extract training embeddings
                    train_embeddings, train_labels = get_embeddings(model, clean_train_loader, device)

                    # Extract test and val embeddings and find nearest neighbors
                    test_embeddings, test_labels = get_embeddings(model, test_loader, device)
                    val_embeddings, val_labels = get_embeddings(model, val_loader, device)

                    predicted_labels_test = find_nearest_neighbors(train_embeddings, test_embeddings, train_labels, k=3)
                    predicted_labels_val = find_nearest_neighbors(train_embeddings, val_embeddings, train_labels, k=3)

                    # Evaluate accuracy
                    accuracy_test = np.mean(predicted_labels_test == test_labels)
                    accuracy_val = np.mean(predicted_labels_val == val_labels)

                    if verbose:
                        print(f"-----> Test Embedding Accuracy: {accuracy_test:.4f}")
                        print(f"-----> Val Embedding Accuracy: {accuracy_val:.4f}")

                    accuracies['test_embed'].append(accuracy_test)
                    accuracies['val_embed'].append(accuracy_val)

        # Extract training embeddings
        train_embeddings, train_labels = get_embeddings(model, clean_train_loader, device)

        # Extract test and val embeddings and find nearest neighbors
        test_embeddings, test_labels = get_embeddings(model, test_loader, device)
        val_embeddings, val_labels = get_embeddings(model, val_loader, device)

        predicted_labels_test = find_nearest_neighbors(train_embeddings, test_embeddings, train_labels, k=3)
        predicted_labels_val = find_nearest_neighbors(train_embeddings, val_embeddings, train_labels, k=3)

        # Evaluate accuracy
        accuracy_test = np.mean(predicted_labels_test == test_labels)
        accuracy_val = np.mean(predicted_labels_val == val_labels)
        print(f"Final Test Embedding Accuracy: {accuracy_test:.4f}")
        print(f"Final Val Embedding Accuracy: {accuracy_val:.4f}")

        accuracies['test_embed'].append(accuracy_test)
        accuracies['val_embed'].append(accuracy_val)


        # Export model and metrics
        torch.save(model.state_dict(), f'../results/weights/{model_name}_{split}.pth')

        del model

        with open(f'../results/training_metrics/losses_{model_name}_{split}.json', 'w') as file:
            json.dump(losses, file)

        with open(f'..results/training_metrics/accuracies_{model_name}_{split}.json', 'w') as file:
            json.dump(accuracies, file)

def crossentropy_train_loop(model_name, training_augments, label_map, batch_size=64, num_epochs=100, lr=1e-3, weight_decay=5e-4, momentum=0.9, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {device}")

    transform_train, transform_clean = get_transforms(training_augments)

    for split in ['time_aware_split', 'encounter_split', 'random_split']:
        # Load the datasets
        train_dataset = models.CustomImageFolder(f'../data/SeaTurtleIDHeads/splits/{split}/train', label_map=label_map[split], transform=transform_train)
        test_dataset = models.CustomImageFolder(f'../data/SeaTurtleIDHeads/splits/{split}/test', label_map=label_map[split], transform=transform_clean)
        val_dataset = models.CustomImageFolder(f'../data/SeaTurtleIDHeads/splits/{split}/val', label_map=label_map[split], transform=transform_clean)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        # Get number of classes
        num_classes = len(set(train_dataset.classes))

        model = models.EfficientNetModel(num_classes)
        model.to(device);

        # Initialize lists to store the losses and accuracies
        losses = {'train': [], 'val': [], 'test': []}
        accuracies = {'train': [], 'val': [], 'test': []}

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()

        # Define the optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

        # Define the CosineAnnealingLR scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}; Split: {split}")

            model.train()
            total_samples = 0
            running_loss = 0.0
            running_accuracy = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                total_samples += images.size(0)  # Increment total samples

                # Forward pass
                outputs = model(images)

                # Compute the cross-entropy loss
                loss = criterion(outputs, labels)

                accuracy = (outputs.argmax(1) == labels).float().mean()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)  # Multiply by batch size
                running_accuracy += accuracy.item() * images.size(0)  # Multiply by batch size

            epoch_loss = running_loss / total_samples
            epoch_accuracy = running_accuracy / total_samples

            losses['train'].append(epoch_loss)
            accuracies['train'].append(epoch_accuracy)
            if verbose:
                print(f"--> Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")

            # Evaluation on the validation set
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_accuracy = 0.0
                val_total_samples = 0

                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    val_total_samples += images.size(0)  # Increment total samples

                    outputs = model(images)

                    loss = criterion(outputs, labels)
                    accuracy = (outputs.argmax(1) == labels).float().mean()

                    val_loss += loss.item() * images.size(0)  # Multiply by batch size
                    val_accuracy += accuracy.item() * images.size(0)  # Multiply by batch size

                val_epoch_loss = val_loss / val_total_samples
                val_epoch_accuracy = val_accuracy / val_total_samples

                losses['val'].append(val_epoch_loss)
                accuracies['val'].append(val_epoch_accuracy)

                if verbose:
                    print(f"--> Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}")

                test_loss = 0.0
                test_accuracy = 0.0
                test_total_samples = 0

                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    test_total_samples += images.size(0)  # Increment total samples

                    outputs = model(images)

                    loss = criterion(outputs, labels)
                    accuracy = (outputs.argmax(1) == labels).float().mean()

                    test_loss += loss.item() * images.size(0)  # Multiply by batch size
                    test_accuracy += accuracy.item() * images.size(0)  # Multiply by batch size

                test_loss /= test_total_samples
                test_accuracy /= test_total_samples

                losses['test'].append(test_loss)
                accuracies['test'].append(test_accuracy)

                if verbose:
                    print(f"--> Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Export model and metrics
        torch.save(model.state_dict(), f'../results/weights/{model_name}_{split}.pth')

        del model

        with open(f'../results/training_metrics/losses_{model_name}_{split}.json', 'w') as file:
            json.dump(losses, file)

        with open(f'../results/training_metrics/accuracies_{model_name}_{split}.json', 'w') as file:
            json.dump(accuracies, file)

def random_motion(steps=16, initial_vector=None, alpha=0.2):
    if initial_vector is None:
        initial_vector = torch.randn(1, dtype=torch.cfloat)

    # Generate the random motion path
    motion = [torch.zeros_like(initial_vector)]
    for _ in range(steps):
        change = torch.randn(initial_vector.shape[0], dtype=torch.cfloat)
        initial_vector = initial_vector + change * alpha
        initial_vector /= initial_vector.abs().add(1e-8)
        motion.append(motion[-1] + initial_vector)

    motion = torch.stack(motion, -1)

    # Find bounding box
    real_min, _ = motion.real.min(dim=-1, keepdim=True)
    real_max, _ = motion.real.max(dim=-1, keepdim=True)
    imag_min, _ = motion.imag.min(dim=-1, keepdim=True)
    imag_max, _ = motion.imag.max(dim=-1, keepdim=True)

    # Scale motion to fit exactly in steps x steps
    real_scale = (steps - 1) / (real_max - real_min)
    imag_scale = (steps - 1) / (imag_max - imag_min)
    scale = torch.min(real_scale, imag_scale)

    real_shift = (steps - (real_max - real_min) * scale) / 2 - real_min * scale
    imag_shift = (steps - (imag_max - imag_min) * scale) / 2 - imag_min * scale

    scaled_motion = motion * scale + (real_shift + 1j * imag_shift)

    # Create kernel
    kernel = torch.zeros(initial_vector.shape[0], 1, steps, steps)

    # Fill kernel
    for s in range(steps + 1):
        v = scaled_motion[:, s]
        x = torch.clamp(v.real, 0, steps - 1)
        y = torch.clamp(v.imag, 0, steps - 1)

        ix = x.long()
        iy = y.long()

        vx = x - ix.float()
        vy = y - iy.float()

        for i in range(initial_vector.shape[0]):
            kernel[i, 0, iy[i], ix[i]] += (1-vx[i]) * (1-vy[i]) / steps
            if ix[i] + 1 < steps:
                kernel[i, 0, iy[i], ix[i]+1] += vx[i] * (1-vy[i]) / steps
            if iy[i] + 1 < steps:
                kernel[i, 0, iy[i]+1, ix[i]] += (1-vx[i]) * vy[i] / steps
            if ix[i] + 1 < steps and iy[i] + 1 < steps:
                kernel[i, 0, iy[i]+1, ix[i]+1] += vx[i] * vy[i] / steps

    # Normalize the kernel
    kernel /= kernel.sum(dim=(-1, -2), keepdim=True)

    return kernel

class RandomDistortion(nn.Module):
    """
    Apply random distortion (no distortion, Gaussian blur, motion blur, or resolution distortion) to input PIL Images.
    """
    def __init__(self, motion_steps=17, motion_alpha=0.2,
                 gaussian_kernel_sizes=[3, 5, 7, 9, 11, 13],
                 gaussian_sigmas=[1, 2, 3, 5],
                 resolution_scale_factors=[0.25, 0.5, 0.75]):
        """
        Initialize the RandomDistortion module.

        Args:
        - motion_steps (int): Number of steps in the motion path
        - motion_alpha (float): Controls the randomness of the motion path
        - gaussian_kernel_sizes (list): List of kernel sizes for Gaussian blur
        - gaussian_sigmas (list): List of sigma values for Gaussian blur
        - resolution_scale_factors (list): List of scale factors for resolution distortion
        """
        super().__init__()
        self.motion_steps = motion_steps
        self.motion_alpha = motion_alpha
        self.gaussian_kernel_sizes = gaussian_kernel_sizes
        self.gaussian_sigmas = gaussian_sigmas
        self.resolution_scale_factors = resolution_scale_factors

    def gaussian_blur(self, img):
        """Apply Gaussian blur to the input PIL Image."""
        kernel_size = random.choice(self.gaussian_kernel_sizes)
        sigma = random.choice(self.gaussian_sigmas)
        return TF.gaussian_blur(img, kernel_size, [sigma, sigma])

    def motion_blur(self, img):
        """Apply motion blur to the input PIL Image."""
        # Convert PIL Image to tensor
        x = TF.to_tensor(img).unsqueeze(0)

        # Generate a random initial vector
        vector = torch.randn(1, dtype=torch.cfloat) / 3
        vector.real /= 2

        # Create the motion blur kernel
        m = random_motion(self.motion_steps, vector, alpha=self.motion_alpha)

        # Pad the input tensor for convolution
        xpad = [m.shape[-1]//2+1] * 2 + [m.shape[-2]//2+1] * 2
        x = F.pad(x, xpad)

        # Pad the kernel to match input size
        mpad = [0, x.shape[-1]-m.shape[-1], 0, x.shape[-2]-m.shape[-2]]
        mp = F.pad(m, mpad)

        # Apply blur in the frequency domain
        fx = torch.fft.fft2(x)  # FFT of input
        fm = torch.fft.fft2(mp)  # FFT of kernel
        fy = fx * fm  # Multiplication in frequency domain
        y = torch.fft.ifft2(fy).real  # Inverse FFT to get blurred result

        # Crop the result to original size
        y = y[..., xpad[2]:-xpad[3], xpad[0]:-xpad[1]]

        # Clip values to [0, 1] range
        y = torch.clamp(y, 0, 1)

        # Convert back to PIL Image
        return TF.to_pil_image(y.squeeze(0))

    def resolution_distortion(self, img):
        """
        Lowers the resolution of the image and then scales it back to original size.
        """
        scale_factor = random.choice(self.resolution_scale_factors)
        w, h = img.size
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)

        transform = T.Compose([
            T.Resize((new_h, new_w), antialias=True),  # Lower resolution
            T.Resize((h, w), antialias=True)  # Scale back to original size
        ])

        return transform(img)

    def forward(self, img):
        """
        Apply random distortion to the input PIL Image.

        Args:
        - img (PIL.Image): Input image

        Returns:
        - PIL.Image: Distorted or original image
        """
        choice = random.random()

        if choice < 1/4:  # No distortion
            return img
        elif choice < 2/4:  # Gaussian blur
            return self.gaussian_blur(img)
        elif choice < 3/4:  # Motion blur
            return self.motion_blur(img)
        else:  # Resolution distortion
            return self.resolution_distortion(img)