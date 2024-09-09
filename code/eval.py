import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

def predict_ce(model, test_loader, device=torch.device('cpu'), train_indices=None, test_indices=None, distortion_func=None):
    model.eval()

    predictions = []
    ground_truth = []

    current_index = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if test_indices is not None:
                batch_indices = [i - current_index for i in test_indices if current_index <= i < current_index + len(images)]
                if batch_indices:
                    images = images[batch_indices]
                    labels = labels[batch_indices]
                else:
                    current_index += len(images)
                    continue
            
            # Apply distortion if a function is provided
            if distortion_func is not None:
                images = distortion_func(images)
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions.append(outputs.argmax(dim=1).cpu().numpy())
            ground_truth.append(labels.cpu().numpy())
            
            current_index += len(images)
            if test_indices is not None and current_index > max(test_indices):
                break
    return np.concatenate(predictions), np.concatenate(ground_truth)


def get_embeddings(model, dataloader, device, indices=None, distortion_func=None):
    model.eval()
    embeddings = []
    labels = []
    current_index = 0
    with torch.no_grad():
        for images, batch_labels in dataloader:
            if indices is not None:
                # Find which indices in this batch we want
                batch_indices = [i - current_index for i in indices if current_index <= i < current_index + len(images)]
                if batch_indices:
                    images = images[batch_indices]
                    batch_labels = batch_labels[batch_indices]
                else:
                    current_index += len(images)
                    continue
            
            # Apply distortion if a function is provided
            if distortion_func is not None:
                images = distortion_func(images)
            
            images = images.to(device)
            batch_embeddings = model(images, None)
            embeddings.append(batch_embeddings.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
            
            current_index += len(images)
            if indices is not None and current_index > max(indices):
                break

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    return embeddings, labels

def find_nearest_neighbors(train_embeddings, test_embeddings, train_labels, train_indices, k=5):
    neigh = NearestNeighbors(n_neighbors=k, metric='cosine')
    neigh.fit(train_embeddings)

    distances, indices = neigh.kneighbors(test_embeddings)

    predicted_labels = []
    for idx in indices:
        labels = train_labels[idx]
        predicted_label = np.bincount(labels).argmax()
        predicted_labels.append(predicted_label)

    return np.array(predicted_labels)

def predict_arc(model, test_loader, clean_train_loader, device=torch.device('cpu'), train_indices=None, test_indices=None, distortion_func=None):
    train_embeddings, train_labels = get_embeddings(model, clean_train_loader, device, indices=train_indices)
    test_embeddings, test_labels = get_embeddings(model, test_loader, device, indices=test_indices, distortion_func=distortion_func)
    predicted_labels_test = find_nearest_neighbors(train_embeddings, test_embeddings, train_labels, train_indices, k=3)
    
    return predicted_labels_test, test_labels

def overall_accuracy(data, split, model_column):
    data = data[data[split] == 'test']

    predictions = data[model_column]
    ground_truth = data['identity']
    return accuracy_score(predictions, ground_truth)

def individual_accuracy(data, split, model_column):
    data = data[data[split] == 'test']
    
    per_individual_accuracy = {}
    for individual in data['identity'].unique():
        individual_data = data[data['identity'] == individual]
        individual_predictions = individual_data[model_column]
        individual_ground_truth = individual_data['identity']
        per_individual_accuracy[individual] = accuracy_score(individual_predictions, individual_ground_truth)

    return per_individual_accuracy

def avg_individual_accuracy(data, split, model_column):
    per_individual_accuracy = individual_accuracy(data, split, model_column)
    return np.mean(list(per_individual_accuracy.values()))