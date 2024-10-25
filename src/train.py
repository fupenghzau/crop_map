import torch
import torch.nn as nn

def mkmmd_loss_fn(source_features, target_features, bandwidths=[1, 5, 10]):
    """
    Compute the multi-kernel maximum mean discrepancy (MK-MMD) loss for domain adaptation.

    Parameters:
    - source_features (torch.Tensor): Features from the source domain of shape (batch_size, feature_dim).
    - target_features (torch.Tensor): Features from the target domain of shape (batch_size, feature_dim).
    - bandwidths (list of int): List of bandwidths for the Gaussian kernels.

    Returns:
    - torch.Tensor: The computed MK-MMD loss.
    """
    loss = 0.0
    for bandwidth in bandwidths:
        gamma = 1 / (2 * bandwidth ** 2)
        loss += torch.exp(-gamma * ((source_features.unsqueeze(1) - target_features.unsqueeze(0)) ** 2).sum(-1)).mean()
    return loss

def train_model_with_mkmmd(model, X_source, y_source, X_target, y_target, 
                           optimizer, lambda_mmd=0.1, num_epochs=1000, 
                           early_stopping_patience=100, lr_scheduler=None):
    """
    Train the model using MK-MMD loss for domain adaptation with optional early stopping.

    Parameters:
    - model (torch.nn.Module): The model to be trained.
    - X_source (torch.Tensor): Source domain data.
    - y_source (torch.Tensor): Labels for the source domain data.
    - X_target (torch.Tensor): Target domain data.
    - y_target (torch.Tensor): Labels for the target domain data.
    - optimizer (torch.optim.Optimizer): Optimizer for the model.
    - lambda_mmd (float): Weight for the MK-MMD loss.
    - num_epochs (int): Maximum number of epochs to train.
    - early_stopping_patience (int): Number of epochs to wait for early stopping.
    - lr_scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.

    Returns:
    - model (torch.nn.Module): The trained model.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    no_improvement_epochs = 0

    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Forward pass: make predictions for each sample
        source_logits = model(X_source)  # Shape should be (batch_size, num_classes, sequence_length)
        target_logits = model(X_target)  # Shape should be (batch_size, num_classes, sequence_length)

        # Compute the classification losses using original logits
        source_loss = criterion(source_logits.view(-1, source_logits.size(1)), y_source)  # Flatten logits for batch
        target_loss = criterion(target_logits.view(-1, target_logits.size(1)), y_target)

        # Compute MK-MMD loss
        source_features = source_logits.mean(dim=-1)  # Shape becomes (batch_size, num_classes)
        target_features = target_logits.mean(dim=-1)  # Shape becomes (batch_size, num_classes)
        mmd_loss = mkmmd_loss_fn(source_features, target_features)

        # Total loss
        total_loss = source_loss + target_loss + lambda_mmd * mmd_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update the learning rate scheduler if available
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Early stopping check
        if total_loss < best_loss:
            best_loss = total_loss
            no_improvement_epochs = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improvement_epochs += 1

        # Print training progress
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Source Loss: {source_loss.item():.4f}, '
                  f'Target Loss: {target_loss.item():.4f}, MMD Loss: {mmd_loss.item():.4f}, '
                  f'Total Loss: {total_loss.item():.4f}')

        # Trigger early stopping if no improvement for 'early_stopping_patience' epochs
        if no_improvement_epochs >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch} with best loss: {best_loss:.4f}')
            break

    return model