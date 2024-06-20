import torch
import torch.nn as nn


class Trainer:
    """
    TODO update docstring

    A generic Trainer class for training.

    Attributes:
        model: The model to be trained.
        config: Configuration settings for training, such as optimizer, learning rate, epochs, etc.
        optimizer: The optimizer used for training.
        loss_fn: The loss function used for training.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
    """

    def __init__(self, model, config):
        """
        Initialize the Trainer with the given model and configuration.

        Args:
            model: The model to be trained.
            config: Configuration settings for training, such as optimizer, learning rate, epochs, etc.
        """
        self.model = model
        self.config = config
        
        # TODO: Initialize the optimizer with the model parameters and config settings.
        # self.optimizer = 
        
        # TODO: Initialize the loss function from the config.
        # self.loss_fn = 

        # TODO: Initialize data loaders for training and validation data.
        # self.train_loader = 
        # self.val_loader = 
        
        # TODO: Set up other necessary attributes, such as learning rate scheduler.
        # self.scheduler = 

    def train(self):
        """
        Train the model for a specified number of epochs.

        This function iterates over the number of epochs defined in the configuration, 
        performing training and validation steps.
        """
        num_epochs = self.config.get('epochs', 10)  # Default to 10 epochs if not specified in config
        
        for epoch in range(num_epochs):
            # TODO: Set the model to training mode
            # self.model.train()
            
            # Perform a single epoch of training
            self._train_epoch()

            # TODO: Set the model to evaluation mode
            # self.model.eval()
            
            # TODO: Perform validation step and calculate metrics
            # self._validate_epoch()
            
            # TODO: Adjust learning rate if using a scheduler
            # if self.scheduler:
            #     self.scheduler.step()

    def _train_epoch(self):
        """
        Perform one epoch of training.

        This function iterates over the training data loader, performing training steps and logging progress.
        """
        # TODO: Initialize running loss and other metrics if needed
        # running_loss = 0.0
        
        for batch in self.train_loader:
            # TODO: Perform a training step with the current batch
            # loss = self._train_step(batch)
            
            # TODO: Accumulate running loss and other metrics
            # running_loss += loss.item()
        
        # TODO: Log the training metrics for the epoch
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(self.train_loader)}")
            pass

    def _train_step(self, batch):
        """
        Perform a single training step with the given batch of data.

        Args:
            batch: A batch of data from the training DataLoader.

        Returns:
            loss: The loss value for the training step.
        """
        # TODO: Zero the parameter gradients
        # self.optimizer.zero_grad()

        # TODO: Get the inputs and targets from the batch
        # inputs, targets = batch

        # TODO: Forward pass
        # outputs = self.model(inputs)
        
        # TODO: Compute the loss
        # loss = self.loss_fn(outputs, targets)
        
        # TODO: Backward pass and optimize
        # loss.backward()
        # self.optimizer.step()
        
        # return loss
        pass

    def _validate_epoch(self):
        """
        Perform one epoch of validation.

        This function iterates over the validation data loader, performing validation steps and logging progress.
        """
        # TODO: Initialize running loss and other metrics if needed
        # running_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # TODO: Perform a validation step with the current batch
                # loss = self._validate_step(batch)
                
                # TODO: Accumulate running loss and other metrics
                # running_loss += loss.item()
        
        # TODO: Log the validation metrics for the epoch
        # print(f"Validation Loss: {running_loss/len(self.val_loader)}")
                pass

    def _validate_step(self, batch):
        """
        Perform a single validation step with the given batch of data.

        Args:
            batch: A batch of data from the validation DataLoader.

        Returns:
            loss: The loss value for the validation step.
        """
        # TODO: Get the inputs and targets from the batch
        # inputs, targets = batch

        # TODO: Forward pass
        # outputs = self.model(inputs)
        
        # TODO: Compute the loss
        # loss = self.loss_fn(outputs, targets)
        
        # return loss
