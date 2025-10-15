"""
LSTM/Transformer model for time series pattern recognition
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Optional, Tuple
import joblib

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: (n_samples, sequence_length, n_features)
            y: (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """
    LSTM model for sequence classification/regression
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        task: str = 'regression'
    ):
        """
        Args:
            input_size: Number of features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Output dimension (1 for regression, 3 for classification)
            task: 'regression' or 'classification'
        """
        super(LSTMModel, self).__init__()

        self.task = task
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

        # Output activation
        if task == 'classification':
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            output: (batch_size, output_size)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq, hidden)

        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden)

        # Fully connected
        out = self.fc1(context_vector)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.output_activation(out)

        return out


class LSTMTrainer:
    """
    Trainer for LSTM model
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        task: str = 'regression',
        device: str = 'cpu'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.task = task
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
            task=task
        ).to(self.device)

        # Loss function with class weighting
        if task == 'classification':
            # Class weights: SELL (0), HOLD (1), BUY (2)
            # HOLD daha az tahmin edildiği için weight artırıldı
            class_weights = torch.tensor([1.0, 2.0, 1.0])
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.MSELoss()

        self.optimizer = None
        self.train_losses = []
        self.val_losses = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10
    ):
        """
        Train the LSTM model

        Args:
            X_train: (n_samples, sequence_length, n_features)
            y_train: (n_samples,)
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
        """
        logger.info(f"Training LSTM model for {epochs} epochs...")

        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_dataset = TimeSeriesDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)

                if self.task == 'classification':
                    loss = self.criterion(outputs, batch_y.long())
                else:
                    loss = self.criterion(outputs.squeeze(), batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = self.model(batch_X)

                        if self.task == 'classification':
                            loss = self.criterion(outputs, batch_y.long())
                        else:
                            loss = self.criterion(outputs.squeeze(), batch_y)

                        val_loss += loss.item()

                val_loss /= len(val_loader)
                self.val_losses.append(val_loss)

                # Logging
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f}")

        logger.info("Training completed")

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions

        Args:
            X: (n_samples, sequence_length, n_features)
            batch_size: Batch size

        Returns:
            Predictions
        """
        self.model.eval()

        # Create dummy labels (not used for prediction)
        dummy_y = np.zeros(len(X))
        dataset = TimeSeriesDataset(X, dummy_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)

                if self.task == 'classification':
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = outputs.squeeze()

                predictions.append(preds.cpu().numpy())

        predictions = np.concatenate(predictions)

        return predictions

    def predict_proba(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Predict probabilities (classification only)

        Args:
            X: Input sequences

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        self.model.eval()

        dummy_y = np.zeros(len(X))
        dataset = TimeSeriesDataset(X, dummy_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())

        predictions = np.concatenate(predictions)

        return predictions

    def save(self, path: str):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'task': self.task,
            'model_config': {
                'input_size': self.model.lstm.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'output_size': self.model.fc2.out_features
            }
        }, path)

        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.task = checkpoint['task']

        logger.info(f"Model loaded from {path}")
