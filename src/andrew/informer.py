import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import InformerConfig, InformerForPrediction
from sklearn.preprocessing import StandardScaler

# Custom Dataset for Time Series
class TimeSeriesDataset(Dataset):
    def __init__(self, data, context_length=96, prediction_length=1):
        """
        Args:
            data: numpy array of shape (timesteps, features)
            context_length: number of past timesteps to use as context
            prediction_length: number of future timesteps to predict
        """
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        
    def __len__(self):
        return len(self.data) - self.context_length - self.prediction_length + 1
    
    def __getitem__(self, idx):
        # Past values (context)
        past_values = self.data[idx:idx + self.context_length]
        
        # Future values (target)
        future_values = self.data[idx + self.context_length:
                                  idx + self.context_length + self.prediction_length]
        
        # Create time features (you can customize these)
        past_time_features = np.arange(self.context_length).reshape(-1, 1)
        future_time_features = np.arange(self.context_length, 
                                        self.context_length + self.prediction_length).reshape(-1, 1)
        
        return {
            'past_values': torch.FloatTensor(past_values),
            'past_time_features': torch.FloatTensor(past_time_features),
            'future_values': torch.FloatTensor(future_values),
            'future_time_features': torch.FloatTensor(future_time_features),
        }

# Example: Load and prepare your data
# Replace this with your actual data loading
def prepare_data(csv_path=None, use_sample=True):
    if use_sample:
        # Sample data: 1000 timesteps, 32 features
        np.random.seed(42)
        data = np.random.randn(1000, 32)
        # Add some temporal pattern
        for i in range(32):
            data[:, i] += np.sin(np.linspace(0, 10*np.pi, 1000) + i*0.1)
    else:
        # Load your actual data
        df = pd.read_csv(csv_path)
        # Assume first column is time, rest are features
        data = df.iloc[:, 1:].values
    
    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    return data_normalized, scaler

# Setup and training
def setup_informer(num_features=32, context_length=96, prediction_length=1):
    """
    Setup Informer model configuration
    
    Args:
        num_features: number of input features (32 in your case)
        context_length: how many past timesteps to look at
        prediction_length: how many future timesteps to predict (1 for next value)
    """
    
    config = InformerConfig(
        # Input/Output configuration
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=num_features,  # Number of features
        
        # Time features
        num_time_features=1,  # Simple time index
        
        # Model architecture
        d_model=128,  # Hidden dimension
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        
        # Informer specific
        attention_dropout=0.1,
        dropout=0.1,
        
        # Distribution head (for probabilistic forecasting)
        distribution_output="student_t",  # Can also use "normal"
        
        # Scaling
        scaling="std",  # Standardize inputs
    )
    
    model = InformerForPrediction(config)
    return model, config

# Training function
def train_model(model, train_loader, num_epochs=10, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            past_values = batch['past_values'].to(device)
            past_time_features = batch['past_time_features'].to(device)
            future_values = batch['future_values'].to(device)
            future_time_features = batch['future_time_features'].to(device)
            
            # Forward pass
            outputs = model(
                past_values=past_values,
                past_time_features=past_time_features,
                future_values=future_values,
                future_time_features=future_time_features,
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    return model

# Prediction function
def predict_next_value(model, recent_data, scaler, context_length=96):
    """
    Predict the next value given recent data
    
    Args:
        model: trained Informer model
        recent_data: numpy array of shape (context_length, num_features)
        scaler: fitted StandardScaler
        context_length: number of past timesteps to use
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Normalize input
    recent_data_normalized = scaler.transform(recent_data)
    
    # Prepare input
    past_values = torch.FloatTensor(recent_data_normalized).unsqueeze(0).to(device)
    past_time_features = torch.FloatTensor(
        np.arange(context_length).reshape(-1, 1)
    ).unsqueeze(0).to(device)
    
    future_time_features = torch.FloatTensor(
        np.array([[context_length]])
    ).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            past_values=past_values,
            past_time_features=past_time_features,
            future_time_features=future_time_features,
        )
    
    # Get prediction (median of distribution)
    prediction = outputs.sequences.cpu().numpy()[0]
    
    # Denormalize
    prediction_original = scaler.inverse_transform(prediction)
    
    return prediction_original

# Main execution
if __name__ == "__main__":
    # Parameters
    NUM_FEATURES = 32
    CONTEXT_LENGTH = 96
    PREDICTION_LENGTH = 1
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    
    print("Loading data...")
    data, scaler = prepare_data(use_sample=True)
    
    print(f"Data shape: {data.shape}")
    
    # Split into train/test
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, CONTEXT_LENGTH, PREDICTION_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Setup model
    print("Setting up Informer model...")
    model, config = setup_informer(NUM_FEATURES, CONTEXT_LENGTH, PREDICTION_LENGTH)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("Training model...")
    model = train_model(model, train_loader, num_epochs=NUM_EPOCHS)
    
    # Make a prediction
    print("\nMaking prediction on test data...")
    recent_data = test_data[:CONTEXT_LENGTH]
    prediction = predict_next_value(model, recent_data, scaler, CONTEXT_LENGTH)
    actual = test_data[CONTEXT_LENGTH]
    
    print(f"Predicted next values (first 5 features): {prediction[0, :5]}")
    print(f"Actual next values (first 5 features): {actual[:5]}")
    
    # Save model
    print("\nSaving model...")
    model.save_pretrained("./informer_model")
    
    print("Done!")