import os
import numpy as np
import librosa
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple, Optional, Union
import pdb


class GTZANDataset(Dataset):
    """
    Dataset class for GTZAN music genre classification using tiktoken tokenizer
    """
    def __init__(self, 
                 data_dir: str,
                 genre: str,  # Specific genre to use
                 tiktoken_encoding: str = "cl100k_base",  # Default GPT-4 encoding
                 sample_rate: int = 22050,
                 duration: int = 30,  # in seconds
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 max_length: int = 1024,  # Maximum sequence length for tokenizer
                 feature_type: str = "mfcc",  # 'mfcc', 'mel', or 'chroma'
                 n_features: int = 20,  # Number of MFCCs or other features
                 include_audio_text: bool = True,  # Whether to include textual representations
                ):
        """
        Initialize the GTZAN dataset for a single genre
        
        Args:
            data_dir: Directory containing audio files organized in genre subfolders
            genre: The specific genre to use (e.g., 'jazz', 'classical', 'rock')
            tiktoken_encoding: Encoding name for tiktoken
            sample_rate: Audio sample rate
            duration: Duration of audio clips in seconds
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length for feature extraction
            max_length: Maximum sequence length for tokenizer
            feature_type: Type of audio feature to extract ('mfcc', 'mel', or 'chroma')
            n_features: Number of features to extract
            include_audio_text: Whether to include textual representation of audio features
        """
        self.data_dir = data_dir
        self.genre = genre
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
        self.feature_type = feature_type
        self.n_features = n_features
        self.include_audio_text = include_audio_text
        
        # Initialize tiktoken tokenizer
        self.tokenizer = tiktoken.get_encoding(tiktoken_encoding)
        
        # Get all files for the specified genre
        self.files = []
        genre_dir = os.path.join(data_dir, genre)
        
        # Verify genre directory exists
        if not os.path.exists(genre_dir) or not os.path.isdir(genre_dir):
            raise ValueError(f"Genre directory not found: {genre_dir}")
        
        # Get all audio files for this genre
        for file in os.listdir(genre_dir):
            if file.endswith('.wav') or file.endswith('.au'):
                self.files.append(os.path.join(genre_dir, file))
        
        if len(self.files) == 0:
            raise ValueError(f"No audio files found for genre '{genre}' in {genre_dir}")
            
        # For a single genre, we don't need label encoding
        # All samples will have the same label
        self.label_id = 0  # Using 0 as the label for our single genre
        
        print(f"Loaded {len(self.files)} audio files for genre '{genre}'")
        
    def __len__(self) -> int:
        """Return the number of audio files in the dataset"""
        return len(self.files)
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract audio features from the given audio file
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            np.ndarray: Extracted features
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        
        # Ensure consistent length
        if len(y) < self.sample_rate * self.duration:
            y = np.pad(y, (0, self.sample_rate * self.duration - len(y)))
        
        # Extract features based on the selected type
        if self.feature_type == 'mfcc':
            features = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=self.n_features,
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )
        elif self.feature_type == 'mel':
            features = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=self.n_features,
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )
            # Convert to dB scale
            features = librosa.power_to_db(features, ref=np.max)
        elif self.feature_type == 'chroma':
            features = librosa.feature.chroma_stft(
                y=y, 
                sr=sr,
                n_chroma=self.n_features,
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
            
        return features
    
    def features_to_text(self, features: np.ndarray) -> str:
        """
        Convert audio features to a textual representation
        
        Args:
            features: Audio features array
            
        Returns:
            str: Textual representation of features
        """
        # Normalize features
        features_norm = (features - np.mean(features)) / np.std(features)
        
        # Convert to string representation
        # Format: "Feature1: [val1, val2, ...], Feature2: [val1, val2, ...], ..."
        text_repr = ""
        for i in range(features.shape[0]):
            feature_values = features_norm[i, :].tolist()
            # Limit the number of values to keep text length reasonable
            feature_values = [round(v, 2) for v in feature_values[:20]]
            text_repr += f"Feature{i+1}: {feature_values}, "
        
        # Add genre information placeholder (will be replaced with actual genre during training)
        text_repr += "Genre: [GENRE]"
        
        return text_repr
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset
        
        Args:
            idx: Index of the item
            
        Returns:
            dict: Dictionary containing input_ids, attention_mask, label_ids, and segment_ids
        """
        file_path = self.files[idx]
        
        # Extract audio features
        features = self.extract_features(file_path)
        
        # Convert features to text if enabled
        if self.include_audio_text:
            text = self.features_to_text(features)
            # Replace genre placeholder with actual genre
            text = text.replace("[GENRE]", self.genre)
        else:
            # Basic text description if we're not including detailed features
            text = f"This is a {self.genre} music track."
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        # Truncate or pad tokens to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token != 0 else 0 for token in tokens]
        
        # Create segment IDs (all 0s for single sequence)
        segment_ids = [0] * len(tokens)
        
        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        label_ids = torch.tensor(self.label_id, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'segment_ids': segment_ids,
            'label_ids': label_ids,
            'genre': self.genre,
            'file_path': file_path
        }


def create_gtzan_dataloaders(
    data_dir: str,
    genre: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for a single genre from GTZAN dataset
    
    Args:
        data_dir: Directory containing audio files
        genre: Specific genre to use (e.g., 'jazz', 'classical', 'rock')
        batch_size: Batch size for dataloaders
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        **dataset_kwargs: Additional arguments for GTZANDataset
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create dataset for the specified genre
    dataset = GTZANDataset(data_dir=data_dir, genre=genre, **dataset_kwargs)
    
    # Split dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_end = int(train_split * dataset_size)
    val_end = train_end + int(val_split * dataset_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create subset samplers
    from torch.utils.data.sampler import SubsetRandomSampler
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler
    )
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Replace with your GTZAN dataset path
    data_dir = "data/genres_original"  # Path to GTZAN dataset
    
    # Specify which genre to use
    genre = "jazz"  # Choose one of: 'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'
    
    # Create dataloaders for the specified genre
    train_loader, val_loader, test_loader = create_gtzan_dataloaders(
        data_dir=data_dir,
        genre=genre,
        batch_size=16,
        feature_type="mfcc",
        n_features=20,
        max_length=512
    )
    
    # Check a batch
    for batch in train_loader:
        pdb.set_trace()
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Attention mask shape:", batch['attention_mask'].shape)
        print("Segment IDs shape:", batch['segment_ids'].shape)
        print("Label IDs shape:", batch['label_ids'].shape)
        print("Genres:", batch['genre'])
        
        # Print a sample tokenized text
        sample_idx = 0
        tokens = batch['input_ids'][sample_idx].tolist()
        # Remove padding tokens
        tokens = [t for t in tokens if t != 0]
        
        # Get the encoding to decode the tokens
        enc = tiktoken.get_encoding("cl100k_base")
        print("Sample text:", enc.decode(tokens))
        
        # Only print first batch
        break