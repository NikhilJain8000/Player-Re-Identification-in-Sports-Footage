# reid.py
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

class ReID:
    """
    A class to handle feature extraction for player re-identification.
    It uses a pre-trained ResNet50 model to generate embeddings.
    """
    def __init__(self):
        # Load the best available pre-trained weights for ResNet50
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        
        # Remove the final classification layer to get feature embeddings
        # The output will be a 2048-dimensional vector
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        
        # Use GPU if available for faster processing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Define transformations
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, player_crop):
        """
        Generates a feature embedding for a single player crop image.
        
        Args:
            player_crop (numpy.ndarray): The image crop of the player (from OpenCV BGR).

        Returns:
            numpy.ndarray: A 1D feature vector (2048 dimensions).
        """
        # Convert BGR to RGB (OpenCV uses BGR, but the model expects RGB)
        img_rgb = player_crop[:, :, ::-1].copy()  # The .copy() prevents negative stride issues
        
        # Apply transformations
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            embedding = self.model(img_tensor)
            
        # Flatten to 1D vector and move to CPU
        return embedding.cpu().numpy().flatten()