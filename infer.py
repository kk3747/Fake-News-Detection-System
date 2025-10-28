import os
import json
import torch
import torch.nn.functional as F
import numpy as np

from improved_gnn import ImprovedNewsGNN


class InferenceModel:
    """Loads trained GNN model and performs inference on precomputed feature vectors.
    Auto-detects feature dimensions from saved state_dict for robust loading.

    API:
      - load(model_path)
      - predict_from_vector(feature_vector) -> {probabilities, pred}
    """

    def __init__(self, model_path='models/gnn_model.pt', device=None):
        self.model_path = model_path
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        print(f"Using device: {self.device}")
        self.model = None
        self._news_feat_dim = None
        self._news_feat_dim = None
        self._tweet_feat_dim = None

    def load(self):
        """Load model, auto-detecting input dimensions from saved weights."""
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"Model checkpoint not found at {self.model_path}")

        try:
            # Load state dict first to inspect shapes
            state = torch.load(self.model_path, map_location=self.device)
            
            # Get news feature dim from first layer weights
            news_encoder_key = 'news_encoder.0.weight'  # First linear layer
            if news_encoder_key not in state:
                raise ValueError("Unexpected model structure - missing news encoder weights")
            self._news_feat_dim = state[news_encoder_key].shape[1]
            
            # Get tweet feature dim similarly
            tweet_encoder_key = 'tweet_encoder.0.weight'
            if tweet_encoder_key not in state:
                raise ValueError("Unexpected model structure - missing tweet encoder weights")
            tweet_feat_dim = state[tweet_encoder_key].shape[1]
            
            # Now create model with correct dimensions
            self.model = ImprovedNewsGNN(
                news_feature_dim=self._news_feat_dim,
                tweet_feature_dim=tweet_feat_dim,
                hidden_channels=128,  # These can be fixed since they're internal
                num_heads=4
            ).to(self.device)
            
            self.model.load_state_dict(state)
            self.model.eval()
            print(f"Model loaded. Expects news feature vectors with {self._news_feat_dim} dimensions")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(traceback.format_exc())
            raise

    def predict_from_vector(self, feature_vector):
        """feature_vector: list or numpy array of size news_feature_dim

        Returns dict: {probabilities: [p0, p1], pred: int}
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            x_news = torch.FloatTensor(np.array(feature_vector))
        except Exception as e:
            raise ValueError(f"Could not convert input to tensor: {e}")

        if x_news.shape[0] != self._news_feat_dim:
            raise ValueError(
                f"Input vector has {x_news.shape[0]} dimensions but model expects {self._news_feat_dim}. "
                "Pass a feature vector matching the training dimension."
            )
        
        x_news = x_news.unsqueeze(0).to(self.device)


def _test():
    # Quick smoke test when running infer.py directly
    m = InferenceModel()
    if not os.path.exists(m.model_path):
        print('Model checkpoint not found at', m.model_path)
        return
    m.load()
    # Create dummy vector matching detected dimension
    dummy = np.random.rand(m._news_feat_dim).tolist()
    print(m.predict_from_vector(dummy))


if __name__ == '__main__':
    _test()
