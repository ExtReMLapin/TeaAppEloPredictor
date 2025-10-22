import os
import sys
import argparse
import torch
import open_clip
from torch import nn, autocast
from PIL import Image

MODEL_NAME = 'hf-hub:timm/PE-Core-bigG-14-448'
EMBEDDING_DIM = 1280
REGRESSOR_PATH = "best_open_clip_regressor.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.fc2(self.act(self.fc1(self.norm(x)))))

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)
    def forward(self, x):
        x, gate = self.fc(x).chunk(2, dim=-1)
        return x * torch.sigmoid(gate)

class EmbeddingRegressor(nn.Module):
    def __init__(self, embedding_dim=1280, width=512, depth=1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, width),
            SwiGLU(width)
        )
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(width, dropout=0.2) for _ in range(depth)]
        )
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.out(x)

def load_models():
    print(f"Loading embedding model : {MODEL_NAME}...")
    
    embedder_model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, precision="fp16", device=DEVICE)
    embedder_model.eval()
    
    if not os.path.exists(REGRESSOR_PATH):
        print(f"Could not find regressor : {REGRESSOR_PATH}")
        sys.exit(1)
        
    checkpoint = torch.load(REGRESSOR_PATH, map_location=DEVICE, weights_only=False)
    
    regressor_model = EmbeddingRegressor(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    regressor_model.load_state_dict(checkpoint['model_state'])
    regressor_model.eval()
    
    mean_elo = checkpoint['mean_elo']
    std_elo = checkpoint['std_elo']
    
    return embedder_model, preprocess, regressor_model, mean_elo, std_elo

def predict_elo(image_path, embedder, preprocess, regressor, mean, std):

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)


    with torch.no_grad(), autocast(DEVICE):
        embedding = embedder.encode_image(image_tensor)
        
        normalized_score = regressor(embedding.float())
        
    predicted_elo = (normalized_score.item() * std) + mean
    return predicted_elo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicts image ELO")
    parser.add_argument("image_path", type=str, help="image path")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"File not found : {args.image_path}")
        sys.exit(1)

    embedder, preprocess, regressor, mean_elo, std_elo = load_models()
    
    print(f"\nPredicting for: {args.image_path}")
    elo_score = predict_elo(args.image_path, embedder, preprocess, regressor, mean_elo, std_elo)
    
    if elo_score is not None:
        print("\n---------------------------------")
        print(f"   ELO Score predicted : {elo_score:.2f}")
        print("---------------------------------")