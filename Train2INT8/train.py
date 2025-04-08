# train.py
import torch
import torch.nn as nn
import torch.optim as optim
#from model import SimpleClassifier

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # dummy data (3채널 이미지, 224x224)
    x = torch.randn(16, 3, 224, 224).to(device)
    y = torch.randint(0, 2, (16, 1)).float().to(device)

    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    # 모델 저장
    torch.save(model.state_dict(), "model_weights.pth")

    return model

if __name__ == "__main__":
    train()
