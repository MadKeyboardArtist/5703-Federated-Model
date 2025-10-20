
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=16):
        
        hidden_dims=[128, 64, 32]
        super(MLPEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建编码器层
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        # 最终的embedding层
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)

def load_and_preprocess_data(file_path):
    """加载和预处理数据"""
    print("Loading and preprocessing data...")
    
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 分离特征和标签
    X = df.drop('Diabetes_012', axis=1).values
    y = df['Diabetes_012'].values
    
    # 转换为张量
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    return X, y

def train_model(model, train_loader, val_loader, device, epochs=50):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            embeddings = model(inputs)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                embeddings = model(inputs)
                loss = criterion(embeddings, labels)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_encoder.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    X, y = load_and_preprocess_data('tabular_dataset/diabetes_012_ready_to_model.csv')
    
    # 分割数据
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型
    input_dim = X.shape[1]  # 特征维度
    model = MLPEncoder(input_dim).to(device)
    
    # 训练模型
    train_model(model, train_loader, val_loader, device)
    
    print("Training completed!")

if __name__ == "__main__":
    main()



