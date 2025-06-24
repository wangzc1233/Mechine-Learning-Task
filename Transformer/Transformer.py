import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import math
import warnings
warnings.filterwarnings('ignore')
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # Linux系统
# plt.rcParams['axes.unicode_minus'] = False
# 设置随机种子保证结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class PowerConsumptionDataset(Dataset):
    """电力消耗数据集类（与之前相同）"""
    def __init__(self, features, target, seq_length):
        self.features = features
        self.target = target
        self.seq_length = seq_length
        
        assert len(features) == len(target), "特征和目标数据长度不匹配"
        assert len(features) > seq_length, f"数据长度({len(features)})必须大于序列长度({seq_length})"
    
    def __len__(self):
        return len(self.features) - self.seq_length
    
    def __getitem__(self, idx):
        x_sequence = self.features[idx:idx + self.seq_length]
        y_target = self.target[idx + self.seq_length]
        return torch.FloatTensor(x_sequence), torch.FloatTensor([y_target])

class PositionalEncoding(nn.Module):
    """位置编码层（Transformer核心组件）"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerPredictor(nn.Module):
    """Transformer预测模型"""
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        # 输入嵌入层
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self.d_model = d_model
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # 输入嵌入
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer编码
        transformer_out = self.transformer_encoder(x)
        
        # 取最后一个时间步的输出
        last_output = transformer_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # 全连接输出层
        output = self.fc(last_output)
        return output

class PowerPredictionSystem:
    """电力预测系统主类（修改为使用Transformer）"""
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
    def load_and_process_data(self):
        """加载和处理数据（与之前相同）"""
        print("正在加载数据...")
        
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        train_df['Date'] = pd.to_datetime(train_df['Date'])
        test_df['Date'] = pd.to_datetime(test_df['Date'])
        
        train_df = train_df.sort_values('Date').reset_index(drop=True)
        test_df = test_df.sort_values('Date').reset_index(drop=True)
        
        print(f"训练集大小: {len(train_df)}")
        print(f"测试集大小: {len(test_df)}")
        
        feature_columns = [col for col in train_df.columns if col != 'Date']
        target_column = 'Global_active_power'
        feature_columns = [col for col in feature_columns if col != target_column]
        
        print(f"特征列: {feature_columns}")
        
        train_features = train_df[feature_columns].values
        train_target = train_df[target_column].values
        
        test_features = test_df[feature_columns].values
        test_target = test_df[target_column].values
        
        train_features_scaled = self.scaler_features.fit_transform(train_features)
        test_features_scaled = self.scaler_features.transform(test_features)
        
        train_target_scaled = self.scaler_target.fit_transform(train_target.reshape(-1, 1)).flatten()
        test_target_scaled = self.scaler_target.transform(test_target.reshape(-1, 1)).flatten()
        
        self.train_data = {
            'features': train_features_scaled,
            'target': train_target_scaled,
            'raw_target': train_target,
            'dates': train_df['Date'].values
        }
        
        self.test_data = {
            'features': test_features_scaled,
            'target': test_target_scaled,
            'raw_target': test_target,
            'dates': test_df['Date'].values
        }
        
        self.input_size = len(feature_columns)
        print(f"输入特征维度: {self.input_size}")
        
    def create_dataset(self, data, seq_length):
        """创建数据集（与之前相同）"""
        features = data['features']
        target = data['target']
        dataset = PowerConsumptionDataset(features, target, seq_length)
        return dataset
    
    def train_model(self, seq_length=90, d_model=128, nhead=8, num_layers=3, 
                   epochs=10, batch_size=32, learning_rate=0.0005):
        """训练Transformer模型（修改以包含验证集）"""
        print(f"\n开始训练Transformer模型 (序列长度: {seq_length})")
        
        # 创建完整训练数据集
        full_train_dataset = self.create_dataset(self.train_data, seq_length)
        print(f"总训练样本数量: {len(full_train_dataset)}")
        
        # 划分训练集和验证集（90%训练，10%验证）
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        print(f"训练集样本数量: {len(train_dataset)}")
        print(f"验证集样本数量: {len(val_dataset)}")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建测试数据集和加载器用于跟踪测试损失
        test_dataset = self.create_dataset(self.test_data, seq_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建Transformer模型
        model = TransformerPredictor(
            input_size=self.input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_size=1
        ).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
        
        # 训练循环
        train_losses = []
        val_losses = []
        test_losses = []
        model.train()
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪防止爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # 测试阶段
            test_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            
            # 更新学习率调度器（基于验证损失）
            scheduler.step(avg_val_loss)
            
            # if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
        
        return model, {'train': train_losses, 'val': val_losses, 'test': test_losses}
    
    def predict_and_evaluate(self, model, seq_length, prediction_days):
        """预测并评估模型（与之前相同）"""
        model.eval()
        test_dataset = self.create_dataset(self.test_data, seq_length)
        actual_prediction_days = min(len(test_dataset), prediction_days)
        print(f"实际预测天数: {actual_prediction_days}")
        
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for i in range(actual_prediction_days):
                x_seq, y_true = test_dataset[i]
                x_seq = x_seq.unsqueeze(0).to(self.device)
                pred = model(x_seq)
                predictions.append(pred.cpu().numpy()[0, 0])
                true_values.append(y_true.numpy()[0])
        
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        predictions_rescaled = self.scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()
        true_values_rescaled = self.scaler_target.inverse_transform(true_values.reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(true_values_rescaled, predictions_rescaled)
        mae = mean_absolute_error(true_values_rescaled, predictions_rescaled)
        
        return predictions_rescaled, true_values_rescaled, mse, mae
    
    def run_experiments(self, num_experiments=5):
        """运行多次实验（减少实验次数以节省时间）"""
        print("开始进行多次实验...")
        
        results = {
            'short_term': {'mse': [], 'mae': []},
            'long_term': {'mse': [], 'mae': []}
        }
        
        best_models = {}
        loss_histories = {}
        
        for exp in range(num_experiments):
            print(f"\n=== 实验 {exp + 1}/{num_experiments} ===")
            set_seed(42 + exp)
            
            # 短期预测 (90天)
            print("训练短期预测模型...")
            # model_short, losses_short = self.train_model(seq_length=90, epochs=60)
            model_short, losses_short = self.train_model(
                seq_length=90,
                d_model=128,
                nhead=8,
                num_layers=3,
                epochs=10
            )
            pred_short, true_short, mse_short, mae_short = self.predict_and_evaluate(
                model_short, seq_length=90, prediction_days=90
            )
            
            results['short_term']['mse'].append(mse_short)
            results['short_term']['mae'].append(mae_short)
            
            if exp == 0:
                best_models['short_term'] = {
                    'model': model_short,
                    'predictions': pred_short,
                    'true_values': true_short
                }
                loss_histories['short_term'] = losses_short
            
            # 长期预测 (365天)
            print("训练长期预测模型...")
            # model_long, losses_long = self.train_model(seq_length=90, epochs=80)
            model_long, losses_long = self.train_model(
                seq_length=90,
                d_model=256,
                nhead=16,
                num_layers=4,
                epochs=40
            )
            pred_long, true_long, mse_long, mae_long = self.predict_and_evaluate(
                model_long, seq_length=90, prediction_days=365
            )
            
            results['long_term']['mse'].append(mse_long)
            results['long_term']['mae'].append(mae_long)
            
            if exp == 0:
                best_models['long_term'] = {
                    'model': model_long,
                    'predictions': pred_long,
                    'true_values': true_long
                }
                loss_histories['long_term'] = losses_long
            
            print(f"短期预测 - MSE: {mse_short:.4f}, MAE: {mae_short:.4f}")
            print(f"长期预测 - MSE: {mse_long:.4f}, MAE: {mae_long:.4f}")
        
        # 计算统计结果
        print("\n=== 实验结果汇总 ===")
        for term in ['short_term', 'long_term']:
            term_name = "短期" if term == "short_term" else "长期"
            mse_mean = np.mean(results[term]['mse'])
            mse_min, mse_max = np.min(results[term]['mse']), np.max(results[term]['mse'])
            mse_std = np.std(results[term]['mse'])
            mae_mean = np.mean(results[term]['mae'])
            mae_min, mae_max = np.min(results[term]['mae']), np.max(results[term]['mae'])
            mae_std = np.std(results[term]['mae'])
            
            print(f"\n{term_name}预测结果:")
            print(f"MSE: {mse_mean:.4f} ± {mse_std:.4f}", f"范围({mse_min:.4f}, {mse_max:.4f})")
            print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}", f"范围({mae_min:.4f}, {mae_max:.4f})")
            
        
        return results, best_models, loss_histories
    
    def visualize_results(self, best_models, loss_histories):
        """可视化结果（修改以包含验证和测试损失）"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 短期预测结果
        ax1 = axes[0, 0]
        pred_short = best_models['short_term']['predictions']
        true_short = best_models['short_term']['true_values']
        days_short = range(1, len(pred_short) + 1)
        
        ax1.plot(days_short, true_short, label='Ground Truth', color='blue', linewidth=2)
        ax1.plot(days_short, pred_short, label='Transformer Prediction', color='red', linewidth=2, alpha=0.8)
        ax1.set_title('Short-term Prediction (90 days)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Days', fontsize=12)
        ax1.set_ylabel('Global Active Power (kW)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 长期预测结果
        ax2 = axes[0, 1]
        pred_long = best_models['long_term']['predictions']
        true_long = best_models['long_term']['true_values']
        days_long = range(1, len(pred_long) + 1)
        
        ax2.plot(days_long, true_long, label='Ground Truth', color='blue', linewidth=2)
        ax2.plot(days_long, pred_long, label='Transformer Prediction', color='red', linewidth=2, alpha=0.8)
        ax2.set_title('Long-term Prediction (365 days)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Days', fontsize=12)
        ax2.set_ylabel('Global Active Power (kW)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 短期训练损失（包含验证和测试损失）
        ax3 = axes[1, 0]
        epochs = range(1, len(loss_histories['short_term']['train']) + 1)
        ax3.plot(epochs, loss_histories['short_term']['train'], color='green', linewidth=2, label='Train Loss')
        ax3.plot(epochs, loss_histories['short_term']['val'], color='orange', linewidth=2, label='Validation Loss')
        ax3.plot(epochs, loss_histories['short_term']['test'], color='red', linewidth=2, label='Test Loss')
        ax3.set_title('Short-term Model Training Loss', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss (MSE)', fontsize=12)
        ax3.set_ylim(0, 0.1)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 长期训练损失（包含验证和测试损失）
        ax4 = axes[1, 1]
        epochs = range(1, len(loss_histories['long_term']['train']) + 1)
        ax4.plot(epochs, loss_histories['long_term']['train'], color='green', linewidth=2, label='Train Loss')
        ax4.plot(epochs, loss_histories['long_term']['val'], color='orange', linewidth=2, label='Validation Loss')
        ax4.plot(epochs, loss_histories['long_term']['test'], color='red', linewidth=2, label='Test Loss')
        ax4.set_title('Long-term Model Training Loss', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Loss (MSE)', fontsize=12)
        ax4.set_ylim(0, 0.1)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/data/zcwang/temp/result/transformer_prediction_resultsdp.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 创建误差分析图
        self.plot_error_analysis(best_models)
    
    def plot_error_analysis(self, best_models):
        """绘制误差分析图（与之前相同）"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 短期预测误差分布
        pred_short = best_models['short_term']['predictions']
        true_short = best_models['short_term']['true_values']
        error_short = pred_short - true_short
        
        axes[0].hist(error_short, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(np.mean(error_short), color='red', linestyle='--', 
                       label=f'Mean Error: {np.mean(error_short):.2f}')
        axes[0].set_title('Short-term Prediction Error Distribution', fontweight='bold')
        axes[0].set_xlabel('Prediction Error (kW)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 长期预测误差分布
        pred_long = best_models['long_term']['predictions']
        true_long = best_models['long_term']['true_values']
        error_long = pred_long - true_long
        
        axes[1].hist(error_long, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1].axvline(np.mean(error_long), color='red', linestyle='--',
                       label=f'Mean Error: {np.mean(error_long):.2f}')
        axes[1].set_title('Long-term Prediction Error Distribution', fontweight='bold')
        axes[1].set_xlabel('Prediction Error (kW)')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/data/zcwang/temp/result/transformer_error_analysisdp.png', dpi=300, bbox_inches='tight')
        plt.show()

# 主执行函数
def main():
    """主函数"""
    # 创建预测系统实例
    system = PowerPredictionSystem(
        train_path='/data/zcwang/temp/data/train_processed_data.csv',
        test_path='/data/zcwang/temp/data/test_processed_data.csv'
    )
    
    # 加载和处理数据
    system.load_and_process_data()
    
    # 运行实验（减少实验次数）
    results, best_models, loss_histories = system.run_experiments(num_experiments=5)
    
    # 可视化结果
    system.visualize_results(best_models, loss_histories)
    
    print("\n实验完成！结果图片已保存为:")
    print("- transformer_prediction_results.png (预测结果对比图)")
    print("- transformer_error_analysis.png (误差分析图)")

if __name__ == "__main__":
    main()

# nohup  python /data/zcwang/temp/ml/Transformer.py  >/data/zcwang/temp/logs/transformer3.out 2>&1 &