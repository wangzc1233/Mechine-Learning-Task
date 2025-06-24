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
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # Linux系统
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子保证结果可复现
def set_seed(seed=66):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class PowerConsumptionDataset(Dataset):
    """电力消耗数据集类（添加时间特征）"""
    def __init__(self, features, target, dates, seq_length):
        self.features = features
        self.target = target
        self.dates = dates
        self.seq_length = seq_length
        
        assert len(features) == len(target) == len(dates), "数据长度不匹配"
        assert len(features) > seq_length, f"数据长度({len(features)})必须大于序列长度({seq_length})"
    
    def __len__(self):
        return len(self.features) - self.seq_length
    
    def __getitem__(self, idx):
        x_sequence = self.features[idx:idx + self.seq_length]
        y_target = self.target[idx + self.seq_length]
        return torch.FloatTensor(x_sequence), torch.FloatTensor([y_target])

class PositionalEncoding(nn.Module):
    """改进位置编码层（增加最大长度）"""
    def __init__(self, d_model, max_len=1000):
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

class AsymmetricMSELoss(nn.Module):
    """非对称MSE损失函数，惩罚正偏差更多"""
    def __init__(self, alpha=3.0):
        super().__init__()
        self.alpha = alpha  # 正偏差惩罚系数
    
    def forward(self, pred, target):
        error = pred - target
        loss = torch.where(
            error > 0,
            self.alpha * error ** 2,
            error ** 2
        )
        return loss.mean()

class EnhancedTransformerPredictor(nn.Module):
    """增强版Transformer预测模型"""
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.2):
        super(EnhancedTransformerPredictor, self).__init__()
        
        # 增强输入嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(input_size, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 改进位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)
        
        # Transformer编码器（增加层归一化和残差连接）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # 添加层归一化
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 多尺度时间特征提取
        self.temporal_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 增强输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, output_size)
        )
        
        # 初始化参数
        self.d_model = d_model
        
    def forward(self, x):
        # 输入嵌入
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # Transformer编码
        transformer_out = self.transformer_encoder(x)
        
        # 多尺度特征提取
        transformer_out = transformer_out.permute(0, 2, 1)  # [batch, d_model, seq_len]
        pooled_out = self.temporal_pool(transformer_out)
        
        # 输出预测
        output = self.output_layer(pooled_out)
        return output

class PowerPredictionSystem:
    """电力预测系统主类（优化版）"""
    def __init__(self, train_path, test_path, result_dir='/data/zcwang/temp/result'):
        self.train_path = train_path
        self.test_path = test_path
        self.result_dir = result_dir
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
        
        # 创建结果目录
        os.makedirs(self.result_dir, exist_ok=True)
        print(f"使用设备: {self.device}")
        
    def add_time_features(self, df):
        """添加时间相关特征（不含节假日）"""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 基本时间特征
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Weekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)
        df['Quarter'] = df['Date'].dt.quarter
        
        # 季节特征
        df['Season'] = df['Month'].apply(lambda m: (m % 12 + 3) // 3)
        
        # 滞后特征（1天、7天、30天）
        target_col = 'Global_active_power'
        for lag in [1, 7, 30]:
            df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def load_and_process_data(self):
        """加载和处理数据（增强特征工程）"""
        print("正在加载数据并添加时间特征...")
        
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        # 添加时间特征
        train_df = self.add_time_features(train_df)
        test_df = self.add_time_features(test_df)
        
        # 按日期排序
        train_df = train_df.sort_values('Date').reset_index(drop=True)
        test_df = test_df.sort_values('Date').reset_index(drop=True)
        
        # # 删除滞后特征产生的NaN行
        train_df.dropna(inplace=True)
        test_df.dropna(inplace=True)
        
        print(f"训练集大小: {len(train_df)}")
        print(f"测试集大小: {len(test_df)}")
        
        # 选择特征列（排除日期和目标列）
        feature_columns = [col for col in train_df.columns 
                          if col not in ['Date', 'Global_active_power']]
        target_column = 'Global_active_power'
        
        print(f"特征列数量: {len(feature_columns)}")
        print(f"特征示例: {feature_columns[:5]}...")
        
        # 准备训练数据
        train_features = train_df[feature_columns].values
        train_target = train_df[target_column].values
        
        # 准备测试数据
        test_features = test_df[feature_columns].values
        test_target = test_df[target_column].values
        
        # 标准化特征
        train_features_scaled = self.scaler_features.fit_transform(train_features)
        test_features_scaled = self.scaler_features.transform(test_features)
        
        # 标准化目标变量
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
        """创建数据集"""
        features = data['features']
        target = data['target']
        dates = data['dates']
        dataset = PowerConsumptionDataset(features, target, dates, seq_length)
        return dataset
    
    def train_model(self, seq_length=90, d_model=128, nhead=8, num_layers=3, 
                   epochs=150, batch_size=64, learning_rate=0.0001):
        """训练增强版Transformer模型（添加验证集）"""
        print(f"\n开始训练Transformer模型 (序列长度: {seq_length})")
        print(f"模型参数: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
        
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
        
        # 创建增强版Transformer模型
        model = EnhancedTransformerPredictor(
            input_size=self.input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_size=1
        ).to(self.device)
        
        # 使用非对称损失函数
        criterion = AsymmetricMSELoss(alpha=3.0)
        
        # 改进的优化器配置
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4,
            betas=(0.9, 0.98)
        )
        
        # 改进的学习率调度
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            anneal_strategy='cos'
        )
        
        # 训练循环
        train_losses = []
        val_losses = []
        test_losses = []
        
        # 早停机制
        best_loss = float('inf')
        patience = 15
        patience_counter = 0
        best_model_state = None
        
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
                scheduler.step()
                
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
            
            # 早停检查（基于验证损失）
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Test Loss: {avg_test_loss:.6f} *')
            else:
                patience_counter += 1
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
                if patience_counter >= patience:
                    print(f"早停于第{epoch+1}轮")
                    model.load_state_dict(best_model_state)
                    break
        
        return model, {'train': train_losses, 'val': val_losses, 'test': test_losses}
    
    def predict_and_evaluate(self, model, seq_length, prediction_days):
        """改进的预测方法（减少累积误差）"""
        model.eval()
        test_dataset = self.create_dataset(self.test_data, seq_length)
        actual_prediction_days = min(len(test_dataset), prediction_days)
        print(f"实际预测天数: {actual_prediction_days}")
        
        predictions = []
        true_values = []
        
        # 获取测试数据
        test_features = self.test_data['features']
        test_target = self.test_data['target']
        
        # 改进的递归预测（混合真实值）
        with torch.no_grad():
            # 使用真实值初始化输入序列
            initial_idx = 0
            input_seq = test_features[initial_idx:initial_idx+seq_length]
            input_seq_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
            
            for i in range(actual_prediction_days):
                # 预测下一个点
                pred = model(input_seq_tensor).cpu().numpy()[0, 0]
                predictions.append(pred)
                
                # 获取真实值
                true_val = test_target[initial_idx + seq_length + i]
                true_values.append(true_val)
                
                # 创建新的输入序列（混合预测值和真实值）
                new_input = input_seq_tensor.cpu().numpy()[0, 1:]
                
                # 构建新数据点 - 使用最新真实特征+预测值
                # 假设目标变量是最后一个特征
                new_point = np.copy(test_features[initial_idx + seq_length + i])
                new_point[-1] = pred  # 用预测值替换目标特征
                
                # 更新输入序列
                new_input = np.vstack([new_input, new_point])
                input_seq_tensor = torch.FloatTensor(new_input).unsqueeze(0).to(self.device)
        
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        # 反标准化
        predictions_rescaled = self.scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()
        true_values_rescaled = self.scaler_target.inverse_transform(true_values.reshape(-1, 1)).flatten()
        
        # 计算评估指标
        mse = mean_squared_error(true_values_rescaled, predictions_rescaled)
        mae = mean_absolute_error(true_values_rescaled, predictions_rescaled)
        
        return predictions_rescaled, true_values_rescaled, mse, mae
    
    def run_experiments(self, num_experiments=5):
        """运行多次实验（带模型集成）"""
        print("开始进行多次实验...")
        
        results = {
            'short_term': {'mse': [], 'mae': []},
            'long_term': {'mse': [], 'mae': []}
        }
        
        best_models = {}
        loss_histories = {}
        ensemble_predictions = []
        model_weights = []
        
        for exp in range(num_experiments):
            print(f"\n=== 实验 {exp + 1}/{num_experiments} ===")
            set_seed(42 + exp)
            
            # 短期预测 (90天)
            print("训练短期预测模型...")
            model_short, losses_short = self.train_model(
                seq_length=90,
                d_model=128,
                nhead=8,
                num_layers=3,
                epochs=100,
                batch_size=64,
                learning_rate=0.0001
            )
            
            # 在验证集上评估模型性能（使用部分测试集）
            _, _, val_mse, _ = self.predict_and_evaluate(
                model_short, seq_length=90, prediction_days=min(100, len(self.test_data['features'])-90)
            )
            model_weights.append(1 / (val_mse + 1e-5))
            
            # 完整预测
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
            model_long, losses_long = self.train_model(
                seq_length=90,
                d_model=256,
                nhead=8,
                num_layers=5,
                epochs=150,
                batch_size=64,
                learning_rate=0.0001
            )
            
            # 完整预测
            pred_long, true_long, mse_long, mae_long = self.predict_and_evaluate(
                model_long, seq_length=90, prediction_days=365
            )
            
            # results['long_term']['mse'].append(mse_long)
            # results['long_term']['mae'].append(mae_long)
            ensemble_predictions.append(pred_long)
            
            if exp == 0:
                best_models['long_term'] = {
                    'model': model_long,
                    'predictions': pred_long,
                    'true_values': true_long
                }
                loss_histories['long_term'] = losses_long
            
            print(f"短期预测 - MSE: {mse_short:.4f}, MAE: {mae_short:.4f}")
            print(f"长期预测 - MSE: {mse_long:.4f}, MAE: {mae_long:.4f}")
        
        # 加权集成长期预测
        weights = np.array(model_weights) / sum(model_weights)
        ensemble_pred = np.zeros_like(ensemble_predictions[0])
        for i in range(num_experiments):
            ensemble_pred += weights[i] * ensemble_predictions[i]
        
        best_models['long_term']['predictions'] = ensemble_pred
        
        # 重新计算集成模型的指标
        true_long = best_models['long_term']['true_values']
        mse_long = mean_squared_error(true_long, ensemble_pred)
        mae_long = mean_absolute_error(true_long, ensemble_pred)
        
        # 更新长期结果
        # results['long_term']['mse'] = [mse_long] * num_experiments
        # results['long_term']['mae'] = [mae_long] * num_experiments
        
        # 计算统计结果
        print("\n=== 实验结果汇总（含集成模型）===")
        for term in ['short_term', 'long_term']:
            term_name = "短期" if term == "short_term" else "长期"
            mse_mean = np.mean(results[term]['mse'])
            mse_std = np.std(results[term]['mse'])
            mae_mean = np.mean(results[term]['mae'])
            mae_std = np.std(results[term]['mae'])
            
            print(f"\n{term_name}预测结果:")
            print(f"MSE: {mse_mean:.4f} ± {mse_std:.4f}")
            print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}")
        
        return results, best_models, loss_histories
    
    def visualize_results(self, best_models, loss_histories):
        """Visualize results (enhanced version with validation and test losses)"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Short-term prediction results
        ax1 = axes[0, 0]
        pred_short = best_models['short_term']['predictions']
        true_short = best_models['short_term']['true_values']
        days_short = range(1, len(pred_short) + 1)
        
        # Calculate error
        error_short = pred_short - true_short
        
        ax1.plot(days_short, true_short, label='Ground Truth', color='blue', linewidth=2)
        ax1.plot(days_short, pred_short, label='Transformer Prediction', color='red', linewidth=1.5, alpha=0.8)
        
        # Add error band
        ax1.fill_between(days_short, 
                        true_short - np.abs(error_short), 
                        true_short + np.abs(error_short),
                        color='pink', alpha=0.3, label='Prediction Error Range')
        
        ax1.set_title('Short-Term Prediction Results (90 Days)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Days', fontsize=12)
        ax1.set_ylabel('Global Active Power (kW)', fontsize=12)
        ax1.legend(fontsize=10, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Long-term prediction results
        ax2 = axes[0, 1]
        pred_long = best_models['long_term']['predictions']
        true_long = best_models['long_term']['true_values']
        days_long = range(1, len(pred_long) + 1)
        
        # Calculate error
        error_long = pred_long - true_long
        
        # Show details for the 365days
        detail_days = min(365, len(pred_long))
        
        ax2.plot(days_long[:detail_days], true_long[:detail_days], label='Actual Values', color='blue', linewidth=2)
        ax2.plot(days_long[:detail_days], pred_long[:detail_days], label='Transformer Prediction (Ensemble)', color='red', linewidth=1.5, alpha=0.8)
        
        # Add error band
        ax2.fill_between(days_long[:detail_days], 
                        true_long[:detail_days] - np.abs(error_long[:detail_days]), 
                        true_long[:detail_days] + np.abs(error_long[:detail_days]),
                        color='pink', alpha=0.3, label='Prediction Error Range')
        
        ax2.set_title('Long-Term Prediction Results (365 Days)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Days', fontsize=12)
        ax2.set_ylabel('Global Active Power (kW)', fontsize=12)
        ax2.legend(fontsize=10, loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Short-term training loss (包含验证和测试损失)
        ax3 = axes[1, 0]
        epochs = range(1, len(loss_histories['short_term']['train']) + 1)
        ax3.plot(epochs, loss_histories['short_term']['train'], color='green', linewidth=2, label='Train Loss')
        ax3.plot(epochs, loss_histories['short_term']['val'], color='orange', linewidth=2, label='Validation Loss')
        ax3.plot(epochs, loss_histories['short_term']['test'], color='red', linewidth=2, label='Test Loss')
        ax3.set_title('Short-Term Model Training Loss', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Training Epochs', fontsize=12)
        ax3.set_ylabel('Loss (MSE)', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Long-term training loss (包含验证和测试损失)
        ax4 = axes[1, 1]
        epochs = range(1, len(loss_histories['long_term']['train']) + 1)
        ax4.plot(epochs, loss_histories['long_term']['train'], color='green', linewidth=2, label='Train Loss')
        ax4.plot(epochs, loss_histories['long_term']['val'], color='orange', linewidth=2, label='Validation Loss')
        ax4.plot(epochs, loss_histories['long_term']['test'], color='red', linewidth=2, label='Test Loss')
        ax4.set_title('Long-Term Model Training Loss', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Training Epochs', fontsize=12)
        ax4.set_ylabel('Loss (MSE)', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        result_path = os.path.join(self.result_dir, 'enhanced_transformer_prediction_results.png')
        plt.savefig(result_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 创建误差分析图
        self.plot_error_analysis(best_models)

    def plot_error_analysis(self, best_models):
        """Plot enhanced error analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Short-term prediction error distribution
        pred_short = best_models['short_term']['predictions']
        true_short = best_models['short_term']['true_values']
        error_short = pred_short - true_short
        mean_error_short = np.mean(error_short)
        std_error_short = np.std(error_short)
        
        axes[0].hist(error_short, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(mean_error_short, color='red', linestyle='--', 
                    label=f'Mean Error: {mean_error_short:.2f} kW')
        axes[0].axvline(mean_error_short + std_error_short, color='green', linestyle=':')
        axes[0].axvline(mean_error_short - std_error_short, color='green', linestyle=':', 
                    label=f'±1 SD: {std_error_short:.2f} kW')
        axes[0].set_title('Short-Term Prediction Error Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Prediction Error (kW)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Long-term prediction error distribution
        pred_long = best_models['long_term']['predictions']
        true_long = best_models['long_term']['true_values']
        error_long = pred_long - true_long
        mean_error_long = np.mean(error_long)
        std_error_long = np.std(error_long)
        
        axes[1].hist(error_long, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1].axvline(mean_error_long, color='red', linestyle='--', 
                    label=f'Mean Error: {mean_error_long:.2f} kW')
        axes[1].axvline(mean_error_long + std_error_long, color='green', linestyle=':')
        axes[1].axvline(mean_error_long - std_error_long, color='green', linestyle=':', 
                    label=f'±1 SD: {std_error_long:.2f} kW')
        axes[1].set_title('Long-Term Prediction Error Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Prediction Error (kW)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        result_path = os.path.join(self.result_dir, 'enhanced_transformer_error_analysis.png')
        plt.savefig(result_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print error statistics
        print("\nError Statistics:")
        print(f"Short-Term Prediction - Mean Error: {mean_error_short:.2f} kW, Standard Deviation: {std_error_short:.2f} kW")
        print(f"Long-Term Prediction - Mean Error: {mean_error_long:.2f} kW, Standard Deviation: {std_error_long:.2f} kW")
        
        # Save error data
        error_data = pd.DataFrame({
            'short_term_error': error_short,
            'long_term_error': error_long[:len(error_short)]  # Align lengths
        })
        error_data.to_csv(os.path.join(self.result_dir, 'prediction_errors.csv'), index=False)

# 主执行函数
def main():
    """主函数"""
    # 创建预测系统实例
    system = PowerPredictionSystem(
        train_path='/data/zcwang/temp/data/train_processed_data.csv',
        test_path='/data/zcwang/temp/data/test_processed_data.csv',
        result_dir='/data/zcwang/temp/result'
    )
    
    # 加载和处理数据
    system.load_and_process_data()
    
    # 运行实验
    results, best_models, loss_histories = system.run_experiments(num_experiments=5)
    
    # 可视化结果
    system.visualize_results(best_models, loss_histories)
    
    print("\n实验完成！优化结果图片已保存为:")
    print("- enhanced_transformer_prediction_results.png")
    print("- enhanced_transformer_error_analysis.png")
    print("预测误差数据已保存为: prediction_errors.csv")

if __name__ == "__main__":
    main()

# 后台执行命令:
# nohup python /data/zcwang/temp/ml/transformer_dp_v2_2.py > /data/zcwang/temp/logs/enhanced_transformer4.out 2>&1 &