import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# =====================全局参数设置=====================
# 模型参数
MODEL_PARAMS = {
    'seq_length': 50,           # 序列长度
    'hidden_size': 128,         # LSTM隐藏层大小
    'num_layers': 2,            # LSTM层数
    'dropout': 0.2,             # Dropout比例
    'learning_rate': 0.005,     # 学习率
    'batch_size': 32,           # 批次大小
    'epochs_short': 100,         # 短期预测训练轮数
    'epochs_long': 100,          # 长期预测训练轮数
}

# 训练参数
TRAIN_PARAMS = {
    'val_split_ratio': 0.1,     # 验证集比例（10%用于验证）
    'num_experiments': 5,       # 实验次数
    'random_seed': 42,          # 随机种子
    'prediction_days_short': 90, # 短期预测天数
    'prediction_days_long': 365, # 长期预测天数
}

# 文件路径
DATA_PATHS = {
    'train_path': 'data/train_processed_data.csv',
    'test_path': 'data/test_processed_data.csv',
}
# ====================================================

# 设置随机种子保证结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class PowerConsumptionDataset(Dataset):
    """
    电力消耗数据集类
    
    设计思路：
    1. 时间序列预测需要"滑动窗口"机制
    2. 每个样本包含：过去seq_length天的特征 -> 下一天的功率值
    3. 自动处理数据的时序性和连续性
    """
    def __init__(self, features, target, seq_length):
        """
        参数:
        - features: 特征数据 (n_samples, n_features)
        - target: 目标变量 (n_samples,)
        - seq_length: 序列长度，即用过去多少天预测下一天
        """
        self.features = features  # 所有特征数据
        self.target = target      # 目标变量（Global_active_power）
        self.seq_length = seq_length
        
        # 验证数据长度
        assert len(features) == len(target), "特征和目标数据长度不匹配"
        assert len(features) > seq_length, f"数据长度({len(features)})必须大于序列长度({seq_length})"
    
    def __len__(self):
        """
        返回数据集大小
        
        为什么是 len(self.features) - self.seq_length？
        - 如果有100天数据，序列长度为90天
        - 第1个样本：用第1-90天预测第91天
        - 第2个样本：用第2-91天预测第92天
        - ...
        - 最后一个样本：用第10-99天预测第100天
        - 总共可以生成 100-90 = 10 个样本
        """
        return len(self.features) - self.seq_length
    
    def __getitem__(self, idx):
        """
        获取第idx个样本
        
        返回格式：
        - x_sequence: (seq_length, n_features) - 过去seq_length天的所有特征
        - y_target: (1,) - 下一天的目标功率值
        """
        # 获取输入序列：从idx开始的seq_length个时间步
        x_sequence = self.features[idx:idx + self.seq_length]
        
        # 获取目标值：序列结束后的下一个时间步的功率值
        y_target = self.target[idx + self.seq_length]
        
        # 转换为PyTorch张量
        return torch.FloatTensor(x_sequence), torch.FloatTensor([y_target])

class LSTMPredictor(nn.Module):
    """LSTM预测模型"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        # Dropout
        lstm_out = self.dropout(lstm_out)
        # 全连接层
        output = self.fc(lstm_out)
        return output

class PowerPredictionSystem:
    """电力预测系统主类"""
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
    def load_and_process_data(self):
        """加载和处理数据"""
        print("正在加载数据...")
        
        # 加载数据
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        # 转换日期列
        train_df['Date'] = pd.to_datetime(train_df['Date'])
        test_df['Date'] = pd.to_datetime(test_df['Date'])
        
        # 按日期排序
        train_df = train_df.sort_values('Date').reset_index(drop=True)
        test_df = test_df.sort_values('Date').reset_index(drop=True)
        
        print(f"原始训练集大小: {len(train_df)}")
        print(f"测试集大小: {len(test_df)}")
        
        # 选择特征列（除了日期）
        feature_columns = [col for col in train_df.columns if col != 'Date']
        target_column = 'Global_active_power'
        
        # 移除目标列从特征中
        feature_columns = [col for col in feature_columns if col != target_column]
        
        print(f"特征列: {feature_columns}")
        
        # 准备训练数据
        train_features = train_df[feature_columns].values
        train_target = train_df[target_column].values
        
        # =====================新增：训练集验证集划分=====================
        # 计算划分点（90%训练，10%验证）
        val_split_ratio = TRAIN_PARAMS['val_split_ratio']
        split_idx = int(len(train_features) * (1 - val_split_ratio))
        
        # 按时间顺序划分（保持时序性）
        train_features_split = train_features[:split_idx]
        train_target_split = train_target[:split_idx]
        val_features_split = train_features[split_idx:]
        val_target_split = train_target[split_idx:]
        
        print(f"训练集大小: {len(train_features_split)}")
        print(f"验证集大小: {len(val_features_split)}")
        # =============================================================
        
        # 准备测试数据
        test_features = test_df[feature_columns].values
        test_target = test_df[target_column].values
        
        # 标准化特征（基于训练集拟合）
        train_features_scaled = self.scaler_features.fit_transform(train_features_split)
        val_features_scaled = self.scaler_features.transform(val_features_split)
        test_features_scaled = self.scaler_features.transform(test_features)
        
        # 标准化目标变量（基于训练集拟合）
        train_target_scaled = self.scaler_target.fit_transform(train_target_split.reshape(-1, 1)).flatten()
        val_target_scaled = self.scaler_target.transform(val_target_split.reshape(-1, 1)).flatten()
        test_target_scaled = self.scaler_target.transform(test_target.reshape(-1, 1)).flatten()
        
        # 存储训练数据
        self.train_data = {
            'features': train_features_scaled,
            'target': train_target_scaled,
            'raw_target': train_target_split,
            'dates': train_df['Date'].values[:split_idx]
        }
        
        # 存储验证数据
        self.val_data = {
            'features': val_features_scaled,
            'target': val_target_scaled,
            'raw_target': val_target_split,
            'dates': train_df['Date'].values[split_idx:]
        }
        
        # 存储测试数据
        self.test_data = {
            'features': test_features_scaled,
            'target': test_target_scaled,
            'raw_target': test_target,
            'dates': test_df['Date'].values
        }
        
        self.input_size = len(feature_columns)
        print(f"输入特征维度: {self.input_size}")
        
    def create_dataset(self, data, seq_length):
        """
        创建PyTorch数据集
        """
        features = data['features']
        target = data['target']
        
        # 直接创建Dataset，让它内部处理序列生成
        dataset = PowerConsumptionDataset(features, target, seq_length)
        
        return dataset
    
    def validate_model(self, model, val_loader, criterion):
        """验证模型性能"""
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train_model(self, seq_length=None, hidden_size=None, num_layers=None, 
                   epochs=None, batch_size=None, learning_rate=None):
        """训练LSTM模型（使用全局参数）"""
        # 使用全局参数作为默认值
        seq_length = seq_length or MODEL_PARAMS['seq_length']
        hidden_size = hidden_size or MODEL_PARAMS['hidden_size']
        num_layers = num_layers or MODEL_PARAMS['num_layers']
        epochs = epochs or MODEL_PARAMS['epochs_short']
        batch_size = batch_size or MODEL_PARAMS['batch_size']
        learning_rate = learning_rate or MODEL_PARAMS['learning_rate']
        dropout = MODEL_PARAMS['dropout']
        
        print(f"\n开始训练模型 (序列长度: {seq_length}, 隐藏层: {hidden_size}, 训练轮数: {epochs})")
        
        # 创建训练和验证数据集
        train_dataset = self.create_dataset(self.train_data, seq_length)
        val_dataset = self.create_dataset(self.val_data, seq_length)
        
        print(f"训练样本数量: {len(train_dataset)}")
        print(f"验证样本数量: {len(val_dataset)}")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        model = LSTMPredictor(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            dropout=dropout
        ).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # 训练循环
        train_losses = []
        val_losses = []
        model.train()
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            epoch_train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            avg_val_loss = self.validate_model(model, val_loader, criterion)
            val_losses.append(avg_val_loss)
            
            # 学习率调整
            scheduler.step(avg_val_loss)
            
            # 每10个epoch输出一次结果
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        return model, train_losses, val_losses
    
    def predict_and_evaluate(self, model, seq_length, prediction_days):
        """预测并评估模型"""
        model.eval()
        
        # 创建测试数据集
        test_dataset = self.create_dataset(self.test_data, seq_length)
        
        # 限制预测天数
        actual_prediction_days = min(len(test_dataset), prediction_days)
        print(f"实际预测天数: {actual_prediction_days}")
        
        # 进行预测
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for i in range(actual_prediction_days):
                x_seq, y_true = test_dataset[i]
                x_seq = x_seq.unsqueeze(0).to(self.device)  # 添加batch维度
                
                pred = model(x_seq)
                predictions.append(pred.cpu().numpy()[0, 0])
                true_values.append(y_true.numpy()[0])
        
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        # 反标准化
        predictions_rescaled = self.scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()
        true_values_rescaled = self.scaler_target.inverse_transform(true_values.reshape(-1, 1)).flatten()
        
        # 计算评估指标
        mse = mean_squared_error(true_values_rescaled, predictions_rescaled)
        mae = mean_absolute_error(true_values_rescaled, predictions_rescaled)
        
        return predictions_rescaled, true_values_rescaled, mse, mae
    
    def run_experiments(self, num_experiments=None):
        """运行多次实验"""
        num_experiments = num_experiments or TRAIN_PARAMS['num_experiments']
        print(f"开始进行{num_experiments}次实验...")
        
        results = {
            'short_term': {'mse': [], 'mae': []},
            'long_term': {'mse': [], 'mae': []}
        }
        
        # 存储最佳模型和损失历史
        best_models = {}
        loss_histories = {}
        
        for exp in range(num_experiments):
            print(f"\n=== 实验 {exp + 1}/{num_experiments} ===")
            set_seed(TRAIN_PARAMS['random_seed'] + exp)  # 每次实验使用不同种子
            
            # 短期预测
            print("训练短期预测模型...")
            model_short, train_losses_short, val_losses_short = self.train_model(
                epochs=MODEL_PARAMS['epochs_short']
            )
            pred_short, true_short, mse_short, mae_short = self.predict_and_evaluate(
                model_short, MODEL_PARAMS['seq_length'], TRAIN_PARAMS['prediction_days_short']
            )
            
            results['short_term']['mse'].append(mse_short)
            results['short_term']['mae'].append(mae_short)
            
            if exp == 0:  # 保存第一次实验的模型用于可视化
                best_models['short_term'] = {
                    'model': model_short,
                    'predictions': pred_short,
                    'true_values': true_short
                }
                loss_histories['short_term'] = {
                    'train_losses': train_losses_short,
                    'val_losses': val_losses_short
                }
            
            # 长期预测
            print("训练长期预测模型...")
            model_long, train_losses_long, val_losses_long = self.train_model(
                epochs=MODEL_PARAMS['epochs_long']
            )
            pred_long, true_long, mse_long, mae_long = self.predict_and_evaluate(
                model_long, MODEL_PARAMS['seq_length'], TRAIN_PARAMS['prediction_days_long']
            )
            
            results['long_term']['mse'].append(mse_long)
            results['long_term']['mae'].append(mae_long)
            
            if exp == 0:  # 保存第一次实验的模型用于可视化
                best_models['long_term'] = {
                    'model': model_long,
                    'predictions': pred_long,
                    'true_values': true_long
                }
                loss_histories['long_term'] = {
                    'train_losses': train_losses_long,
                    'val_losses': val_losses_long
                }
            
            print(f"短期预测 - MSE: {mse_short:.4f}, MAE: {mae_short:.4f}")
            print(f"长期预测 - MSE: {mse_long:.4f}, MAE: {mae_long:.4f}")
        
        # 计算统计结果
        print("\n=== 实验结果汇总 ===")
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
        """可视化结果（包含验证损失）"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 修改字体设置
        plt.rcParams['axes.unicode_minus'] = False
        
        
        # 短期预测结果
        ax1 = axes[0, 0]
        pred_short = best_models['short_term']['predictions']
        true_short = best_models['short_term']['true_values']
        days_short = range(1, len(pred_short) + 1)
        
        ax1.plot(days_short, true_short, label='Ground Truth', color='blue', linewidth=2)
        ax1.plot(days_short, pred_short, label='LSTM Prediction', color='red', linewidth=2, alpha=0.8)
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
        ax2.plot(days_long, pred_long, label='LSTM Prediction', color='red', linewidth=2, alpha=0.8)
        ax2.set_title('Long-term Prediction (365 days)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Days', fontsize=12)
        ax2.set_ylabel('Global Active Power (kW)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 参数设置可视化
        ax3 = axes[0, 2]
        ax3.axis('off')
        param_text = f"""
        Model Configuration:

        Sequence Length: {MODEL_PARAMS['seq_length']}
        Hidden Size: {MODEL_PARAMS['hidden_size']}
        LSTM Layers: {MODEL_PARAMS['num_layers']}
        Dropout: {MODEL_PARAMS['dropout']}
        Learning Rate: {MODEL_PARAMS['learning_rate']}
        Batch Size: {MODEL_PARAMS['batch_size']}

        Short-term Epochs: {MODEL_PARAMS['epochs_short']}
        Long-term Epochs: {MODEL_PARAMS['epochs_long']}

        Validation Split: {TRAIN_PARAMS['val_split_ratio']*100}%
        """
        ax3.text(0.05, 0.95, param_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax3.set_title('Configuration', fontsize=14, fontweight='bold')
        
        # 短期训练损失（包含验证损失）
        ax4 = axes[1, 0]
        train_losses_short = loss_histories['short_term']['train_losses']
        val_losses_short = loss_histories['short_term']['val_losses']
        
        ax4.plot(train_losses_short, label='Train Loss', color='green', linewidth=2)
        ax4.plot(val_losses_short, label='Val Loss', color='orange', linewidth=2)
        ax4.set_title('Short-term Model Training & Validation Loss', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Loss (MSE)', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 长期训练损失（包含验证损失）
        ax5 = axes[1, 1]
        train_losses_long = loss_histories['long_term']['train_losses']
        val_losses_long = loss_histories['long_term']['val_losses']
        
        ax5.plot(train_losses_long, label='Train Loss', color='green', linewidth=2)
        ax5.plot(val_losses_long, label='Val Loss', color='orange', linewidth=2)
        ax5.set_title('Long-term Model Training & Validation Loss', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('Loss (MSE)', fontsize=12)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # 损失对比（训练vs验证）
        ax6 = axes[1, 2]
        final_train_short = train_losses_short[-1]
        final_val_short = val_losses_short[-1]
        final_train_long = train_losses_long[-1]
        final_val_long = val_losses_long[-1]
        
        categories = ['Short-term\nTrain', 'Short-term\nVal', 'Long-term\nTrain', 'Long-term\nVal']
        values = [final_train_short, final_val_short, final_train_long, final_val_long]
        colors = ['green', 'orange', 'green', 'orange']
        
        bars = ax6.bar(categories, values, color=colors, alpha=0.7)
        ax6.set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Loss (MSE)', fontsize=12)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值标签
        for bar, value in zip(bars, values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('lstm_prediction_results_improved.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 创建误差分析图
        self.plot_error_analysis(best_models)
    
    def plot_error_analysis(self, best_models):
        """绘制误差分析图"""
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
        plt.savefig('lstm_error_analysis_improved.png', dpi=300, bbox_inches='tight')
        plt.show()

# 主执行函数
def main():
    """主函数"""
    print("=== LSTM电力预测系统 - 改进版 ===")
    print("全局参数配置:")
    print(f"模型参数: {MODEL_PARAMS}")
    print(f"训练参数: {TRAIN_PARAMS}")
    print(f"数据路径: {DATA_PATHS}")
    
    # 创建预测系统实例
    system = PowerPredictionSystem(
        train_path=DATA_PATHS['train_path'],
        test_path=DATA_PATHS['test_path']
    )
    
    # 加载和处理数据
    system.load_and_process_data()
    
    # 运行实验
    results, best_models, loss_histories = system.run_experiments()
    
    # 可视化结果
    system.visualize_results(best_models, loss_histories)
    
    print("\n实验完成！结果图片已保存为:")
    print("- lstm_prediction_results_improved.png (预测结果对比图)")
    print("- lstm_error_analysis_improved.png (误差分析图)")

if __name__ == "__main__":
    main()