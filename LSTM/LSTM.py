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
        
        print(f"训练集大小: {len(train_df)}")
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
        """
        创建PyTorch数据集
        
        这里不再手动创建序列，而是直接使用PowerConsumptionDataset
        让Dataset类自动处理滑动窗口逻辑
        """
        features = data['features']
        target = data['target']
        
        # 直接创建Dataset，让它内部处理序列生成
        dataset = PowerConsumptionDataset(features, target, seq_length)
        
        return dataset
    
    def train_model(self, seq_length=90, hidden_size=128, num_layers=2, 
                   epochs=100, batch_size=32, learning_rate=0.001):
        """训练LSTM模型"""
        print(f"\n开始训练模型 (序列长度: {seq_length})")
        
        # 创建数据集（使用优化后的方法）
        train_dataset = self.create_dataset(self.train_data, seq_length)
        print(f"训练样本数量: {len(train_dataset)}")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 创建模型
        model = LSTMPredictor(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1
        ).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # 训练循环
        train_losses = []
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
        
        return model, train_losses
    
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
    
    def run_experiments(self, num_experiments=5):
        """运行多次实验"""
        print("开始进行多次实验...")
        
        results = {
            'short_term': {'mse': [], 'mae': []},
            'long_term': {'mse': [], 'mae': []}
        }
        
        # 存储最佳模型和损失历史
        best_models = {}
        loss_histories = {}
        
        for exp in range(num_experiments):
            print(f"\n=== 实验 {exp + 1}/{num_experiments} ===")
            set_seed(42 + exp)  # 每次实验使用不同种子
            
            # 短期预测 (90天)
            print("训练短期预测模型...")
            model_short, losses_short = self.train_model(seq_length=90, epochs=50)
            pred_short, true_short, mse_short, mae_short = self.predict_and_evaluate(
                model_short, seq_length=90, prediction_days=90
            )
            
            results['short_term']['mse'].append(mse_short)
            results['short_term']['mae'].append(mae_short)
            
            if exp == 0:  # 保存第一次实验的模型用于可视化
                best_models['short_term'] = {
                    'model': model_short,
                    'predictions': pred_short,
                    'true_values': true_short
                }
                loss_histories['short_term'] = losses_short
            
            # 长期预测 (365天)
            print("训练长期预测模型...")
            model_long, losses_long = self.train_model(seq_length=90, epochs=80)
            pred_long, true_long, mse_long, mae_long = self.predict_and_evaluate(
                model_long, seq_length=90, prediction_days=365
            )
            
            results['long_term']['mse'].append(mse_long)
            results['long_term']['mae'].append(mae_long)
            
            if exp == 0:  # 保存第一次实验的模型用于可视化
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
            mse_std = np.std(results[term]['mse'])
            mae_mean = np.mean(results[term]['mae'])
            mae_std = np.std(results[term]['mae'])
            
            print(f"\n{term_name}预测结果:")
            print(f"MSE: {mse_mean:.4f} ± {mse_std:.4f}")
            print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}")
        
        return results, best_models, loss_histories
    
    def visualize_results(self, best_models, loss_histories):
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
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
        
        # 短期训练损失
        ax3 = axes[1, 0]
        ax3.plot(loss_histories['short_term'], color='green', linewidth=2)
        ax3.set_title('Short-term Model Training Loss', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss (MSE)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 长期训练损失
        ax4 = axes[1, 1]
        ax4.plot(loss_histories['long_term'], color='orange', linewidth=2)
        ax4.set_title('Long-term Model Training Loss', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Loss (MSE)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lstm_prediction_results.png', dpi=300, bbox_inches='tight')
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
        plt.savefig('lstm_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# 主执行函数
def main():
    """主函数"""
    # 创建预测系统实例
    system = PowerPredictionSystem(
        train_path='data/train_processed_data.csv',
        test_path='data/test_processed_data.csv'
    )
    
    # 加载和处理数据
    system.load_and_process_data()
    
    # 运行实验
    results, best_models, loss_histories = system.run_experiments(num_experiments=5)
    
    # 可视化结果
    system.visualize_results(best_models, loss_histories)
    
    print("\n实验完成！结果图片已保存为:")
    print("- lstm_prediction_results.png (预测结果对比图)")
    print("- lstm_error_analysis.png (误差分析图)")

if __name__ == "__main__":
    main()