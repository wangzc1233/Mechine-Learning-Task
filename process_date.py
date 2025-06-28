import pandas as pd
from datetime import datetime
from collections import defaultdict
import pdb
def preprocess_csv(input_path, output_path):
    """
    使用基础循环实现的预处理：
    1. 按天合并数据
    2. 计算各指标总和/平均值
    3. 计算剩余能耗
    4. 保存为CSV
    """
    try:
        # 1. 加载数据
        print(f"加载数据: {input_path}")
        df = pd.read_csv(input_path)
        
        # 2. 校验必要列
        required_cols = {'DateTime', 'Global_active_power', 'Sub_metering_1',
                        'Sub_metering_2', 'Sub_metering_3'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"缺少必要列: {missing}")

        # 3. 准备按日聚合的数据结构
        daily_data = defaultdict(lambda: {
            'Global_active_power': 0.0,
            'Global_reactive_power': 0.0,
            'Voltage': [],
            'Global_intensity': [],
            'Sub_metering_1': 0.0,
            'Sub_metering_2': 0.0,
            'Sub_metering_3': 0.0,
            'RR': None,
            'NBJRR1': None,
            'NBJRR5': None,
            'NBJRR10': None,
            'NBJBROU': None,
            'count': 0
        })

        # 4. 遍历每一行数据进行聚合
        print("正在聚合数据...")
        for _, row in df.iterrows():
            try:
                date = datetime.strptime(row['DateTime'], '%Y-%m-%d %H:%M:%S').date()
                date_str = date.isoformat()
                
                # 累加数值型数据
                daily = daily_data[date_str]
                daily['Global_active_power'] += float(row['Global_active_power'])
                daily['Global_reactive_power'] += float(row['Global_reactive_power'])
                daily['Sub_metering_1'] += float(row['Sub_metering_1'])
                daily['Sub_metering_2'] += float(row['Sub_metering_2'])
                daily['Sub_metering_3'] += float(row['Sub_metering_3'])
                
                # 收集需要平均的数值
                daily['Voltage'].append(float(row['Voltage']))
                daily['Global_intensity'].append(float(row['Global_intensity']))
                
                # 保留第一个非空天气数据
                if daily['RR'] is None and pd.notna(row['RR']):
                    daily['RR'] = float(row['RR'])

                if daily['NBJRR1'] is None and pd.notna(row['NBJRR1']):
                    daily['NBJRR1'] = float(row['NBJRR1'])

                if daily['NBJRR5'] is None and pd.notna(row['NBJRR5']):
                    daily['NBJRR5'] = float(row['NBJRR5'])

                if daily['NBJRR10'] is None and pd.notna(row['NBJRR10']):
                    daily['NBJRR10'] = float(row['NBJRR10'])

                if daily['NBJBROU'] is None and pd.notna(row['NBJBROU']):
                    daily['NBJBROU'] = float(row['NBJBROU'])

                daily['count'] += 1

            except Exception as e:
                print(f"跳过无效行: {row} | 错误: {str(e)}")
                continue

        # 5. 计算最终结果
        print("计算统计量...")
        result = []
        for date_str, data in daily_data.items():
            if data['count'] == 0:
                continue
                
            # 计算平均值
            avg_voltage = round(sum(data['Voltage']) / len(data['Voltage']), 3) if data['Voltage'] else 0.0
            avg_intensity = round(sum(data['Global_intensity']) / len(data['Global_intensity']), 3) if data['Global_intensity'] else 0.0
            
            # 计算剩余能耗
            remainder = round((data['Global_active_power'] * 1000 / 60 - 
                            data['Sub_metering_1'] - 
                            data['Sub_metering_2'] - 
                            data['Sub_metering_3']), 3)
            
            result.append({
                'Date': date_str,
                'Global_active_power': round(data['Global_active_power'], 3),
                'Global_reactive_power': round(data['Global_reactive_power'], 3),
                'Voltage': avg_voltage,
                'Global_intensity': avg_intensity,
                'Sub_metering_1': round(data['Sub_metering_1'], 3),
                'Sub_metering_2': round(data['Sub_metering_2'], 3),
                'Sub_metering_3': round(data['Sub_metering_3'], 3),
                'RR': data['RR'],
                'NBJRR1': data['NBJRR1'],
                'NBJRR5': data['NBJRR5'],
                'NBJRR10': data['NBJRR10'],
                'NBJBROU': data['NBJBROU'],
                'Sub_metering_remainder': remainder
            })

        # 6. 转换为DataFrame并保存
        result_df = pd.DataFrame(result)
        result_df.to_csv(output_path, index=False , float_format='%.3f')
        
        print(f"处理完成！原始记录: {len(df)}条 → 聚合天数: {len(result)}天")
        return True
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return False
    
if __name__ == "__main__":
    # 输入输出文件配置
    input_csv = "data/train.csv"         # 替换为您的原始数据路径
    output_csv = "data/train_processed_data.csv"  # 处理后的数据保存路径
    
    # 执行预处理
    success = preprocess_csv(input_csv, output_csv)
    
    if success:
        print(f"数据已成功处理并保存到 {output_csv}")
        print("可以开始进行LSTM模型训练了！")
    else:
        print("数据处理失败，请检查错误信息")