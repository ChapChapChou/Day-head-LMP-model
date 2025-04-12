import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hybrid_prediction as hp
import pywt

# 加载处理好的数据
def load_processed_data(file_path='process_data.csv'):
    """
    加载预处理好的数据
    """
    try:
        # 尝试加载数据，并将日期时间列作为索引
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"数据加载成功，共 {len(data)} 行，列名: {data.columns.tolist()}")
        return data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

# 分离高低频数据演示
def demonstrate_wavelet_decomposition(data, target_col, wavelet='db4', level=2):
    """
    展示如何使用小波变换将数据分解为高频和低频部分
    """
    # 获取目标列的数据
    y = data[target_col].values
    
    # 使用小波变换分解数据
    low_freq, high_freq = hp.wavelet_decompose(y, wavelet=wavelet, level=level)
    
    # 绘制原始数据和分解后的高低频部分
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(y, label='原始价格数据', color='blue')
    plt.title('原始LMP价格数据')
    plt.xlabel('时间')
    plt.ylabel('价格 ($/MWh)')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(low_freq, label='低频部分', color='green')
    plt.title('低频部分 (PSO-LSSVM预测)')
    plt.xlabel('时间')
    plt.ylabel('幅度')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(high_freq, label='高频部分', color='red')
    plt.title('高频部分 (ARIMA预测)')
    plt.xlabel('时间')
    plt.ylabel('幅度')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return low_freq, high_freq

# 执行混合预测并评估结果
def run_hybrid_prediction(data, target_col='total_lmp_da_log_norm', 
                         feature_cols=None,
                         test_size=0.2, 
                         wavelet='db4', 
                         level=2,
                         arima_order=(5,1,0)):
    """
    运行混合预测模型并评估结果
    """
    if feature_cols is None:
        # 默认使用标准化后的负载和时间特征作为输入
        feature_cols = ['zone_load_log_norm', 'hour', 'day_of_week', 'month', 'is_weekend']
    
    print("正在运行混合预测模型...")
    print(f"目标变量: {target_col}")
    print(f"特征变量: {feature_cols}")
    print(f"测试集比例: {test_size}")
    print(f"小波函数: {wavelet}，分解级别: {level}")
    print(f"ARIMA模型参数: {arima_order}")
    
    # 确保数据集的索引是datetime类型
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # 运行混合预测
    results = hp.hybrid_price_prediction(
        data=data,
        target_col=target_col,
        feature_cols=feature_cols,
        test_size=test_size,
        wavelet=wavelet,
        level=level,
        arima_order=arima_order,
        plot_results=True
    )
    
    # 输出评估指标
    print("\n混合预测模型总体评估:")
    print(f"均方误差 (MSE): {results['metrics']['mse']:.4f}")
    print(f"均方根误差 (RMSE): {results['metrics']['rmse']:.4f}")
    print(f"平均绝对误差 (MAE): {results['metrics']['mae']:.4f}")
    print(f"决定系数 (R²): {results['metrics']['r2']:.4f}")
    
    return results

if __name__ == "__main__":
    # 1. 加载处理好的数据
    data = load_processed_data('process_data.csv')
    
    if data is not None:
        # 2. 演示小波分解
        print("\n演示小波分解将数据分为高频和低频部分...")
        low_freq, high_freq = demonstrate_wavelet_decomposition(
            data, 
            target_col='total_lmp_da_log_norm'
        )
        
        # 3. 运行混合预测模型
        print("\n开始运行混合预测模型...")
        results = run_hybrid_prediction(data)
        
        # 4. 可视化预测结果（在hybrid_prediction.py中已经实现）
        print("\n混合预测完成！请查看可视化结果。")
