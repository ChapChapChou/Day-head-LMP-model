import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

# PSO-LSSVM Implementation
class LSSVM:
    def __init__(self, gamma=1.0, sigma=1.0):
        self.gamma = gamma  # regularization parameter
        self.sigma = sigma  # RBF kernel parameter
        self.alpha = None
        self.b = None
        self.X_train = None
        
    def rbf_kernel(self, x1, x2):
        """Radial Basis Function Kernel"""
        return np.exp(-np.sum((x1 - x2) ** 2) / (2 * self.sigma ** 2))
    
    def compute_kernel_matrix(self, X):
        """Compute the kernel matrix for the training data"""
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.rbf_kernel(X[i], X[j])
        return K
    
    def fit(self, X, y):
        """Train the LS-SVM model"""
        self.X_train = X
        n_samples = X.shape[0]
        
        # Compute the kernel matrix
        K = self.compute_kernel_matrix(X)
        
        # Solve the linear system
        J = np.block([
            [0, np.ones((1, n_samples))],
            [np.ones((n_samples, 1)), K + np.eye(n_samples) / self.gamma]
        ])
        
        # Target vector
        target = np.block([0, y])
        
        # Solve the system for alpha and b
        solution = np.linalg.solve(J, target)
        self.b = solution[0]
        self.alpha = solution[1:]
        
        return self
    
    def predict(self, X):
        """Make predictions for new data"""
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        
        for i in range(n_samples):
            y_pred[i] = self.b
            for j in range(len(self.X_train)):
                y_pred[i] += self.alpha[j] * self.rbf_kernel(X[i], self.X_train[j])
        
        return y_pred
    
class PSO:
    def __init__(self, num_particles=20, max_iter=100, c1=2.0, c2=2.0, w=0.7):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.c1 = c1  # cognitive weight
        self.c2 = c2  # social weight
        self.w = w    # inertia weight
        
    def optimize(self, X_train, y_train, X_val, y_val, bounds=((0.01, 100), (0.01, 100))):
        """
        Use PSO to find optimal gamma and sigma parameters for LSSVM
        
        Parameters:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        bounds: Range for gamma and sigma parameters
        
        Returns:
        best_gamma, best_sigma: Best parameters found
        """
        # Initialize particles
        particles = []
        for _ in range(self.num_particles):
            # Random initial position for gamma and sigma
            position = np.array([
                random.uniform(bounds[0][0], bounds[0][1]),  # gamma
                random.uniform(bounds[1][0], bounds[1][1])   # sigma
            ])
            velocity = np.zeros(2)
            best_position = position.copy()
            best_score = float('inf')
            particles.append({
                'position': position,
                'velocity': velocity,
                'best_position': best_position,
                'best_score': best_score
            })
        
        global_best_position = None
        global_best_score = float('inf')
        
        # Main PSO loop
        for iteration in range(self.max_iter):
            for particle in particles:
                # Evaluate current position
                gamma, sigma = particle['position']
                model = LSSVM(gamma=gamma, sigma=sigma)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = mean_squared_error(y_val, y_pred)
                
                # Update personal best
                if score < particle['best_score']:
                    particle['best_score'] = score
                    particle['best_position'] = particle['position'].copy()
                    
                    # Update global best
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = particle['position'].copy()
            
            # Update particle velocities and positions
            for particle in particles:
                inertia = self.w * particle['velocity']
                cognitive = self.c1 * random.random() * (particle['best_position'] - particle['position'])
                social = self.c2 * random.random() * (global_best_position - particle['position'])
                
                particle['velocity'] = inertia + cognitive + social
                particle['position'] = np.clip(particle['position'] + particle['velocity'], 
                                              [bounds[0][0], bounds[1][0]], 
                                              [bounds[0][1], bounds[1][1]])
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best score: {global_best_score}")
        
        return global_best_position[0], global_best_position[1]  # best gamma, best sigma

def wavelet_decompose(data, wavelet='db4', level=2):
    """
    Use wavelet transform to decompose the data
    
    Parameters:
        data: input data
        wavelet: wavelet function
        level: decomposition level
    Returns:
        low_freq: low frequency part
        high_freq: high frequency part
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)

    low_freq = coeffs[0]  # low frequency part
    
    # Create a new coeffs list with zeros for low frequency parts only
    high_freq = [np.zeros_like(coeffs[0])]+coeffs[1:]
    high_freq_coeffs = pywt.waverec(high_freq, wavelet)
    high_freq = high_freq_coeffs[:len(data)]  # Cut off to ensure the length is same as original data

    # Rebuild the low frequency part
    # Create a new coeffs list with zeros for high frequency parts
    low_freq_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    low_freq = pywt.waverec(low_freq_coeffs, wavelet)
    low_freq = low_freq[:len(data)]
    
    return low_freq, high_freq

def train_pso_lssvm_model(X_train, y_train, X_val, y_val):
    """
    Train PSO-optimized LSSVM model
    
    Parameters:
        X_train, y_train: Training data
        X_val, y_val: Validation data
    
    Returns:
        Trained LSSVM model
    """
    # Optimize LSSVM hyperparameters using PSO
    pso = PSO(num_particles=20, max_iter=50)
    best_gamma, best_sigma = pso.optimize(X_train, y_train, X_val, y_val)
    
    print(f"Best parameters found: gamma={best_gamma}, sigma={best_sigma}")
    
    # Train LSSVM with best parameters
    model = LSSVM(gamma=best_gamma, sigma=best_sigma)
    model.fit(X_train, y_train)
    
    return model

def train_arima_model(time_series, order=(5,1,0)):
    """
    Train ARIMA model for high frequency data
    
    Parameters:
        time_series: Time series data
        order: ARIMA order (p,d,q)
    
    Returns:
        Trained ARIMA model
    """
    model = ARIMA(time_series, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate model performance
    
    Parameters:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model for display
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

def create_time_features(df):
    """
    Create time features from datetime index
    
    Parameters:
        df: DataFrame with datetime index
    
    Returns:
        DataFrame with added time features
    """
    df_features = df.copy()
    # Extract time features
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)
    
    return df_features

def hybrid_price_prediction(data, target_col, feature_cols, test_size=0.2, wavelet='db4', level=2, 
                          arima_order=(5,1,0), plot_results=True):
    """
    Main function for hybrid price prediction using PSO-LSSVM for low frequency data
    and ARIMA for high frequency data
    
    Parameters:
        data: Input DataFrame with features and target
        target_col: Column name of the price to predict
        feature_cols: List of feature column names
        test_size: Proportion of data to use for testing
        wavelet: Wavelet function to use for decomposition
        level: Decomposition level
        arima_order: ARIMA model order (p,d,q)
        plot_results: Whether to plot results
    
    Returns:
        Dictionary with evaluation metrics and predictions
    """
    # Get series to predict
    y = data[target_col].values
    
    # Decompose the series into low and high frequency components
    low_freq, high_freq = wavelet_decompose(y, wavelet=wavelet, level=level)
    
    if plot_results:
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(y, label='Original LMP Data')
        plt.title('Original Price Data')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(low_freq, label='Low Frequency Component')
        plt.title('Low Frequency Component')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(high_freq, label='High Frequency Component')
        plt.title('High Frequency Component')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    # Split the data
    train_size = int(len(data) * (1 - test_size))
    
    X_train = data[feature_cols].values[:train_size]
    y_train_low = low_freq[:train_size]
    y_train_high = high_freq[:train_size]
    
    X_test = data[feature_cols].values[train_size:]
    y_test_low = low_freq[train_size:]
    y_test_high = high_freq[train_size:]
    y_test = y[train_size:]
    
    # Further split training data to get validation set for PSO
    X_train_lssvm, X_val, y_train_low_lssvm, y_val_low = train_test_split(
        X_train, y_train_low, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_lssvm)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Train PSO-LSSVM for low frequency component
    print("Training PSO-LSSVM model for low frequency component...")
    lssvm_model = train_pso_lssvm_model(X_train_scaled, y_train_low_lssvm, X_val_scaled, y_val_low)
    
    # 2. Train ARIMA for high frequency component
    print("Training ARIMA model for high frequency component...")
    arima_model = train_arima_model(y_train_high, order=arima_order)
    
    # 3. Make predictions
    # Low frequency predictions using PSO-LSSVM
    low_freq_preds = lssvm_model.predict(X_test_scaled)
    
    # High frequency predictions using ARIMA
    high_freq_preds = arima_model.forecast(len(y_test_high))
    
    # Combine predictions
    combined_preds = low_freq_preds + high_freq_preds
    
    # 4. Evaluate models
    evaluate_model(y_test_low, low_freq_preds, "PSO-LSSVM (Low Frequency)")
    evaluate_model(y_test_high, high_freq_preds, "ARIMA (High Frequency)")
    metrics = evaluate_model(y_test, combined_preds, "Hybrid Model (Combined)")
    
    if plot_results:
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(y_test)), y_test, label='Actual Price')
        plt.plot(range(len(combined_preds)), combined_preds, label='Predicted Price')
        plt.title('Hybrid Model Prediction Results')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    
    return {
        "metrics": metrics,
        "predictions": combined_preds,
        "actual": y_test,
        "low_freq_preds": low_freq_preds,
        "high_freq_preds": high_freq_preds
    }

# Example of how to use the hybrid_price_prediction function:
"""
if __name__ == "__main__":
    # Load processed data
    data = pd.read_csv('process_data.csv', index_col=0, parse_dates=True)
    
    # Define features and target
    target_col = 'total_lmp_da_log_norm'
    feature_cols = ['zone_load_log_norm', 'hour', 'day_of_week', 'month', 'is_weekend']
    
    # Run hybrid prediction
    results = hybrid_price_prediction(
        data=data,
        target_col=target_col,
        feature_cols=feature_cols,
        test_size=0.2,
        wavelet='db4',
        level=2,
        arima_order=(5,1,0),
        plot_results=True
    )
    
    print(f"Overall RMSE: {results['metrics']['rmse']:.4f}")
""" 