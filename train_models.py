import numpy as np
import pickle
import warnings
import os
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import data_processing as dp
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

warnings.filterwarnings('ignore')

# Create Report directory if it doesn't exist
os.makedirs('Report/Train', exist_ok=True)

def extract_features_batch(paths_batch):
    """Process a batch of images for parallelization"""
    return dp.extract_features(paths_batch)

def save_training_time(task, start_time, end_time):
    """Save training time to report folder"""
    report_dir = 'Report/Train'
    training_time = end_time - start_time
    
    # If file exists, load and append
    time_log_file = f'{report_dir}/training_times.json'
    if os.path.exists(time_log_file):
        with open(time_log_file, 'r') as f:
            time_log = json.load(f)
    else:
        time_log = {}
    
    time_log[task] = training_time
    
    with open(time_log_file, 'w') as f:
        json.dump(time_log, f, indent=4)
    
    return training_time

def save_age_model_report(y_true, y_pred, model, X_train, y_train):
    """Save age model performance report"""
    report_dir = 'Report/Train'
    
    # Calculate performance metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': range(len(model.feature_importances_)),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save performance metrics as JSON
    performance = {
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'R2_score': float(model.score(X_train, y_train)),
        'top_10_features': feature_importance.head(10).to_dict(orient='records')
    }
    
    with open(f'{report_dir}/age_model_performance.json', 'w') as f:
        json.dump(performance, f, indent=4)
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title('Age Prediction Performance')
    plt.tight_layout()
    plt.savefig(f'{report_dir}/age_prediction_performance.png')
    plt.close()
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Features for Age Prediction')
    plt.tight_layout()
    plt.savefig(f'{report_dir}/age_feature_importance.png')
    plt.close()
    
    # Plot distribution of errors
    plt.figure(figsize=(10, 6))
    errors = y_pred - y_true
    sns.histplot(errors, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Count')
    plt.title('Distribution of Age Prediction Errors')
    plt.tight_layout()
    plt.savefig(f'{report_dir}/age_prediction_errors.png')
    plt.close()
    
    return performance

def save_gender_model_report(y_true, y_pred, model, X_val):
    """Save gender model performance report"""
    report_dir = 'Report/Train'
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Save performance metrics as JSON
    performance = {
        'accuracy': float(accuracy),
        'classification_report': class_report
    }
    
    with open(f'{report_dir}/gender_model_performance.json', 'w') as f:
        json.dump(performance, f, indent=4)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Gender Prediction Confusion Matrix')
    labels = ['Male', 'Female']
    plt.xticks([0.5, 1.5], labels)
    plt.yticks([0.5, 1.5], labels, rotation=0)
    plt.tight_layout()
    plt.savefig(f'{report_dir}/gender_confusion_matrix.png')
    plt.close()
    
    # For LinearSVC, we can get decision function scores
    try:
        # Calculate ROC curve
        y_scores = model.decision_function(X_val)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Gender Classification')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'{report_dir}/gender_roc_curve.png')
        plt.close()
        
        # Update performance metrics with AUC
        performance['roc_auc'] = float(roc_auc)
        with open(f'{report_dir}/gender_model_performance.json', 'w') as f:
            json.dump(performance, f, indent=4)
    except:
        pass
    
    return performance

def save_emotion_model_report(y_true, y_pred, model, X_val):
    """Save emotion model performance report"""
    report_dir = 'Report/Train'
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Emotion labels mapping
    emotion_labels = {
        0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 
        4: 'happy', 5: 'sadness', 6: 'surprise'
    }
    
    # Save performance metrics as JSON
    performance = {
        'accuracy': float(accuracy),
        'classification_report': class_report
    }
    
    with open(f'{report_dir}/emotion_model_performance.json', 'w') as f:
        json.dump(performance, f, indent=4)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Emotion Recognition Confusion Matrix')
    sorted_labels = [emotion_labels[i] for i in range(len(emotion_labels))]
    plt.xticks(np.arange(len(sorted_labels)) + 0.5, sorted_labels, rotation=45)
    plt.yticks(np.arange(len(sorted_labels)) + 0.5, sorted_labels, rotation=0)
    plt.tight_layout()
    plt.savefig(f'{report_dir}/emotion_confusion_matrix.png')
    plt.close()
    
    # Plot feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': range(len(model.feature_importances_)),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Top 20 Features for Emotion Recognition')
        plt.tight_layout()
        plt.savefig(f'{report_dir}/emotion_feature_importance.png')
        plt.close()
        
        # Update performance metrics with feature importance
        performance['top_10_features'] = feature_importance.head(10).to_dict(orient='records')
        with open(f'{report_dir}/emotion_model_performance.json', 'w') as f:
            json.dump(performance, f, indent=4)
    
    # Save per-class metrics visualization
    class_metrics = pd.DataFrame(class_report).drop('accuracy', axis=1)
    if 'macro avg' in class_metrics.columns:
        class_metrics = class_metrics.drop(['macro avg', 'weighted avg'], axis=1)
    
    # Transpose to have metrics as columns and classes as rows
    class_metrics = class_metrics.T
    
    # Create separate plots for precision, recall, and f1-score
    metrics = ['precision', 'recall', 'f1-score']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i+1)
        bars = plt.bar(class_metrics.index, class_metrics[metric].values)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} by Emotion Class')
        plt.ylim(0, 1.1)  # Set y-axis limit
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{report_dir}/emotion_class_metrics.png')
    plt.close()
    
    return performance

def train_and_save_models(use_cache=True):
    start_time = time.time()
    print("Bắt đầu quá trình huấn luyện mô hình...")
    
    # Check for cached features
    cache_dir = "feature_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Prepare datasets
    datasets = dp.prepare_datasets()
    
    # Extract features and train age model
    print("Đang huấn luyện mô hình dự đoán tuổi...")
    age_start_time = time.time()
    x_train_age, x_val_age, y_train_age, y_val_age = datasets['age']
    
    # Check for cached features
    train_age_cache = os.path.join(cache_dir, "train_age_features.npy")
    val_age_cache = os.path.join(cache_dir, "val_age_features.npy")
    
    if use_cache and os.path.exists(train_age_cache) and os.path.exists(val_age_cache):
        print("Đang tải đặc trưng đã lưu trong bộ nhớ đệm cho mô hình tuổi...")
        X_train_age_features = np.load(train_age_cache)
        X_val_age_features = np.load(val_age_cache)
    else:
        print("Đang trích xuất đặc trưng cho mô hình tuổi với đa luồng...")
        # Determine optimal number of processes (use 75% of available cores)
        num_processes = max(1, int(cpu_count() * 0.75))
        print(f"Sử dụng {num_processes} tiến trình cho việc trích xuất đặc trưng")
        
        # Split data into chunks for parallel processing
        batch_size = max(1, len(x_train_age) // num_processes)
        train_batches = [x_train_age[i:i + batch_size] for i in range(0, len(x_train_age), batch_size)]
        
        # Process batches in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.map(extract_features_batch, train_batches)
            X_train_age_features = np.vstack(results)
        
        # Process validation data
        X_val_age_features = dp.extract_features(x_val_age)
        
        # Cache the features
        np.save(train_age_cache, X_train_age_features)
        np.save(val_age_cache, X_val_age_features)
    
    # Standardize features
    scaler_age = StandardScaler()
    X_train_age_scaled = scaler_age.fit_transform(X_train_age_features)
    X_val_age_scaled = scaler_age.transform(X_val_age_features)
    
    # Train age model with optimized parameters
    print("Đang huấn luyện mô hình tuổi...")
    age_model = RandomForestRegressor(n_estimators=100, 
                                     n_jobs=-1,  # Use all available cores
                                     random_state=42,
                                     min_samples_split=5,
                                     max_depth=25)
    age_model.fit(X_train_age_scaled, y_train_age)
    
    # Evaluate age model
    y_pred_age = age_model.predict(X_val_age_scaled)
    mae_age = mean_absolute_error(y_val_age, y_pred_age)
    print(f"MAE cho mô hình dự đoán tuổi: {mae_age}")
    
    # Save age model performance report
    save_age_model_report(y_val_age, y_pred_age, age_model, X_train_age_scaled, y_train_age)
    age_end_time = time.time()
    save_training_time('age_model', age_start_time, age_end_time)
    
    # Extract features and train gender model
    print("Đang huấn luyện mô hình dự đoán giới tính...")
    gender_start_time = time.time()
    x_train_gender, x_val_gender, y_train_gender, y_val_gender = datasets['gender']
    
    # Check for cached features
    train_gender_cache = os.path.join(cache_dir, "train_gender_features.npy")
    val_gender_cache = os.path.join(cache_dir, "val_gender_features.npy")
    
    if use_cache and os.path.exists(train_gender_cache) and os.path.exists(val_gender_cache):
        print("Đang tải đặc trưng đã lưu trong bộ nhớ đệm cho mô hình giới tính...")
        X_train_gender_features = np.load(train_gender_cache)
        X_val_gender_features = np.load(val_gender_cache)
    else:
        print("Đang trích xuất đặc trưng cho mô hình giới tính với đa luồng...")
        # Split data into chunks for parallel processing
        batch_size = max(1, len(x_train_gender) // num_processes)
        train_batches = [x_train_gender[i:i + batch_size] for i in range(0, len(x_train_gender), batch_size)]
        
        # Process batches in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.map(extract_features_batch, train_batches)
            X_train_gender_features = np.vstack(results)
        
        # Process validation data
        X_val_gender_features = dp.extract_features(x_val_gender)
        
        # Cache the features
        np.save(train_gender_cache, X_train_gender_features)
        np.save(val_gender_cache, X_val_gender_features)
    
    # Standardize features
    scaler_gender = StandardScaler()
    X_train_gender_scaled = scaler_gender.fit_transform(X_train_gender_features)
    X_val_gender_scaled = scaler_gender.transform(X_val_gender_features)
    
    # Train gender model - use faster LinearSVC instead of SVC when possible
    print("Đang huấn luyện mô hình giới tính...")
    gender_model = LinearSVC(dual=False, random_state=42)
    gender_model.fit(X_train_gender_scaled, y_train_gender)
    
    # Evaluate gender model
    y_pred_gender = gender_model.predict(X_val_gender_scaled)
    accuracy_gender = accuracy_score(y_val_gender, y_pred_gender)
    print(f"Độ chính xác cho mô hình dự đoán giới tính: {accuracy_gender}")
    
    # Save gender model performance report
    save_gender_model_report(y_val_gender, y_pred_gender, gender_model, X_val_gender_scaled)
    gender_end_time = time.time()
    save_training_time('gender_model', gender_start_time, gender_end_time)
    
    # Train emotion model
    print("Đang huấn luyện mô hình nhận diện cảm xúc...")
    emotion_start_time = time.time()
    X_train_emotion, X_val_emotion, y_train_emotion, y_val_emotion = datasets['emotion']
    
    # Cache emotion features
    train_emotion_cache = os.path.join(cache_dir, "train_emotion_features.npy")
    val_emotion_cache = os.path.join(cache_dir, "val_emotion_features.npy")
    
    if use_cache and os.path.exists(train_emotion_cache) and os.path.exists(val_emotion_cache):
        print("Đang tải đặc trưng đã lưu trong bộ nhớ đệm cho mô hình cảm xúc...")
        X_train_emotion = np.load(train_emotion_cache)
        X_val_emotion = np.load(val_emotion_cache)
    else:
        # If we have the emotion data but not cached, save it for next time
        print("Đang lưu đặc trưng cảm xúc vào bộ nhớ đệm...")
        np.save(train_emotion_cache, X_train_emotion)
        np.save(val_emotion_cache, X_val_emotion)
    
    # Standardize features
    scaler_emotion = StandardScaler()
    X_train_emotion_scaled = scaler_emotion.fit_transform(X_train_emotion)
    X_val_emotion_scaled = scaler_emotion.transform(X_val_emotion)
    
    # Train emotion model with optimized parameters
    print("Đang huấn luyện mô hình cảm xúc...")
    emotion_model = RandomForestClassifier(n_estimators=100, 
                                         n_jobs=-1,  # Use all available cores
                                         random_state=42,
                                         min_samples_split=5,
                                         max_depth=25)
    emotion_model.fit(X_train_emotion_scaled, y_train_emotion)
    
    # Evaluate emotion model
    y_pred_emotion = emotion_model.predict(X_val_emotion_scaled)
    accuracy_emotion = accuracy_score(y_val_emotion, y_pred_emotion)
    print(f"Độ chính xác cho mô hình nhận diện cảm xúc: {accuracy_emotion}")
    
    # Save emotion model performance report
    save_emotion_model_report(y_val_emotion, y_pred_emotion, emotion_model, X_val_emotion_scaled)
    emotion_end_time = time.time()
    save_training_time('emotion_model', emotion_start_time, emotion_end_time)
    
    # Save models and scalers
    print("Đang lưu các mô hình và bộ chuẩn hóa...")
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    with open(os.path.join(models_dir, 'age_model.pkl'), 'wb') as f:
        pickle.dump(age_model, f)
    
    with open(os.path.join(models_dir, 'gender_model.pkl'), 'wb') as f:
        pickle.dump(gender_model, f)
        
    with open(os.path.join(models_dir, 'emotion_model.pkl'), 'wb') as f:
        pickle.dump(emotion_model, f)
    
    with open(os.path.join(models_dir, 'scaler_age.pkl'), 'wb') as f:
        pickle.dump(scaler_age, f)
        
    with open(os.path.join(models_dir, 'scaler_gender.pkl'), 'wb') as f:
        pickle.dump(scaler_gender, f)
        
    with open(os.path.join(models_dir, 'scaler_emotion.pkl'), 'wb') as f:
        pickle.dump(scaler_emotion, f)
    
    # Generate summary report with all model performances
    generate_summary_report({
        'age': {
            'mae': float(mae_age),
            'model_type': 'RandomForestRegressor',
            'features': X_train_age_scaled.shape[1]
        },
        'gender': {
            'accuracy': float(accuracy_gender),
            'model_type': 'LinearSVC',
            'features': X_train_gender_scaled.shape[1]
        },
        'emotion': {
            'accuracy': float(accuracy_emotion),
            'model_type': 'RandomForestClassifier',
            'features': X_train_emotion_scaled.shape[1]
        }
    })
    
    elapsed_time = time.time() - start_time
    print(f"Huấn luyện và lưu đã hoàn thành trong {elapsed_time:.2f} giây!")
    save_training_time('total', start_time, time.time())
    
    return {
        'age_model': age_model,
        'gender_model': gender_model,
        'emotion_model': emotion_model,
        'scaler_age': scaler_age,
        'scaler_gender': scaler_gender,
        'scaler_emotion': scaler_emotion
    }

def generate_summary_report(performance_data):
    """Generate a summary report of all model performances"""
    report_dir = 'Report/Train'
    
    # Save as JSON
    with open(f'{report_dir}/models_summary.json', 'w') as f:
        json.dump(performance_data, f, indent=4)
    
    # Create comparative bar chart for classification models
    plt.figure(figsize=(10, 6))
    tasks = ['Gender', 'Emotion']
    accuracies = [performance_data['gender']['accuracy'], performance_data['emotion']['accuracy']]
    
    plt.bar(tasks, accuracies, color=['lightblue', 'lightgreen'])
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess (Binary)')
    plt.ylim(0, 1.0)
    
    # Add value labels on top of each bar
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center')
    
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{report_dir}/model_accuracy_comparison.png')
    plt.close()
    
    # Create pie charts for training time distribution
    time_log_file = f'{report_dir}/training_times.json'
    if os.path.exists(time_log_file):
        with open(time_log_file, 'r') as f:
            time_log = json.load(f)
        
        if 'total' in time_log:
            # Remove total from the pie chart
            model_times = {k: v for k, v in time_log.items() if k != 'total'}
            
            # Create pie chart
            plt.figure(figsize=(10, 10))
            plt.pie(model_times.values(), labels=model_times.keys(), autopct='%1.1f%%')
            plt.title('Training Time Distribution')
            plt.savefig(f'{report_dir}/training_time_distribution.png')
            plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Huấn luyện các mô hình phân tích khuôn mặt')
    parser.add_argument('--no-cache', action='store_true', help='Tắt bộ nhớ đệm đặc trưng')
    parser.add_argument('--fast', action='store_true', help='Sử dụng tập dữ liệu nhỏ hơn để kiểm thử nhanh hơn')
    args = parser.parse_args()
    
    # Modify data processing function if fast mode is enabled
    if args.fast:
        # Monkey patch the prepare_datasets function to return smaller datasets
        original_prepare_datasets = dp.prepare_datasets
        
        def fast_prepare_datasets():
            datasets = original_prepare_datasets()
            # Use only 10% of the data for fast testing
            for key in datasets:
                if key != 'emotion':  # emotion is already processed
                    # Unpack
                    x_train, x_val, y_train, y_val = datasets[key]
                    # Take smaller subset 
                    fast_size = max(100, len(x_train) // 10)
                    x_train = x_train[:fast_size]
                    y_train = y_train[:fast_size]
                    fast_val_size = max(50, len(x_val) // 10)
                    x_val = x_val[:fast_val_size]
                    y_val = y_val[:fast_val_size]
                    # Repack
                    datasets[key] = (x_train, x_val, y_train, y_val)
                else:
                    # Emotion features
                    X_train, X_val, y_train, y_val = datasets[key]
                    fast_size = max(100, len(X_train) // 10)
                    X_train = X_train[:fast_size]
                    y_train = y_train[:fast_size]
                    fast_val_size = max(50, len(X_val) // 10)
                    X_val = X_val[:fast_val_size]
                    y_val = y_val[:fast_val_size]
                    datasets[key] = (X_train, X_val, y_train, y_val)
            return datasets
        
        # Replace with our patched version
        dp.prepare_datasets = fast_prepare_datasets
        print("Chế độ nhanh đã được bật - sử dụng kích thước tập dữ liệu được giảm")
    
    # Train and save models
    train_and_save_models(use_cache=not args.no_cache) 