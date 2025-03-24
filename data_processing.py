import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import cv2
from glob import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time

# Create Report directory if it doesn't exist
os.makedirs('Report/DataProcessing', exist_ok=True)

# Function to extract HOG features from images
def extract_features(images_paths, size=(64, 64)):
    features = []
    
    for img_path in tqdm(images_paths):
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cảnh báo: Không thể đọc ảnh {img_path}")
            continue
            
        img = cv2.resize(img, size)
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Extract HOG features
        hog_features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=False, 
                        block_norm='L2-Hys', transform_sqrt=True)
        features.append(hog_features)
    
    return np.array(features)

# Function to extract and save HOG visualization for reporting
def extract_hog_visualization(img_path, save_path, size=(64, 64)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features with visualization
    hog_features, hog_image = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, 
                        block_norm='L2-Hys', transform_sqrt=True)
    
    # Create figure for report
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return hog_features

# Function to preprocess a single image for prediction
def preprocess_image(file_path, size=(64, 64)):
    im = Image.open(file_path)
    width, height = im.size
    
    # Process the image
    if width == height:
        im = im.resize(size)
    else:
        if width > height:
            left = width/2 - height/2
            right = width/2 + height/2
            top = 0
            bottom = height
            im = im.crop((left, top, right, bottom))
            im = im.resize(size)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            im = im.crop((left, top, right, bottom))
            im = im.resize(size)
    
    # Convert to numpy array
    img_array = np.array(im)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # Extract HOG features
    hog_features = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False)
    
    return hog_features, im

# Load and process UTKFace dataset
def load_utk_face_data():
    images = []
    ages = []
    genders = []
    
    for i in os.listdir('UTKFace/'):
        split = i.split('_')
        ages.append(int(split[0]))
        genders.append(int(split[1]))
        images.append(f'UTKFace/{i}')
        
    images = pd.Series(images, name='Name')
    ages = pd.Series(ages, name='Age')
    genders = pd.Series(genders, name='Gender')
    df = pd.concat([images, ages, genders], axis=1)

    # Filter data
    df = df[df['Age'] <= 90]
    df = df[df['Age'] >= 0]
    df = df.reset_index(drop=True)
    
    # Save age and gender distribution for reporting
    save_demographic_distribution(df)
    
    return df

# Function to save demographic distribution for reporting
def save_demographic_distribution(df):
    report_dir = 'Report/DataProcessing'
    
    # Age distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Age'], bins=20, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig(f'{report_dir}/age_distribution.png')
    plt.close()
    
    # Gender distribution
    gender_counts = df['Gender'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(gender_counts, labels=['Male', 'Female'], autopct='%1.1f%%', colors=['lightblue', 'lightpink'])
    plt.title('Gender Distribution')
    plt.savefig(f'{report_dir}/gender_distribution.png')
    plt.close()
    
    # Save statistics
    stats = {
        'total_samples': len(df),
        'age_mean': df['Age'].mean(),
        'age_median': df['Age'].median(),
        'age_std': df['Age'].std(),
        'gender_distribution': {
            'male': int(gender_counts.get(0, 0)),
            'female': int(gender_counts.get(1, 0))
        }
    }
    
    with open(f'{report_dir}/demographic_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)

# Load and process emotion data from CK+48
def load_emotion_data():
    classes_list = {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 
                   'happy': 4, 'sadness': 5, 'surprise': 6}
    X_emotion = []
    y_emotion = []
    train_dir = 'CK+48/'
    
    # For reporting
    emotion_counts = {emotion: 0 for emotion in classes_list.keys()}

    for f in os.listdir(train_dir):
        files = glob(pathname=str(train_dir + f + '/*.png'))
        emotion_counts[f] = len(files)
        for file in files:
            image = cv2.imread(file)
            image_resized = cv2.resize(image, (64, 64))
            gray_img = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            hog_features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=False)
            X_emotion.append(hog_features)
            y_emotion.append(classes_list[f])
    
    # Save emotion distribution for reporting
    save_emotion_distribution(emotion_counts)
    
    # Save sample HOG visualization
    if len(os.listdir(train_dir)) > 0:
        for emotion in classes_list.keys():
            files = glob(pathname=str(train_dir + emotion + '/*.png'))
            if files:
                sample_img = files[0]
                extract_hog_visualization(sample_img, f'Report/DataProcessing/hog_{emotion}.png')
                break
    
    return np.array(X_emotion), np.array(y_emotion)

# Function to save emotion distribution for reporting
def save_emotion_distribution(emotion_counts):
    report_dir = 'Report/DataProcessing'
    
    # Emotion distribution
    plt.figure(figsize=(12, 8))
    plt.bar(emotion_counts.keys(), emotion_counts.values())
    plt.title('Emotion Class Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{report_dir}/emotion_distribution.png')
    plt.close()
    
    # Save statistics
    with open(f'{report_dir}/emotion_stats.json', 'w') as f:
        json.dump(emotion_counts, f, indent=4)

# Prepare train and validation sets
def prepare_datasets():
    # Load UTK Face data
    df = load_utk_face_data()
    
    # Split data
    x = df['Name']
    y_age = df['Age']
    y_gender = df['Gender']
    
    x_train_age, x_val_age, y_train_age, y_val_age = train_test_split(x, y_age, test_size=0.1)
    x_train_gender, x_val_gender, y_train_gender, y_val_gender = train_test_split(
        x, y_gender, test_size=0.1, stratify=y_gender
    )
    
    # Load emotion data
    X_emotion, y_emotion = load_emotion_data()
    
    # Split emotion data
    X_train_emotion, X_val_emotion, y_train_emotion, y_val_emotion = train_test_split(
        X_emotion, y_emotion, test_size=0.1, random_state=42, stratify=y_emotion
    )
    
    # Save dataset split info for reporting
    save_dataset_split_info(
        {
            'age': (len(x_train_age), len(x_val_age)),
            'gender': (len(x_train_gender), len(x_val_gender)),
            'emotion': (len(X_train_emotion), len(X_val_emotion))
        }
    )
    
    return {
        'age': (x_train_age, x_val_age, y_train_age, y_val_age),
        'gender': (x_train_gender, x_val_gender, y_train_gender, y_val_gender),
        'emotion': (X_train_emotion, X_val_emotion, y_train_emotion, y_val_emotion)
    }

# Function to save dataset split information for reporting
def save_dataset_split_info(split_info):
    report_dir = 'Report/DataProcessing'
    
    # Save as JSON
    with open(f'{report_dir}/dataset_split_info.json', 'w') as f:
        json.dump({
            'age': {
                'train_samples': int(split_info['age'][0]),
                'validation_samples': int(split_info['age'][1]),
                'train_percent': round(100 * split_info['age'][0] / (split_info['age'][0] + split_info['age'][1]), 2),
                'validation_percent': round(100 * split_info['age'][1] / (split_info['age'][0] + split_info['age'][1]), 2)
            },
            'gender': {
                'train_samples': int(split_info['gender'][0]),
                'validation_samples': int(split_info['gender'][1]),
                'train_percent': round(100 * split_info['gender'][0] / (split_info['gender'][0] + split_info['gender'][1]), 2),
                'validation_percent': round(100 * split_info['gender'][1] / (split_info['gender'][0] + split_info['gender'][1]), 2)
            },
            'emotion': {
                'train_samples': int(split_info['emotion'][0]),
                'validation_samples': int(split_info['emotion'][1]),
                'train_percent': round(100 * split_info['emotion'][0] / (split_info['emotion'][0] + split_info['emotion'][1]), 2),
                'validation_percent': round(100 * split_info['emotion'][1] / (split_info['emotion'][0] + split_info['emotion'][1]), 2)
            }
        }, f, indent=4)
    
    # Create plot
    tasks = ['Age', 'Gender', 'Emotion']
    train_samples = [split_info['age'][0], split_info['gender'][0], split_info['emotion'][0]]
    val_samples = [split_info['age'][1], split_info['gender'][1], split_info['emotion'][1]]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(tasks))
    width = 0.35
    
    plt.bar(x - width/2, train_samples, width, label='Train')
    plt.bar(x + width/2, val_samples, width, label='Validation')
    
    plt.xlabel('Task')
    plt.ylabel('Number of Samples')
    plt.title('Dataset Split by Task')
    plt.xticks(x, tasks)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{report_dir}/dataset_split.png')
    plt.close()

# Function to generate processing time report
def report_processing_time(task_name, start_time, end_time):
    report_dir = 'Report/DataProcessing'
    
    processing_time = end_time - start_time
    
    # If file exists, load and append
    time_log_file = f'{report_dir}/processing_times.json'
    if os.path.exists(time_log_file):
        with open(time_log_file, 'r') as f:
            time_log = json.load(f)
    else:
        time_log = {}
    
    time_log[task_name] = processing_time
    
    with open(time_log_file, 'w') as f:
        json.dump(time_log, f, indent=4)
    
    return processing_time 