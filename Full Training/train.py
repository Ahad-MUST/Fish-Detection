import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import cv2
from pathlib import Path
import warnings
import json
from datetime import datetime
import pickle
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class FishFreshnessClassifier:
    def __init__(self, dataset_path, img_size=(224, 224), batch_size=32, output_dir="results"):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['Fresh', 'Medium', 'Rotten']
        self.class_mapping = {
            'fresh': 0,
            'medium': 1, 
            'rotten': 2
        }
        self.model = None
        self.history = None
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organization
        (self.output_dir / "graphs").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        # Initialize results dictionary
        self.results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'configuration': {
                'img_size': img_size,
                'batch_size': batch_size,
                'dataset_path': str(dataset_path)
            }
        }
        
    def extract_day_number(self, dir_name):
        """Extract day number from directory name (handles 'Day X' format)"""
        if dir_name.lower().startswith('day'):
            try:
                day_num = int(dir_name.split()[-1])
                return day_num
            except (ValueError, IndexError):
                return None
        elif dir_name.isdigit():
            return int(dir_name)
        else:
            return None
        
    def map_day_to_class(self, day):
        """Map day number to freshness class - UPDATED: Fresh is now days 1-4"""
        if day in [1, 2, 3, 4]:  # Updated: Fresh is now days 1-4
            return 'fresh'
        elif day in [5, 6, 7, 8]:  # Medium freshness
            return 'medium'
        elif day in [9, 10, 11, 12]:  # Extended rotten range
            return 'rotten'
        else:
            return None
    
    def load_dataset(self):
        """Load and organize the dataset"""
        print("Loading dataset...")
        print(f"Dataset path: {self.dataset_path}")
        images = []
        labels = []
        file_paths = []
        day_info = []
        
        dataset_path = Path(self.dataset_path)
        
        if not dataset_path.exists():
            print(f"ERROR: Dataset path does not exist: {dataset_path}")
            return None, None
        
        print(f"Found dataset directory: {dataset_path}")
        
        # Store dataset info
        dataset_info = {
            'total_directories': 0,
            'processed_directories': 0,
            'day_distribution': {}
        }
        
        # Iterate through day directories
        for day_dir in dataset_path.iterdir():
            if day_dir.is_dir():
                dataset_info['total_directories'] += 1
                day_num = self.extract_day_number(day_dir.name)
                
                if day_num is not None:
                    class_label = self.map_day_to_class(day_num)
                    
                    if class_label is None:
                        print(f"  Skipping day {day_num} (not in classification range)")
                        continue
                    
                    print(f"  Processing Day {day_num} - Class: {class_label}")
                    dataset_info['processed_directories'] += 1
                    
                    # Load images from this day directory
                    day_images_loaded = self.load_images_from_directory(
                        day_dir, class_label, images, labels, file_paths, day_info, day_num
                    )
                    
                    dataset_info['day_distribution'][f'Day_{day_num}'] = {
                        'class': class_label,
                        'images_loaded': day_images_loaded
                    }
                    
                    print(f"  Loaded {day_images_loaded} images from Day {day_num}")
        
        if len(images) == 0:
            print("\nERROR: No images were loaded!")
            return None, None
        
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.file_paths = file_paths
        self.day_info = day_info
        
        # Store dataset info in results
        self.results['dataset_info'] = dataset_info
        
        print(f"\nDataset loaded successfully!")
        print(f"Total images: {len(self.images)}")
        
        # Print and save class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        class_distribution = {}
        for i, (cls, count) in enumerate(zip(unique, counts)):
            class_distribution[self.class_names[cls]] = int(count)
            print(f"{self.class_names[cls]}: {count} images")
        
        self.results['class_distribution'] = class_distribution
        
        return self.images, self.labels
    
    def load_images_from_directory(self, directory, class_label, images, labels, file_paths, day_info, day_num):
        """Recursively load images from a directory and its subdirectories"""
        images_loaded = 0
        
        for item in directory.rglob('*'):
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                try:
                    img = cv2.imread(str(item))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.img_size)
                        img = img.astype(np.float32) / 255.0
                        
                        images.append(img)
                        labels.append(self.class_mapping[class_label])
                        file_paths.append(str(item))
                        day_info.append(day_num)
                        images_loaded += 1
                    else:
                        print(f"    Warning: Could not load image {item.name}")
                except Exception as e:
                    print(f"    Error loading {item}: {e}")
        
        return images_loaded
    
    def create_train_val_test_split(self, test_size=0.2, val_size=0.2):
        """Split dataset into train, validation, and test sets"""
        print("\nSplitting dataset...")
        
        if len(self.images) == 0:
            print("ERROR: No images to split!")
            return None, None, None, None, None, None
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.images, self.labels, 
            test_size=test_size, 
            stratify=self.labels, 
            random_state=42
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size_adjusted, 
            stratify=y_temp, 
            random_state=42
        )
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        # Store split information
        split_info = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'split_ratios': {
                'train': len(X_train) / len(self.images),
                'val': len(X_val) / len(self.images),
                'test': len(X_test) / len(self.images)
            }
        }
        
        self.results['split_info'] = split_info
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples") 
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_generators(self):
        """Create data generators with augmentation"""
        print("\nCreating data generators with augmentation...")
        
        # Enhanced augmentation parameters
        train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.3,
            shear_range=0.2,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest'
        )
        
        val_test_datagen = ImageDataGenerator()
        
        self.train_generator = train_datagen.flow(
            self.X_train, self.y_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.val_generator = val_test_datagen.flow(
            self.X_val, self.y_val,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.test_generator = val_test_datagen.flow(
            self.X_test, self.y_test,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        print("Data generators created successfully!")
        
    def visualize_sample_images(self, save=True):
        """Visualize sample images from each class"""
        print("\nVisualizing sample images...")
        
        fig, axes = plt.subplots(3, 6, figsize=(18, 12))
        fig.suptitle('Sample Images from Each Class (Updated Fresh: Days 1-4)', fontsize=16)
        
        for class_idx in range(3):
            class_indices = np.where(self.y_train == class_idx)[0]
            
            if len(class_indices) == 0:
                continue
            
            num_samples = min(6, len(class_indices))
            sample_indices = np.random.choice(class_indices, num_samples, replace=False)
            
            for i in range(6):
                if i < len(sample_indices):
                    idx = sample_indices[i]
                    axes[class_idx, i].imshow(self.X_train[idx])
                    axes[class_idx, i].set_title(f'{self.class_names[class_idx]}')
                axes[class_idx, i].axis('off')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "graphs" / "sample_images.png", dpi=300, bbox_inches='tight')
            print(f"Sample images saved to: {self.output_dir / 'graphs' / 'sample_images.png'}")
        
        plt.show()
    
    def visualize_augmented_images(self, save=True):
        """Visualize augmented images"""
        print("\nVisualizing data augmentation effects...")
        
        sample_img = self.X_train[0:1]
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Enhanced Data Augmentation Examples', fontsize=16)
        
        temp_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.3,
            shear_range=0.2,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest'
        )
        
        axes[0, 0].imshow(sample_img[0])
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        temp_generator = temp_datagen.flow(sample_img, batch_size=1)
        
        for i in range(1, 10):
            aug_img = next(temp_generator)[0]
            row, col = i // 5, i % 5
            axes[row, col].imshow(aug_img)
            axes[row, col].set_title(f'Augmented {i}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "graphs" / "augmentation_examples.png", dpi=300, bbox_inches='tight')
            print(f"Augmentation examples saved to: {self.output_dir / 'graphs' / 'augmentation_examples.png'}")
        
        plt.show()
    
    def build_model(self):
        """Build enhanced ResNet50-based model"""
        print("\nBuilding enhanced ResNet50-based model...")
        
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        base_model.trainable = False
        
        # Enhanced model architecture
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        # Store model info
        self.results['model_info'] = {
            'total_params': model.count_params(),
            'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
            'architecture': 'ResNet50 + Custom Head'
        }
        
        print("Enhanced model built successfully!")
        print(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, epochs=50, patience=10):
        """Train the model with enhanced callbacks"""
        print(f"\nStarting training for {epochs} epochs...")
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                str(self.output_dir / 'models' / 'best_fish_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        steps_per_epoch = len(self.X_train) // self.batch_size
        validation_steps = len(self.X_val) // self.batch_size
        
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        
        # Store training info
        self.results['training_info'] = {
            'epochs_trained': len(history.history['accuracy']),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'best_train_accuracy': float(max(history.history['accuracy'])),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_train_loss': float(history.history['loss'][-1])
        }
        
        print("Training completed!")
        return history
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\nEvaluating model...")
        
        test_steps = len(self.X_test) // self.batch_size + 1
        predictions = self.model.predict(self.test_generator, steps=test_steps, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        
        min_len = min(len(self.y_test), len(y_pred))
        y_true = self.y_test[:min_len]
        y_pred = y_pred[:min_len]
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Store evaluation results
        self.test_accuracy = accuracy
        self.y_true = y_true
        self.y_pred = y_pred
        self.predictions_proba = predictions[:min_len]
        
        # Store detailed metrics
        self.results['evaluation_metrics'] = {
            'test_accuracy': float(accuracy),
            'per_class_precision': [float(p) for p in precision],
            'per_class_recall': [float(r) for r in recall],
            'per_class_f1': [float(f) for f in f1],
            'per_class_support': [int(s) for s in support]
        }
        
        return accuracy, y_true, y_pred
    
    def plot_training_history(self, save=True):
        """Plot and save training history"""
        if self.history is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Training & validation accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy Over Time', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training & validation loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss Over Time', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'], linewidth=2, color='orange')
            axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy difference (overfitting indicator)
        acc_diff = np.array(self.history.history['accuracy']) - np.array(self.history.history['val_accuracy'])
        axes[1, 1].plot(acc_diff, linewidth=2, color='red')
        axes[1, 1].set_title('Overfitting Indicator (Train - Val Accuracy)', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Difference')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "graphs" / "training_history.png", dpi=300, bbox_inches='tight')
            print(f"Training history saved to: {self.output_dir / 'graphs' / 'training_history.png'}")
        
        plt.show()
    
    def plot_confusion_matrix(self, save=True):
        """Plot and save detailed confusion matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # Create subplots for different views
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[0])
        axes[0].set_title('Confusion Matrix (Raw Counts)')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Reds', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        # Per-class accuracy
        class_acc = np.diag(cm_norm)
        axes[2].bar(self.class_names, class_acc, color=['green', 'orange', 'red'])
        axes[2].set_title('Per-Class Accuracy')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_ylim(0, 1)
        
        # Add accuracy values on bars
        for i, acc in enumerate(class_acc):
            axes[2].text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "graphs" / "confusion_matrix.png", dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {self.output_dir / 'graphs' / 'confusion_matrix.png'}")
        
        plt.show()
    
    def plot_class_distribution(self, save=True):
        """Plot class distribution across all splits"""
        splits = ['Train', 'Validation', 'Test']
        y_splits = [self.y_train, self.y_val, self.y_test]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Distribution across splits
        for i, (split, y_split) in enumerate(zip(splits, y_splits)):
            row, col = i // 2, i % 2
            unique, counts = np.unique(y_split, return_counts=True)
            class_names_split = [self.class_names[j] for j in unique]
            
            bars = axes[row, col].bar(class_names_split, counts, color=['green', 'orange', 'red'])
            axes[row, col].set_title(f'{split} Set Distribution')
            axes[row, col].set_ylabel('Number of Images')
            
            # Add count labels on bars
            for j, count in enumerate(counts):
                axes[row, col].text(j, count + 0.5, str(count), ha='center')
        
        # Overall distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        class_names_all = [self.class_names[j] for j in unique]
        
        bars = axes[1, 1].bar(class_names_all, counts, color=['green', 'orange', 'red'])
        axes[1, 1].set_title('Overall Dataset Distribution')
        axes[1, 1].set_ylabel('Number of Images')
        
        # Add count labels and percentages
        total = sum(counts)
        for j, count in enumerate(counts):
            percentage = (count / total) * 100
            axes[1, 1].text(j, count + 0.5, f'{count}\n({percentage:.1f}%)', ha='center')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "graphs" / "class_distribution.png", dpi=300, bbox_inches='tight')
            print(f"Class distribution saved to: {self.output_dir / 'graphs' / 'class_distribution.png'}")
        
        plt.show()
    
    def plot_prediction_confidence(self, save=True):
        """Plot prediction confidence analysis"""
        max_probs = np.max(self.predictions_proba, axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overall confidence distribution
        axes[0, 0].hist(max_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Prediction Confidence Distribution')
        axes[0, 0].set_xlabel('Maximum Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confidence by correctness
        correct_mask = self.y_true == self.y_pred
        axes[0, 1].hist(max_probs[correct_mask], bins=15, alpha=0.7, label='Correct', 
                       color='green', edgecolor='black')
        axes[0, 1].hist(max_probs[~correct_mask], bins=15, alpha=0.7, label='Incorrect', 
                       color='red', edgecolor='black')
        axes[0, 1].set_title('Confidence by Prediction Correctness')
        axes[0, 1].set_xlabel('Maximum Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confidence by class
        for class_idx in range(len(self.class_names)):
            class_mask = self.y_true == class_idx
            if np.any(class_mask):
                axes[1, 0].hist(max_probs[class_mask], bins=15, alpha=0.6, 
                               label=self.class_names[class_idx], edgecolor='black')
        
        axes[1, 0].set_title('Confidence Distribution by True Class')
        axes[1, 0].set_xlabel('Maximum Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confidence statistics
        confidence_stats = {
            'Mean': np.mean(max_probs),
            'Std': np.std(max_probs),
            'Min': np.min(max_probs),
            'Max': np.max(max_probs),
            'Correct Mean': np.mean(max_probs[correct_mask]),
            'Incorrect Mean': np.mean(max_probs[~correct_mask])
        }
        
        # Create text plot for statistics
        axes[1, 1].text(0.1, 0.9, 'Confidence Statistics:', fontsize=14, fontweight='bold',
                       transform=axes[1, 1].transAxes)
        
        y_pos = 0.75
        for stat, value in confidence_stats.items():
            axes[1, 1].text(0.1, y_pos, f'{stat}: {value:.4f}', fontsize=12,
                           transform=axes[1, 1].transAxes)
            y_pos -= 0.1
        
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "graphs" / "prediction_confidence.png", dpi=300, bbox_inches='tight')
            print(f"Prediction confidence analysis saved to: {self.output_dir / 'graphs' / 'prediction_confidence.png'}")
        
        plt.show()
        
        # Store confidence statistics
        self.results['confidence_stats'] = confidence_stats
    
    def analyze_misclassifications(self, save=True):
        """Analyze and visualize misclassified images"""
        print("\nAnalyzing misclassifications...")
        
        # Find misclassified samples
        misclassified_indices = np.where(self.y_true != self.y_pred)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return
        
        print(f"Found {len(misclassified_indices)} misclassified samples")
        
        # Show up to 12 misclassified examples
        num_examples = min(12, len(misclassified_indices))
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('Misclassified Examples', fontsize=16)
        
        for i in range(num_examples):
            idx = misclassified_indices[i]
            row, col = i // 4, i % 4
            
            # Get the original image from test set
            img = self.X_test[idx]
            true_class = self.class_names[self.y_true[idx]]
            pred_class = self.class_names[self.y_pred[idx]]
            confidence = self.predictions_proba[idx][self.y_pred[idx]]
            
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(num_examples, 12):
            row, col = i // 4, i % 4
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "graphs" / "misclassifications.png", dpi=300, bbox_inches='tight')
            print(f"Misclassification analysis saved to: {self.output_dir / 'graphs' / 'misclassifications.png'}")
        
        plt.show()
        
        # Store misclassification info
        self.results['misclassification_info'] = {
            'total_misclassified': len(misclassified_indices),
            'misclassification_rate': len(misclassified_indices) / len(self.y_true)
        }
    
    def fine_tune_model(self, epochs=20, learning_rate=1e-5):
        """Fine-tune the pre-trained layers"""
        print("\nFine-tuning the model...")
        
        # Unfreeze the base model
        self.model.layers[0].trainable = True
        
        # Use a very low learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks for fine-tuning
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-8,
                verbose=1
            ),
            ModelCheckpoint(
                str(self.output_dir / 'models' / 'best_fish_model_finetuned.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        steps_per_epoch = len(self.X_train) // self.batch_size
        validation_steps = len(self.X_val) // self.batch_size
        
        history_ft = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        if self.history is not None:
            combined_history = {}
            for key in self.history.history.keys():
                combined_history[key] = self.history.history[key] + history_ft.history[key]
            self.history.history = combined_history
        
        print("Fine-tuning completed!")
        return history_ft
    
    def predict_single_image(self, image_path):
        """Predict freshness for a single image"""
        try:
            # Load and preprocess image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            prediction = self.model.predict(img, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            result = {
                'predicted_class': self.class_names[predicted_class],
                'confidence': float(confidence),
                'all_probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.class_names, prediction[0])
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            return None
    
    def save_results(self):
        """Save all results to files"""
        print("\nSaving results...")
        
        # Save main results as JSON
        results_file = self.output_dir / "reports" / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {results_file}")
        
        # Save model summary
        model_summary_file = self.output_dir / "reports" / "model_summary.txt"
        with open(model_summary_file, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"Model summary saved to: {model_summary_file}")
        
        # Save predictions
        predictions_file = self.output_dir / "data" / "test_predictions.csv"
        predictions_df = pd.DataFrame({
            'true_class': [self.class_names[i] for i in self.y_true],
            'predicted_class': [self.class_names[i] for i in self.y_pred],
            'confidence': np.max(self.predictions_proba, axis=1),
            'correct': self.y_true == self.y_pred
        })
        
        # Add probability columns
        for i, class_name in enumerate(self.class_names):
            predictions_df[f'prob_{class_name}'] = self.predictions_proba[:, i]
        
        predictions_df.to_csv(predictions_file, index=False)
        print(f"Predictions saved to: {predictions_file}")
        
        # Save training history
        if self.history is not None:
            history_file = self.output_dir / "data" / "training_history.pkl"
            with open(history_file, 'wb') as f:
                pickle.dump(self.history.history, f)
            print(f"Training history saved to: {history_file}")
        
        print(f"\nAll results saved to: {self.output_dir}")
    
    def generate_report(self):
        """Generate a comprehensive report"""
        print("\nGenerating comprehensive report...")
        
        report_file = self.output_dir / "reports" / "classification_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FISH FRESHNESS CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset Information
            f.write("DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Dataset Path: {self.dataset_path}\n")
            f.write(f"Total Images: {len(self.images)}\n")
            f.write(f"Image Size: {self.img_size}\n")
            f.write(f"Batch Size: {self.batch_size}\n\n")
            
            # Class Distribution
            f.write("CLASS DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            for class_name, count in self.results['class_distribution'].items():
                percentage = (count / len(self.images)) * 100
                f.write(f"{class_name}: {count} images ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Model Information
            f.write("MODEL INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Architecture: {self.results['model_info']['architecture']}\n")
            f.write(f"Total Parameters: {self.results['model_info']['total_params']:,}\n")
            f.write(f"Trainable Parameters: {self.results['model_info']['trainable_params']:,}\n\n")
            
            # Training Results
            f.write("TRAINING RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Epochs Trained: {self.results['training_info']['epochs_trained']}\n")
            f.write(f"Best Validation Accuracy: {self.results['training_info']['best_val_accuracy']:.4f}\n")
            f.write(f"Best Training Accuracy: {self.results['training_info']['best_train_accuracy']:.4f}\n")
            f.write(f"Final Validation Loss: {self.results['training_info']['final_val_loss']:.4f}\n")
            f.write(f"Final Training Loss: {self.results['training_info']['final_train_loss']:.4f}\n\n")
            
            # Test Results
            f.write("TEST RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Test Accuracy: {self.results['evaluation_metrics']['test_accuracy']:.4f}\n\n")
            
            # Per-class metrics
            f.write("PER-CLASS METRICS\n")
            f.write("-" * 40 + "\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {self.results['evaluation_metrics']['per_class_precision'][i]:.4f}\n")
                f.write(f"  Recall: {self.results['evaluation_metrics']['per_class_recall'][i]:.4f}\n")
                f.write(f"  F1-Score: {self.results['evaluation_metrics']['per_class_f1'][i]:.4f}\n")
                f.write(f"  Support: {self.results['evaluation_metrics']['per_class_support'][i]}\n\n")
            
            # Updated classification mapping
            f.write("CLASSIFICATION MAPPING (UPDATED)\n")
            f.write("-" * 40 + "\n")
            f.write("Fresh: Days 1-4\n")
            f.write("Medium: Days 5-8\n")
            f.write("Rotten: Days 9-12\n\n")
            
            # Confidence Statistics
            if 'confidence_stats' in self.results:
                f.write("CONFIDENCE STATISTICS\n")
                f.write("-" * 40 + "\n")
                for stat, value in self.results['confidence_stats'].items():
                    f.write(f"{stat}: {value:.4f}\n")
                f.write("\n")
            
            # Misclassification Information
            if 'misclassification_info' in self.results:
                f.write("MISCLASSIFICATION ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Misclassified: {self.results['misclassification_info']['total_misclassified']}\n")
                f.write(f"Misclassification Rate: {self.results['misclassification_info']['misclassification_rate']:.4f}\n\n")
        
        print(f"Comprehensive report saved to: {report_file}")
    
    def run_complete_pipeline(self, epochs=50, fine_tune=False, fine_tune_epochs=20):
        """Run the complete training and evaluation pipeline"""
        print("Starting Fish Freshness Classification Pipeline")
        print("=" * 60)
        
        try:
            # 1. Load dataset
            images, labels = self.load_dataset()
            if images is None:
                print("Failed to load dataset. Exiting.")
                return
            
            # 2. Create train/val/test splits
            splits = self.create_train_val_test_split()
            if splits[0] is None:
                print("Failed to create data splits. Exiting.")
                return
            
            # 3. Create data generators
            self.create_data_generators()
            
            # 4. Visualize data
            self.visualize_sample_images()
            self.visualize_augmented_images()
            self.plot_class_distribution()
            
            # 5. Build model
            self.build_model()
            
            # 6. Train model
            self.train_model(epochs=epochs)
            
            # 7. Fine-tune if requested
            if fine_tune:
                self.fine_tune_model(epochs=fine_tune_epochs)
            
            # 8. Evaluate model
            self.evaluate_model()
            
            # 9. Generate visualizations
            self.plot_training_history()
            self.plot_confusion_matrix()
            self.plot_prediction_confidence()
            self.analyze_misclassifications()
            
            # 10. Save results
            self.save_results()
            self.generate_report()
            
            print("\n" + "=" * 60)
            print("Pipeline completed successfully!")
            print(f"Results saved to: {self.output_dir}")
            print(f"Test Accuracy: {self.test_accuracy:.4f}")
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = FishFreshnessClassifier(
        dataset_path="./Dataset",
        img_size=(224, 224),
        batch_size=32,
        output_dir="fish_classification_results"
    )
    
    # Run complete pipeline
    classifier.run_complete_pipeline(
        epochs=50,
        fine_tune=True,
        fine_tune_epochs=20
    )
    
    # Example of predicting a single image
    # result = classifier.predict_single_image("path/to/test/image.jpg")
    # if result:
    #     print(f"Predicted class: {result['predicted_class']}")
    #     print(f"Confidence: {result['confidence']:.4f}")
    #     print(f"All probabilities: {result['all_probabilities']}")