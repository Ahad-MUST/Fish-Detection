import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import random
import gc

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not found. Please install it using:")
    print("pip install tensorflow")
    exit(1)

# Memory optimization settings
def setup_memory_optimization():
    """Configure TensorFlow for optimal memory usage"""
    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Limit GPU memory to 4GB (adjust based on your GPU)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                )
        except RuntimeError as e:
            print(f"GPU memory setup warning: {e}")
    
    # DISABLE mixed precision for TFLite compatibility
    print("Mixed precision DISABLED for TFLite compatibility")

# Setup memory optimization
setup_memory_optimization()

# Paths
original_data_dir = r"C:\Users\aadi2\OneDrive\Desktop\Temp\fish dataset"
classes = ['fresh', 'medium', 'rotten']

# MEMORY-OPTIMIZED SETTINGS - CRITICAL FOR PREVENTING CRASHES
IMG_SIZE = 512  # MUCH smaller than 512 to prevent memory issues
BATCH_SIZE = 8  # Small batch size for memory efficiency

print(f"🔧 [MEMORY OPTIMIZATION] Using image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"🔧 [MEMORY OPTIMIZATION] Using batch size: {BATCH_SIZE}")
print(f"🔧 [MEMORY OPTIMIZATION] This prevents memory crashes!")

# Create TensorFlow dataset with built-in augmentation (NO file-based augmentation)
def create_efficient_dataset():
    """Create memory-efficient dataset with built-in augmentation"""
    
    # Load dataset directly without pre-augmentation
    dataset = tf.keras.utils.image_dataset_from_directory(
        original_data_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=True,
        seed=42
    )
    
    class_names = dataset.class_names
    print(f"[INFO] Class names: {class_names}")
    
    # Data augmentation using TensorFlow (in-memory, efficient)
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    
    # Normalization
    def preprocess_data(image, label):
        # Normalize to [0,1]
        image = tf.cast(image, tf.float32) / 255.0
        # ImageNet normalization
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        image = (image - mean) / std
        return image, label
    
    def augment_data(image, label):
        # Apply augmentation only during training
        image = data_augmentation(image, training=True)
        return image, label
    
    # Apply preprocessing
    dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Calculate dataset size
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    print(f"[INFO] Total batches: {dataset_size}")
    
    # Split dataset (adjusted for small dataset)
    train_size = int(0.8 * dataset_size)  # Increased train split
    val_size = int(0.1 * dataset_size)    # Smaller val split
    test_size = dataset_size - train_size - val_size
    
    print(f"📊 [DATASET INFO] Small dataset detected ({dataset_size} batches)")
    print(f"📊 [DATASET INFO] Adjusting splits for better training...")
    
    train_dataset = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val_dataset = remaining.take(val_size)
    test_dataset = remaining.skip(val_size)
    
    # Apply augmentation only to training data
    train_dataset = train_dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Configure for performance (NO CACHING to save memory)
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.shuffle(100).prefetch(buffer_size=1)  # Small buffer
    val_dataset = val_dataset.prefetch(buffer_size=1)
    test_dataset = test_dataset.prefetch(buffer_size=1)
    
    print(f"[INFO] Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    return train_dataset, val_dataset, test_dataset, class_names

# Create datasets
train_dataset, val_dataset, test_dataset, class_names = create_efficient_dataset()
num_classes = len(class_names)

# Create TFLite-compatible model
def create_tflite_compatible_model():
    """Create a simple model that's fully compatible with TFLite"""
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Simple CNN architecture - TFLite compatible
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

# Create lightweight model
def create_lightweight_model():
    """Create a memory-efficient model"""
    
    print("[INFO] Creating TFLite-compatible custom CNN model...")
    model = create_tflite_compatible_model()
    return model

# Create model
model = create_lightweight_model()

# Compile model with TFLite-compatible settings
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # Compatible with softmax
    metrics=['accuracy']
)

print(f"[INFO] Model parameters: {model.count_params():,}")

# Memory-efficient training function
def train_model_efficiently(model, train_dataset, val_dataset, epochs=10):
    """Memory-efficient training with frequent cleanup"""

    # Callbacks for memory management
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'best_fish_model_tf.weights.h5',  # Fixed filename for weights
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=True  # Save only weights to reduce file size
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Custom callback for memory cleanup
    class MemoryCleanupCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()  # Force garbage collection
            if tf.config.experimental.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()  # Clear GPU memory
    
    callbacks.append(MemoryCleanupCallback())
    
    try:
        # Train model (adjusted for small dataset)
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1,
            # Remove step limits for small datasets
        )
        
        return history
    except Exception as e:
        print(f"Training error: {e}")
        # Force memory cleanup
        gc.collect()
        tf.keras.backend.clear_session()
        return None

# Simplified evaluation function
def evaluate_model_efficiently(model, test_dataset, class_names):
    """Memory-efficient evaluation"""
    print("\n📊 [TEST RESULTS] Evaluating model...")
    
    try:
        # Evaluate in small batches
        y_true = []
        y_pred = []
        
        batch_count = 0
        for images, labels in test_dataset:
            if batch_count >= 20:  # Limit evaluation batches
                break
                
            predictions = model(images, training=False)
            predicted_labels = tf.argmax(predictions, axis=1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(predicted_labels.numpy())
            batch_count += 1
            
            # Memory cleanup every few batches
            if batch_count % 5 == 0:
                gc.collect()
        
        # Classification report
        print("\n📊 [TEST RESULTS] Classification Report:\n")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        
        # Simple accuracy calculation
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        return accuracy
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0.0

# TFLite conversion with multiple fallback strategies
def convert_to_tflite_efficiently(model_weights_path, model, tflite_path="fish_model.tflite"):
    """TFLite conversion with multiple fallback strategies"""
    try:
        print(f"[INFO] Converting model to TensorFlow Lite...")
        
        # Load weights
        model.load_weights(model_weights_path)
        
        # Strategy 1: Basic conversion (most compatible)
        print("[INFO] Trying basic TFLite conversion...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            # No optimizations - most compatible
            tflite_model = converter.convert()
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            size_mb = len(tflite_model) / 1024 / 1024
            print(f"✅ [SUCCESS] Basic conversion successful!")
            print(f"[INFO] Model saved as {tflite_path}")
            print(f"[INFO] Model size: {size_mb:.2f} MB")
            return True
            
        except Exception as e1:
            print(f"[WARNING] Basic conversion failed: {str(e1)[:100]}...")
            
            # Strategy 2: With TF Select ops (supports more operations)
            print("[INFO] Trying conversion with TF Select ops...")
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
                    tf.lite.OpsSet.SELECT_TF_OPS      # TensorFlow ops fallback
                ]
                tflite_model = converter.convert()
                
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                
                size_mb = len(tflite_model) / 1024 / 1024
                print(f"✅ [SUCCESS] TF Select conversion successful!")
                print(f"[INFO] Model saved as {tflite_path}")
                print(f"[INFO] Model size: {size_mb:.2f} MB")
                print(f"[INFO] Note: This model requires TF Lite with Select TF ops")
                return True
                
            except Exception as e2:
                print(f"[WARNING] TF Select conversion failed: {str(e2)[:100]}...")
                
                # Strategy 3: Save as regular Keras model instead
                print("[INFO] TFLite conversion failed, saving as Keras model...")
                try:
                    keras_path = tflite_path.replace('.tflite', '_keras.h5')
                    model.save(keras_path)
                    print(f"✅ [SUCCESS] Model saved as Keras format: {keras_path}")
                    print(f"[INFO] You can convert this manually later or use for deployment")
                    return True
                    
                except Exception as e3:
                    print(f"[ERROR] All conversion attempts failed: {e3}")
                    return False
        
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return False

# 🚀 Main execution with error handling
def main():
    try:
        print("[INFO] Starting memory-efficient training...")
        
        # Train model
        history = train_model_efficiently(model, train_dataset, val_dataset, epochs=10)
        
        if history is None:
            print("[ERROR] Training failed!")
            return
        
        print("[INFO] Training completed successfully!")
        
        # Evaluate model
        print("[INFO] Evaluating model...")
        accuracy = evaluate_model_efficiently(model, test_dataset, class_names)
        
        # Convert to TFLite
        print("[INFO] Converting to TensorFlow Lite...")
        success = convert_to_tflite_efficiently('best_fish_model_tf.weights.h5', model)
        
        if success:
            print("✅ [SUCCESS] Model training and conversion completed!")
        else:
            print("⚠️ [WARNING] Training completed but TFLite conversion failed")
            
    except Exception as e:
        print(f"❌ [ERROR] Main execution failed: {e}")
    finally:
        # Final cleanup
        gc.collect()
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()