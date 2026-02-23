# 🎥 Wildfire Detection using CNN - Computer Vision

A **deep learning system for detecting wildfires** from satellite and aerial imagery using convolutional neural networks and real-time alert systems.

## 🎯 Overview

This project provides:
- ✅ CNN for fire detection
- ✅ Object detection (YOLO)
- ✅ Satellite/aerial image processing
- ✅ Smoke detection algorithms
- ✅ Geolocation services
- ✅ Real-time alerting
- ✅ Multiple model architectures

## 📷 Image Preprocessing

```python
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

class FireImageProcessor:
    """Process satellite imagery"""
    
    def __init__(self):
        self.target_size = (224, 224)
    
    def load_image(self, image_path):
        """Load and resize"""
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.target_size)
        return image
    
    def extract_bands(self, image):
        """Extract color channels"""
        r, g, b = cv2.split(image)
        
        # Normalized Difference Vegetation Index
        ndvi = (b.astype(float) - r.astype(float)) / (b + r + 1e-6)
        
        # Normalized Difference Burn Ratio (fire indicator)
        ndbr = ((g - r) / (g + r + 1e-6) + 1) / 2
        
        return {
            'red': r,
            'green': g,
            'blue': b,
            'ndvi': ndvi,
            'ndbr': ndbr
        }
    
    def contrast_enhancement(self, image):
        """Enhance image contrast"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def normalize(self, image):
        """Normalize for model"""
        image = image.astype('float32') / 255.0
        return image
```

## 🔥 Fire Detection CNN

```python
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.applications import ResNet50, MobileNetV2

class FireDetectionCNN:
    """Custom CNN for fire detection"""
    
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
    
    def build_model(self):
        """Build custom architecture"""
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        return history
```

## 🔧 Transfer Learning

```python
class FireDetectionTransferLearning:
    """Pre-trained models for fire detection"""
    
    @staticmethod
    def resnet50_model(input_shape=(224, 224, 3)):
        """Fine-tune ResNet50"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base layers
        base_model.trainable = False
        
        # Add custom head
        inputs = Input(shape=input_shape)
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model, base_model
    
    @staticmethod
    def mobilenet_model(input_shape=(224, 224, 3)):
        """Lightweight MobileNet"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        base_model.trainable = False
        
        inputs = Input(shape=input_shape)
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def unfreeze_layers(base_model, num_layers=50):
        """Progressive unfreezing"""
        for layer in base_model.layers[-num_layers:]:
            layer.trainable = True
```

## 🔍 Object Detection (YOLO)

```python
class FireObjectDetector:
    """Detect fire locations"""
    
    def __init__(self):
        # Would use pre-trained YOLO weights
        self.detector = None
    
    def detect_fire_regions(self, image, confidence_threshold=0.5):
        """Find fire bounding boxes"""
        # Using a simplified version
        results = self.detector.predict(image)
        
        fire_boxes = []
        for detection in results:
            if detection['class'] == 'fire' and detection['confidence'] > confidence_threshold:
                fire_boxes.append({
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'area': self._calculate_area(detection['bbox'])
                })
        
        return fire_boxes
    
    def _calculate_area(self, bbox):
        """Calculate bounding box area"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
```

## 💨 Smoke Detection

```python
class SmokeDetector:
    """Detect wildfire smoke"""
    
    @staticmethod
    def detect_smoke_color(image):
        """Smoke by color analysis"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Gray/white smoke
        lower_smoke = np.array([0, 0, 100])
        upper_smoke = np.array([180, 50, 255])
        
        mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
        smoke_percentage = np.sum(mask > 0) / mask.size * 100
        
        return smoke_percentage
    
    @staticmethod
    def detect_smoke_motion(frame1, frame2):
        """Motion-based smoke detection"""
        diff = cv2.absdiff(frame1, frame2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        _, motion_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        motion_percentage = np.sum(motion_mask > 0) / motion_mask.size * 100
        
        return motion_percentage
```

## 🌍 Geolocation

```python
class GeoLocationService:
    """Convert pixel coordinates to GPS"""
    
    def __init__(self, satellite_metadata):
        self.lat_min = satellite_metadata['lat_min']
        self.lat_max = satellite_metadata['lat_max']
        self.lon_min = satellite_metadata['lon_min']
        self.lon_max = satellite_metadata['lon_max']
        self.image_height = satellite_metadata['image_height']
        self.image_width = satellite_metadata['image_width']
    
    def pixel_to_gps(self, pixel_x, pixel_y):
        """Convert pixel to coordinates"""
        lat = self.lat_max - (pixel_y / self.image_height) * (self.lat_max - self.lat_min)
        lon = self.lon_min + (pixel_x / self.image_width) * (self.lon_max - self.lon_min)
        
        return lat, lon
```

## 🚨 Real-Time Alert System

```python
class FireAlertSystem:
    """Alert generation"""
    
    def __init__(self):
        self.alert_threshold = 0.7  # 70% fire confidence
    
    def generate_alert(self, detection, gps_coords):
        """Create alert"""
        if detection['confidence'] > self.alert_threshold:
            return {
                'severity': self._calculate_severity(detection['area']),
                'latitude': gps_coords[0],
                'longitude': gps_coords[1],
                'confidence': detection['confidence'],
                'timestamp': datetime.utcnow(),
                'action': 'Contact emergency services'
            }
        
        return None
    
    def _calculate_severity(self, area):
        """Severity based on fire area"""
        if area < 1000:
            return 'LOW'
        elif area < 5000:
            return 'MEDIUM'
        else:
            return 'HIGH'
```

## 💡 Interview Talking Points

**Q: Why use transfer learning?**
```
Answer:
- Limited satellite fire data
- ResNet50/MobileNet pre-trained on ImageNet
- Faster training, better accuracy
- Can fine-tune with small dataset
```

**Q: Real-time inference challenges?**
```
Answer:
- Large satellite images
- Multiple model inference stages
- Network latency issues
- MobileNet lighter than ResNet
- Edge deployment considerations
```

## 🌟 Portfolio Value

✅ Computer vision fundamentals
✅ CNN architectures
✅ Transfer learning
✅ Object detection
✅ Real-time systems
✅ Geospatial data
✅ Critical infrastructure

---

**Technologies**: TensorFlow, OpenCV, YOLO, NumPy

