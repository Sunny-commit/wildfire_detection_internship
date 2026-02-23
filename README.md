# 🔥 Wildfire Detection - Deep Learning Computer Vision

A **computer vision system using deep learning** to detect and locate wildfires from satellite imagery and aerial footage, enabling early warning and forest fire prevention.

## 🎯 Overview

This project provides:
- ✅ Real-time fire detection
- ✅ CNN-based image classification
- ✅ Object detection (YOLO/Faster R-CNN)
- ✅ Satellite imagery processing
- ✅ Smoke detection
- ✅ Geographic localization
- ✅ Alert system

## 📸 Data & Preprocessing

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np

class WildfireDataPreprocessor:
    """Prepare satellite/aerial imagery"""
    
    @staticmethod
    def load_and_preprocess(image_path, target_size=(256, 256)):
        """Load and preprocess satellite image"""
        # Read image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Normalize
        img = img / 255.0
        
        return img
    
    @staticmethod
    def enhance_fire_visibility(img):
        """Enhance fire regions for detection"""
        # Convert to HSV (fire has high red/orange)
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Red and orange lower/upper bounds
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        mask2 = cv2.inRange(hsv, lower_orange, upper_orange)
        
        fire_mask = cv2.bitwise_or(mask1, mask2)
        
        return fire_mask
    
    @staticmethod
    def apply_edge_detection(img):
        """Detect fire edges"""
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges
```

## 🏗️ Fire Detection CNN

```python
class WildfireDetectionCNN:
    """Binary classifier: Fire vs No Fire"""
    
    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self):
        """Build fire detection model"""
        model = keras.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # FC Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
```

## 🎯 Object Detection - Fire Localization

```python
class WildfireObjectDetector:
    """Detect and localize fire regions"""
    
    def __init__(self, num_classes=2):  # background, fire
        self.num_classes = num_classes
        self.model = self._build_detector()
    
    def _build_detector(self):
        """Build object detection model"""
        # Simple region-based approach
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Output: 4 bbox coordinates per region
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(4)  # Bounding box: x, y, w, h
        ])
        
        return model
    
    @staticmethod
    def post_process_detections(predictions, confidence_threshold=0.5):
        """Filter and process detections"""
        filtered = []
        
        for pred in predictions:
            if pred[-1] >= confidence_threshold:  # confidence score
                filtered.append(pred)
        
        return filtered
    
    @staticmethod
    def draw_bboxes(image, bboxes):
        """Draw bounding boxes on image"""
        img_copy = image.copy()
        
        for bbox in bboxes:
            x, y, w, h = bbox[:4]
            cv2.rectangle(img_copy, (int(x), int(y)), 
                         (int(x + w), int(y + h)), (0, 255, 0), 2)
        
        return img_copy
```

## 💨 Smoke Detection

```python
class SmokeDetector:
    """Detect smoke regions"""
    
    @staticmethod
    def detect_smoke(image):
        """Detect gray/white smoke regions"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Smoke: Low saturation, medium-high value
        lower_smoke = np.array([0, 0, 100])
        upper_smoke = np.array([255, 50, 255])
        
        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
        
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
        
        return smoke_mask
    
    @staticmethod
    def find_smoke_contours(smoke_mask, min_area=100):
        """Find smoke regions"""
        contours, _ = cv2.findContours(smoke_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                detected_regions.append((x, y, w, h))
        
        return detected_regions
```

## 📍 Geographic Localization

```python
class GeoLocalization:
    """Map fire locations using GPS/coordinates"""
    
    def __init__(self, satellite_bounds):
        """
        satellite_bounds: (lon_min, lon_max, lat_min, lat_max)
        """
        self.bounds = satellite_bounds
    
    def pixel_to_geo_coords(self, pixel_x, pixel_y, image_shape):
        """Convert pixel coordinates to lat/lon"""
        img_height, img_width = image_shape
        
        lon_min, lon_max, lat_min, lat_max = self.bounds
        
        # Map pixel to geographic coordinates
        lon = lon_min + (pixel_x / img_width) * (lon_max - lon_min)
        lat = lat_max - (pixel_y / img_height) * (lat_max - lat_min)
        
        return lat, lon
    
    def generate_alerts(self, detections, image_shape):
        """Generate location-based alerts"""
        alerts = []
        
        for det in detections:
            pixel_x, pixel_y = det[:2]
            lat, lon = self.pixel_to_geo_coords(pixel_x, pixel_y, image_shape)
            
            alert = {
                'latitude': lat,
                'longitude': lon,
                'confidence': det[4] if len(det) > 4 else None
            }
            alerts.append(alert)
        
        return alerts
```

## 🚨 Alert & Monitoring System

```python
class FireAlertSystem:
    """Real-time monitoring and alerts"""
    
    def __init__(self, confidence_threshold=0.7):
        self.threshold = confidence_threshold
        self.alerts_history = []
    
    def process_image(self, image, detector_model):
        """Process incoming image"""
        # Predict
        prediction = detector_model.predict(np.expand_dims(image, 0))
        
        # Generate alert if fire detected
        if prediction[0][0] > self.threshold:
            alert = {
                'timestamp': pd.Timestamp.now(),
                'confidence': float(prediction[0][0]),
                'status': 'FIRE DETECTED'
            }
            self.alerts_history.append(alert)
            return alert
        
        return None
    
    def send_alert(self, alert, channels=['email', 'sms', 'push']):
        """Send multi-channel alert"""
        for channel in channels:
            if channel == 'email':
                # Email notification
                pass
            elif channel == 'sms':
                # SMS notification
                pass
            elif channel == 'push':
                # Push notification
                pass
```

## 📊 Model Training

```python
class WildfireTrainer:
    """Train detection model"""
    
    def __init__(self, model):
        self.model = model
    
    def train(self, train_gen, val_gen, epochs=50):
        """Train with generators"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3
            ),
            keras.callbacks.ModelCheckpoint(
                'best_fire_detector.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
```

## 💡 Interview Talking Points

**Q: Why critical for early detection?**
```
Answer:
- Minutes matter in wildfire prevention
- Quick response maximizes containment
- Reduces property/life loss
- AI enables 24/7 monitoring
```

**Q: Challenges in this domain?**
```
Answer:
- Variable conditions (weather, time of day)
- Class imbalance (few fire images)
- Real-time processing requirements
- Geographic variation
```

## 🌟 Portfolio Value

✅ CNN for fire detection
✅ Object detection & localization
✅ Smoke pattern recognition
✅ Real-time processing
✅ Geographic information systems
✅ Critical infrastructure application
✅ Alert system design

---

**Technologies**: TensorFlow, Keras, OpenCV, NumPy

