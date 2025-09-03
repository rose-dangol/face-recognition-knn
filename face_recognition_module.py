import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from collections import Counter
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

class CustomKNeighborsClassifier:
    """
    Custom K-Nearest Neighbors Classifier implemented from scratch.
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.n_samples_fit_ = None
        
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
            
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        self.n_samples_fit_ = X.shape[0]
        
        return self
    
    def _calculate_distances(self, X):
        X = np.asarray(X)
        distances = np.zeros((X.shape[0], self.X_train.shape[0]))
        
        for i, x in enumerate(X):
            for j, x_train in enumerate(self.X_train):
                distances[i, j] = np.sqrt(np.sum((x - x_train) ** 2))
                
        return distances
    
    def _get_neighbors(self, distances):
        neighbors = np.zeros((distances.shape[0], self.n_neighbors), dtype=int)
        
        for i in range(distances.shape[0]):
            neighbor_indices = np.argpartition(distances[i], self.n_neighbors)[:self.n_neighbors]
            neighbors[i] = neighbor_indices
            
        return neighbors
    
    def predict(self, X):
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")
            
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        distances = self._calculate_distances(X)
        neighbors = self._get_neighbors(distances)
        
        predictions = []
        for neighbor_indices in neighbors:
            neighbor_labels = self.y_train[neighbor_indices]
            label_counts = Counter(neighbor_labels)
            predicted_label = max(label_counts, key=label_counts.get)
            predictions.append(predicted_label)
            
        return np.array(predictions)
    
    def predict_proba(self, X):
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")
            
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        distances = self._calculate_distances(X)
        neighbors = self._get_neighbors(distances)
        
        probabilities = []
        for neighbor_indices in neighbors:
            neighbor_labels = self.y_train[neighbor_indices]
            label_counts = Counter(neighbor_labels)
            
            proba = np.zeros(len(self.classes_))
            for i, class_label in enumerate(self.classes_):
                proba[i] = label_counts.get(class_label, 0) / self.n_neighbors
                
            probabilities.append(proba)
            
        return np.array(probabilities)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

class ModelEvaluator:
    """Model evaluation system for Face Recognition."""
    
    def __init__(self, results_folder='model_evaluation_results'):
        self.results_folder = results_folder
        self.ensure_results_folder()
        
    def ensure_results_folder(self):
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
            print(f"[INFO] Created results folder: {self.results_folder}")
    
    def evaluate_model(self, model, X_train, y_train, model_name="Custom_KNN"):
        print(f"[INFO] Evaluating {model_name} model...")
        
        y_pred = model.predict(X_train)
        y_true = y_train
        
        metrics = self._calculate_metrics(y_true, y_pred)
        
        self._generate_confusion_matrix(y_true, y_pred, model_name)
        self._generate_accuracy_metrics(metrics, model_name)
        self._save_detailed_report(metrics, y_true, y_pred, model_name)
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        total_samples = len(y_true)
        correct_predictions = np.sum(y_true == y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'confusion_matrix': cm
        }
    
    def _generate_confusion_matrix(self, y_true, y_pred, model_name):
        try:
            classes = np.unique(np.concatenate([y_true, y_pred]))
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=classes, yticklabels=classes)
            
            plt.title(f'Confusion Matrix - {model_name}\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            accuracy = accuracy_score(y_true, y_pred)
            plt.text(0.02, 0.98, f'Accuracy: {accuracy:.2%}', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white'))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_confusion_matrix_{timestamp}.png"
            filepath = os.path.join(self.results_folder, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[SUCCESS] Confusion matrix saved: {filepath}")
            
        except Exception as e:
            print(f"[ERROR] Failed to generate confusion matrix: {e}")
    
    def _generate_accuracy_metrics(self, metrics, model_name):
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Model Performance - {model_name}\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
            
            bars = ax1.bar(metric_names, metric_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
            ax1.set_title('Overall Performance')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            sample_data = [metrics['correct_predictions'], 
                          metrics['total_samples'] - metrics['correct_predictions']]
            sample_labels = ['Correct', 'Incorrect']
            colors = ['#2ECC71', '#E74C3C']
            
            ax2.pie(sample_data, labels=sample_labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Prediction Distribution')
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_accuracy_metrics_{timestamp}.png"
            filepath = os.path.join(self.results_folder, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[SUCCESS] Accuracy metrics saved: {filepath}")
            
        except Exception as e:
            print(f"[ERROR] Failed to generate accuracy metrics: {e}")
    
    def _save_detailed_report(self, metrics, y_true, y_pred, model_name):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_detailed_report_{timestamp}.txt"
            filepath = os.path.join(self.results_folder, filename)
            
            with open(filepath, 'w') as f:
                f.write(f"MODEL EVALUATION REPORT\n")
                f.write(f"========================\n\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Samples: {metrics['total_samples']}\n\n")
                
                f.write(f"PERFORMANCE METRICS\n")
                f.write(f"------------------\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"F1-Score: {metrics['f1']:.4f}\n\n")
                
                f.write(f"PREDICTIONS\n")
                f.write(f"-----------\n")
                f.write(f"Correct: {metrics['correct_predictions']}\n")
                f.write(f"Incorrect: {metrics['total_samples'] - metrics['correct_predictions']}\n")
                f.write(f"Success Rate: {metrics['correct_predictions']/metrics['total_samples']:.2%}\n\n")
                
                f.write(f"CONFUSION MATRIX\n")
                f.write(f"----------------\n")
                f.write(f"{metrics['confusion_matrix']}\n\n")
                
                f.write(f"CLASSIFICATION REPORT\n")
                f.write(f"--------------------\n")
                f.write(classification_report(y_true, y_pred))
            
            print(f"[SUCCESS] Detailed report saved: {filepath}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save detailed report: {e}")
    
    def cleanup_old_results(self, keep_latest=3):
        try:
            all_files = [f for f in os.listdir(self.results_folder) if f.endswith(('.png', '.txt'))]
            
            file_groups = {}
            for file in all_files:
                if '_confusion_matrix_' in file:
                    key = 'confusion_matrix'
                elif '_accuracy_metrics_' in file:
                    key = 'accuracy_metrics'
                elif '_detailed_report_' in file:
                    key = 'detailed_report'
                else:
                    continue
                
                if key not in file_groups:
                    file_groups[key] = []
                file_groups[key].append(file)
            
            for key, files in file_groups.items():
                if len(files) > keep_latest:
                    def extract_timestamp(filename):
                        try:
                            parts = filename.split('_')
                            for part in parts:
                                if len(part) == 15 and part.replace('.png', '').replace('.txt', '').isdigit():
                                    return part.replace('.png', '').replace('.txt', '')
                            return filename.split('_')[-1].replace('.png', '').replace('.txt', '')
                        except:
                            return filename
                    
                    files.sort(key=extract_timestamp, reverse=True)
                    files_to_delete = files[keep_latest:]
                    
                    for old_file in files_to_delete:
                        filepath = os.path.join(self.results_folder, old_file)
                        os.remove(filepath)
                        print(f"[INFO] Removed old file: {old_file}")
            
            print(f"[INFO] Cleanup completed. Kept latest {keep_latest} files of each type.")
            
        except Exception as e:
            print(f"[ERROR] Failed to cleanup old results: {e}")

class FaceRecognitionModule:
    def __init__(self, faces_dir='static/faces', model_path='face_recognition_model.pkl'):
        """
        Initialize Face Recognition Module
        
        Args:
            faces_dir: Directory containing user face folders
            model_path: Path to save/load the trained model
        """
        self.faces_dir = faces_dir
        self.model_path = model_path
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = None
        
        # Create directories if they don't exist
        os.makedirs(faces_dir, exist_ok=True)
    
    def extract_hog_features(self, face_img):
        """
        Extract HOG (Histogram of Oriented Gradients) features from face image
        """
        try:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            hog_features = hog(
                gray_face,
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=False,
                feature_vector=True
            )
            
            return hog_features
            
        except Exception as e:
            print(f"[ERROR] extract_hog_features: Failed to extract HOG features - {e}")
            return face_img.ravel()
    
    def extract_faces(self, img):
        """
        Extract faces from image using Haar Cascade classifier
        """
        try:
            if img is None:
                print("[DEBUG] extract_faces: Input image is None")
                return np.array([])
            
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_points = self.face_detector.detectMultiScale(gray_img, 1.5, 7)
            
            if len(face_points) > 0:
                print(f"[DEBUG] extract_faces: Detected {len(face_points)} face(s)")
            
            return face_points
            
        except Exception as e:
            print(f"[ERROR] extract_faces: Face detection failed - {e}")
            return np.array([])
    
    def train_model(self):
        """
        Train KNN model using available face images
        """
        print("[INFO] train_model: Starting model training...")
        
        if os.path.exists(self.model_path):
            print("[INFO] train_model: Removing old model file...")
            os.remove(self.model_path)

        if not os.path.exists(self.faces_dir) or len(os.listdir(self.faces_dir)) == 0:
            print(f"[WARNING] train_model: No face images found in {self.faces_dir}")
            return

        print(f"[INFO] train_model: Found {len(os.listdir(self.faces_dir))} user directories")
        
        faces = []
        labels = []
        user_list = os.listdir(self.faces_dir)
        
        for user in user_list:
            user_path = os.path.join(self.faces_dir, user)
            if not os.path.isdir(user_path):
                continue
                
            user_images = os.listdir(user_path)
            print(f"[INFO] train_model: Processing user '{user}' with {len(user_images)} images")
            
            for img_name in user_images:
                img_path = os.path.join(user_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    resized_face = cv2.resize(img, (50, 50))
                    try:
                        hog_features = self.extract_hog_features(resized_face)
                        faces.append(hog_features)
                        labels.append(user)
                        print(f"[DEBUG] train_model: Successfully extracted HOG features for {img_path}")
                    except Exception as e:
                        print(f"[ERROR] train_model: Failed to extract HOG features for {img_path}: {e}")
                        continue
                else:
                    print(f"[WARNING] train_model: Failed to load image {img_path}")

        print(f"[INFO] train_model: Total faces processed: {len(faces)}")
        
        if len(faces) == 0:
            print("[ERROR] train_model: No valid faces found for training")
            return

        faces = np.array(faces)
        labels = np.array(labels)
        
        print(f"[INFO] train_model: Training KNN model with {len(faces)} samples...")
        print(f"[INFO] train_model: HOG feature vector length: {faces.shape[1]} features per face")
        
        # Split data into training and testing sets
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                faces, labels, 
                test_size=0.2,
                random_state=42,
                stratify=labels
            )
        except Exception as e:
            print(f"[WARNING] train_model: Train/test split failed, using all data for training: {e}")
            X_train, X_test, y_train, y_test = faces, np.array([]), labels, np.array([])
        
        # Train the model
        print(f"[INFO] train_model: Training KNN model on {len(X_train)} samples...")
        knn = CustomKNeighborsClassifier(n_neighbors=7)
        knn.fit(X_train, y_train)
        
        # Evaluate model performance
        print(f"[INFO] train_model: Evaluating model performance...")
        evaluator = ModelEvaluator()
        
        train_metrics = evaluator.evaluate_model(knn, X_train, y_train, "Custom_KNN_Train")
        
        if len(X_test) > 0:
            test_metrics = evaluator.evaluate_model(knn, X_test, y_test, "Custom_KNN_Test")
            print(f"[INFO] train_model: Training accuracy: {train_metrics['accuracy']:.2%}")
            print(f"[INFO] train_model: Testing accuracy: {test_metrics['accuracy']:.2%}")
        else:
            print(f"[INFO] train_model: Training accuracy: {train_metrics['accuracy']:.2%}")
            print(f"[WARNING] train_model: No test data available for validation")
        
        # Cleanup old results
        evaluator.cleanup_old_results(keep_latest=3)
        
        # Save model
        joblib.dump(knn, self.model_path)
        self.model = knn
        print(f"[SUCCESS] train_model: Model saved to {self.model_path}")
    
    def load_model(self):
        """
        Load trained model from file
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"[INFO] load_model: Model loaded from {self.model_path}")
            return True
        else:
            print(f"[WARNING] load_model: Model file not found at {self.model_path}")
            return False
    
    def identify_face(self, face_array):
        """
        Identify face using trained KNN model
        """
        try:
            if self.model is None:
                if not self.load_model():
                    print("[ERROR] identify_face: No model available")
                    return ["Unknown"]
            
            prediction = self.model.predict(face_array)
            predicted_person = prediction[0]
            print(f"[DEBUG] identify_face: Prediction result: {predicted_person}")
            
            return prediction
            
        except Exception as e:
            print(f"[ERROR] identify_face: Face identification failed - {e}")
            return ["Unknown"]
    
    def capture_and_identify(self):
        """
        Capture video from camera and identify faces in real-time
        """
        print("[INFO] capture_and_identify: Starting face recognition...")
        
        # Load or train model
        if not self.load_model():
            print("[INFO] capture_and_identify: Training new model...")
            self.train_model()
            if not self.load_model():
                print("[ERROR] capture_and_identify: Failed to load model")
                return
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if cap is None or not cap.isOpened():
            print("[ERROR] capture_and_identify: Failed to open camera")
            return
        
        print("[SUCCESS] capture_and_identify: Camera opened successfully")
        print("[INFO] Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = self.extract_faces(frame)
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = frame[y:y + h, x:x + w]
                    face_resized = cv2.resize(face_roi, (50, 50))
                    
                    try:
                        # Extract HOG features and identify
                        hog_features = self.extract_hog_features(face_resized)
                        identified_person = self.identify_face(hog_features.reshape(1, -1))[0]
                        
                        # Parse identification result
                        if '$' in identified_person:
                            name = identified_person.split('$')[0]
                            user_id = identified_person.split('$')[1] if len(identified_person.split('$')) > 1 else "N/A"
                        else:
                            name = identified_person
                            user_id = "N/A"
                        
                        # Draw rectangle and labels
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                        cv2.putText(frame, f'Name: {name}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 20), 2)
                        cv2.putText(frame, f'ID: {user_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 20), 2)
                        
                    except Exception as e:
                        print(f"[ERROR] capture_and_identify: Error processing face: {e}")
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, 'Error', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Face Recognition', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] capture_and_identify: Camera released")
    
    def add_user_images(self, user_name, user_id, user_section, num_images=50):
        """
        Capture face images for a new user
        """
        user_folder = f"{user_name}${user_id}${user_section}"
        user_path = os.path.join(self.faces_dir, user_folder)
        
        if not os.path.exists(user_path):
            os.makedirs(user_path)
            print(f"[INFO] add_user_images: Created user folder: {user_path}")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if cap is None or not cap.isOpened():
            print("[ERROR] add_user_images: Failed to open camera")
            return False
        
        print(f"[INFO] add_user_images: Starting image capture for {user_name}")
        print(f"[INFO] Target: {num_images} images")
        
        images_captured = 0
        
        while images_captured < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = self.extract_faces(frame)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face_img = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                    image_path = os.path.join(user_path, f'{images_captured}.jpg')
                    cv2.imwrite(image_path, face_img)
                    images_captured += 1
                    
                    if images_captured % 10 == 0:
                        print(f"[INFO] add_user_images: Captured {images_captured}/{num_images} images")
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                    
                    if images_captured >= num_images:
                        break
            
            # Display progress
            cv2.putText(frame, f'Images Captured: {images_captured}/{num_images}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
            cv2.putText(frame, 'Press q to stop early', (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Capturing Face Data", frame)
            
            # Exit early on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"[SUCCESS] add_user_images: Captured {images_captured} images for {user_name}")
        
        # Retrain model with new user
        print("[INFO] add_user_images: Retraining model with new user...")
        self.train_model()
        
        return True


# Example usage
if __name__ == "__main__":
    # Initialize face recognition module
    fr = FaceRecognitionModule()
    
    print("Face Recognition Module")
    print("1. Add new user")
    print("2. Start face recognition")
    print("3. Train model")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == '1':
        name = input("Enter user name: ")
        user_id = input("Enter user ID: ")
        section = input("Enter user section: ")
        fr.add_user_images(name, user_id, section)
    elif choice == '2':
        fr.capture_and_identify()
    elif choice == '3':
        fr.train_model()
    else:
        print("Invalid choice")