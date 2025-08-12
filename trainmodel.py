import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Define dataset paths
folders = {
    'glioma': r'C:\Users\Lenovo\Desktop\miniproject\dataset\Training\glioma',
    'meningioma': r'C:\Users\Lenovo\Desktop\miniproject\dataset\Training\meningioma',
    'pituitary': r'C:\Users\Lenovo\Desktop\miniproject\dataset\Training\pituitary',
    'notumor': r'C:\Users\Lenovo\Desktop\miniproject\dataset\Training\notumor',
}

class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Step 2: Load and preprocess images
def load_images(folder, label):
    data = []
    for file in os.listdir(folder):
        try:
            img_path = os.path.join(folder, file)
            image = load_img(img_path, target_size=(128, 128))
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append((image, label))
        except:
            continue
    return data

print("Loading images...")
data = []
data += load_images(folders['glioma'], 0)
data += load_images(folders['meningioma'], 1)
data += load_images(folders['notumor'], 2)
data += load_images(folders['pituitary'], 3)

np.random.shuffle(data)
X = np.array([img for img, label in data])
y = np.array([label for img, label in data])

# Step 3: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 4: Feature extraction with VGG16 (fine-tuned)
print("Extracting features...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Fine-tune last 4 layers
for layer in base_model.layers[:-4]:
    layer.trainable = False
for layer in base_model.layers[-4:]:
    layer.trainable = True

model = Model(inputs=base_model.input, outputs=base_model.output)

# Extract features for train and test
features_train = model.predict(X_train, batch_size=32, verbose=1)
features_test = model.predict(X_test, batch_size=32, verbose=1)

X_train_flat = features_train.reshape(features_train.shape[0], -1)
X_test_flat = features_test.reshape(features_test.shape[0], -1)

# Step 5: Train RandomForest with tuned parameters
print("Training classifier...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train_flat, y_train)

# Step 6: Evaluate
y_pred = clf.predict(X_test_flat)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_labels))

# Step 7: 4x4 confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (4 Classes)")
plt.tight_layout()
plt.show()

# Step 8: Save model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(clf, f)
print("Model saved as rf_model.pkl")
