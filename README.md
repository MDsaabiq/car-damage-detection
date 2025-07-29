---
annotations_creators:
- manual
language:
- en
license: mit
multilinguality: monolingual
pretty_name: Car Front and Rear Damage Detection
size_categories:
- 1K<n<10K
source_datasets: []
task_categories:
- image-classification
task_ids:
- multi-class-image-classification
---

# 🚗 Comprehensive Car Damage Detection Dataset

A high-quality labeled dataset for training machine learning models to detect and classify car damage in front and rear views.

## 📊 Dataset Overview

This dataset contains **6 distinct categories** for comprehensive car damage assessment:

### Categories
- **F_Normal** - Front view of undamaged cars
- **F_Crushed** - Front view with crushed/dented damage  
- **F_Breakage** - Front view with broken parts (lights, bumper, etc.)
- **R_Normal** - Rear view of undamaged cars
- **R_Crushed** - Rear view with crushed/dented damage
- **R_Breakage** - Rear view with broken parts (lights, bumper, etc.)

## 🎥 Demo

<video width="600" controls>
  <source src="https://res.cloudinary.com/dy4nhsvfm/video/upload/v1753790734/demo_szzmgv.mp4" type="video/mp4">
  Your browser does not support the video tag. <a href="https://res.cloudinary.com/dy4nhsvfm/video/upload/v1753790734/demo_szzmgv.mp4">Download video</a>
</video>

*Car damage detection demonstration*

## 🎯 Use Cases

Perfect for building AI solutions for:
- **Insurance Claim Assessment** - Automated damage evaluation
- **Vehicle Inspection Systems** - Pre-purchase inspections
- **Fleet Management** - Regular vehicle condition monitoring
- **Accident Analysis** - Damage severity assessment
- **Quality Control** - Manufacturing defect detection

## 📁 Dataset Structure

```
comprehensive-car-damage/
├── F_Normal/           # Front view - No damage
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── F_Crushed/          # Front view - Crushed damage
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── F_Breakage/         # Front view - Broken parts
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── R_Normal/           # Rear view - No damage
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── R_Crushed/          # Rear view - Crushed damage
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── R_Breakage/         # Rear view - Broken parts
    ├── image_001.jpg
    ├── image_002.jpg
    └── ...
```

## 🔧 Usage with PyTorch

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(
    root='./comprehensive-car-damage',
    transform=transform
)

# Create data loader
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True
)

# Check classes
print("Classes:", dataset.classes)
print("Number of samples:", len(dataset))
```

## 📈 Dataset Statistics

| Category | Description | Sample Count |
|----------|-------------|--------------|
| F_Normal | Front - No damage | ~XXX images |
| F_Crushed | Front - Crushed | ~XXX images |
| F_Breakage | Front - Broken | ~XXX images |
| R_Normal | Rear - No damage | ~XXX images |
| R_Crushed | Rear - Crushed | ~XXX images |
| R_Breakage | Rear - Broken | ~XXX images |
| **Total** | | **~XXX images** |

## 🏷️ Label Mapping

```python
class_to_idx = {
    'F_Breakage': 0,
    'F_Crushed': 1, 
    'F_Normal': 2,
    'R_Breakage': 3,
    'R_Crushed': 4,
    'R_Normal': 5
}
```

## 🚀 Quick Start

### 1. Clone or Download Dataset
```bash
# If using git
git clone <repository-url>
cd comprehensive-car-damage

# Or download and extract ZIP file
```

### 2. Verify Dataset Structure
```python
import os

dataset_path = "./comprehensive-car-damage"
categories = ['F_Normal', 'F_Crushed', 'F_Breakage', 
              'R_Normal', 'R_Crushed', 'R_Breakage']

for category in categories:
    path = os.path.join(dataset_path, category)
    if os.path.exists(path):
        count = len(os.listdir(path))
        print(f"{category}: {count} images")
    else:
        print(f"{category}: Directory not found!")
```

### 3. Train Your Model
```python
# Example training setup
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 6)  # 6 classes

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        # Your training code here
        pass
```

## 📋 Data Quality Guidelines

### Image Requirements
- **Format**: JPG, PNG
- **Resolution**: Minimum 224x224 pixels
- **Quality**: Clear, well-lit images
- **Angle**: Front/rear view of vehicles
- **Content**: Single vehicle per image

### Damage Categories
- **Normal**: No visible damage, minor scratches acceptable
- **Crushed**: Dented, deformed body parts
- **Breakage**: Broken lights, missing parts, cracked components

## 🔍 Model Performance Tips

### Data Augmentation
```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### Recommended Models
- **ResNet50/101** - Good balance of accuracy and speed
- **EfficientNet** - Optimal efficiency
- **Vision Transformer** - State-of-the-art accuracy

## 📊 Evaluation Metrics

Recommended metrics for this dataset:
- **Accuracy** - Overall classification performance
- **Precision/Recall** - Per-class performance
- **F1-Score** - Balanced metric
- **Confusion Matrix** - Detailed error analysis

## 🤝 Contributing

To contribute to this dataset:
1. Follow the naming convention: `category/image_XXX.jpg`
2. Ensure image quality meets guidelines
3. Verify correct categorization
4. Submit pull request with new images

## 📄 License

This dataset is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Contributors who provided and labeled images
- Automotive industry partners
- Computer vision research community

## 📞 Contact

For questions about this dataset:
- Create an issue in this repository
- Contact: saabiqcs@gmail.com

---

**Perfect for training robust car damage detection models! 🚗✨**
