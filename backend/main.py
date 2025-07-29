from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition (same as your training code)
class CarClassifierResNet(nn.Module):
    def __init__(self, dropout_rate): 
        super(CarClassifierResNet, self).__init__()
        self.model = models.resnet50(weights='DEFAULT')
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
           nn.Dropout(dropout_rate),
           nn.Linear(self.model.fc.in_features, 6)
        )
    
    def forward(self, x):
        return self.model(x)

# Load new model
model = CarClassifierResNet(0.6).to(device)
model.load_state_dict(torch.load('saved_newmodel.pth', map_location=device))
model.eval()

# Updated class names
class_names = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict_damage(file: UploadFile = File(None)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(outputs, dim=1).item()
            max_prob = float(probabilities[predicted_class])
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
            confidence_score = max_prob * (1 - entropy.item() / torch.log(torch.tensor(6.0)))
        return {
            "prediction": class_names[predicted_class],
            "confidence": float(confidence_score),
            "max_probability": max_prob,
            "entropy": float(entropy),
            "probabilities": {
                class_names[i]: float(probabilities[i]) 
                for i in range(len(class_names))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Car Damage Classification API"}

@app.get("/v1/models")
async def models_endpoint():
    return {"message": "This endpoint is not implemented"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}