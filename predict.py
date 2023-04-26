#Importaciones
import glob
import cv2
import torch
import torch.nn as nn
from numpy import argmax
from pymongo import MongoClient
from torchvision import transforms
import math

try:
	# Conectar a la db, host y puerto
	conn = MongoClient(host='localhost', port=27017)
	# Obtener base de datos
	db = conn.local
except:
	pass

# Definir modelo
class scratch_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=100, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(100, 200, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(200, 400, 3, stride=1, padding=0)
        self.mpool = nn.MaxPool2d(kernel_size=3)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(19600,1024)
        self.linear2 = nn.Linear(1024,512)
        self.linear3 = nn.Linear(512,7)
        self.classifier = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.mpool( self.relu(self.conv1(x)) )
        x = self.mpool( self.relu(self.conv2(x)) )
        x = self.mpool( self.relu(self.conv3(x)) )
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.classifier(x)
        return x

# Cargar modelo entrenado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = scratch_nn()
model.load_state_dict(torch.load("vehicles_model.pth", map_location=torch.device('cpu')))
model.eval()
model = model.to(device)

# Definir preprocesados de la imagen
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224),antialias=None),
])


def selection(url):
    label=open(url, 'r').read()
    array=label.split("\n")
    array.pop()
    dist=100
    for coord in array:
        coord=coord.split(" ")
        distaux=math.sqrt((0.5-float(coord[1]))**2+(0.5-float(coord[2]))**2)
        if distaux<dist:
            dist=distaux
            eleccion=coord
    return int(eleccion[0])


# Realizar la prediccion de todas las imagenes en la carpeta
aciertos=0
casi_aciertos=0
fallos=0
for image_path in glob.glob("test/images/*.jpg"):
    img_orig = cv2.imread(image_path)
    img = data_transform(img_orig).unsqueeze(0).to(device)
    outputs = model(img)
    outputs = outputs.detach().cpu().numpy()
    output = argmax(outputs, axis=1)[0]
    label_path=image_path.replace('images','labels').replace(".jpg",".txt")
    label=int(selection(label_path))
    outputs2=outputs.tolist()
    outputs2[0].pop(output)
    output2=argmax(outputs2, axis=1)[0]
    if int(output)==label:
        aciertos+=1
    elif int(output2)==label:
        casi_aciertos+=1
    else:
        fallos+=1

print("Aciertos: ",aciertos)
print("Casi aciertos: ",casi_aciertos)
print("Fallos: ",fallos)
