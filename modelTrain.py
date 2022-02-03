from detecto import core, utils
from torchvision import transforms, torch
import matplotlib.pyplot as plt
import cv2

modelName = './model_weights_advanced_setembro_24.pth'
fruitsName = ['abacaxi', 'banana','laranja','limao_verde','maca','morango','pimentao','tomate','tomate_cereja']

torch.cuda.memory_summary()
# Custom transforms

custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(400),
    transforms.ColorJitter(saturation=0.3),
    transforms.ToTensor(),
    utils.normalize_transform(),
])

dataset = core.Dataset('training_labels/', transform=custom_transforms)

# Validation dataset

val_dataset = core.Dataset('validation_labels/')
loader_val = core.DataLoader(val_dataset)

# Customize training options
loader_train = core.DataLoader(dataset, batch_size=2, shuffle=True) #Limitado a 2, devido a memória disponivel na GPU


model = core.Model(fruitsName,torch.device('cuda')) #Carrega as classes (nomes das frutas)
validatonLosses, trainLosses = model.fit(loader_train, loader_val, epochs=100, learning_rate=0.001, verbose=True) #Funcao modificada para funcionar conforme necessario

# Save model
print("Training")
print(trainLosses)

print("#######################")

print("Validation")
print(validatonLosses)

model.save('model_weights_advanced_setembro_16.pth')

# Visualize loss during training
plt.plot(trainLosses, label='Loss/Epochs - Validation')
plt.ylabel('Loss')
plt.xlabel('Epoch(s)')
plt.show()

plt.plot(validatoinLosses, label='Loss/Epochs - Train')
plt.ylabel('Loss')
plt.xlabel('Epoch(s)')
plt.show()
