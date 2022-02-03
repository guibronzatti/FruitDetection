from detecto import core, utils, visualize
from playsound import playsound
import torch
import cv2
import numpy as np
import time
import os
#from picamera import PiCamera

top_labels = {}
dictFiltrado = {}
pathImages = './Fotos_Teste/'
lista = list(os.listdir(pathImages))

modelName = './model_weights_advanced_setembro_24.pth'
fruitsName = ['abacaxi', 'banana','laranja','limao_verde','maca','morango','pimentao','tomate','tomate_cereja']

#def capturePictureRaspi(): #Para usar no Raspberry
    #camera = PiCamera()
    #camera.start_preview()

    #time.sleep(5)
    #camera.capture('./image.jpg', resize=(640, 480))
    #camera.stop_preview()

   # picName = './image.jpg'

   # return picName

def capturePicturePc(): #Para usar no PC
    cap = cv2.VideoCapture(0)
    time.sleep(3)
    cap.set(3,640) #width=640
    cap.set(4,480) #height=480
    print("Capturando")
    if cap.isOpened():
        _,frame = cap.read()
        cap.release() #releasing camera immediately after capturing picture
        if _ and frame is not None:
            cv2.imwrite('./imagens/image.jpg', frame)
    picName = './imagens/image.jpg'
    return picName

def convScore(val): #converte de TensorFlow para inteiro
    scores = []
    for integer in val:
        scores.append(integer.item()) 
    return scores

def filterValues(dictfilter): #filtra os valores acima de 0.7, valor vai mudar...
    dictPos = {}
    for name, value in dictfilter.items():
        if value >= 0.7:
            dictPos.update({name:value})  
    return dictPos

def checkName(dictfilter): #Reproduz nome da fruta
    for name, value in dictfilter.items():
        if name == 'abacaxi':
            print(name)
            print(value)
            playsound('./sons/abacaxi.mp3')
            time.sleep(2)
        elif name == 'banana':
            print(name)
            print(value)
            playsound('./sons/banana.mp3')
            time.sleep(2)
        elif name == 'laranja':
            print(name)
            print(value)
            playsound('./sons/laranja.mp3')
            time.sleep(2)
        elif name == 'limao_verde':
            print(name)
            print(value)
            playsound('./sons/limaoverde.mp3')
            time.sleep(2)
        elif name == 'maca':
            print(name)
            print(value)
            playsound('./sons/maca.mp3')
            time.sleep(2)
        elif name == 'morango':
            print(name)
            print(value)
            playsound('./sons/morango.mp3')
            time.sleep(2)
        elif name == 'pimentao':
            print(name)
            print(value)
            playsound('./sons/pimentaovermelho.mp3')
            time.sleep(2)
        elif name == 'tomate':
            print(name)
            print(value)
            playsound('./sons/tomate.mp3')
            time.sleep(2)
        elif name == 'tomate_cereja':
            print(name)
            print(value)
            playsound('./sons/tomatecereja.mp3')
            time.sleep(2)

def toList(box):
    newList = box.tolist()
    return newList

def compareDict(dict1, dict2):
    dict3 = {}
    for name, value in dict2.items(): 
        if name in dict1:
            dict3.update({name:value})
    return dict3

def listTensor(dict1): #Converter de lista para Tensor novamente
    list1 = []
    list2 = []
    for name, value in dict1.items():
        list2.append(name)
        list1.append(value)
    list3 = torch.Tensor(list1)
    return list3, list2

def main():
    i = 0
    print('######################')
    if (os.path.isdir(pathImages) == False):
        os.mkdir(pathImages)

    while i<len(lista):
        
        image_name=pathImages+lista[i]
        model = core.Model.load(modelName,fruitsName)

        image = utils.read_image(image_name)

        predictions = model.predict_top(image)

        # predictions format: (labels, boxes, scores)
        label, box, score = predictions

        newBox = toList(box)
        scores = convScore(score) 

        dictBoxes = dict(zip(label,newBox))
        dictValue = dict(zip(label, scores))

        dictFiltrado = filterValues(dictValue)
        boxesFilter = compareDict(dictFiltrado,dictBoxes)
        boxFilter, newName = listTensor(boxesFilter)
        checkName(dictFiltrado)
        print('######################') 

        i += 1

if __name__ == "__main__":
    main()