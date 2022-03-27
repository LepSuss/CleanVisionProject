from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np 
import time
import pickle
from tfcamera import takepic

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)

    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    #print(output_details)
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    #scale, zero_point = output_details['quantization']
    #output = scale * (output - zero_point)

    #print(output)

    id = np.rint(output)
    id = np.int_(id)
    print(id)
    print(output)
    if id == 0:
        output = 1-output
    #print(id)

    #ordered = np.argpartition(-output, 1)
    return id, output

def runmodel():
    takepic()
    
    data_folder = "/home/pi/CleanVision/TF/"

    model_path = data_folder + "NewBinaryModel_Lite/model.tflite"
    label_path = data_folder + "labels.text"

    interpreter = Interpreter(model_path)
    print("Model loaded successfully")

    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    print("Image shape (", width, ",", height, ")")

    #load image
    image = Image.open(data_folder + "test.jpg").convert('RGB').resize((width, height))

    #calssify image
    time1 = time.time()
    label_id, prob = classify_image(interpreter, image)
    time2 = time.time()
    classification_time = np.round(time2-time1, 3)
    print("Classification Time =", classification_time, "seconds.")

    #read class labels.
    labels = load_labels(label_path)

    classification_label = labels[label_id]
    print("Image label is: ", classification_label, ", with accuracy: ", np.round(prob*100, 2), "%.")

    result = f'{classification_label}, with accuracy: {np.round(prob*100, 2)}%'
    return result

if __name__ == '__main__':
    runmodel()
