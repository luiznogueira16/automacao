from fileinput import filename
import os
import pathlib
import pickle
import glob
from turtle import width
import pathlib
import cv2
from cv2 import THRESH_OTSU
import keras
from numpy import character, expand_dims
import numpy as np
from keras import *
from sklearn import neural_network
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sympy import flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow import *
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from imutils import paths
from imutils import resize

TRAINING_FOLDER = "training_data"
MODEL_LABELS = "model_labels.dat"
MODEL_FILENAME = "captcha_model.hdf5"
OUTPUT_FOLDER = "extracted_images"
TEST_DATA = "test_data"


"""image_files = glob.glob(os.path.join(TRAINING_FOLDER, "*"))
for(i, captcha_file) in enumerate(image_files):
    img = cv2.imread(path)
    img = cv2.resize(img, None, fx=2, fx=3, interpolation=cv2.INTER_LINEAR_EXACT)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
finename = os.path.basename(captcha_file)
captcha_text = os.path.splitext(filename)[0]"""

def pre_process(img):
    img = cv2.medianBlur(img, 9)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + THRESH_OTSU)
    kernel = np.ones((2,2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.medianBlur(img, 5)
    img = cv2.dilate(img, np.ones((3,3), np.uint8), iterations=1)
    img = cv2.medianBlur(img, 5)
    img = cv2.erode(img, np.ones((3,3), np.uint8), iterations=2)
    img = cv2.dilate(img, np.ones((3,3), np.uint8), iterations=1)
    return img

def extract_chars(contours):
    char_regions = []
    
    for contour in contours:
        (x, y, width, height) = cv2.boundingRect(contour)
        if(width < 10 or height < 10):
            return []
        char_regions.append((x, y, width, height))
    
    return char_regions

def segment_image(img, num_chars):
    pre_processed = pre_process(img)
    contours, _ = cv2.findContours(pre_processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    char_regions = extract_chars(contours)
    print(contours)
    print(char_regions)
    if(num_chars > 0):
        if (len(char_regions) != num_chars):
                return []
    char_images = sorted(char_regions, key=lambda x: x[0])
    
    return char_images

def generate_characters(img, max_batches):
    data = keras.preprocessing.image.img_to_graph(img)
    samples = expand_dims(data, 0)
    generator = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center = False,
        featurewise_std_normalization = False,
        horizontal_flip = False,
        vertical_flip = False,
        rotation_range = 15,
        width_shift_range = .05,
        height_shift_range = .1,)
    batches = 0
    augmented_images = []
    for batch in generator.flow(samples, batch_size=max_batches):
        augmented_images.append(batch[0])
        batches += 1
        if batches >= max_batches:
            break
    
    return augmented_images

def data_argumentation():
    count_synt = []
    letter_num = []
    remaining = []
    for char in range(10):
        path = os.path.join(OUTPUT_FOLDER, str(char))
        print(path)
        count_synt.append(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]))
        remaining.append(1000-count_synt[char])
        letter_num.append(math.ceil(remaining[char]/count_synt[char]))
        
    extracted_images = glob.glob(os.path.join(OUTPUT_FOLDER, "**/*.png"), recursive=True)
    for (i, captcha_char) in enumerate(extract_images):
        print("[INFO] augmenting image {}/{}".format(i + 1, len(extract_images)))
        folder = pathlib.PurePath(captcha_char)
        
        iterations = letter_num[int(folder.parent.name)]
        if iterations > remaining[int(folder.parent.name)]:
            iterations = remaining[int(folder.parent.name)]
        remaining[int(folder.parent.name)] -= iterations
        
        if iterations > 0:
            img = cv2.imread(captcha_char)
            augmented_chars = generate_characters(img, iterations)
            for augmented_img in augmented_chars:
                count_synt[int(folder.parent.name)] += 1
                filename = os.path.join(folder.parent, "{}_synt.png".format(str(count_synt[int(folder.parent.name)]).zfill(6)))
                cv2.imwrite(filename, augmented_img)

def create_neural_network():
    model = Sequential()
    
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(100,100,1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(50,(5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), striders=(2,2)))
    
    #model.add(flatten())
    model.add(Dense(500, activation="relu"))
    
    model.add(Dense(10, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image

def load_training_data(path):
    data = []
    labels = []
    
    for image_file in os.pathsep.list_images(path):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = resize_to_fit(image, 100, 100)
        if image is not None:
            image = np.extand_dims(image, axis=2)
            label = image_file.split(os.path.sep)[-2]
            data.append(image)
            labels.append(label)
    
    return data, labels

MODEL_LABELS_FILENAME = "model_labels.dat"

def configure_training_data(data, labels):
          
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    
    (x_training, x_validation, y_training, y_validation) = train_test_split(data, labels, test_size=0.25, random_state=0)
    
    lb = LabelBinarizer().fit(y_training)
    y_training = lb.transform(y_training)
    y_validation = lb.transform(y_validation)
    
    with open(MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)
        
    return x_training, x_validation, y_training, y_validation

def train_neural_network(network, train_dataset, validation_dataset, train_labels, validation_labels):
    
    network.fit(train_dataset, train_labels, validation_data=(validation_dataset, validation_labels), batch_size=32, epochs=10, verbose=1)

def save_neural_network(network):
    
    network.save(MODEL_FILENAME)
    
def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR_EXACT)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def solve_captcha(image):
    
    with open(MODEL_LABELS_FILENAME, "rb") as f:
        lb = pickle.load(f)
    
    model = load_model(MODEL_FILENAME)

    chars = segment_image(image, -1)
    
    if len(chars) > 0:
        output = cv2.merge([image] * 3)
        predictions = []
        
        for bounding_box in chars:
            x, y, w, h = bounding_box
            char_image = image[y-2:y+h+2, x-2:x+w+2]
            char_image = resize_to_fit(char_image, 100, 100)
            
            if char_image is not None:
                char_image = np.expand_dims(char_image, axis=2)
                char_image = np.expand_dims(char_image, axis=0)
                
                predictions = model.predict(char_image)
                
                label = lb.inverse_transform(predictions)[0]
                predictions.append(label)
                
                cv2.rectangle(output, (x-2, y-2), (x+w+4, y+h+4), (0,255,0), 1)
                cv2.putText(output, label, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        
        captcha_text = "".join(predictions)
        print("CAPTCHA is: {}".format(captcha_text))
        
        return output, captcha_text
    
    return None, ''

def segmentation():
    print("Segmentation data")
    image_qty = 0
    char_qty = 0
    image_files = glob.glob(os.path.join(TRAINING_FOLDER, "*"))
    counts = {}
    
    for(i, captcha_file) in enumerate(image_files):
        print("[INFO] processing image {}/{}".format(i+1, len(image_files)))
        
        img = load_image(captcha_file)
        filename = os.path.basename(captcha_file)
        captcha_text = os.path.splitext(filename)[0]
        
        chars = segment_image(img, len(filename) - 4)
       
        
        if len(chars) > 0:
            image_qty += 1
            char_qty += len(chars)
            
            for letter_bounding_box, letter_text in zip(chars, captcha_text):
                x, y, w, h = letter_bounding_box
                
                letter_image = img[y:y + h + 2, x:x + w + 2]
                print(letter_bounding_box)
                print(letter_image)
                save_path = os.path.join(OUTPUT_FOLDER, letter_text)
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                count = counts.get(letter_text, 1)
                p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
                print(letter_bounding_box)
                cv2.imwrite(p, letter_image)
                
                counts[letter_text] = count + 1
                
    print(image_qty, char_qty)
    
def train_neural_network():
    print("Train neural network")
    (data, labels) = neural_network.load_training_data(OUTPUT_FOLDER)
    (training_dataset, validation_dataset, training_lab, validation_lab) = configure_training_data(data, labels)
    network = neural_network.create_neural_network()
    neural_network.train_neural_network(network, training_dataset, validation_dataset, training_lab, validation_lab)
    neural_network.save_neural_network(network)
    
def get_captcha():
    print("Testing neural network")
    false_negative = 0
    true_positive = 0
    false_positive = 0
    
    captcha_image_files = glob.glob(TEST_DATA, "*")
    for (i, captcha_file) in enumerate(captcha_image_files):
        print("[INFO] testing image {}/{}".format(i + 1, len(captcha_image_files)))
        captcha = load_image(captcha_image_files)
        _, result = solve_captcha(captcha)
        filename = os.path.basename(captcha_image_files)
        captcha_correct_text = os.path.splitext(filename)[0]
        
        if result == "":
            false_negative += 1
        elif captcha_correct_text == result:
            true_positive += 1
        else:
            false_positive += 1
        
        print(true_positive, false_positive, false_negative, true_positive / (true_positive + false_positive + false_negative))
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = true_positive / (true_positive + false_positive + false_negative)
    f1_score = 2 * ((precision*recall)/(precision+recall))
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"accuracy: {accuracy}")
    print(f"f1_score: {f1_score}")
    
if __name__ == "__main__":
    segmentation()
    data_argumentation()
    train_neural_network()
    get_captcha()