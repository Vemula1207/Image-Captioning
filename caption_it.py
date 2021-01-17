from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.models import load_model, Model

import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#load model
model = load_model("C:/Users/Akhil/anaconda3/envs/deeplearning\image project/model_9.h5")
model._make_predict_function()
#data preprocess
model_temp = ResNet50(weights="imagenet",input_shape=(224,224,3))

#final model by removing last layer
model_resnet = Model(model_temp.input, model_temp.layers[-2].output)
model_resnet._make_predict_function()
#loading word to idx and idx to word 
with open("C:/Users/Akhil/anaconda3/envs/deeplearning/image project/word_to_idx.pkl", "rb") as w2i:
    word_to_idx = pickle.load(w2i)

with open("C:/Users/Akhil/anaconda3/envs/deeplearning/image project/idx_to_word.pkl", "rb") as i2w:
    idx_to_word = pickle.load(i2w)

    max_len = 35
#preprocess image 
def preprocess_image(img):
    img = image.load_img(img, 
                         target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
#encode image
def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])
    return feature_vector

#predict caption
def predict_caption(photo):
    in_text = "startseq"

    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word

        if word =='endseq':
            break


    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)

    return final_caption
#caption this image 
def caption_this_image(input_img): 

    photo = encode_image(input_img)
    

    caption = predict_caption(photo)
    
    return caption    

    