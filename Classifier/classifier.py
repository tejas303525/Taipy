from taipy import Gui
from tensorflow.keras import models
from PIL import Image
import numpy as np


class_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

model=models.load_model("/home/profbubs/dev/taipy/Classifier/starterFiles/baseline_mariya.keras")

def predict_image(model,path_to_img):
    img=Image.open(path_to_img)
    img=img.convert("RGB")
    img=img.resize((32,32))
    data=np.asarray(img)
    data=data/255
    probs=model.predict(np.array([data])[:1])
    top_prob=probs.max()
    top_pred=class_names[np.argmax(probs)]

    return top_prob,top_pred


content=""

image_path="/home/profbubs/dev/taipy/Classifier/starterFiles/placeholder_image.png"
prob=0
pred=""
index="""
<|text-center|
#Image prediction

<|{content}|file_selector|extensions=.png|>
Select a file

<|{pred}|>

<|{image_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>

<i>~By Tejas</i>
"""

def on_change(state,var_name,var_val):
    if var_name:
        top_prob,top_pred=predict_image(model,var_val)
        state.prob=round(top_prob*100)
        state.pred="this is an "+ top_pred
        state.image_path=var_val
Gui(page=index).run(use_reloader=True)