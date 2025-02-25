import numpy as np
from numpy import asarray
from flask import Flask, request, jsonify, render_template, make_response
import pickle
from PIL import Image 
import torchvision
from torchvision import datasets, transforms
from werkzeug.utils import secure_filename
import io
import torch
#import pdfkit
import base64
from io import BytesIO
import torchvision.transforms.functional as TF
from flask import render_template_string
import os
from pathlib import Path
import tempfile
#from models import ResNet, BasicBlock
#from models import ResNet101
import ast
#from models import get_resnet101_model
from models import CustomResNet101, get_resnet101_model



model = get_resnet101_model(2)
app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))
model_path = 'model/best_model4.pth'
#model = ResNet101(in_channels=3, num_classes=2, use_se=True)


# Load the state_dict
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
#model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['net'])
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#model = torch.load(model_path, map_location=torch.device('cpu'))



# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/')
def home():
    return render_template('main.html')


# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/q-and-a')
# def q_and_a():
#     return render_template('q_and_a.html')

# @app.route('/contact-us')
# def contact_us():
#     return render_template('contact_us.html')

# this one is not needed
@app.route('/upload', methods=['POST'])
def upload():
  print('test')

  file = request.files['file']
  filename = secure_filename(file.filename)
  
  print('uploading: ',filename)
  file_contents = file.read()
  text = request.form['text']
  return file_contents



@app.route('/predict',methods=['POST'])
def predict():

    if 'file' not in request.files:
        return render_template('index.html', error_text='No file uploaded. Please upload an image file.')


    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error_text='No file selected. Please select an image file.')
    
    filename = secure_filename(file.filename)
    
    print('uploading: ',filename)
    file_contents = file.read()

    data = Image.open(io.BytesIO(file_contents)).convert('RGB')

    ori = Image.open(io.BytesIO(file_contents)).convert('RGB')

    train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 调整饱和度
    transforms.RandomAffine(degrees=0, shear=0.3, translate=(0.2, 0.2)),
    transforms.RandomVerticalFlip(),
    #transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomCrop(224),  # 随机裁剪并缩放
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



    train_transform2 = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(), # to tensor
    ])



    data = train_transform(data)
    ori = train_transform2(ori)

    print(data.shape)

    # torch.reshape(data,(1,3,224,224))
    data = data.unsqueeze(0)
    ori = ori.unsqueeze(0)

    print('new shape: ',data.shape)

    #logits= model.forward(data)
    model.eval()
    logits= model(data)
    probs = torch.argmax(logits, dim=1)
    print(probs)  #see the output of probability
    preds = torch.argmax(logits, dim=1)
    pro = probs.detach().numpy()
    Face_type = ['Fake', 'Real']
    result = Face_type[preds]
    print(result)

    #  ##########

    pil_image = TF.to_pil_image(ori.squeeze(0))

    image_data_uri = image_to_data_uri(pil_image)

    pil_image2 = TF.to_pil_image(data.squeeze(0))

    image_data_uri2 = image_to_data_uri(pil_image2)

    return render_template('main.html', prediction_text=result,image_data_uri=image_data_uri,image_data_uri2=image_data_uri2,pro=pro)


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    # 這個是用來收request那裏傳過來的數據，然後predict的
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

@app.route('/download_prediction', methods=['POST'])
def download_prediction():
    prediction_text = request.form['prediction_text']
    image_data_uri = request.form['image_data_uri']
    image_data_uri2 = request.form['image_data_uri2']
    pro_string = request.form['pro']
    pro_string = pro_string.replace("  ", " ")
    pro_string = pro_string.replace(" ", ", ")
    pro = ast.literal_eval(pro_string)   

    with open('./templates/prediction_template.html', 'r') as f:
        template = f.read()

    html = render_template_string(template,pro=pro, prediction_text=prediction_text, image_data_uri=image_data_uri,image_data_uri2=image_data_uri2)

    # pdf = pdfkit.from_string(html, False)

    # response = make_response(pdf)
    # response.headers['Content-Type'] = 'application/pdf'
    # response.headers['Content-Disposition'] = 'attachment; filename=prediction.pdf'

    return response

def image_to_data_uri(image):
    buffered_image = BytesIO()
    image.save(buffered_image, format="JPEG")
    buffered_image.seek(0)
    img_data = buffered_image.getvalue()
    b64_data = base64.b64encode(img_data).decode('utf-8')
    return f'data:image/png;base64,{b64_data}'


def save_image_to_temp_file(image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file, format="JPEG")
        return temp_file.name

if __name__ == "__main__":
    app.run(debug=True)
