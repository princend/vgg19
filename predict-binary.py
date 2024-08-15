from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

# 加载保存的模型
model = load_model('my_model.h5')
img_path = 'jaychou-predict.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
predictions = model.predict(img_array)

# 打印预测结果

if predictions[0] > 0.5:
    print("實際歌手為周杰倫,预测类别:蔡依林")
else:
    print("實際歌手為周杰倫,预测类别:周杰倫")

img_path = 'jaychou-predict-2.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
predictions = model.predict(img_array)

# 打印预测结果
if predictions[0] > 0.5:
    print("實際歌手為周杰倫,预测类别:蔡依林")
else:
    print("實際歌手為周杰倫,预测类别:周杰倫")
    
    
img_path = 'jolin-predict.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
predictions = model.predict(img_array)

# 打印预测结果
if predictions[0] > 0.5:
    print("實際歌手為蔡依林,预测类别:蔡依林")
else:
    print("實際歌手為蔡依林,预测类别:周杰倫")
    
    