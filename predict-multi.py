import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# 加载模型
model = tf.keras.models.load_model('my_multiclass_model.h5')


# 加载并预处理图像
# img_path = 'jjlin-predict.jpg'  # 替换为你要预测的图片路径
# img = image.load_img(img_path, target_size=(224, 224))  # 调整图像大小为模型输入尺寸
# img_array = image.img_to_array(img)  # 将图像转换为数组
# img_array = np.expand_dims(img_array, axis=0)  # 增加一个维度以符合模型输入
# img_array = img_array / 255.0  # 缩放像素值到[0, 1]之间

# # 预测
# predictions = model.predict(img_array)
# predicted_class_index = np.argmax(predictions[0])  # 找到概率最高的类别索引
# predicted_class_prob = predictions[0][predicted_class_index]  # 概率值

# # 打印预测结果
# print(f'Predicted class index: {predicted_class_index}')
# print(f'Confidence: {predicted_class_prob:.2f}')



# img_path = 'jaychou-predict.jpg'  # 替换为你要预测的图片路径
# img = image.load_img(img_path, target_size=(224, 224))  # 调整图像大小为模型输入尺寸
# img_array = image.img_to_array(img)  # 将图像转换为数组
# img_array = np.expand_dims(img_array, axis=0)  # 增加一个维度以符合模型输入
# img_array = img_array / 255.0  # 缩放像素值到[0, 1]之间

# # 预测
# predictions = model.predict(img_array)
# predicted_class_index = np.argmax(predictions[0])  # 找到概率最高的类别索引
# predicted_class_prob = predictions[0][predicted_class_index]  # 概率值

# # 打印预测结果
# print(f'Predicted class index: {predicted_class_index}')
# print(f'Confidence: {predicted_class_prob:.2f}')


img_path = 'jolin-predict.jpg'  # 替换为你要预测的图片路径
img = image.load_img(img_path, target_size=(224, 224))  # 调整图像大小为模型输入尺寸
img_array = image.img_to_array(img)  # 将图像转换为数组
img_array = np.expand_dims(img_array, axis=0)  # 增加一个维度以符合模型输入
img_array = img_array / 255.0  # 缩放像素值到[0, 1]之间

# 预测
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])  # 找到概率最高的类别索引
predicted_class_prob = predictions[0][predicted_class_index]  # 概率值

# 打印预测结果
print(f'Predicted class index: {predicted_class_index}')
print(f'Confidence: {predicted_class_prob:.2f}')