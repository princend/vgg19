import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np

# pip install keras-vggface keras-applications

# 数据路径
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# 数据预处理
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # 多元分类
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # 多元分类
)

# 加载 VGGFace 模型，不包括顶部的全连接层
base_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3))

# 添加自定义的顶层
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # 输出层，类别数为num_classes

# 创建模型
model = Model(inputs=base_model.input, outputs=predictions)

# 冻结卷积基
for layer in base_model.layers:
    layer.trainable = False

# 编译模型
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator
)

# 评估模型
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation loss: {loss}')
print(f'Validation accuracy: {accuracy}')
model.save('my_vggface_model.h5')

# 单张图片预测
def predict_image(img_path):
    # 加载并预处理图像
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array, version=1)  # 使用 VGGFace 的预处理函数

    # 预测
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_prob = predictions[0][predicted_class_index]

    # 打印预测结果
    print(f'Predicted class index: {predicted_class_index}')
    print(f'Confidence: {predicted_class_prob:.2f}')

# 示例使用
# predict_image('path_to_your_image.jpg')  # 替换为你要预测的图片路径
