## 簡介
用最新的Keras當作Tensorflow的front-end可以輕鬆架構出convolutions的模型。  
例如在Fashion MNIST的分類的應用上，要使用第一層為2D Convolution且包含32個3*3的filter、第二層為2D Max Pooling層、第五層為分類總數的softmax層時，只需用以下程式碼即可架構好。[1][2]  

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
```

## 使用callback來中斷訓練
在訓練模型時，當模型準度已經被訓練夠時，可以選擇中斷訓練以節省時間及避免overfitting。[1]  
需先設定callback並asign到model.fit裡面。    
```python
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.9):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
```

## 使用內建的image data generator來前處理圖片
在做圖片分類應用時常常會使用不同尺寸的圖片，Keras的ImageDataGenerator可以事先處理圖片資料方便後續訓練模型。[4]  
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
        "train_dir",  
        target_size=(150, 150),
        batch_size=128,
        class_mode='binary')
```

當圖片資料不夠時，可以使用image augmentation來解決overfitting。[4]  
```python
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40, # rotate between 0 ~ 40
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
```

## Transfer Learning
Transfer Learning可直接使用暨存的模型，凍結各層避免被retrained。在模型後面可以加入自己的DNN來retrain自己的圖片來應用。[5]  
Dropout為regularization的方法，可以避免overfitting。  
```python
from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape = (150, 150, 3),
                                include_top = False,
                                weights = None)
pre_trained_model.load_weights(local_weights_file)
#freeze the pre-trained layers
for layer in pre_trained_model.layers:
  layer.trainable = False

pre_trained_model.summary()
#take the 'mixed7' as the last layer of the pre-trained model
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['acc'])
```

## Binary-class VS Multi-class
在做Multi-class和Binary-class分類時的設定有些不同，如在ImageDataGenerator、模型最後一層的Dense layer、loss function。[1]  
ImageDataGenerator在Binary-class時  
```python
training_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='binary'
)
```
ImageDataGenerator在Multi-class時  
```python
training_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical'
)
```
最後一層的Dense layer在Binary-class時  
```python
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
最後一層的Dense layer在Multi-class時  
```python
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
```
loss function在Binary-class時  
```python
model.compile(loss = 'binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
```
loss function在Multi-class時  
```python
model.compile(loss = 'categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
```

## References
[1] https://www.coursera.org/learn/introduction-tensorflow  
[2] https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D?version=stable  
[3] https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/RMSPropOptimizer  
[4] https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator?version=stable  
[5] https://www.tensorflow.org/tutorials/images/transfer_learning  
