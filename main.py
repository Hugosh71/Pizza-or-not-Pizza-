const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const pd = require('pandas-js');

const Image = require('image-js').Image;

const { plot } = require('nodeplotlib');

const { 
  Dense,
  Conv2D,
  Flatten,
  BatchNormalization,
  Dropout,
  MaxPooling2D,
  GlobalAvgPool2D,
} = require('@tensorflow/tfjs-layers');

const {
  InceptionV3,
} = require('@tensorflow-models/inceptionv3');

const { ResNet50 } = require('@tensorflow-models/resnet50');

const {
  Adam,
} = require('@tensorflow/tfjs-optimizers');

const { ImageDataGenerator } = require('@tensorflow/tfjs-node');

const {
  TensorBoard,
  ModelCheckpoint,
  EarlyStopping,
} = require('@tensorflow/tfjs');

const { 
  join,
  basename,
} = require('path');

const fs = require('fs');

const os = require('os');

const randomSeed = 32;
tf.setSeed(randomSeed);
const np = tf.numpy();

const imageDir = 'pizza_not_pizza/';

const notPizza = fs.readdirSync(join(imageDir, 'not_pizza'))
  .filter(file => file.split('.')[1] === 'jpg')
  .map(file => [join(imageDir, 'not_pizza', file), 0]);

const pizza = fs.readdirSync(join(imageDir, 'pizza'))
  .filter(file => file.split('.')[1] === 'jpg')
  .map(file => [join(imageDir, 'pizza', file), 1]);

const imageList = [...notPizza, ...pizza];
const columns = ['filename', 'category'];
const df = new pd.DataFrame(imageList, { columns });

console.log(df.sample(5).toString());

const [trainDF, dummyDF] = df.iloc(null, [0, 1]).trainTestSplit(0.7, randomSeed, true);
const [valDF, testDF] = dummyDF.iloc(null, [0, 1]).trainTestSplit(0.6, randomSeed, true);

trainDF.set('category', trainDF.get('category').astype('string'));
valDF.set('category', valDF.get('category').astype('string'));
testDF.set('category', testDF.get('category').astype('string'));

const datagen = new ImageDataGenerator({
  rescale: 1. / 255,
  shearRange: 0.2,
  zoomRange: 0.2,
  horizontalFlip: true,
});

const targetSize = [256, 256];
const batchSize = 32;
const classMode = 'binary';
const shuffle = true;

const trainGenerator = datagen.flowFromDataFrame(trainDF.toDict(), {
  xCol: 'filename',
  yCol: 'category',
  targetSize,
  batchSize,
  classMode,
  shuffle,
});

const valGenerator = datagen.flowFromDataFrame(valDF.toDict(), {
  xCol: 'filename',
  yCol: 'category',
  targetSize,
  batchSize,
  classMode,
  shuffle,
});

const testGenerator = datagen.flowFromDataFrame(testDF.toDict(), {
  xCol: 'filename',
  yCol: 'category',
  targetSize,
  batchSize,
  classMode,
  shuffle,
});

const inputShape = [256, 256, 3];
const baseModel = await InceptionV3.create({inputShape, weights: 'imagenet'});
for (const layer of

# model.add(Dropout(0.2))
model.add(Dense(256,activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

model.build(input_shape=(None, 256, 256, 3))
model.summary()

model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])

modelcheck = ModelCheckpoint(filepath='model.h5',monitor='val_loss',save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
tensorboard = TensorBoard(log_dir='logs')

history = model.fit(train_generator,validation_data=val_generator,epochs=20,callbacks=[modelcheck,earlystop,tensorboard])

fig,ax = plt.subplots(1,2,figsize=(15,9))
total_epochs = [i for i in range(len(history.history['loss']))]
fig.suptitle("CNN Performance")

ax[0].plot(total_epochs,history.history['loss'],label='train')
ax[0].plot(total_epochs,history.history['val_loss'],label='val')
ax[0].set_title("Training Loss vs Validation Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss %")
ax[0].legend(loc='best')

ax[1].plot(total_epochs,history.history['accuracy'],label='train')
ax[1].plot(total_epochs,history.history['val_accuracy'],label='val')
ax[1].set_title("Training Accuracy vs Validation Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy %")
ax[1].legend(loc='best')
plt.show()