const tf = require('@tensorflow/tfjs');
const { promisify } = require('util');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const { createCanvas, loadImage } = require('canvas');
const fetch = require('node-fetch');
const FormData = require('form-data');

// Import the required TensorFlow.js packages
require('@tensorflow/tfjs-node');
require('@tensorflow/tfjs-node-gpu');

const {
  ImageDataGenerator,
  Sequential,
  ModelCheckpoint,
  EarlyStopping,
  TensorBoard,
  Adam,
  InceptionV3,
  GlobalAvgPool2D,
  Dense,
  BatchNormalization,
} = tf.keras;

const pandas = require('pandas-js');

async function main() {
  console.log('HEllo world');
  // Set the random seed
  tf.setRandomSeed(32);

  // Define the image directory
  const imageDir = 'pizza_not_pizza/';

  // Create a function to load images and their labels
  const loadImageAndLabel = async (imagePath, label) => {
    const img = await loadImage(imagePath);
    const canvas = createCanvas(256, 256);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 256, 256);
    const imageData = ctx.getImageData(0, 0, 256, 256).data;
    const imageArray = Array.from(imageData);
    const imageTensor = tf.tensor4d(imageArray, [1, 256, 256, 4]);
    const resizedImageTensor = tf.image.resizeBilinear(imageTensor, [256, 256]);
    return { xs: resizedImageTensor, ys: tf.oneHot(tf.tensor1d([label]), 2) };
  };

  // Load and preprocess the images
  const loadImages = async () => {
    const notPizzaDir = path.join(imageDir, 'not_pizza');
    const notPizzaImages = fs.readdirSync(notPizzaDir).filter((file) => file.split('.')[1] === 'jpg');
    const notPizzaData = await Promise.all(
      notPizzaImages.map(async (image) => loadImageAndLabel(path.join(notPizzaDir, image), 0))
    );

    const pizzaDir = path.join(imageDir, 'pizza');
    const pizzaImages = fs.readdirSync(pizzaDir).filter((file) => file.split('.')[1] === 'jpg');
    const pizzaData = await Promise.all(
      pizzaImages.map(async (image) => loadImageAndLabel(path.join(pizzaDir, image), 1))
    );

    return { xs: notPizzaData.concat(pizzaData).map((data) => data.xs), ys: notPizzaData.concat(pizzaData).map((data) => data.ys) };
  };

  const data = await loadImages();
  const { xs, ys } = data;

  // Split the data into training, validation, and test sets
  const [trainXs, valTestXs, trainYs, valTestYs] = tf.split(xs, [Math.floor(xs.shape[0] * 0.7), Math.floor(xs.shape[0] * 0.3)]);
  const [valXs, testXs] = tf.split(valTestXs, [Math.floor(valTestXs.shape[0] * 0.6)]);
  const [valYs, testYs] = tf.split(valTestYs, [Math.floor(valTestYs.shape[0] * 0.6)]);

  // Create the data generators
  const datagen = new ImageDataGenerator({
    rescale: 1.0 / 255,
    shearRange: 0.2,
    zoomRange: 0.2,
    horizontalFlip: true,
  });

  const trainGenerator = datagen.flow(trainXs, trainYs, {
    targetSize: [256, 256],
    batchSize: 32,
    classMode: 'categorical',
    shuffle: true,
  });

  const valGenerator = datagen.flow(valXs, valYs, {
    targetSize: [256, 256],
    batchSize: 32,
    classMode: 'categorical',
    shuffle: true,
  });

  const testGenerator = datagen.flow(testXs, testYs, {
    targetSize: [256, 256],
    batchSize: 32,
    classMode: 'categorical',
    shuffle: true,
  });

  // Create the base model
  const baseModel = await InceptionV3.create({ weights: 'imagenet', inputShape: [256, 256, 3] });
  for (const layer of baseModel.layers) {
    layer.trainable = false;
  }

  // Create the model architecture
  const model = new Sequential();
  model.add(baseModel);
  model.add(new GlobalAvgPool2D());
  model.add(new Dense({ units: 512, activation: 'relu', kernelInitializer: 'heNormal' }));
  model.add(new BatchNormalization());
  model.add(new Dense({ units: 256, activation: 'relu', kernelInitializer: 'heNormal' }));
  model.add(new BatchNormalization());
  model.add(new Dense({ units: 1, activation: 'sigmoid' }));

  model.summary();

  // Compile the model
  model.compile({ loss: 'binaryCrossentropy', optimizer: new Adam(0.001), metrics: ['accuracy'] });

  // Define the callbacks
  const modelCheck = new ModelCheckpoint('model.h5', { monitor: 'valLoss', saveBestOnly: true });
  const earlyStop = new EarlyStopping({ monitor: 'valLoss', patience: 3, restoreBestWeights: true });
  const tensorboard = new TensorBoard('logs');

  // Train the model
  const history = await model.fit(trainGenerator, {
    validationData: valGenerator,
    epochs: 20,
    callbacks: [modelCheck, earlyStop, tensorboard],
  });

  // Plot the training history
  const plt = require('matplotlib.pyplot');
  const totalEpochs = Array.from({ length: history.epoch.length }, (_, i) => i);
  plt.plot(totalEpochs, history.history.loss, 'b', 'Training Loss');
  plt.plot(totalEpochs, history.history.valLoss, 'r', 'Validation Loss');
  plt.xlabel('Epochs');
  plt.ylabel('Loss %');
  plt.title('Training Loss vs Validation Loss');
  plt.legend();
  plt.show();
}

main().catch((error) => console.error(error));
