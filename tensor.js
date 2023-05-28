const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const path = require('path');
const fs = require('fs');

const { promisify } = require('util');
const readdir = promisify(fs.readdir);

const imageDir = 'pizza_not_pizza/';

async function importData() {
  const notPizza = (await readdir(path.join(imageDir, 'not_pizza')))
    .filter((file) => file.split('.')[1] === 'jpg')
    .map((file) => [path.join(imageDir, 'not_pizza', file), 0]);

  const pizza = (await readdir(path.join(imageDir, 'pizza')))
    .filter((file) => file.split('.')[1] === 'jpg')
    .map((file) => [path.join(imageDir, 'pizza', file), 1]);

  const data = [...notPizza, ...pizza];
  const totalSize = data.length;
  const trainSize = Math.floor(totalSize * 0.7);
  const valSize = Math.floor(totalSize * 0.15);
  const testSize = totalSize - trainSize - valSize;

  const df = tf.data.array(data).shuffle(1000);
  const trainData = df.take(trainSize).batch(32);
  const valData = df.skip(trainSize).take(valSize).batch(32);
  const testData = df.skip(trainSize + valSize).take(testSize).batch(32);

  return [trainData, valData, testData];
}

function createModel() {
  const baseModel = tf.keras.applications.inception_v3.InceptionV3({
    weights: 'imagenet',
    includeTop: false,
    inputShape: [256, 256, 3],
  });

  for (let layer of baseModel.layers) {
    layer.trainable = false;
  }

  const model = tf.sequential();
  model.add(baseModel);
  model.add(tf.layers.globalAveragePooling2d());
  model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function trainModel(model, trainData, valData) {
  const history = await model.fitDataset(trainData, {
    epochs: 20,
    validationData: valData,
    callbacks: [
      tf.callbacks.earlyStopping({
        monitor: 'val_loss',
        patience: 3,
        restoreBestWeights: true,
      }),
      tf.callbacks.modelCheckpoint({
        filepath: './model.h5',
        monitor: 'val_loss',
        saveBestOnly: true,
      }),
      tf.callbacks.tensorBoard({ logDir: './logs' }),
    ],
  });
}

async function testModel(model, testData) {
  const results = await model.evaluateDataset(testData);
}

async function main() {
  const [trainData, valData, testData] = await importData();
  const model = createModel();
  await trainModel(model, trainData, valData);
}
main()
  .then(() => console.log('Training complete'))
  .catch((err) => console.error(err));
