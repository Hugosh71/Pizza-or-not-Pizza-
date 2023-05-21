const R = require('ramda');
const tf = require('@tensorflow/tfjs-node');
const { InceptionV3, preprocess } = require('@tensorflow-models/inception-v3');
const { plot } = require('nodeplotlib');
const path = require('path');
const fs = require('fs');
const { promisify } = require('util');
const readdir = promisify(fs.readdir);

const dataDir = path.join(__dirname, 'pizza_not_pizza');
const imageDir = path.join(dataDir, 'images');

const getImageData = async (dir, category) => {
  const files = await readdir(dir);
  const images = await Promise.all(
    files
      .filter((file) => file.endsWith('.jpg'))
      .map(async (file) => {
        const imageBuffer = await fs.promises.readFile(path.join(dir, file));
        return { input: preprocess(imageBuffer), output: category };
      })
  );
  return images;
};

const trainData = await Promise.all([
  getImageData(path.join(imageDir, 'pizza'), 1),
  getImageData(path.join(imageDir, 'not_pizza'), 0),
]).then((results) => R.flatten(results));

const testData = await getImageData(path.join(imageDir, 'test'), null);

const inceptionV3 = await InceptionV3.create();

const flatten = (input) => tf.layers.flatten().apply(input);

const dense = (units, activation, input) => {
  const layer = tf.layers.dense({ units, activation });
  return layer.apply(input);
};

const dropout = (rate, input) => {
  const layer = tf.layers.dropout({ rate });
  return layer.apply(input);
};

const compileModel = (model) => {
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
};

const createModel = () => {
  const input = tf.input({ shape: [299, 299, 3] });
  const model = tf.model({
    inputs: input,
    outputs: R.pipe(
      R.applySpec({
        inceptionOutput: () => inceptionV3.predict(input),
        flattenOutput: (inceptionOutput) => flatten(inceptionOutput),
        denseOutput: (flattenOutput) => dense(512, 'relu', flattenOutput),
        dropoutOutput: (denseOutput) => dropout(0.5, denseOutput),
        predictionOutput: (dropoutOutput) => dense(1, 'sigmoid', dropoutOutput),
      }),
      R.values,
      R.reduce((acc, layer) => layer.apply(acc), input)
    ),
  });
  return model;
};

const model = createModel();
const compiledModel = compileModel(model);

const trainXs = tf.stack(trainData.map((data) => data.input));
const trainYs = tf.tensor1d(trainData.map((data) => data.output));

const testXs = tf.stack(testData.map((data) => data.input));
const testYs = tf.tensor1d(testData.map((data) => data.output));

const batchSize = 32;
const epochs = 10;

await compiledModel.fit(trainXs, trainYs, {
  batchSize,
  epochs,
  validationData: [testXs, testYs],
  callbacks: [
    tf.callbacks.earlyStopping({ patience: 3 }),
    tf.callbacks.tensorBoard({ logDir: path.join(__dirname, 'logs') }),
    tf.callbacks.modelCheckpoint({
      filepath: path.join(__dirname, 'model.{epoch}.h5'),
    }),
  ],
});

const history = await compiledModel.evaluate(testData);

console.log("Test Loss:", history[0]);
console.log("Test Accuracy:", history[1]);

// Predict on test data
const predictions = await compiledModel.predict(testData);

// Display some predictions
R.range(0, 10).forEach((i) => {
  console.log("Prediction", i, predictions[i].dataSync());
});

// Save the model
const savePath = "file:///path/to/model";
await tfn.io.saveModel(savePath, compiledModel);

// Load the model
const loadedModel = await tfn.io.loadModel(savePath);

// Make predictions using the loaded model
const loadedPredictions = await loadedModel.predict(testData);
console.log("Loaded Model Predictions", loadedPredictions[0].dataSync());
