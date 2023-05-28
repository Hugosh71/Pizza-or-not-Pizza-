const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
const predictionResultElement = document.getElementById('predictionResult'); // Sélectionner l'élément pour les résultats

let net;

async function app() {
  console.log('Loading mobilenet..');

  net = await mobilenet.load();
  console.log('Successfully loaded model');

  // Charger les images du dataset
  const dataset = new Array();
  const classNames = ['Pizza', 'Not Pizza'];

  // Charger les images de classe 'Pizza'
  for (let i = 1; i <= 200; i++) {
    const img = new Image();
    img.src = 'pizza_full/pizza' + i + '.jpg';
    await new Promise((resolve) => (img.onload = resolve)); // Vérifier le chargement de l'image
    const activation = net.infer(img, true);
    classifier.addExample(activation, 0); // Ajouter l'image à la classe 'Pizza'
    dataset.push(activation);
  }

  // Charger les images de classe 'Not Pizza'
  for (let i = 1; i <= 400; i++) {
    const img = new Image();
    img.src = 'not_pizza_full/notpizza' + i + '.jpg';
    await new Promise((resolve) => (img.onload = resolve)); // Vérifier le chargement de l'image
    const activation = net.infer(img, true);
    classifier.addExample(activation, 1); // Ajouter l'image à la classe 'Not Pizza'
    dataset.push(activation);
  }

  console.log('Dataset loaded');

  const webcam = await tf.data.webcam(webcamElement);

  const addExample = async (classId) => {
    const img = await webcam.capture();
    const activation = net.infer(img, true);
    classifier.addExample(activation, classId);
    img.dispose();
  };

  document.getElementById('pizza').addEventListener('click', () => addExample(0));
  document.getElementById('notpizza').addEventListener('click', () => addExample(1));

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();
      const activation = net.infer(img, 'conv_preds');
      const result = await classifier.predictClass(activation);

      predictionResultElement.innerText = `Prediction: ${classNames[result.label]}, Probability: ${result.confidences[result.label]}`; // Mettre à jour le contenu avec les résultats

      img.dispose();
    }
    await tf.nextFrame();
  }
}

app();