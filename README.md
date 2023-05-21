Pizza Or Not Pizza?  
===================
A simple image classifier using Ramda to determine if an image is a Pizza ( or not ?) 

Structure  
---------
The repository contains following modules:  (à faire)
- *Main*: it contains the instructions to run the project;  

Dataset  
--------
We used a Kaggle dataset for our project, this dataset contains 983 images of pizza and 983 images of dishes other than pizza.
[Dataset's Web site](https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza)  

install js package :
--------
- npm install @tensorflow/tfjs
- npm install ramda
- npm install scikitjs  (train_test_split)

--------
## GitHub Action - Reconnaissance d'image de pizza
### Utilisation

1. Créez un fichier de workflow `.github/workflows/pizzaNotpizza.yml` dans votre dépôt.
2. Ajoutez le contenu suivant pour configurer votre GitHub Action :

```yaml
name: PizzaNotpizza

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  pizza-recognition:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v2

      - name: Set up Node.js
        uses: actions/setup-node@v2
        with:
          node-version: 14

      - name: Install dependencies
        run: npm ci

      - name: Run PizzaNotpizza
        run: node main.js
       
```
L'action est déclenchée à chaque push ou pull request sur la branche "main". 
Elle effectue les étapes suivantes :

1.Récupération du code source du dépôt.
2.Configuration de Node.js avec la version 14.
3.Installation des dépendances à partir du fichier package.json.
4.Exécution du script main.js qui contient l'algorithme.

## GitHub Action - Evaluation de performances
### Utilisation
```yaml
name: Surveillance de performances

on:
  schedule:
    - cron: "0 0 * * *"  # L'action s'éxecutera tout les jours a minuit.

jobs:
  evaluer-modele:
    runs-on: ubuntu-latest

    steps:
      - name: Récupérer le code source
        uses: actions/checkout@v2

      - name: Installer les dépendances
        run: npm ci

      - name: Évaluer les performances
        run: node performances.js

