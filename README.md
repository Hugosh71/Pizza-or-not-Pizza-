Pizza Or Not Pizza?  
===================
Un simple classifieur d'images déterminant si une image est une pizza (ou pas ?)

Structure  
---------
le Répetoire contient les modules suivant :  (à faire)


Dataset  
--------
Nous avons utilisés un dataset via Kaggle pour notre projet, ce dataset contient 983 images de pizzas et 983 images d'autres plats.
[Dataset's Web site](https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza)  

install js package :
--------
- npm install @tensorflow/tfjs
- npm install ramda
- npm install scikitjs  (train_test_split)

--------
## GitHub Action - Reconnaissance d'image de pizza
### Utilisation

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

