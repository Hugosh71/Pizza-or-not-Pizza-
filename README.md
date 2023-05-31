Pizza Or Not Pizza?  
===================
Un simple classifieur d'images déterminant si une image est une pizza (ou pas ?).
[![Vérification du style de code](https://github.com/Hugosh71/Pizza-or-not-Pizza-/actions/workflows/editorconfig-checker.yml/badge.svg)](https://github.com/Hugosh71/Pizza-or-not-Pizza-/actions/workflows/editorconfig-checker.yml)


Structure du projet
---------
le répertoire contient les modules suivant :

- *Index.js* : ce fichier contient le code principal en JS, permettant de charger les images pré-entrainées et de prédire si l'image affichée a la caméra sera une pizza ou pas.

- *Index.html*: ce fichier contient l'interface de la page HTML, il appelle le script *Index.js*, les modèles nécessaires à la prediction (knn,tfjs et mobilenet) et contient l'id des boutons Add pizza et Add food, permettant d'ajouter une image au dataset correct au cas ou celle-ci serait mal reconnu par le programme.

- *Rename.py*: ce fichier Python contient une fonction permettant de renommer les fichier d'un dataset en ajoutant a chaque image un numéro qui s'incrémente.

- *pizza_not_pizza*: ce dossier contient les dataset complet utilisé pour notre projet, un sous dossier pizza comportant les images de pizzas et un sous dossier not_pizza comportant les images de non-pizzas.

- *pizza_full*: ce dossier contient toutes les images de pizzas du dataset après modifications par le fichier *rename.py*.

- *not_pizza_full*: ce dossier contient toutes les images de non-pizzas du dataset après modifications par le fichier *rename.py*.

Dataset  
--------
Nous avons utilisés un dataset via Kaggle pour notre projet, ce dataset contient 983 images de pizzas et 983 images d'autres plats.
[Dataset's Web site](https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza)  


## GitHub Action 
--------
## Editor config utilisé
```
# EditorConfig is awesome: https://EditorConfig.org

# top-most EditorConfig file
root = true

# Unix-style newlines with a newline ending every file
[*]
end_of_line = lf
insert_final_newline = true
charset = utf-8
```

