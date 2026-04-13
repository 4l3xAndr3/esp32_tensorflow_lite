# Projet de Classification par Machine Learning

Ce projet est conçu pour la classification de données à l'aide de techniques d'apprentissage automatique (machine learning). Il comprend une approche structurée pour gérer les jeux de données bruts et traités, l'entraînement du modèle et son évaluation.

## Structure du Projet

```
ml-classification-project
├── data
│   ├── raw                # Contient les fichiers du jeu de données brut
│   └── processed          # Contient les fichiers du jeu de données traité prêts pour l'entraînement
├── src
│   ├── model.py           # Définit l'architecture du modèle
│   ├── train.py           # Contient la logique d'entraînement
│   └── evaluate.py        # Responsable de l'évaluation du modèle
├── requirements.txt       # Liste les dépendances du projet
└── README.md              # Documentation du projet
```

## Instructions d'Installation

1. **Cloner le dépôt** :
   ```
   git clone <URL-du-depot>
   cd ml-classification-project
   ```

2. **Créer un environnement virtuel** (optionnel mais recommandé) :
   ```
   python -m venv venv
   source venv/bin/activate  # Sur Windows, utilisez `venv\Scripts\activate`
   ```

3. **Installer les dépendances requises** :
   ```
   pip install -r requirements.txt
   ```

## Utilisation

### Entraîner le Modèle

Pour entraîner le modèle, exécutez la commande suivante :
```
python src/train.py
```

Ce script chargera le jeu de données traité, entraînera le modèle et sauvegardera le modèle entraîné sur le disque.

### Évaluer le Modèle

Pour évaluer le modèle entraîné, utilisez la commande suivante :
```
python src/evaluate.py
```

Ce script chargera le modèle entraîné et le jeu de données de test, fera des prédictions et calculera les métriques d'évaluation.

## Jeu de Données

- Placez vos fichiers de données bruts dans le répertoire `data/raw`.
- Après traitement, les fichiers doivent être placés dans le répertoire `data/processed`.

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.
