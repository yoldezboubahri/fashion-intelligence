# Fashion Intelligence

**Du comportement d'achat à la composition de collection : un pipeline de Machine Learning en trois modèles.**

Projet de Machine Learning · E-commerce & Mode — réalisé par **Yoldez Boubahri**.

## Problématique

> Peut-on, à partir des seuls avis clients, identifier à l'avance les articles qui vont gagner en popularité tout en restant appréciés, et en composer une collection plus performante qu'un choix fondé sur des règles simples ?

## Pipeline

| Modèle | Rôle | Sortie |
|---|---|---|
| **1. Préférences** | Mesurer la satisfaction (filtrage collaboratif, SVD) | Note attendue de chaque article |
| **2. Tendances** | Anticiper la demande (régression logistique, gradient boosting) | Probabilité qu'un article décolle dans les 3 mois |
| **3. Collection** | Décider sous contraintes | Sélection de 100 articles (max 3 par marque, note attendue ≥ 4) |

## Données

*Amazon Reviews 2023* (McAuley Lab), catégorie **Clothing, Shoes & Jewelry**.

Les fichiers de données ne sont pas versionnés (trop volumineux). Place-les dans le dossier `data/` :

- `amazon_reviews.csv.gz` — colonnes `user_id`, `parent_asin`, `rating`, `timestamp`
- `meta_Clothing_Shoes_and_Jewelry.jsonl.gz` — métadonnées (facultatif)

Source : https://amazon-reviews-2023.github.io/

> Dans le notebook, adapte la variable `DOSSIER` (cellule de configuration) à `"data"` si tu exécutes en local.

## Installation et exécution

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook projet_ml_ecommerce_mode.ipynb
```

Le notebook est conçu pour Google Colab (montage de Drive) mais fonctionne aussi en local.

## Structure

```
.
├── projet_ml_ecommerce_mode.ipynb
├── data/                # données (ignorées par Git)
├── requirements.txt
└── README.md
```
