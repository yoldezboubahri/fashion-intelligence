

#  Fashion Intelligence

### Du comportement d'achat à la composition de collection
*Un pipeline de Machine Learning en trois modèles*

Projet de Machine Learning · E-commerce & Mode<br>
Réalisé par **Yoldez Boubahri**

Données : *Amazon Reviews 2023* (McAuley Lab) · catégorie *Clothing, Shoes & Jewelry* · **24 M d'avis**


---

##  Problématique

> **Peut-on, à partir des seuls avis clients, identifier à l'avance les articles qui vont gagner en popularité tout en restant appréciés, et en composer une collection plus performante qu'un choix fondé sur des règles simples ?**

**L'idée.** Chaque avis client contient deux signaux :

| Signal | Ce qu'il mesure |
|---|---|
| La **note** (1 à 5) | la **satisfaction** |
| Le **nombre d'avis** dans le temps | la **demande** (on laisse un avis après un achat) |

Un article très vendu aujourd'hui ne sera pas forcément demandé demain, et un article populaire n'est pas forcément apprécié. Le projet combine les deux signaux.

---

##  Démarche

```
 1. DONNÉES          2. MODÈLE 1            3. MODÈLE 2             4. MODÈLE 3
 Nettoyer et   ──►   Prédire la note   ──►  Prédire les       ──►  Composer la collection
 explorer            (satisfaction)         tendances (demande)     de 100 articles
```

| Étape | Question | Résultat |
|---|---|---|
| **1. Données** | Les données sont-elles fiables ? | 24 M d'avis nettoyés (doublons, notes invalides, lignes incomplètes) |
| **2. Modèle 1** | Quelle note un client donnera-t-il ? | Note attendue de chaque article |
| **3. Modèle 2** | Quels articles vont décoller dans les 3 mois ? | Probabilité de tendance |
| **4. Modèle 3** | Quels 100 articles choisir ? | Collection finale, vérifiée sur une période réelle |

---

##  Les trois modèles

### Modèle 1 : Préférences
**Question :** quelle note un client donnera-t-il à un article ?
**Méthode :** biais régularisés (client + article), puis SVD pour les goûts latents.

| Modèle | MAE | AUC |
|---|---|---|
| Moyenne globale | 1,014 | 0,500 |
| **Biais** (retenu) | **0,908** | **0,693** |
| Biais + SVD | 0,908 | 0,693 |

 Les biais réduisent l'erreur de **10 %**. La SVD n'apporte rien de plus : le modèle le plus simple suffit.

### Modèle 2 : Tendances
**Question :** quels articles vont devenir plus populaires dans les 3 prochains mois ?
**Définition :** un article est *en tendance* s'il croît au moins **1,5 fois plus vite que le marché** et reçoit au moins 5 avis.
**Méthode :** régression logistique et Gradient Boosting, entraînés avant 2022 et testés sur 2022-2023.

| Modèle | AUC | Vraies tendances dans le top 100 |
|---|---|---|
| Hasard | 0,50 | 10 % |
| Règle naïve (croissance récente) | 0,47 | 13 % |
| Régression logistique | 0,76 | 60 % |
| **Gradient Boosting** | **0,80** | **74 %** |

 La règle « ce qui monte continue de monter » fait **moins bien que le hasard**. Le volume d'avis sur 12 mois est la variable la plus utile.

### Modèle 3 : Collection
**Question :** quels 100 articles mettre en avant ?
**Score :** `P(tendance) × (note attendue − 1) / 4`
**Méthode :** sélection gloutonne, avec une note attendue ≥ 4 et au plus 3 articles par marque.

---

##  Résultats clés

**Backtest** (2022-2023) : la collection est comparée à des règles qu'un acheteur peut appliquer sans ML.

| Stratégie | Vraies tendances | Croissance vs marché | Note réelle |
|---|---|---|---|
| Hasard | 10 % | ×1,31 | 4,19 |
| Les plus populaires | 4 % | ×0,76 | 4,28 |
| Les mieux notés | 10 % | ×1,14 | 4,81 |
| **Notre score** | **70 %** | **×3,17** | **4,47** |

**Vérification sur la réalité** : prédiction faite fin février 2023, résultats observés de mars à mai 2023 → **81 articles sur 100** sont réellement entrés en tendance (contre 10 au hasard).

**Réponse à la problématique : oui.** La collection contient 7 à 8 fois plus d'articles en tendance qu'un choix au hasard, avec une note réelle supérieure. Les best-sellers, eux, régressent.

---

##  Limites et perspectives

| Limites | Perspectives |
|---|---|
| Le nombre d'avis remplace les ventes réelles | Ajouter prix, catégorie et marque |
| Marque et prix non exploités | Traiter les nouveaux articles (texte, images) |
| Modèle 1 entraîné sur 300 000 clients | Personnaliser par segment de clients |
| Fin des données 2023 incomplète | Optimisation en nombres entiers |

---

##  Lancer le projet

**1. Données** : télécharge-les sur [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/) et place-les dans `data/` (non versionnées, trop volumineuses) :
- `amazon_reviews.csv.gz`
- `meta_Clothing_Shoes_and_Jewelry.jsonl.gz` (facultatif)

**2. Installation**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook projet_ml_ecommerce_mode.ipynb
```

> Dans la cellule de configuration du notebook, mets `DOSSIER = "data"` pour une exécution en local. Le notebook fonctionne aussi sur Google Colab. En cas de manque de mémoire, réduis `N_USERS_M1`.

---

##  Structure

```
.
├── projet_ml_ecommerce_mode.ipynb   # analyse complète
├── data/                            # données (ignorées par Git)
├── requirements.txt
├── .gitignore
└── README.md
```

##  Référence principale

Hou, Y., et al. (2024). *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952. Liste complète des références dans la section 13 du notebook.
