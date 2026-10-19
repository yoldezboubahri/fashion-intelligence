# Fashion Intelligence

**Du comportement d'achat à la composition de collection : un pipeline de Machine Learning en trois modèles.**

Projet de Machine Learning · E-commerce & Mode — réalisé par **Yoldez Boubahri**.

Sommaire
Contexte
Problématique et objectifs
Données
Pipeline
Méthodologie et résultats
Résultats clés
Limites et perspectives
Installation et exécution
Structure
Références
Contexte

Chaque saison, une enseigne de mode doit décider quels articles mettre en avant. Ce choix repose souvent sur l'intuition des acheteurs ou sur les ventes passées, alors que :

un article qui se vend bien aujourd'hui n'est pas forcément celui qui se vendra demain ;
un article populaire n'est pas forcément un article apprécié.

Les avis clients portent les deux signaux à la fois : la note (1 à 5) mesure la satisfaction, et le nombre d'avis dans le temps est un indicateur de la demande (un client laisse un avis après un achat).

Problématique et objectifs

Peut-on, à partir des seuls avis clients, identifier à l'avance les articles qui vont gagner en popularité tout en restant appréciés, et en composer une collection plus performante qu'un choix fondé sur des règles simples ?

#	Objectif	Critère de réussite
O1	Nettoyer et comprendre les données	Données fiables et exploitables
O2	Prédire la note qu'un client donnera à un article	Erreur plus faible que la moyenne globale
O3	Prédire quels articles seront en tendance dans les 3 mois	Meilleur classement que le hasard
O4	Composer une collection de 100 articles	Meilleure que les best-sellers, avec une bonne note
O5	Vérifier les prédictions sur une période réelle	Comparaison prédiction / réalité article par article
Données

Amazon Reviews 2023 (McAuley Lab), catégorie Clothing, Shoes & Jewelry : environ 24,3 millions d'avis nettoyés, 2,5 millions d'utilisateurs et 715 000 articles.

Variable	Type	Description
user_id	qualitative	identifiant anonyme du client
parent_asin	qualitative	identifiant de l'article (tailles et couleurs regroupées)
rating	ordinale	note de 1 à 5 étoiles
timestamp	quantitative	date de l'avis, en millisecondes depuis 1970

Les fichiers de données ne sont pas versionnés (trop volumineux). Place-les dans le dossier data/ :

amazon_reviews.csv.gz — colonnes user_id, parent_asin, rating, timestamp
meta_Clothing_Shoes_and_Jewelry.jsonl.gz — métadonnées marque/prix (facultatif)

Source : https://amazon-reviews-2023.github.io/

Dans le notebook, adapte la variable DOSSIER (cellule de configuration) à "data" si tu exécutes en local.

Pipeline
Modèle	Rôle	Sortie
1. Préférences	Mesurer la satisfaction (biais régularisés, SVD)	Note attendue de chaque article
2. Tendances	Anticiper la demande (régression logistique, gradient boosting)	Probabilité qu'un article décolle dans les 3 mois
3. Collection	Décider sous contraintes (sélection gloutonne)	Sélection de 100 articles (max 3 par marque, note attendue ≥ 4)
Avis clients ──► Modèle 1 : note attendue ───────┐
                                                 ├──► Modèle 3 : collection de 100 articles
Avis clients ──► Modèle 2 : P(tendance) ─────────┘
Méthodologie et résultats
1. Nettoyage et exploration
Doublons exacts supprimés (~1,16 M, soit 4,8 % des lignes).
Notes invalides (-1, 0, 6, 7, 10, 99…) remplacées par une valeur manquante, sans supprimer la ligne : elle reste utile pour compter les volumes d'avis (modèle 2).
Lignes sans utilisateur, article ou date supprimées.
Constats clés : 65,7 % des avis sont des 5 étoiles (note moyenne 4,29), la matrice utilisateurs × articles est vide à 99,999 %, la saisonnalité est marquée (décembre ×1,25 et janvier ×1,15), et les données de fin 2023 sont incomplètes.
2. Modèle 1 : prédiction des préférences

Modèle : note = μ + b_client + b_article + p_client · q_article, avec des biais régularisés (λ) et une SVD tronquée (k facteurs latents). Entraîné sur un échantillon de 300 000 clients. Découpage temporel par client : dernier avis = test, avant-dernier = validation, le reste = entraînement.

Modèle	RMSE	MAE	AUC (avis ≥ 4)
Moyenne globale	1,303	1,014	0,500
Biais (λ = 5)	1,231	0,908	0,693
Biais + SVD	1,231	0,908	0,693
Les biais réduisent la MAE de 10 % par rapport à la moyenne.
La SVD n'apporte rien de plus sur ces données (notes concentrées sur 5 étoiles, clients peu actifs) : le modèle le plus simple est retenu.
En ranking (HitRate@10), la popularité (61,9 %) bat le modèle à biais (22 %) et le hasard (10 %) : prédire la note donnée et prédire le choix d'achat sont deux problèmes différents.
3. Modèle 2 : prévision des tendances

Un article est en tendance si, sur les 3 mois suivants, il reçoit au moins 1,5 fois la croissance moyenne du marché et au moins 5 avis. La cible est relative au marché pour ne pas confondre tendance et saisonnalité.

169 689 articles suivis (≥ 20 avis depuis 2015) sur 104 mois.
Entraînement avant 2022, test 2022-2023. Environ 10 % d'articles en tendance (classes déséquilibrées, donc AUC / AP plutôt que l'accuracy).
Variables : volume récent (1, 3, 12 mois, en log), croissance 3 mois, part de marché, ancienneté, note moyenne 6 mois, mois codé sur un cercle (sin/cos).
Modèle	AUC	AP	Précision des 100 premiers
Hasard	0,50	0,10	10 %
Règle naïve (croissance récente)	0,47	0,10	13 %
Régression logistique	0,76	0,31	60 %
Gradient Boosting	0,80	0,36	74 %
La règle naïve fait moins bien que le hasard : un article qui vient de croître tend à redescendre (retour à la moyenne).
Importance par permutation : le volume sur 12 mois domine, puis le volume sur 3 mois. La note n'a aucun effet sur la tendance, ce qui justifie de combiner les deux signaux.
4. Modèle 3 : composition de la collection

Score d'un article :

score = P(tendance) × (note attendue − 1) / 4

Sélection gloutonne des 100 meilleurs scores, avec contraintes métier : note attendue ≥ 4 et au plus 3 articles par marque (si les métadonnées sont disponibles). Les modèles sont recalculés à chaque date avec les seules données connues à cette date, pour ne jamais utiliser le futur.

Backtest (plusieurs dates de 2022 à 2023, comparé à des règles qu'un acheteur pourrait appliquer sans ML) :

Stratégie	% de vraies tendances	Croissance vs marché	Note réelle
Hasard	10 %	×1,31	4,19
Les plus populaires	4 %	×0,76	4,28
Les mieux notés	10 %	×1,14	4,81
Notre score	70 %	×3,17	4,47

Les best-sellers régressent : suivre les ventes passées est la pire stratégie pour anticiper. Les mieux notés satisfont le plus mais leur demande n'augmente pas.

Vérification prédiction contre réalité (prédiction fin février 2023, période mars-mai 2023) :

Stratégie	Vraies tendances sur 100	Croissance médiane vs marché	Note réelle
Hasard	10	×1,10	4,29
Les plus populaires	1	×0,66	4,31
Ancienne méthode (proba × note)	78	×2,22	4,44
Nouvelle méthode (proba, note ≥ 4)	81	×2,63	4,32

Le choix entre les deux dernières méthodes est un arbitrage métier : volume de demande ou satisfaction.

Recommandation : une collection de 100 articles pour septembre-novembre 2023 (parmi 88 307 candidats) est exportée dans collection_recommandee.csv à l'exécution du notebook.

Résultats clés
Objectif	Résultat
O1 : données	24,3 M d'avis nettoyés ; notes très positives, matrice creuse, saisonnalité de fin d'année
O2 : préférences	MAE réduite de 10 % vs la moyenne, AUC 0,69 ; la SVD n'apporte rien de plus
O3 : tendances	Gradient Boosting : AUC 0,80, 74 % de vraies tendances dans le top 100 (10 % au hasard)
O4 : collection	Backtest : 70 % de vraies tendances, croissance ×3,2 vs marché, note réelle 4,47
O5 : vérification	Mars-mai 2023 : 81 articles sur 100 réellement en tendance

Réponse à la problématique : oui. À partir des seuls avis clients, la collection proposée contient 7 à 8 fois plus d'articles en tendance qu'un choix au hasard, tout en restant mieux notée.

Limites et perspectives

Limites

Le nombre d'avis est un indicateur des ventes, pas les ventes elles-mêmes.
Les métadonnées (marque, prix, catégorie) n'ont pas été utilisées : la contrainte de diversité par marque n'a pas pu s'appliquer.
Le modèle 1 est entraîné sur un échantillon de 300 000 clients (contrainte de mémoire).
Les articles de moins de 6 mois ne sont pas candidats, faute d'historique.
La fin des données (2023) est incomplète, ce qui limite la vérification de la dernière collection.
La croissance est mesurée relativement au marché : environ 55 des 81 articles vérifiés ont aussi progressé en valeur absolue.

Perspectives

Intégrer les métadonnées : catégorie et prix comme variables, contraintes de budget et de diversité.
Traiter les nouveaux articles à partir de leur description (texte, images).
Personnaliser la collection par segment de clients.
Remplacer la sélection gloutonne par une optimisation linéaire en nombres entiers.
Vérifier la collection septembre-novembre 2023 avec une version plus récente des données.
Installation et exécution
bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook projet_ml_ecommerce_mode.ipynb

Le notebook est conçu pour Google Colab (montage de Drive) mais fonctionne aussi en local. Le chargement et la SVD sont gourmands en mémoire : prévoir une machine avec suffisamment de RAM, ou réduire N_USERS_M1 dans la cellule de configuration.

Structure
.
├── projet_ml_ecommerce_mode.ipynb   # analyse complète (3 modèles)
├── data/                            # données (ignorées par Git)
├── requirements.txt
├── .gitignore
└── README.md
Références
Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952.
Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. Computer, 42(8).
Cremonesi, P., Koren, Y., & Turrin, R. (2010). Performance of Recommender Algorithms on Top-N Recommendation Tasks. RecSys.
Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5).
Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3e éd.). OTexts.
Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12.

(Liste complète dans la section 13 du notebook.)
