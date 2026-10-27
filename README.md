# Annexe numérique — modèle de Monte-Carlo

Ce dépôt contient les scripts utilisés pour l'analyse de réseau, le modèle de
Monte-Carlo et l'analyse de robustesse relative à la pondération entre
proximité et qualité.

## Contenu

- `montecarlo.py` : construction du graphe, calcul des itinéraires, recalcul de
  la pente depuis swissALTI3D, simulations et sorties TOP1/TOP3 ;
- `analyse_sensibilite_proximite_qualite.py` : scénarios 0/100, 30/70, 50/50,
  70/30 et 100/0, ainsi que diagnostics de stabilité ;
- `requirements.txt` : principales dépendances Python.

Lors de l'exécution de l'analyse de robustesse, le dossier `data/` doit être
créé par l'utilisateur pour y placer les entrées protégées. Le dossier
`outputs/` est créé automatiquement par le script.

## Données attendues

Le script principal attend, dans ce dossier :

1. `montecarlo.gpkg`, avec les couches `rete_strade`, `abitazioni` et `poi` ;
2. un sous-dossier `swissALTI3D` contenant les tuiles GeoTIFF 2022, ou la
   variable d'environnement `SWISSALTI3D_DIR` indiquant leur emplacement.

Les champs utilisés sont `DTV_FZG`, `marcia`, `green_score` et `slope_score`
pour `rete_strade` ; `id` et `quartiere` pour `abitazioni` ; et `poi_id`,
`type` et `cat_funz` pour `poi`. Chaque couche doit posséder une géométrie
valide dans le même système de coordonnées projeté.

Le script d'analyse de robustesse attend :

1. `data/routes_recalculated_20m.csv` ;
2. `data/declared_pairs.csv`.

Le premier fichier est la table intermédiaire des itinéraires recalculés. Il
doit contenir les colonnes `casa_id`, `quartiere`, `poi_id`, `poi_name`,
`cat_funz`, `tempo_min`, `length_m`, `traf_mean`, `marcia_mean`, `green_mean`
et `slope_mean`. Le second associe chaque enfant à ses destinations déclarées
au moyen des deux colonnes `casa_id` et `poi_id`.

Les fichiers individualisés ne sont pas inclus dans cette annexe en raison de
la confidentialité des domiciles et des réponses. Les identifiants sont
pseudonymisés dans les fichiers de travail.

## Exécution

Le script principal nécessite un environnement géospatial dans lequel GDAL,
Fiona, Rasterio et GeoPandas sont compatibles. L'environnement Python fourni
avec QGIS LTR sur macOS a été utilisé pour le calcul définitif.

```text
python montecarlo.py
python analyse_sensibilite_proximite_qualite.py
```

Le premier script produit `risultati_top1.csv`, `risultati_top3.csv` et
`risultati_score_tutti_poi.csv`. Le second produit les tableaux agrégés des
cinq scénarios et les diagnostics de stabilité dans
`outputs/sensibilite_modele/`. Son exécution suppose que les deux fichiers
protégés décrits ci-dessus ont été préparés dans `data/`.

Les paramètres centraux sont une vitesse de 1,2 m/s, un seuil de 15 minutes,
1 000 simulations, une graine égale à 42 et une loi
Dirichlet(1,1,1,1). La spécification centrale combine 30 % de proximité et 70 %
de qualité.

## Confidentialité et reproductibilité

Le code, les paramètres et la structure des entrées sont diffusables. Les
couches contenant les domiciles et les associations enfant–destination ne le
sont pas. Le paquet permet donc d'auditer la méthode et de la réexécuter avec
les données protégées conservées par l'auteur, ou avec des données de même
structure obtenues légalement ; il ne permet pas, à lui seul, de reproduire
les résultats individualisés du mémoire.
