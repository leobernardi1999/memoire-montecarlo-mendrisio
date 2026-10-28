# Annexe numérique — modèle de Monte-Carlo

Ce dépôt contient le script utilisé pour exécuter le modèle de Monte-Carlo du
scénario central présenté dans le mémoire.

## Contenu

- `montecarlo.py` : lecture des données, création du réseau, calcul des
  itinéraires, recalcul de la pente avec swissALTI3D, simulations et création
  des résultats TOP1 et TOP3 ;
- `requirements.txt` : principales bibliothèques nécessaires ;
- `.gitignore` : empêche l'ajout des données et des résultats confidentiels.

## Données nécessaires

Le script cherche dans son dossier :

1. un fichier `montecarlo.gpkg`, avec les couches `rete_strade`, `abitazioni`
   et `poi` ;
2. un dossier `swissALTI3D` contenant les fichiers d'altitude GeoTIFF 2022.

Si les fichiers swissALTI3D sont conservés ailleurs, leur dossier peut être
indiqué au moment de lancer le script avec `SWISSALTI3D_DIR`.

Le GeoPackage et les fichiers contenant les domiciles ne sont pas publiés afin
de protéger les données individuelles.

## Calcul de la pente

La pente est recalculée à partir de swissALTI3D. L'ancien indicateur du
GeoPackage est conservé seulement comme valeur de sécurité lorsqu'une route ne
possède aucune altitude valide. Le programme s'arrête si un itinéraire retenu
utilise cette valeur. Ce cas ne se produit pas dans les résultats définitifs.

## Paramètres principaux

- vitesse de marche : 1,2 m/s ;
- durée maximale : 15 minutes ;
- simulations : 1 000 ;
- graine aléatoire : 42 ;
- répartition entre proximité et qualité : 30 % et 70 % ;
- pas maximal utilisé pour lire les altitudes : 20 m.

Les quatre poids de qualité sont tirés avec une loi Dirichlet(1,1,1,1).

## Exécution

Le script demande un environnement Python géographique compatible avec GDAL,
Fiona, Rasterio et GeoPandas. Il peut notamment être exécuté avec Python dans
QGIS.

```text
python montecarlo.py
```

Les fichiers créés sont :

- `risultati_top1.csv` ;
- `risultati_top3.csv` ;
- `risultati_score_tutti_poi.csv`.

Ces résultats ne sont pas inclus dans le dépôt.
