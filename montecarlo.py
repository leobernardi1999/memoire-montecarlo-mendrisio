# ÉTAPE 1 - Chargement des données
# ============================================================
# Cette partie charge les fichiers nécessaires et contrôle que les données
# principales sont disponibles avant de commencer les calculs.
import math  # Pour calculer les angles de pente.
import os  # Pour travailler avec les dossiers et les chemins des fichiers.
import tempfile  # Pour créer un dossier temporaire.

# Indique où trouver PROJ (outil qui permet de comprendre un système de
# coordonnées comme EPSG:2056). Ce chemin correspond à QGIS sur macOS.
# Si QGIS se trouve ailleurs, seul le chemin entre guillemets doit être adapté.
CARTELLA_PROJ_QGIS = "/Applications/QGIS-LTR.app/Contents/Resources/proj"
if "PROJ_LIB" not in os.environ and os.path.isdir(CARTELLA_PROJ_QGIS):
    os.environ["PROJ_LIB"] = CARTELLA_PROJ_QGIS  # Utilise ce chemin s'il existe.

import fiona  # Lit la liste des couches du GeoPackage.
import geopandas as gpd  # Pour travailler avec les données géographiques.
import networkx as nx  # Pour créer le réseau routier et calculer les parcours.
import numpy as np  # Pour les calculs et les tirages aléatoires.
import pandas as pd  # Pour travailler avec les tableaux.
import rasterio  # Pour lire les altitudes de swissALTI3D.

from osgeo import gdal  # Pour réunir les fichiers d'altitude sans les modifier.
from shapely.geometry import LineString, MultiLineString  # Pour lire les routes.
from scipy.spatial import cKDTree  # Pour trouver rapidement le nœud le plus proche.

CARTELLA_SCRIPT = os.path.dirname(os.path.abspath(__file__))  # Dossier de ce script.

# Par défaut, le GeoPackage doit être à côté du script et porter ce nom.
# S'il se trouve ailleurs, ce chemin doit être remplacé par son chemin complet.
PERCORSO_GPKG = os.path.join(CARTELLA_SCRIPT, "montecarlo.gpkg")

# Par défaut, les fichiers swissALTI3D sont cherchés dans un dossier
# "swissALTI3D" placé dans le même dossier que le script.
# S'ils restent dans un autre dossier, son chemin doit être indiqué au moment
# de lancer le script.
CARTELLA_DEM = os.environ.get(
    "SWISSALTI3D_DIR",
    os.path.join(CARTELLA_SCRIPT, "swissALTI3D")
)

# Les routes ne sont pas coupées tous les 20 m. Cette valeur sert seulement à
# décider tous les combien de mètres les altitudes sont lues.
PASSO_MASSIMO_PENDENZA_M = 20.0

# Le programme s'arrête si le GeoPackage n'est pas trouvé.
if not os.path.exists(PERCORSO_GPKG):
    raise FileNotFoundError(f"GeoPackage non trovato: {PERCORSO_GPKG}")

# Ces lignes affichent le fichier effectivement utilisé.
print("GeoPackage trovato:")
print(PERCORSO_GPKG)

# Vérifie les couches disponibles dans le GeoPackage.
livelli = fiona.listlayers(PERCORSO_GPKG)
print("\nLayer presenti:")
print(livelli)

# Ces trois noms doivent correspondre aux couches du GeoPackage.
# Avec un autre GeoPackage, les noms entre guillemets sont à adapter ici.
rete = gpd.read_file(PERCORSO_GPKG, layer="rete_strade")  # Réseau routier.
abitazioni = gpd.read_file(PERCORSO_GPKG, layer="abitazioni")  # Points de départ des domiciles.
poi = gpd.read_file(PERCORSO_GPKG, layer="poi")  # Points d'intérêt.

# Le score de trafic utilise le logarithme naturel. Dans ce script, les valeurs
# DTV absentes sont considérées comme nulles.
# La colonne DTV_FZG contient le trafic journalier moyen de chaque route.
# Dans ces données, la valeur maximale doit être 16495.
# Ce contrôle permet de repérer si un autre fichier a été chargé par erreur.
DTV_MASSIMO = float(pd.to_numeric(rete["DTV_FZG"], errors="coerce").max())
if DTV_MASSIMO != 16495.0:
    raise ValueError(f"Valore massimo DTV inatteso: {DTV_MASSIMO}")

dtv = (
    pd.to_numeric(rete["DTV_FZG"], errors="coerce")
    .fillna(0.0)
    .clip(lower=0.0)
)
# Plus le trafic augmente, plus le score se rapproche de 0.
rete["traf_score"] = 1.0 - np.log1p(dtv) / np.log1p(DTV_MASSIMO)

# Résumé simple pour contrôler le nombre d'éléments chargés.
print("\nElementi caricati:")
print(f"Rete stradale: {len(rete)} archi")
print(f"Abitazioni: {len(abitazioni)} punti")
print(f"POI: {len(poi)} punti")

# Affiche les systèmes de coordonnées pour vérifier qu'ils sont compatibles.
print("\nCRS:")
print(f"Rete: {rete.crs}")
print(f"Abitazioni: {abitazioni.crs}")
print(f"POI: {poi.crs}")

# ============================================================
# ÉTAPE 2 - Recalcul de la pente et création du graphe
# ============================================================
# Utilise les routes chargées avant et les altitudes swissALTI3D pour calculer
# la pente. Le résultat est ensuite transformé en réseau de nœuds et de routes.

# Vitesse de marche retenue (m/s).
VELOCITA_CAMMINO = 1.2


def iter_lines(geometria):
    """Permet de lire de la même façon une route formée d'une ou de plusieurs lignes."""
    if isinstance(geometria, LineString):
        yield geometria
    elif isinstance(geometria, MultiLineString):
        yield from geometria.geoms


def build_edge_records(strade):
    """Prépare chaque morceau de route pour construire le graphe."""
    dati_archi = []

    # Parcourt toutes les routes et leurs coordonnées.
    for _, riga in strade.iterrows():
        for linea in iter_lines(riga.geometry):
            coordinate = list(linea.coords)

            # Deux coordonnées consécutives forment un morceau de route.
            for indice in range(len(coordinate) - 1):
                inizio = tuple(coordinate[indice][:2])
                fine = tuple(coordinate[indice + 1][:2])
                lunghezza_m = float(LineString([inizio, fine]).length)

                if lunghezza_m <= 0:
                    continue

                # Chaque morceau garde les facteurs de qualité de sa route d'origine.
                dati_archi.append({
                    "start": inizio,
                    "end": fine,
                    "length_m": lunghezza_m,
                    "traf_score": float(riga["traf_score"]),
                    "marcia": float(riga["marcia"]),
                    "green_score": float(riga["green_score"]),
                    # Ancien score utilisé seulement si une altitude manque.
                    "old_slope_score": float(riga["slope_score"])
                })

    return dati_archi


def recalculate_slope_scores(dati_archi, raster_quote, passo_massimo_m):
    """
    Calcule la pente de chaque route avec les altitudes swissALTI3D.
    Pour lire les altitudes, chaque route est considérée par parties égales de
    20 m au maximum. Une valeur proche de 1 indique une route plate et une
    valeur proche de 0 une forte pente.
    """
    # Points où les altitudes devront être lues.
    coordinate_quote = []
    # Pour chaque partie, garde la route, la longueur et les deux points utiles.
    parti_da_calcolare = []

    for indice_arco, arco in enumerate(dati_archi):
        lunghezza_m = arco["length_m"]
        # Nombre de parties nécessaire pour ne jamais dépasser 20 m.
        numero_parti = max(
            1,
            int(math.ceil(lunghezza_m / passo_massimo_m))
        )
        lunghezza_parte = lunghezza_m / numero_parti  # Longueur réelle d'une partie.
        x_inizio, y_inizio = arco["start"]  # Début du morceau de route.
        x_fine, y_fine = arco["end"]  # Fin du morceau de route.

        # Calcule les coordonnées du début et de la fin de chaque partie.
        for numero in range(numero_parti):
            frazione_inizio = numero / numero_parti  # Début entre 0 et 1.
            frazione_fine = (numero + 1) / numero_parti  # Fin entre 0 et 1.

            # Garde la position du premier point dans la liste des coordonnées.
            indice_inizio = len(coordinate_quote)
            # Place le début de cette partie le long du morceau de route.
            # Une fraction de 0 correspond au début et une fraction de 1 à la fin.
            coordinate_quote.append((
                x_inizio + (x_fine - x_inizio) * frazione_inizio,
                y_inizio + (y_fine - y_inizio) * frazione_inizio
            ))

            # Garde ensuite la position du deuxième point dans la même liste.
            indice_fine = len(coordinate_quote)
            # Place la fin de cette partie de la même manière.
            coordinate_quote.append((
                x_inizio + (x_fine - x_inizio) * frazione_fine,
                y_inizio + (y_fine - y_inizio) * frazione_fine
            ))

            # Mémorise à quelle route appartient cette partie, sa longueur et
            # les deux positions qui permettront de retrouver ses altitudes.
            parti_da_calcolare.append((
                indice_arco,
                lunghezza_parte,
                indice_inizio,
                indice_fine
            ))

    # Lit toutes les altitudes en une seule fois.
    quote_lette = np.asarray(
        [valore[0] for valore in raster_quote.sample(coordinate_quote)],
        dtype=float
    )

    # Les points sans altitude valide ne sont pas utilisés dans la moyenne.
    if raster_quote.nodata is not None:
        quote_lette[
            np.isclose(quote_lette, raster_quote.nodata)
        ] = np.nan

    # Sommes utilisées pour calculer la pente moyenne de chaque route.
    somma_angoli_pesata = np.zeros(len(dati_archi), dtype=float)  # Angles et longueurs.
    somma_lunghezze_valide = np.zeros(len(dati_archi), dtype=float)  # Parties avec deux altitudes.
    somma_lunghezze_totali = np.zeros(len(dati_archi), dtype=float)  # Toutes les parties.

    for indice_arco, lunghezza_m, indice_inizio, indice_fine in parti_da_calcolare:
        somma_lunghezze_totali[indice_arco] += lunghezza_m
        quota_inizio = quote_lette[indice_inizio]  # Altitude au début de la partie.
        quota_fine = quote_lette[indice_fine]  # Altitude à la fin de la partie.

        if not (np.isfinite(quota_inizio) and np.isfinite(quota_fine)):
            continue

        # Angle calculé avec la différence d'altitude et la longueur.
        angolo_gradi = math.degrees(
            math.atan(abs(quota_fine - quota_inizio) / lunghezza_m)
        )
        somma_angoli_pesata[indice_arco] += angolo_gradi * lunghezza_m
        somma_lunghezze_valide[indice_arco] += lunghezza_m

    # Moyenne des angles en donnant plus de poids aux parties plus longues.
    angoli_medi = np.divide(
        somma_angoli_pesata,
        somma_lunghezze_valide,
        out=np.full(len(dati_archi), np.nan, dtype=float),
        where=somma_lunghezze_valide > 0
    )

    if not np.isfinite(angoli_medi).any():
        raise ValueError("Nessuna quota swissALTI3D valida sulla rete")

    angolo_massimo = float(np.nanmax(angoli_medi))
    if angolo_massimo <= 0:
        raise ValueError("Pendenza massima non valida")

    # Transforme les angles en scores compris entre 0 et 1.
    score_pendenze = 1.0 - angoli_medi / angolo_massimo
    # Part de chaque route pour laquelle une altitude valide a été trouvée.
    copertura_dem = np.divide(
        somma_lunghezze_valide,
        somma_lunghezze_totali,
        out=np.zeros(len(dati_archi), dtype=float),
        where=somma_lunghezze_totali > 0
    )

    return score_pendenze, copertura_dem, angolo_massimo


# Transforme rete en morceaux de route et conserve leurs facteurs de qualité.
dati_archi = build_edge_records(rete)

# Le VRT est un fichier temporaire qui permet de lire toutes les tuiles
# swissALTI3D comme une seule image, sans modifier les fichiers originaux.
# Seuls les fichiers .tif dont le nom commence par "swissalti3d_2022_" sont lus.
# Si les fichiers portent un autre nom, ce début de nom doit être adapté ici.
tessere_dem = sorted([
    os.path.join(CARTELLA_DEM, nome_file)
    for nome_file in os.listdir(CARTELLA_DEM)
    if nome_file.startswith("swissalti3d_2022_") and nome_file.endswith(".tif")
]) if os.path.isdir(CARTELLA_DEM) else []

if not tessere_dem:
    raise FileNotFoundError(
        "Nessun raster swissALTI3D trovato. Inserire le tessere nella cartella "
        f"{CARTELLA_DEM} oppure definire SWISSALTI3D_DIR."
    )

# Le dossier et le VRT temporaires sont supprimés automatiquement à la fin.
with tempfile.TemporaryDirectory() as cartella_temporanea:
    percorso_vrt = os.path.join(cartella_temporanea, "swissalti3d_mosaic.vrt")
    mosaico_vrt = gdal.BuildVRT(percorso_vrt, tessere_dem)

    if mosaico_vrt is None:
        raise RuntimeError("Impossibile creare il mosaico virtuale swissALTI3D")

    mosaico_vrt.FlushCache()
    mosaico_vrt = None

    with rasterio.open(percorso_vrt) as raster_quote:
        # Les altitudes et les routes doivent utiliser le même système de coordonnées.
        if rete.crs is not None and raster_quote.crs != rete.crs:
            raise ValueError(
                "Il CRS dei raster swissALTI3D non coincide con quello della rete"
            )

        score_pendenze, copertura_dem, angolo_massimo_pendenza = recalculate_slope_scores(
            dati_archi,
            raster_quote,
            PASSO_MASSIMO_PENDENZA_M
        )

# Chaque route peut être parcourue dans les deux sens.
grafo = nx.Graph()
archi_senza_quote = 0

# Ajoute au réseau chaque morceau de route et ses caractéristiques.
for indice_arco, arco in enumerate(dati_archi):
    if np.isfinite(score_pendenze[indice_arco]):
        score_pendenza = float(score_pendenze[indice_arco])
        fonte_pendenza = "swissALTI3D"
    else:
        # Utilise l'ancien score seulement si swissALTI3D ne donne rien.
        score_pendenza = arco["old_slope_score"]
        fonte_pendenza = "fallback_vecchio_indicatore"
        archi_senza_quote += 1

    grafo.add_edge(
        arco["start"],
        arco["end"],
        length_m=arco["length_m"],
        time_min=arco["length_m"] / VELOCITA_CAMMINO / 60,  # Temps à pied en minutes.
        traf_score=arco["traf_score"],
        marcia=arco["marcia"],
        green_score=arco["green_score"],
        slope_score=score_pendenza,
        copertura_dem=float(copertura_dem[indice_arco]),
        slope_source=fonte_pendenza
    )

# Affiche les contrôles sur les altitudes et le réseau obtenu.
print("\nPendenza ricostruita da swissALTI3D:")
print(f"Tessere DEM: {len(tessere_dem)}")
print(f"Passo massimo di campionamento: {PASSO_MASSIMO_PENDENZA_M:.0f} m")
print(f"Angolo medio massimo osservato: {angolo_massimo_pendenza:.6f}°")
print(f"Archi senza alcuna quota valida: {archi_senza_quote}")

print("\nGrafo creato:")
print(f"Nodi: {grafo.number_of_nodes()}")
print(f"Archi: {grafo.number_of_edges()}")

# ============================================================
# ÉTAPE 3 - Rattachement des domiciles et des POI au réseau
# ============================================================
# Utilise le réseau créé à l'étape 2. Chaque point de départ et chaque POI est
# associé au nœud du réseau qui se trouve le plus près.

# Récupère la liste des nœuds du réseau et leurs coordonnées.
nodi = list(grafo.nodes())
coordinate_nodi = np.array(nodi)

# Prépare une recherche rapide du nœud le plus proche.
albero_ricerca = cKDTree(coordinate_nodi)


def find_nearest_node(punto):
    """Trouve le nœud le plus proche et calcule la distance en mètres."""

    distanza, indice = albero_ricerca.query([punto.x, punto.y])
    return nodi[indice], distanza


def snap_points_to_network(punti_geografici):
    """Ajoute à chaque point le nœud le plus proche et sa distance."""

    punti_geografici = punti_geografici.copy()  # La table originale n'est pas modifiée.
    punti_geografici["nearest_node"] = None  # Nœud trouvé sur le réseau.
    punti_geografici["snap_dist_m"] = None  # Distance entre le point et ce nœud.

    # Cherche le nœud le plus proche pour chaque point de départ ou POI.
    for indice, riga in punti_geografici.iterrows():
        nodo, distanza = find_nearest_node(riga.geometry)
        punti_geografici.at[indice, "nearest_node"] = nodo
        punti_geografici.at[indice, "snap_dist_m"] = distanza

    return punti_geografici


abitazioni = snap_points_to_network(abitazioni)
poi = snap_points_to_network(poi)

# Affiche les distances obtenues et les points situés à plus de 70 m.
print("\nAggancio alla rete completato.")

print("\nDistanze di aggancio abitazioni:")
print(abitazioni["snap_dist_m"].describe())

print("\nDistanze di aggancio POI:")
print(poi["snap_dist_m"].describe())

print("\nAbitazioni con distanza > 70 m:")
print(abitazioni.loc[
    abitazioni["snap_dist_m"] > 70,
    ["id", "quartiere", "snap_dist_m"]
])

print("\nPOI con distanza > 70 m:")
print(poi.loc[
    poi["snap_dist_m"] > 70,
    ["poi_id", "type", "snap_dist_m"]
])

# ============================================================
# ÉTAPE 4 - Parcours domicile-POI accessibles en 15 minutes
# ============================================================
# Utilise les points reliés au réseau à l'étape 3. Les parcours les plus rapides
# sont calculés et seuls ceux qui durent au maximum 15 minutes sont conservés.

TEMPO_MASSIMO = 15  # Temps maximal retenu pour atteindre un POI à pied.

righe_percorsi = []  # Parcours qui respectent la limite de 15 minutes.
archi_senza_quote_usati = set()  # Contrôle des routes sans altitude valide.


def calculate_path_means(percorso):
    """
    Calcule la moyenne des quatre critères sur un parcours. Les routes plus
    longues comptent davantage que les routes plus courtes.
    """

    # Chaque somme commence à 0, puis reçoit les valeurs de chaque route.
    lunghezza_totale = 0  # Longueur totale du parcours.
    somma_traffico = 0  # Trafic multiplié par la longueur des routes.
    somma_marciapiede = 0  # Trottoirs multipliés par la longueur des routes.
    somma_verde = 0  # Végétation multipliée par la longueur des routes.
    somma_pendenza = 0  # Pente multipliée par la longueur des routes.

    # Parcourt toutes les routes qui composent le parcours.
    for indice in range(len(percorso) - 1):
        nodo_inizio = percorso[indice]  # Premier nœud du morceau.
        nodo_fine = percorso[indice + 1]  # Nœud suivant.

        arco = grafo[nodo_inizio][nodo_fine]  # Informations entre les deux nœuds.
        lunghezza = arco["length_m"]  # Longueur de ce morceau.

        lunghezza_totale += lunghezza
        somma_traffico += arco["traf_score"] * lunghezza
        somma_marciapiede += arco["marcia"] * lunghezza
        somma_verde += arco["green_score"] * lunghezza
        somma_pendenza += arco["slope_score"] * lunghezza

        # Note les routes qui utilisent encore l'ancien score de pente.
        if arco["slope_source"] != "swissALTI3D":
            archi_senza_quote_usati.add(tuple(sorted((nodo_inizio, nodo_fine))))

    # Divise les sommes par la longueur totale pour obtenir les quatre moyennes.
    return {
        "length_m": lunghezza_totale,
        "traf_mean": somma_traffico / lunghezza_totale,
        "marcia_mean": somma_marciapiede / lunghezza_totale,
        "green_mean": somma_verde / lunghezza_totale,
        "slope_mean": somma_pendenza / lunghezza_totale
    }


# Essaie chaque combinaison entre un domicile et un POI.
for _, casa in abitazioni.iterrows():

    casa_id = casa["id"]
    quartiere = casa["quartiere"]
    nodo_partenza = casa["nearest_node"]

    for _, punto_poi in poi.iterrows():

        poi_id = punto_poi["poi_id"]
        nome_poi = punto_poi["type"]
        nodo_arrivo = punto_poi["nearest_node"]

        try:
            # Liste des nœuds du parcours le plus rapide à pied.
            percorso = nx.shortest_path(
                grafo,
                source=nodo_partenza,
                target=nodo_arrivo,
                weight="time_min"
            )

            # Durée totale de ce même parcours.
            tempo_min = nx.shortest_path_length(
                grafo,
                source=nodo_partenza,
                target=nodo_arrivo,
                weight="time_min"
            )

            # Seuls les parcours de 15 minutes ou moins sont conservés.
            if tempo_min <= TEMPO_MASSIMO:

                # Calcule les quatre facteurs moyens du parcours retenu.
                medie_percorso = calculate_path_means(percorso)

                righe_percorsi.append({
                    "casa_id": casa_id,
                    "quartiere": quartiere,
                    "poi_id": poi_id,
                    "poi_name": nome_poi,
                    "tempo_min": tempo_min,
                    **medie_percorso
                })

        except nx.NetworkXNoPath:
            # Le couple domicile-POI est ignoré si aucun parcours n'existe.
            continue


percorsi = gpd.GeoDataFrame(righe_percorsi)  # Réunit tous les parcours dans un tableau.

# Arrête le programme si un parcours utilise une pente non recalculée.
if archi_senza_quote_usati:
    raise ValueError(
        "Almeno un percorso analizzato usa un arco senza copertura swissALTI3D"
    )

# Affiche les contrôles et un aperçu des parcours conservés.
print("\nControllo copertura DEM dei percorsi analizzati:")
print("Nessun percorso entro 15 minuti usa il valore precedente.")

print("\nPercorsi casa -> POI entro 15 minuti:")
print(f"Numero righe: {len(percorsi)}")
print(percorsi.head())

print("\nNumero POI raggiungibili per abitazione:")
print(percorsi.groupby("casa_id")["poi_id"].count())

print("\nNumero totale POI nel layer originale:")
print(poi["poi_id"].nunique())

print("\nNumero totale di POI diversi entro 15 minuti:")
print(percorsi["poi_id"].nunique())

poi_raggiunti = percorsi["poi_id"].unique()

# Cherche les POI qui ne sont accessibles depuis aucun domicile.
poi_non_raggiunti = poi[~poi["poi_id"].isin(poi_raggiunti)]

print("\nPOI mai raggiunti entro 15 minuti da nessuna abitazione:")
print(poi_non_raggiunti[["poi_id", "type"]])
# ============================================================
# ÉTAPE 5 - Séparation de l'école des autres POI
# ============================================================
# Part des parcours conservés à l'étape 4. L'école sert seulement à mesurer
# l'accessibilité; les autres POI passent à la simulation Monte Carlo.

# Repère les parcours qui conduisent à une école.
e_scuola = percorsi["poi_name"].str.contains("scuola", case=False, na=False)

# Sépare les parcours vers l'école de ceux utilisés dans le Monte Carlo.
percorsi_scuola = percorsi[e_scuola].copy()
percorsi_mc = percorsi[~e_scuola].copy()

# Affiche l'accès à l'école et le contenu utilisé dans le Monte Carlo.
print("\nAccessibilità scuola entro 15 minuti:")
print(
    percorsi_scuola[
        ["casa_id", "quartiere", "poi_id", "poi_name", "tempo_min"]
    ]
)

print("\nNumero abitazioni con almeno una scuola entro 15 minuti:")
print(percorsi_scuola["casa_id"].nunique())

print("\nPercorsi usati per il Monte Carlo:")
print(f"Numero righe: {len(percorsi_mc)}")
print(f"Numero abitazioni: {percorsi_mc['casa_id'].nunique()}")
print(f"Numero POI: {percorsi_mc['poi_id'].nunique()}")

# ============================================================
# ÉTAPE 6 - Monte Carlo: TOP1, TOP3 et score de tous les POI
# ============================================================
# Utilise les parcours sans l'école préparés à l'étape 5. Les quatre critères
# de qualité reçoivent de nouveaux poids à chacune des 1 000 simulations.

NUMERO_SIMULAZIONI = 1000  # Nombre de simulations réalisées.

# Permet d'obtenir les mêmes tirages à chaque exécution du script.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Le scénario principal donne 30 % à la proximité et 70 % à la qualité.
PESO_VICINANZA = 0.3
PESO_QUALITA = 0.7

# Listes utilisées pour conserver les résultats de chaque simulation.
risultati_top1 = []
risultati_top3 = []
risultati_score = []


for numero_simulazione in range(1, NUMERO_SIMULAZIONI + 1):

    # Produit quatre poids aléatoires dont la somme vaut toujours 1.
    peso_traffico, peso_marciapiede, peso_verde, peso_pendenza = np.random.dirichlet(
        [1, 1, 1, 1]
    )

    tabella_simulazione = percorsi_mc.copy()  # Copie pour la simulation en cours.

    # Plus le temps est court, plus la proximité est proche de 1.
    tabella_simulazione["vicinanza"] = 1 - (
        tabella_simulazione["tempo_min"] / TEMPO_MASSIMO
    )

    # Combine trafic, trottoir, espaces verts et pente avec les poids tirés.
    tabella_simulazione["qualita"] = (
        peso_traffico * tabella_simulazione["traf_mean"] +
        peso_marciapiede * tabella_simulazione["marcia_mean"] +
        peso_verde * tabella_simulazione["green_mean"] +
        peso_pendenza * tabella_simulazione["slope_mean"]
    )

    # Combine la proximité et la qualité pour obtenir le score final.
    tabella_simulazione["score"] = (
        PESO_VICINANZA * tabella_simulazione["vicinanza"] +
        PESO_QUALITA * tabella_simulazione["qualita"]
    )

    # Conserve le numéro de la simulation et les quatre poids utilisés.
    tabella_simulazione["sim"] = numero_simulazione
    tabella_simulazione["w_traf"] = peso_traffico
    tabella_simulazione["w_marcia"] = peso_marciapiede
    tabella_simulazione["w_green"] = peso_verde
    tabella_simulazione["w_slope"] = peso_pendenza

    risultati_score.append(tabella_simulazione)  # Garde le score de tous les POI.

    # Sélectionne le POI avec le score le plus élevé pour chaque domicile.
    top1_simulazione = (
        tabella_simulazione.sort_values(["casa_id", "score"], ascending=[True, False])
            .groupby("casa_id")
            .head(1)
    )

    risultati_top1.append(top1_simulazione)

    # Même classement, mais en conservant les trois premiers POI.
    top3_simulazione = (
        tabella_simulazione.sort_values(["casa_id", "score"], ascending=[True, False])
            .groupby("casa_id")
            .head(3)
    )

    risultati_top3.append(top3_simulazione)
    
# ============================================================
# ÉTAPE 7 - Synthèse du Monte Carlo
# ============================================================
# Réunit les résultats des 1 000 simulations pour calculer la fréquence du
# TOP1, du TOP3 et la variation du score de chaque POI.

# Réunit les TOP1 produits séparément dans les 1 000 simulations.
risultati_top1 = pd.concat(risultati_top1, ignore_index=True)

# Réunit de la même manière tous les TOP3.
risultati_top3 = pd.concat(risultati_top3, ignore_index=True)

# Réunit enfin les scores de tous les POI.
risultati_score = pd.concat(
    risultati_score,
    ignore_index=True
)

# TOP1: compte le nombre de fois où chaque POI arrive en première position.

riepilogo_top1 = (
    risultati_top1
    .groupby(["casa_id", "poi_id", "poi_name"])
    .size()
    .reset_index(name="n_top1")
)

riepilogo_top1["prob_top1"] = (
    riepilogo_top1["n_top1"] / NUMERO_SIMULAZIONI  # Transforme le nombre en probabilité.
)

riepilogo_top1 = riepilogo_top1.sort_values(
    ["casa_id", "prob_top1"],
    ascending=[True, False]
)

# TOP3: compte le nombre de fois où chaque POI apparaît dans les trois premiers.

riepilogo_top3 = (
    risultati_top3
    .groupby(["casa_id", "poi_id", "poi_name"])
    .size()
    .reset_index(name="n_top3")
)

riepilogo_top3["prob_top3"] = (
    riepilogo_top3["n_top3"] / NUMERO_SIMULAZIONI  # Transforme le nombre en probabilité.
)

riepilogo_top3 = riepilogo_top3.sort_values(
    ["casa_id", "prob_top3"],
    ascending=[True, False]
)

# Calcule la moyenne et la variation du score pour chaque couple domicile-POI.

riepilogo_score = (

    risultati_score

    .groupby(["casa_id", "poi_id", "poi_name"])

    .agg(
        score_medio=("score", "mean"),  # Score moyen.
        score_min=("score", "min"),  # Score le plus faible.
        score_max=("score", "max"),  # Score le plus élevé.
        score_std=("score", "std"),  # Variation du score.

        # Ces valeurs ne changent pas entre les simulations.
        tempo_min=("tempo_min", "first"),

        traf_mean=("traf_mean", "first"),
        marcia_mean=("marcia_mean", "first"),
        green_mean=("green_mean", "first"),
        slope_mean=("slope_mean", "first")
    )

    .reset_index()
)

riepilogo_score = riepilogo_score.sort_values(
    ["casa_id", "score_medio"],
    ascending=[True, False]
)
# ============================================================
# ÉTAPE 8 - Export des résultats
# ============================================================
# Enregistre les trois tableaux finaux dans le dossier du script et vérifie
# que tous les domiciles sont encore présents dans les résultats.
# Les noms des fichiers peuvent être changés dans les lignes ci-dessous.

# Enregistre les probabilités TOP1.
riepilogo_top1.to_csv(
    os.path.join(CARTELLA_SCRIPT, "risultati_top1.csv"),
    index=False
)

# Enregistre les probabilités TOP3.
riepilogo_top3.to_csv(
    os.path.join(CARTELLA_SCRIPT, "risultati_top3.csv"),
    index=False
)

# Enregistre les statistiques de score de tous les POI.
riepilogo_score.to_csv(
    os.path.join(CARTELLA_SCRIPT, "risultati_score_tutti_poi.csv"),
    index=False
)

# Affiche les noms des fichiers créés.
print("\nFile salvati:")
print("- risultati_top1.csv")
print("- risultati_top3.csv")
print("- risultati_score_tutti_poi.csv")

# Vérifie que les 40 domiciles sont présents dans chaque résultat.
print("\nControllo numero abitazioni:")

print("Case iniziali:",
      abitazioni["id"].nunique())

print("Case Monte Carlo:",
      percorsi_mc["casa_id"].nunique())

print("Case TOP1:",
      riepilogo_top1["casa_id"].nunique())

print("Case TOP3:",
      riepilogo_top3["casa_id"].nunique())

print("Case score medio:",
      riepilogo_score["casa_id"].nunique())
