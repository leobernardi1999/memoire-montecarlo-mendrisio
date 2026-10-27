# STEP 1 - Importazione librerie e caricamento dei dati
# ============================================================
import math #Per le funzioni trigonometriche usate nel calcolo della pendenza
import os #Per lavorare con cartelle e percorsi del computer
import tempfile #Per creare temporaneamente il mosaico dei raster altimetrici

# Quando lo script è eseguito con il Python incluso in QGIS su macOS, indico
# esplicitamente dove si trova il database dei sistemi di coordinate.
QGIS_PROJ_DIR = "/Applications/QGIS-LTR.app/Contents/Resources/proj"
if "PROJ_LIB" not in os.environ and os.path.isdir(QGIS_PROJ_DIR):
    os.environ["PROJ_LIB"] = QGIS_PROJ_DIR

import fiona #PEr leggere i layer dentro un GeoPackage
import geopandas as gpd #Per gestire dati geografici
import networkx as nx #Per analizzare il grafo stradale. Trasforma le strade in una rete di nodi ed archi
import numpy as np #Per calcoli numerici
import pandas as pd #Per tabelle e riassunti finali
import rasterio #Per campionare le quote del modello numerico di terreno

from osgeo import gdal #Per riunire virtualmente i raster swissALTI3D senza modificarli
from shapely.geometry import LineString, MultiLineString #Per riconoscere le geometrie lineari della rete stradale
from scipy.spatial import cKDTree #Per trovare rapidamente il nodo della rete più vicino a un punto

# Percorso del GeoPackage nella stessa cartella dello script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) #Trova la cartella in cui si trova il file montecarlo.py. SCRIPT_DIR diventa quella cartella    
GPKG_PATH = os.path.join(SCRIPT_DIR, "montecarlo.gpkg") #Costruisce il pecorso completo del Geopackage. GPKG_PATH diventa il percorso completo del file montecarlo.gpkg. Cerca il file .gpkg nella stessa cartella dello script montecarlo.py

# Cartella contenente le tessere swissALTI3D in formato GeoTIFF.
# Per rendere il codice trasferibile, si può:
# 1) creare una sottocartella "swissALTI3D" accanto allo script; oppure
# 2) definire la variabile d'ambiente SWISSALTI3D_DIR con il percorso dei raster.
DEM_DIR = os.environ.get(
    "SWISSALTI3D_DIR",
    os.path.join(SCRIPT_DIR, "swissALTI3D")
)

# Le strade non vengono spezzate nel grafo: 20 m è soltanto il passo massimo
# usato per campionare le quote lungo ciascun arco esistente.
MAX_SLOPE_SUBSEGMENT_M = 20.0

# Controllo che il file esista
if not os.path.exists(GPKG_PATH): #Esiste questo file nel computer?
    raise FileNotFoundError(f"GeoPackage non trovato: {GPKG_PATH}")  #Interrompi il programma e mostra un errore

print("GeoPackage trovato:")
print(GPKG_PATH) #Stampa il percorso

# Controllo dei layer presenti
layers = fiona.listlayers(GPKG_PATH) #Apre il GeoPackage e legge i nomi dei layer presenti. layers diventa una lista di nomi dei layer. Li salva nella variabile layers.
print("\nLayer presenti:")
print(layers)

# Caricamento dei layer principali
rete = gpd.read_file(GPKG_PATH, layer="rete_strade") #Apri il layer "rete_strade" del GeoPackage e salvalo nella variabile rete. 
abitazioni = gpd.read_file(GPKG_PATH, layer="abitazioni") #Apri il layer "abitazioni" del GeoPackage e salvalo nella variabile abitazioni.
poi = gpd.read_file(GPKG_PATH, layer="poi") #Apri il layer "poi" del GeoPackage e salvalo nella variabile poi.

# Trasformazione logaritmica del traffico.
# Il calcolo avviene sulla copia caricata in memoria: il GeoPackage sorgente
# non viene modificato. I segmenti senza DTV sono trattati come DTV = 0,
# conformemente all'ipotesi dichiarata nella tesi.
DTV_MAX = float(pd.to_numeric(rete["DTV_FZG"], errors="coerce").max())
if DTV_MAX != 16495.0:
    raise ValueError(f"Valore massimo DTV inatteso: {DTV_MAX}")

dtv = (
    pd.to_numeric(rete["DTV_FZG"], errors="coerce")
    .fillna(0.0)
    .clip(lower=0.0)
)
rete["traf_score"] = 1.0 - np.log1p(dtv) / np.log1p(DTV_MAX)

# Controllo rapido dei dati caricati
print("\nElementi caricati:")
print(f"Rete stradale: {len(rete)} archi") #Quante righe contiene rete? Ogni riga è un arco (edge, tratto di strada che collega due nodi) della rete stradale. Stampa il numero di archi.
print(f"Abitazioni: {len(abitazioni)} punti") #Quante righe contiene abitazioni? Ogni riga è un punto che rappresenta un'abitazione. Stampa il numero di abitazioni.
print(f"POI: {len(poi)} punti") #Quante righe contiene poi? Ogni riga è un punto che rappresenta un POI. Stampa il numero di POI.

# Controllo del sistema di coordinate
print("\nCRS:")
print(f"Rete: {rete.crs}")
print(f"Abitazioni: {abitazioni.crs}")
print(f"POI: {poi.crs}")

# ============================================================
# STEP 2 - Ricostruzione della pendenza e creazione del grafo
# ============================================================

# Velocità media di cammino del bambino in metri/secondo
WALK_SPEED = 1.2


def iter_lines(geometry):
    """Restituisce le singole linee contenute in una geometria stradale."""
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def build_edge_records(roads):
    """
    Prepara gli archi del grafo senza modificarne la topologia.
    Ogni coppia di coordinate consecutive rimane un unico arco.
    """
    edge_records = []

    for _, row in roads.iterrows():
        for line in iter_lines(row.geometry):
            coords = list(line.coords)

            for index in range(len(coords) - 1):
                start = tuple(coords[index][:2])
                end = tuple(coords[index + 1][:2])
                length_m = float(LineString([start, end]).length)

                if length_m <= 0:
                    continue

                edge_records.append({
                    "start": start,
                    "end": end,
                    "length_m": length_m,
                    "traf_score": float(row["traf_score"]),
                    "marcia": float(row["marcia"]),
                    "green_score": float(row["green_score"]),
                    # Il vecchio valore è conservato soltanto come controllo di
                    # sicurezza per un eventuale arco fuori dalla copertura DEM.
                    "old_slope_score": float(row["slope_score"])
                })

    return edge_records


def recalculate_slope_scores(edge_records, raster, maximum_subsegment_length):
    """
    Calcola la pendenza assoluta di ogni arco stradale.

    Per il solo campionamento altimetrico, ciascun arco i viene suddiviso in
    sotto-tronconi j lunghi al massimo 20 m. L'angolo di ciascun sotto-troncone
    è calcolato dalle quote delle sue estremità; gli angoli sono poi aggregati
    sull'arco originale con una media ponderata per la lunghezza. Infine la
    pendenza media è invertita e normalizzata tra 0 e 1.
    """
    sample_coordinates = []
    sample_jobs = []

    for edge_index, edge in enumerate(edge_records):
        length_m = edge["length_m"]
        subsegment_count = max(
            1,
            int(math.ceil(length_m / maximum_subsegment_length))
        )
        subsegment_length = length_m / subsegment_count
        x_start, y_start = edge["start"]
        x_end, y_end = edge["end"]

        for number in range(subsegment_count):
            fraction_start = number / subsegment_count
            fraction_end = (number + 1) / subsegment_count

            first_index = len(sample_coordinates)
            sample_coordinates.append((
                x_start + (x_end - x_start) * fraction_start,
                y_start + (y_end - y_start) * fraction_start
            ))

            second_index = len(sample_coordinates)
            sample_coordinates.append((
                x_start + (x_end - x_start) * fraction_end,
                y_start + (y_end - y_start) * fraction_end
            ))

            sample_jobs.append((
                edge_index,
                subsegment_length,
                first_index,
                second_index
            ))

    sampled_elevations = np.asarray(
        [value[0] for value in raster.sample(sample_coordinates)],
        dtype=float
    )

    if raster.nodata is not None:
        sampled_elevations[
            np.isclose(sampled_elevations, raster.nodata)
        ] = np.nan

    weighted_angle_sum = np.zeros(len(edge_records), dtype=float)
    valid_length_sum = np.zeros(len(edge_records), dtype=float)
    total_length_sum = np.zeros(len(edge_records), dtype=float)

    for edge_index, length_m, first_index, second_index in sample_jobs:
        total_length_sum[edge_index] += length_m
        z_start = sampled_elevations[first_index]
        z_end = sampled_elevations[second_index]

        if not (np.isfinite(z_start) and np.isfinite(z_end)):
            continue

        angle_deg = math.degrees(
            math.atan(abs(z_end - z_start) / length_m)
        )
        weighted_angle_sum[edge_index] += angle_deg * length_m
        valid_length_sum[edge_index] += length_m

    mean_angles = np.divide(
        weighted_angle_sum,
        valid_length_sum,
        out=np.full(len(edge_records), np.nan, dtype=float),
        where=valid_length_sum > 0
    )

    if not np.isfinite(mean_angles).any():
        raise ValueError("Nessuna quota swissALTI3D valida sulla rete")

    maximum_angle = float(np.nanmax(mean_angles))
    if maximum_angle <= 0:
        raise ValueError("Pendenza massima non valida")

    slope_scores = 1.0 - mean_angles / maximum_angle
    dem_coverage = np.divide(
        valid_length_sum,
        total_length_sum,
        out=np.zeros(len(edge_records), dtype=float),
        where=total_length_sum > 0
    )

    return slope_scores, dem_coverage, maximum_angle


edge_records = build_edge_records(rete)

# Le tessere restano file separati: il VRT è soltanto un mosaico virtuale
# temporaneo e non modifica i raster originali.
dem_tiles = sorted([
    os.path.join(DEM_DIR, filename)
    for filename in os.listdir(DEM_DIR)
    if filename.startswith("swissalti3d_2022_") and filename.endswith(".tif")
]) if os.path.isdir(DEM_DIR) else []

if not dem_tiles:
    raise FileNotFoundError(
        "Nessun raster swissALTI3D trovato. Inserire le tessere nella cartella "
        f"{DEM_DIR} oppure definire SWISSALTI3D_DIR."
    )

with tempfile.TemporaryDirectory() as temporary_directory:
    vrt_path = os.path.join(temporary_directory, "swissalti3d_mosaic.vrt")
    vrt = gdal.BuildVRT(vrt_path, dem_tiles)

    if vrt is None:
        raise RuntimeError("Impossibile creare il mosaico virtuale swissALTI3D")

    vrt.FlushCache()
    vrt = None

    with rasterio.open(vrt_path) as elevation_raster:
        if rete.crs is not None and elevation_raster.crs != rete.crs:
            raise ValueError(
                "Il CRS dei raster swissALTI3D non coincide con quello della rete"
            )

        slope_scores, dem_coverage, maximum_slope_angle = recalculate_slope_scores(
            edge_records,
            elevation_raster,
            MAX_SLOPE_SUBSEGMENT_M
        )

# Grafo non orientato: ogni strada può essere percorsa nei due sensi.
# La topologia, le lunghezze e i tempi sono identici alla versione precedente.
G = nx.Graph()
fallback_edges = 0

for edge_index, edge in enumerate(edge_records):
    if np.isfinite(slope_scores[edge_index]):
        slope_score = float(slope_scores[edge_index])
        slope_source = "swissALTI3D"
    else:
        slope_score = edge["old_slope_score"]
        slope_source = "fallback_vecchio_indicatore"
        fallback_edges += 1

    G.add_edge(
        edge["start"],
        edge["end"],
        length_m=edge["length_m"],
        time_min=edge["length_m"] / WALK_SPEED / 60,
        traf_score=edge["traf_score"],
        marcia=edge["marcia"],
        green_score=edge["green_score"],
        slope_score=slope_score,
        slope_dem_coverage=float(dem_coverage[edge_index]),
        slope_source=slope_source
    )

print("\nPendenza ricostruita da swissALTI3D:")
print(f"Tessere DEM: {len(dem_tiles)}")
print(f"Passo massimo di campionamento: {MAX_SLOPE_SUBSEGMENT_M:.0f} m")
print(f"Angolo medio massimo osservato: {maximum_slope_angle:.6f}°")
print(f"Archi senza alcuna quota valida (fallback): {fallback_edges}")

print("\nGrafo creato:")
print(f"Nodi: {G.number_of_nodes()}")
print(f"Archi: {G.number_of_edges()}")

# ============================================================
# STEP 3 - Aggancio di abitazioni e POI alla rete
# ============================================================

# Lista dei nodi del grafo
nodes = list(G.nodes()) #Prendo tutti i nodi creati prima. Nodes diventa una lista di tutte le coordinate dei nodi del grafo. Ogni nodo è una tupla (x, y).
node_coords = np.array(nodes) #Trasformo lista in un array numpy. Più comodo per dare i calcoli spaziali.

# Struttura spaziale per trovare rapidamente il nodo più vicino
tree = cKDTree(node_coords) #Crea una struttura spaziale per trovare rapidamente il nodo più vicino a un punto. tree diventa un oggetto che permette di fare ricerche spaziali veloci.


def find_nearest_node(point): #Funzione riceve un punto (una casa o un POI)
    """
    Trova il nodo della rete più vicino a un punto.
    Restituisce il nodo e la distanza in metri.
    """

    dist, idx = tree.query([point.x, point.y]) #Trova il nodo più vicino al punto dato. dist diventa la distanza in metri tra il punto e il nodo più vicino. idx diventa il nodo più vicino nella lista dei nodi.
    return nodes[idx], dist


def snap_points_to_network(gdf): #Prende un GeoDataFrame di punti. L'intera tabella per esempio tutte la abitazioni. Per ogni ra della tabella usa findnearest_node() per trovare il nodo più vicino e la distanza. Aggiunge due colonne al GeoDataFrame: nearest_node e snap_dist_m.
    """
    Aggancia ogni punto del GeoDataFrame al nodo più vicino della rete.
    Aggiunge due colonne:
    - nearest_node: nodo più vicino
    - snap_dist_m: distanza dal nodo più vicino
    """

    gdf = gdf.copy() #Crea una copia del GeoDataFrame per non modificare l'originale. gdf diventa una copia del GeoDataFrame originale.
    gdf["nearest_node"] = None #Crea una nuova colonna nel GeoDataFrame per salvare il nodo più vicino a ogni punto. Inizialmente è vuota (None).
    gdf["snap_dist_m"] = None #Crea una nuova colonna nel GeoDataFrame per salvare la distanza dal nodo più vicino a ogni punto. Inizialmente è vuota (None).

    for idx, row in gdf.iterrows(): #Scorre ogni punto della tabella per ogni abitazione o POI.
        node, dist = find_nearest_node(row.geometry) #Chiama la funzione di prima per trovare il nodo più vicino e la distanza. node diventa il nodo più vicino al punto. dist diventa la distanza in metri tra il punto e il nodo più vicino.
        gdf.at[idx, "nearest_node"] = node #Salva il nodo più vicino nella colonna "nearest_node" del GeoDataFrame.
        gdf.at[idx, "snap_dist_m"] = dist #Salva la distanza dal nodo più vicino nella colonna "snap_dist_m" del GeoDataFrame.

    return gdf


# Aggancio alla rete
abitazioni = snap_points_to_network(abitazioni) #Aggancia le abitazioni alla rete stradale. La funzione snap_points_to_network() prende il GeoDataFrame delle abitazioni e aggiunge due colonne: nearest_node e snap_dist_m. La colonna nearest_node contiene il nodo più vicino della rete stradale per ogni abitazione, mentre la colonna snap_dist_m contiene la distanza in metri tra l'abitazione e il nodo più vicino.
poi = snap_points_to_network(poi) #Aggancia i POI alla rete stradale nello stesso modo delle abitazioni. 

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
# STEP 4 - Calcolo dei percorsi casa -> POI entro 15 minuti
# ============================================================

MAX_TIME = 15  # soglia massima di accessibilità in minuti

records = [] #Lista vuota per salvare i risultati. 
fallback_edges_used = set() #Controllo: nessun percorso analizzato deve usare il fallback.


def calculate_path_means(path): #Path è una lista di nodi che rappresenta il percorso più breve tra una casa e un POI. La funzione calcola le caratteristiche medie del percorso, ponderate per la lunghezza dei segmenti attraversati.
    """
    Calcola le caratteristiche medie del percorso.
    Ogni score viene ponderato per la lunghezza dei segmenti attraversati.
    """

    length_tot = 0 #Si inizializzano dei contenitori, sono tutti uguale a zero perchè nessun segmento è stato analizzato. Lenght_tot serv per sommare la lunghezza totale del percorso.
    traf_sum = 0 #Serve per sommare il punteggio di traffico ponderato per la lunghezza dei segmenti attraversati.
    marcia_sum = 0 #Serve per sommare il punteggio di marciapiede ponderato per la lunghezza dei segmenti attraversati.
    green_sum = 0 #Serve per sommare il punteggio di verde ponderato per la lunghezza dei segmenti attraversati.
    slope_sum = 0 #Serve per sommare il punteggio di pendenza ponderato per la lunghezza dei segmenti attraversati.

    for i in range(len(path) - 1): #Serve per scorrere tutti i segmenti del percorso. Ogni segmento è definito da due nodi consecutivi nella lista path.
        u = path[i] #Nodi consecutivi.
        v = path[i + 1]

        edge = G[u][v] #Prende il pezzo di strada tra questi due nodi del Grafo. 
        length = edge["length_m"] #Prende lunghezza del segmento.

        length_tot += length #Aggiorna la lunghezza totale del percorso. Diventa length_tot = length_tot + length.
        traf_sum += edge["traf_score"] * length #Aggiorna la somma ponderata del punteggio di traffico. 
        marcia_sum += edge["marcia"] * length #Aggiorna la somma ponderata del punteggio di marciapiede.
        green_sum += edge["green_score"] * length #Aggiorna la somma ponderata del punteggio di verde.
        slope_sum += edge["slope_score"] * length #Aggiorna la somma ponderata del punteggio di pendenza.

        if edge["slope_source"] != "swissALTI3D":
            fallback_edges_used.add(tuple(sorted((u, v))))

    return { #Restituisce un dizionario con le caratteristiche medie del percorso. Ogni punteggio medio è calcolato come la somma ponderata divisa per la lunghezza totale del percorso.
        "length_m": length_tot,
        "traf_mean": traf_sum / length_tot,
        "marcia_mean": marcia_sum / length_tot,
        "green_mean": green_sum / length_tot,
        "slope_mean": slope_sum / length_tot
    }


for _, casa in abitazioni.iterrows(): #Passa una per una tutte le abitazioni, riga per riga. - indica che non ci interessa l'indice della riga. Ci interessa casa, cioè la riga dell'abitazione.

    casa_id = casa["id"] #Recupera informazioni della casa.
    quartiere = casa["quartiere"]
    start_node = casa["nearest_node"]

    for _, p in poi.iterrows(): #Dentro ogni casa, guarda tutti i POI

        poi_id = p["poi_id"] #Recupera le info dei POI.
        poi_name = p["type"]
        end_node = p["nearest_node"]

        try: #Non sempre esiste, dunque si fa try.
            # Percorso più breve in termini di tempo
            path = nx.shortest_path( #si usa NetworkX per calcolare il percorso più breve tra due nodi del grafo. Restituisce il percorso.
                G,
                source=start_node,
                target=end_node,
                weight="time_min" #Si cerca il percorso più veloce in minuti.
            )

            tempo_min = nx.shortest_path_length( #Restituisce il costo del percorso più breve tra due nodi del grafo. In questo caso il tempo in minuti.
                G,
                source=start_node,
                target=end_node,
                weight="time_min"
            )

            # Tengo solo i POI raggiungibili entro 15 minuti
            if tempo_min <= MAX_TIME:

                path_stats = calculate_path_means(path) #Richiamo la funzione di prima per calcolare le caratteristiche medie del percorso. path_stats diventa un dizionario con le caratteristiche medie del percorso.

                records.append({ #Aggiunge il percorso nella lista dei risultati, se è un POI entro 15 minuti. Ogni percorso è rappresentato da un dizionario con le informazioni della casa, del POI e le caratteristiche medie del percorso.
                    "casa_id": casa_id,
                    "quartiere": quartiere,
                    "poi_id": poi_id,
                    "poi_name": poi_name,
                    "tempo_min": tempo_min,
                    **path_stats # Aggiungi dentro il dizionario anche tutte le coppie contenute in path_stats.
                })

        except nx.NetworkXNoPath:
            # Se non esiste un percorso tra casa e POI, il caso viene ignorato
            continue


percorsi = gpd.GeoDataFrame(records) # Trasformo la lista records in una tabllea GeoPandas. Però non è una tabella con la geometria vera e propria.

if fallback_edges_used:
    raise ValueError(
        "Almeno un percorso analizzato usa un arco senza copertura swissALTI3D"
    )

print("\nControllo copertura DEM dei percorsi analizzati:")
print("Nessun percorso entro 15 minuti usa il valore di fallback.")

print("\nPercorsi casa -> POI entro 15 minuti:")
print(f"Numero righe: {len(percorsi)}") #Quante righe ha la tabella percorsi.
print(percorsi.head())

print("\nNumero POI raggiungibili per abitazione:")
print(percorsi.groupby("casa_id")["poi_id"].count()) #Stampo una tabella raggruppata per casa che conta quanti POI entro 15 minuti ha ogni casa.

print("\nNumero totale POI nel layer originale:")
print(poi["poi_id"].nunique()) #Stampo il numero di POI unici nel layer originale.

print("\nNumero totale di POI diversi entro 15 minuti:")
print(percorsi["poi_id"].nunique()) #Stampo il numero di POI unici raggiungibili.

# Identifico quali POI sono stati raggiunti almeno una volta
poi_raggiunti = percorsi["poi_id"].unique() 

# Cerco nel layer originale i POI che non compaiono mai nei percorsi entro 15 minuti
poi_non_raggiunti = poi[~poi["poi_id"].isin(poi_raggiunti)]

print("\nPOI mai raggiunti entro 15 minuti da nessuna abitazione:")
print(poi_non_raggiunti[["poi_id", "type"]])
# ============================================================
# STEP 5 - Separazione della scuola dagli altri POI
# ============================================================

# La scuola viene usata solo come indicatore di accessibilità.
# Non entra nella simulazione Monte Carlo.
is_school = percorsi["poi_name"].str.contains("scuola", case=False, na=False) #Faccio mashera booleana.Prendo la colonna dei POI e cerco la parola scuola, ignorando maiuscole e minuscole e righe vuote. Dunque una lista di vero o falso.

percorsi_scuola = percorsi[is_school].copy() #Raggruppo solo i percorsi che portano a una scuola. Creo una copia della tabella percorsi con solo le righe dove is_school è True. Con copy credo una tabell autonoma.
percorsi_mc = percorsi[~is_school].copy() #Qua raggruppo tutto ciò che non è scuola. Creo una copia della tabella percorsi con solo le righe dove is_school è False. Con copy creo una tabella autonoma. Verà usata nel MonteCarlo.

print("\nAccessibilità scuola entro 15 minuti:")
print( #Stampo solo alcune colonne della tabella percorsi_scuola. Dunque le case che hanno almeno una scuola entro 15 minuti.
    percorsi_scuola[
        ["casa_id", "quartiere", "poi_id", "poi_name", "tempo_min"]
    ]
)

print("\nNumero abitazioni con almeno una scuola entro 15 minuti:")
print(percorsi_scuola["casa_id"].nunique()) #Conta i valori unici, per calcolare quante case hanno accesso ad almeno una scuola entro 15 minuti.

print("\nPercorsi usati per il Monte Carlo:")
print(f"Numero righe: {len(percorsi_mc)}") #prendo il numero delle righe.
print(f"Numero abitazioni: {percorsi_mc['casa_id'].nunique()}") #Prendo i valori unici.
print(f"Numero POI: {percorsi_mc['poi_id'].nunique()}") #Prendo i valori unici.

# ============================================================
# STEP 6 - Monte Carlo: TOP1, TOP3 e score di tutti i POI
# ============================================================

# Numero di simulazioni Monte Carlo
N_SIM = 1000 #1000 modi diversi di pesare le caratteristiche del percorso. Ogni simulazione genera 4 pesi casuali che sommano a 1. I pesi rappresentano l'importanza assegnata a: traffico, marciapiede, verde e pendenza.

# Seme casuale fisso per rendere la simulazione riproducibile.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Peso dato alla vicinanza e alla qualità del percorso. Per la scelta del POI conta di più la qualità del tragitto che la sola distanza. Lo score finale del POI è calcolato come una media pesata tra la vicinanza e la qualità del percorso. 
ALPHA_VICINANZA = 0.3
ALPHA_QUALITA = 0.7

# Liste dove salvo i risultati delle simulazioni
risultati_top1 = [] #Salva il POI migliore per ogni casa in ogni simulazione. Dunque 1000 POI migliori per ogni casa.
risultati_top3 = [] #Salva i primi tre POI migliori per ogni casa in ogni simulazione. 
risultati_tutti_score = [] #Salva lo score di tutti i POI raggiungibili per ogni casa in ogni simulazione. 


for sim in range(1, N_SIM + 1): #Inizia ciclo per fare 1000 simulazioni. sim diventa il numero della simulazione corrente, da 1 a 1000.

    # Genera 4 pesi casuali che sommano sempre a 1
    # I pesi rappresentano l'importanza assegnata a:
    # traffico, marciapiede, verde e pendenza
    w_traf, w_marcia, w_green, w_slope = np.random.dirichlet([1, 1, 1, 1]) #Utile visto che non sappiamo quale criterio pesa di più nella scelta di un bambino.

    temp = percorsi_mc.copy() #Creo copia della tabella con tutti i percorsi casa -> POI raggiungibili entro 15 minuti.

    # Vicinanza: più il tempo è basso, più il valore è alto
    temp["vicinanza"] = 1 - (temp["tempo_min"] / MAX_TIME) #Calcolo la vicinanza. Più il tempo è basso più questa sarà alta.

    # Qualità del percorso con pesi casuali
    temp["qualita"] = ( #Calcolo la qualità del percorso come media ponderata dei 4 criteri. Più il punteggio è alto più il percorso è considerato di qualità.
        w_traf * temp["traf_mean"] +
        w_marcia * temp["marcia_mean"] +
        w_green * temp["green_mean"] +
        w_slope * temp["slope_mean"]
    )

    # Score finale del POI 
    temp["score"] = ( #Calcolo lo score finale del POI che combina la vicinanza e la qualità del percorso. Più lo score è alto più il POI è considerato interessante per il bambino.
        ALPHA_VICINANZA * temp["vicinanza"] +
        ALPHA_QUALITA * temp["qualita"]
    )

    # Salvo anche il numero della simulazione e i pesi usati. Così so quali pesi hanno portato a quale risultato. Utile per capire se un certo POI è sempre il migliore o se dipende dai pesi.
    temp["sim"] = sim
    temp["w_traf"] = w_traf
    temp["w_marcia"] = w_marcia
    temp["w_green"] = w_green
    temp["w_slope"] = w_slope

    # Salvo lo score di tutti i POI raggiungibili
    risultati_tutti_score.append(temp)

    # TOP1: POI con score più alto per ogni abitazione
    top1 = (
        temp.sort_values(["casa_id", "score"], ascending=[True, False]) #Ordina la tabella prima per id della casa con ordine crescente e poi per score del POI con ordine decrescente. 
            .groupby("casa_id") #Raggruppa per abitazione
            .head(1) #Prende solo il primo POI per ogni abitazione, cioè quello con lo score più alto.
    )

    risultati_top1.append(top1) #Salva i risultati della simulazione corrente nella lista dei risultati top1.

    # TOP3: primi tre POI con score più alto per ogni abitazione
    top3 = (
        temp.sort_values(["casa_id", "score"], ascending=[True, False])
            .groupby("casa_id")
            .head(3) #Stessa cosa, ma mette 3 POI al posto di quello più alto e basta. Ottiene dunque i 3 POI con lo score più alto.
    )

    risultati_top3.append(top3) #Salva i risultati della simulazione corrente nella lista dei risultati top3.
    
# ============================================================
# STEP 7 - Riassunti finali del Monte Carlo
# ============================================================

# Unisco le 1000 simulazioni
risultati_top1 = pd.concat(risultati_top1, ignore_index=True) #Qui unisce tutte le tabelle TOP1 salvate nelle 1000 simulazioni. pd.concat() prende una lista di DataFrame e li unisce in un unico DataFrame. ignore_index=True significa che l'indice del nuovo DataFrame sarà riassegnato da 0 a n-1, ignorando gli indici originali dei DataFrame concatenati.

risultati_top3 = pd.concat(risultati_top3, ignore_index=True) #Stessa cosa ma per TOP3.

risultati_tutti_score = pd.concat( #Stessa cosa ma per tutti i POI raggiungibili. Qui unisce tutte le tabelle con lo score di tutti i POI salvate nelle 1000 simulazioni.
    risultati_tutti_score,
    ignore_index=True
)

# ------------------------------------------------
# TOP1
# ------------------------------------------------

summary_top1 = ( #Calcola quante volte ogni POI è stato il migliore per ogni casa nelle 1000 simulazioni. Raggruppa per casa e POI e conta quante volte il POI è stato il migliore. Salva il risultato in summary_top1.
    risultati_top1
    .groupby(["casa_id", "poi_id", "poi_name"]) #Raggruppa
    .size() #Conta quante volte ogni POI è stato il migliore per ogni casa. Conta le righe di ogni gruppo.
    .reset_index(name="n_top1") #Trasforma il risultato in un DataFrame con una nuova colonna chiamata "n_top1" che contiene il conteggio.
)

# Probabilità di essere il POI migliore
summary_top1["prob_top1"] = (
    summary_top1["n_top1"] / N_SIM #Viene trasformato il numero di vittorie in probabilità.
)

summary_top1 = summary_top1.sort_values( #Ordina la tabella prima per casa con ordine crescente e poi per probabilità di essere il POI migliore con ordine decrescente. Così per ogni casa vedo subito qual è il POI più probabile.
    ["casa_id", "prob_top1"],
    ascending=[True, False]
)

# ------------------------------------------------
# TOP3
# ------------------------------------------------

summary_top3 = ( #Stessa logica del TOP1 ma quante quante volte compare nei primi 3.
    risultati_top3
    .groupby(["casa_id", "poi_id", "poi_name"])
    .size()
    .reset_index(name="n_top3")
)

# Probabilità di comparire nei primi 3 POI
summary_top3["prob_top3"] = (
    summary_top3["n_top3"] / N_SIM
)

summary_top3 = summary_top3.sort_values(
    ["casa_id", "prob_top3"],
    ascending=[True, False]
)

# ------------------------------------------------
# Score medio di tutti i POI
# ------------------------------------------------

summary_score = (

    risultati_tutti_score

    .groupby(["casa_id", "poi_id", "poi_name"]) #Raggruppa

    .agg( #Qui calcoli statistiche sullo score. PEr ogni casa->POI hai mille score, uno per simulazione.
        score_medio=("score", "mean"), #Qui calcoli la media dello score per ogni casa->POI. score_medio diventa la media dei mille score. Se Alto allora è un POI generalmente buono. Da guardare sempre con la deviazione standard. esempi omedie o.78 e std 0.03 allora buono. se media 0.60 e std 0.18 allora può esseere buono o meno a seonda dei pesi assegnati.
        score_min=("score", "min"), #Qui calcoli il minimo dello score per ogni casa->POI. score_min diventa il minimo dei mille score.
        score_max=("score", "max"), #Qui calcoli il massimo dello score per ogni casa->POI. score_max diventa il massimo dei mille score.
        score_std=("score", "std"),#Qui calcoli la deviazione standard dello score per ogni casa->POI. score_std diventa la deviazione standard dei mille score. Se alto, vuol dire che il poi è molto senbile dal cambiamento dei pesi. Dunque da cosa viene considerato importante. Se basso allora resta buono anche cxambiando i pesi.

        tempo_min=("tempo_min", "first"), #Conservo il tempo di percorrenza. Uso first perchè il tempo non cambia.

        traf_mean=("traf_mean", "first"), #Questi valori non cambiano
        marcia_mean=("marcia_mean", "first"),
        green_mean=("green_mean", "first"),
        slope_mean=("slope_mean", "first")
    )

    .reset_index() #Riporta il risultato in forma di tabella normale.
)

summary_score = summary_score.sort_values( #Ordino per casa poi per score medio decrescente.
    ["casa_id", "score_medio"],
    ascending=[True, False]
)
# ============================================================
# STEP 8 - Salvataggio dei risultati
# ============================================================

summary_top1.to_csv( #Salvo i file in csv.
    os.path.join(SCRIPT_DIR, "risultati_top1.csv"), #SCRIPT_DIR contiene il percordo della cartella dove sto lavorando.
    index=False
)

summary_top3.to_csv(
    os.path.join(SCRIPT_DIR, "risultati_top3.csv"),
    index=False
)

summary_score.to_csv(
    os.path.join(SCRIPT_DIR, "risultati_score_tutti_poi.csv"),
    index=False
)

print("\nFile salvati:")
print("- risultati_top1.csv")
print("- risultati_top3.csv")
print("- risultati_score_tutti_poi.csv")

# Controllo finale
print("\nControllo numero abitazioni:") #Controllo che le case sono tutte in tutte le analisi fatte.

print("Case iniziali:",
      abitazioni["id"].nunique())

print("Case Monte Carlo:",
      percorsi_mc["casa_id"].nunique())

print("Case TOP1:",
      summary_top1["casa_id"].nunique())

print("Case TOP3:",
      summary_top3["casa_id"].nunique())

print("Case score medio:",
      summary_score["casa_id"].nunique())
