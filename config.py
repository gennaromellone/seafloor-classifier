# ---------------------------------------------------------------------------
# GPS georeferencing — opzionale
# ---------------------------------------------------------------------------

# Percorso del CSV GPS del drone Argo (formato $GGLV, separatore ';').
# Impostare a None per disabilitare la georeferenziazione.
# Esempio: GPS_CSV = "../Posidonia/Punta-Licosa-2022-06-14.csv"
GPS_CSV = "../Posidonia/Punta-Licosa-2022-06-14.csv"

# Correzione fuso orario da applicare al tag creation_time del video per
# ricavare l'ora UTC reale.
# La Yi Camera salva l'ora locale (CEST = UTC+2) come se fosse UTC:
# serve sottrarre 2 ore → offset = -2.
CAMERA_TZ_OFFSET_HOURS = -2

# Scarto massimo (secondi) tra timestamp del frame e riga GPS più vicina.
# Se il gap supera questa soglia il punto viene emesso senza coordinate
# (lat/lon vuoti) per evitare di propagare fix GPS con buchi.
MAX_GPS_GAP_SECS = 5
