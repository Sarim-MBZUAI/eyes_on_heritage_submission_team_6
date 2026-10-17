"""Point-level change classification for the Eyes on Heritage hackathon."""

CLASSES = ["construction", "destruction", "soil_sealing", "no_change"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

SITE_DIRS = {
    "agadez": "agadez-niger",
    "aleppo": "aleppo-syria",
    "derna": "derna-libya",
    "marrakesh": "marrakesh-morocco",
    "timbuktu": "timbuktu-mali",
    "damascus": "damascus-syria",
    "herat": "herat-afghanistan",
}
