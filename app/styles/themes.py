"""
Thèmes CSS pour l'application Streamlit
"""

SOFT_PURPLE = """
<style>
/* SUPPRIMER LA BARRE DU HAUT */
header[data-testid="stHeader"] {
    display: none !important;
}

.stApp {
    background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 20%, #ddd6fe 40%, #c4b5fd 60%, #a78bfa 80%, #8b5cf6 100%);
}

.stSidebar > div:first-child {
    background: linear-gradient(180deg, #f3e8ff 0%, #ddd6fe 100%);
    border-right: 2px solid #8b5cf6;
}

/* TOUT LE TEXTE SIDEBAR EN NOIR */
.stSidebar .stMarkdown h1, 
.stSidebar .stMarkdown h2, 
.stSidebar .stMarkdown h3,
.stSidebar .stMarkdown p,
.stSidebar .stMarkdown li,
.stSidebar label,
.stSidebar .stSlider label,
.stSidebar .stSelectbox label,
.stSidebar .stMultiSelect label,
.stSidebar .stCheckbox label,
.stSidebar span,
.stSidebar div {
    color: #000000 !important;
    font-weight: 600;
}

/* TITRE PRINCIPAL EN NOIR - TOUTES LES VARIANTES */
.stMarkdown h1,
h1,
[data-testid="stHeader"] h1,
.css-1v0mbdj h1,
.css-1629p8f h1 {
    color: #000000 !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    font-weight: bold !important;
}

.stMarkdown h2, .stMarkdown h3 {
    color: #374151 !important;
}

.stMarkdown p, .stMarkdown li {
    color: #111827 !important;
    font-weight: 500;
}

.stMetric {
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid #c4b5fd;
    border-radius: 12px;
    padding: 12px;
}

.stMetric label {
    color: #374151 !important;
    font-weight: 600;
}

.stMetric div {
    color: #1f2937 !important;
    font-weight: bold;
}

.stButton > button {
    background: linear-gradient(45deg, #8b5cf6, #a78bfa);
    color: white;
    border: none;
    border-radius: 20px;
    font-weight: bold;
}

.stButton > button:hover {
    background: linear-gradient(45deg, #7c3aed, #8b5cf6);
    transform: translateY(-1px);
}
</style>
"""

# Autres thèmes possibles
NEON_DANCE = """
<style>
/* Theme néon pour la danse */
header[data-testid="stHeader"] {
    display: none !important;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a0a2e 50%, #16213e 100%);
}

.stSidebar > div:first-child {
    background: linear-gradient(180deg, #1a0a2e 0%, #0a0a0a 100%);
    border-right: 2px solid #ff006e;
}

.stSidebar * {
    color: #00ff89 !important;
}

.stMarkdown h1 {
    color: #ff006e !important;
    text-shadow: 0 0 20px #ff006e !important;
}

.stButton > button {
    background: linear-gradient(45deg, #ff006e, #00ff89);
    color: black;
    font-weight: bold;
}
</style>
"""