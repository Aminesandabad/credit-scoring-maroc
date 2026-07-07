import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime
from PIL import Image
import base64

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# ==========================================================
# CONFIGURATION GENERALE
# ==========================================================

st.set_page_config(
    page_title="Système de Scoring Enrichi",
    page_icon="🏦",
    layout="wide"
)

# ==========================================================
# DESIGN PROFESSIONNEL
# ==========================================================

st.markdown("""
<style>
.stApp {
    background: #f7f8fc;
    color: #1e293b;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 100%;
}

.main-header {
    background: white;
    padding: 22px 20px;
    border-radius: 22px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
}

.logo-box {
    width: 160px;
    height: 95px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.logo-box img {
    max-width: 145px;
    max-height: 85px;
    object-fit: contain;
}

.header-center {
    flex: 1;
    text-align: center;
}

.main-title {
    font-size: 34px;
    font-weight: 850;
    color: #0f4c81;
    text-align: center;
    margin-bottom: 6px;
}

.main-subtitle {
    font-size: 20px;
    font-weight: 650;
    color: #475569;
    text-align: center;
}

.section-title {
    background: #eef5ff;
    padding: 11px 14px;
    border-radius: 12px;
    font-size: 23px;
    font-weight: 800;
    color: #0f4c81;
    margin-top: 16px;
    margin-bottom: 18px;
    border: 1px solid #dbeafe;
}

.result-card {
    background: white;
    border-radius: 20px;
    padding: 24px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.07);
    text-align: center;
    margin-bottom: 18px;
}

.result-s1 {
    color: #16a34a;
    font-size: 76px;
    font-weight: 950;
    line-height: 1;
}

.result-s2 {
    color: #f59e0b;
    font-size: 76px;
    font-weight: 950;
    line-height: 1;
}

.result-s3 {
    color: #dc2626;
    font-size: 76px;
    font-weight: 950;
    line-height: 1;
}

.stButton > button {
    background: linear-gradient(90deg, #0f4c81, #1d4ed8);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    font-weight: 800;
    font-size: 17px;
    box-shadow: 0 8px 18px rgba(15, 76, 129, 0.25);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0b3a63, #0f4c81);
    color: white;
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] {
    background-color: white !important;
    color: #0f172a !important;
    border-radius: 12px !important;
}

label {
    color: #334155 !important;
    font-weight: 600 !important;
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding-top: 20px;
}

section[data-testid="stSidebar"] {
    background: #f1f5f9;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# OUTILS DE CODIFICATION
# ==========================================================

COLONNES_ALTERNATIVES = [
    "Paiement_eau",
    "Paiement_electricite",
    "Paiement_telecom",
    "Paiement_loyer",
    "Paiement_taxe_habitation",
    "Paiement_TSC"
]

SCENARIOS = {
    "Alternatives faibles (10%)": 0.10,
    "Alternatives moyennes (20%)": 0.20,
    "Alternatives renforcées (30%)": 0.30
}

ALGORITHMES = [
    "Arbre de décision",
    "Random Forest",
    "Régression Logistique"
]


def coder_historique(valeur):
    return {"Mauvais": 1, "Moyen": 2, "Bon": 3}[valeur]


def coder_paiement(valeur):
    return {
        "Retards fréquents": 1,
        "Quelques retards": 2,
        "Régulier": 3
    }[valeur]


def label_classe(code):
    return {1: "S1", 2: "S2", 3: "S3"}[int(code)]


def infos_risque(classe):
    if classe == "S1":
        return "Faible", "Crédit recommandé", "#16a34a", "result-s1"
    if classe == "S2":
        return "Modéré", "Analyse complémentaire recommandée", "#f59e0b", "result-s2"
    return "Élevé", "Crédit refusé ou garanties exigées", "#dc2626", "result-s3"


def proba_par_classe(modele, ligne_client):
    proba = modele.predict_proba(ligne_client)[0]
    classes = modele.classes_

    resultat = {"S1": 0.0, "S2": 0.0, "S3": 0.0}
    for classe, p in zip(classes, proba):
        resultat[label_classe(classe)] = float(p)

    return resultat


# ==========================================================
# CHARGEMENT ET PREPARATION DE LA BASE
# ==========================================================

@st.cache_data
def charger_base_codee():
    df = pd.read_excel("base_scoring_credit_maroc_classe.xlsx", engine="openpyxl")

    df["Historique_remboursement"] = df["Historique_remboursement"].map({
        "Mauvais": 1,
        "Moyen": 2,
        "Bon": 3
    })

    mapping_paiement = {
        "Retards frequents": 1,
        "Quelques retards": 2,
        "Regulier": 3
    }

    for col in COLONNES_ALTERNATIVES:
        df[col] = df[col].map(mapping_paiement)

    return df


def creer_classe_scoring(df_base, poids_alternatif):
    """
    Crée Score_Enrichi et Classe_Scoring selon un scénario de pondération.

    poids_alternatif = 0.10, 0.20 ou 0.30.
    """
    df = df_base.copy()
    poids_traditionnel = 1 - poids_alternatif

    df["Taux_endettement"] = df["Charges_mensuelles_MAD"] / df["Revenu_mensuel_MAD"]

    score_traditionnel = (
            df["Historique_remboursement"] * 15
            + (1 - df["Incidents_paiement"]) * 15
            + (1 - df["Prets_souffrance"].clip(0, 1)) * 10
            + (1 - df["Taux_endettement"].clip(0, 1)) * 15
    )

    score_alternatif = (
            df["Paiement_eau"] * 5
            + df["Paiement_electricite"] * 5
            + df["Paiement_telecom"] * 5
            + df["Paiement_loyer"] * 10
            + df["Paiement_taxe_habitation"] * 5
            + df["Paiement_TSC"] * 5
    )

    # Normalisation séparée des deux blocs
    score_traditionnel = (score_traditionnel / score_traditionnel.max()) * 100
    score_alternatif = (score_alternatif / score_alternatif.max()) * 100

    df["Score_Enrichi"] = (
            score_traditionnel * poids_traditionnel
            + score_alternatif * poids_alternatif
    )

    df["Classe_Scoring"] = pd.cut(
        df["Score_Enrichi"],
        bins=[0, 60, 80, 100],
        labels=["S3", "S2", "S1"],
        include_lowest=True
    )

    df["Classe_Scoring"] = df["Classe_Scoring"].map({
        "S1": 1,
        "S2": 2,
        "S3": 3
    })

    return df


def preparer_base_modele(df_scenario):
    df = df_scenario.copy()

    df = df.drop(
        columns=["Client_ID", "Score_Enrichi", "Defaut_paiement"],
        errors="ignore"
    )

    df = pd.get_dummies(
        df,
        columns=["Situation_familiale", "Adresse_zone", "Type_emploi"],
        drop_first=False
    )

    X_enrichi = df.drop(columns=["Classe_Scoring"])

    X_traditionnel = df.drop(
        columns=["Classe_Scoring"] + COLONNES_ALTERNATIVES,
        errors="ignore"
    )

    y = df["Classe_Scoring"]

    return df, X_enrichi, X_traditionnel, y


def construire_modele(nom):
    if nom == "Arbre de décision":
        return DecisionTreeClassifier(max_depth=5, random_state=42)

    if nom == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42))
    ])


@st.cache_resource
def entrainer_tous_les_scenarios():
    df_base = charger_base_codee()

    resultats = []
    objets_modeles = {}

    for nom_scenario, poids_alt in SCENARIOS.items():
        df_scenario = creer_classe_scoring(df_base, poids_alt)
        df_final, X_enrichi, X_traditionnel, y = preparer_base_modele(df_scenario)

        X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
            X_enrichi, y, test_size=0.20, random_state=42, stratify=y
        )

        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
            X_traditionnel, y, test_size=0.20, random_state=42, stratify=y
        )

        objets_modeles[nom_scenario] = {
            "colonnes_enrichies": X_enrichi.columns.tolist(),
            "colonnes_trad": X_traditionnel.columns.tolist(),
            "modeles_enrichis": {},
            "modeles_trad": {},
            "tests": {}
        }

        for nom_algo in ALGORITHMES:
            modele_e = construire_modele(nom_algo)
            modele_t = construire_modele(nom_algo)

            modele_e.fit(X_train_e, y_train_e)
            modele_t.fit(X_train_t, y_train_t)

            pred_e = modele_e.predict(X_test_e)
            pred_t = modele_t.predict(X_test_t)

            acc_e = accuracy_score(y_test_e, pred_e)
            acc_t = accuracy_score(y_test_t, pred_t)

            precision_e, recall_e, f1_e, _ = precision_recall_fscore_support(
                y_test_e, pred_e, average="weighted", zero_division=0
            )
            precision_t, recall_t, f1_t, _ = precision_recall_fscore_support(
                y_test_t, pred_t, average="weighted", zero_division=0
            )

            resultats.append({
                "Scénario": nom_scenario,
                "Poids alternatives": poids_alt,
                "Algorithme": nom_algo,
                "Accuracy traditionnel": acc_t,
                "Accuracy enrichi": acc_e,
                "Gain": acc_e - acc_t,
                "Precision traditionnel": precision_t,
                "Precision enrichi": precision_e,
                "Recall traditionnel": recall_t,
                "Recall enrichi": recall_e,
                "F1 traditionnel": f1_t,
                "F1 enrichi": f1_e
            })

            objets_modeles[nom_scenario]["modeles_enrichis"][nom_algo] = modele_e
            objets_modeles[nom_scenario]["modeles_trad"][nom_algo] = modele_t
            objets_modeles[nom_scenario]["tests"][nom_algo] = {
                "y_test_e": y_test_e,
                "pred_e": pred_e,
                "y_test_t": y_test_t,
                "pred_t": pred_t
            }

    resultats_df = pd.DataFrame(resultats)
    return resultats_df, objets_modeles


# ==========================================================
# CREATION D'UN CLIENT POUR LA PREDICTION
# ==========================================================

def creer_client(
        colonnes_enrichies,
        colonnes_trad,
        age, situation, zone, emploi, anciennete, revenu, encours, charges,
        nb_credits, nb_demandes, historique, incidents, decouverts, souffrance,
        eau, electricite, telecom, loyer, taxe_habitation, tsc
):
    base = pd.DataFrame([{
        "Age": age,
        "Situation_familiale": situation,
        "Adresse_zone": zone,
        "Type_emploi": emploi,
        "Anciennete_emploi_ans": anciennete,
        "Revenu_mensuel_MAD": revenu,
        "Encours_credits_MAD": encours,
        "Charges_mensuelles_MAD": charges,
        "Nombre_credits_en_cours": nb_credits,
        "Nombre_demandes_credit_12m": nb_demandes,
        "Historique_remboursement": coder_historique(historique),
        "Incidents_paiement": incidents,
        "Decouverts_retards": decouverts,
        "Prets_souffrance": souffrance,
        "Paiement_eau": coder_paiement(eau),
        "Paiement_electricite": coder_paiement(electricite),
        "Paiement_telecom": coder_paiement(telecom),
        "Paiement_loyer": coder_paiement(loyer),
        "Paiement_taxe_habitation": coder_paiement(taxe_habitation),
        "Paiement_TSC": coder_paiement(tsc),
        "Taux_endettement": charges / revenu if revenu > 0 else 1
    }])

    base = pd.get_dummies(
        base,
        columns=["Situation_familiale", "Adresse_zone", "Type_emploi"],
        drop_first=False
    )

    client_enrichi = base.reindex(columns=colonnes_enrichies, fill_value=0)
    client_trad = base.reindex(columns=colonnes_trad, fill_value=0)

    return client_enrichi, client_trad


# ==========================================================
# PDF
# ==========================================================

def creer_pdf(resultat_trad, resultat_enrichi, scenario, modele):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, "Rapport - Application de Scoring Enrichi")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, "Simulation de l'évaluation du risque de crédit")
    c.drawString(50, 745, f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    c.drawString(50, 720, f"Scénario : {scenario}")
    c.drawString(50, 695, f"Algorithme utilisé : {modele}")

    c.drawString(50, 655, "Modèle traditionnel")
    c.drawString(70, 635, f"Classe : {resultat_trad['classe']}")
    c.drawString(70, 615, f"Risque : {resultat_trad['risque']}")
    c.drawString(70, 595, f"Décision : {resultat_trad['decision']}")
    c.drawString(70, 575, f"Confiance : {resultat_trad['confiance']:.1f} %")

    c.drawString(50, 535, "Modèle enrichi")
    c.drawString(70, 515, f"Classe : {resultat_enrichi['classe']}")
    c.drawString(70, 495, f"Risque : {resultat_enrichi['risque']}")
    c.drawString(70, 475, f"Décision : {resultat_enrichi['decision']}")
    c.drawString(70, 455, f"Confiance : {resultat_enrichi['confiance']:.1f} %")

    c.drawString(50, 410, "Remarque : prototype académique développé dans le cadre d'un mémoire.")
    c.drawString(50, 390, "Les résultats ne constituent pas une décision bancaire réelle.")

    c.save()
    buffer.seek(0)
    return buffer


# ==========================================================
# HEADER
# ==========================================================

def image_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def afficher_header():
    logo_bp = image_to_base64("BP.png")
    logo_iscae = image_to_base64("ISCAE.png")

    bp_html = (
        f'<img src="data:image/png;base64,{logo_bp}" alt="Banque Populaire">'
        if logo_bp else '<div style="font-weight:700;color:#0f4c81;">Banque Populaire</div>'
    )

    iscae_html = (
        f'<img src="data:image/png;base64,{logo_iscae}" alt="ISCAE">'
        if logo_iscae else '<div style="font-weight:700;color:#0f4c81;">ISCAE</div>'
    )

    st.markdown(f"""
    <div class="main-header">
        <div class="logo-box">{bp_html}</div>
        <div class="header-center">
            <div class="main-title">Application de Scoring Enrichi</div>
            <div class="main-subtitle">Simulation de l'évaluation du risque de crédit</div>
        </div>
        <div class="logo-box">{iscae_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# PAGE 1 : NOUVELLE SIMULATION
# ==========================================================

def page_simulation(resultats_df, objets_modeles):
    st.markdown('<div class="section-title">⚙️ Paramètres du modèle</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        scenario_choisi = st.selectbox(
            "Scénario de pondération",
            list(SCENARIOS.keys())
        )

    with c2:
        modele_choisi = st.selectbox(
            "Algorithme utilisé",
            ALGORITHMES
        )

    st.info(
        "Cette partie sert à évaluer un client individuel. "
        "Le scénario choisi détermine le poids accordé aux données alternatives dans le score simulé."
    )

    left, right = st.columns([1.25, 0.75])

    with left:
        st.markdown('<div class="section-title">👤 Informations personnelles</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Âge", 18, 80, 35)
            situation = st.selectbox("Situation familiale", ["Celibataire", "Marie", "Divorce/Veuf"])
        with c2:
            zone = st.selectbox("Zone géographique", ["Urbain", "Semi-urbain", "Rural"])
            emploi = st.selectbox("Type d'emploi", ["Salarie", "Fonctionnaire", "Independant"])
        with c3:
            anciennete = st.number_input("Ancienneté emploi", 0.0, 45.0, 5.0)
            historique = st.selectbox("Historique remboursement", ["Bon", "Moyen", "Mauvais"])

        st.markdown('<div class="section-title">💰 Situation financière</div>', unsafe_allow_html=True)

        f1, f2, f3 = st.columns(3)
        with f1:
            revenu = st.number_input("Revenu mensuel MAD", 1000, 50000, 9000)
            encours = st.number_input("Encours crédits MAD", 0, 500000, 50000)
        with f2:
            charges = st.number_input("Charges mensuelles MAD", 0, 30000, 2500)
            nb_credits = st.number_input("Nombre crédits en cours", 0, 10, 2)
        with f3:
            nb_demandes = st.number_input("Demandes crédit 12 mois", 0, 10, 1)
            incidents = st.selectbox("Incidents paiement", [0, 1])

        decouverts = st.number_input("Découverts / retards", 0, 10, 0)
        souffrance = st.number_input("Prêts en souffrance", 0, 5, 0)

        st.markdown('<div class="section-title">📌 Données alternatives</div>', unsafe_allow_html=True)

        a1, a2, a3 = st.columns(3)
        with a1:
            eau = st.selectbox("Paiement eau", ["Régulier", "Quelques retards", "Retards fréquents"])
            electricite = st.selectbox("Paiement électricité", ["Régulier", "Quelques retards", "Retards fréquents"])
        with a2:
            telecom = st.selectbox("Paiement télécom", ["Régulier", "Quelques retards", "Retards fréquents"])
            loyer = st.selectbox("Paiement loyer", ["Régulier", "Quelques retards", "Retards fréquents"])
        with a3:
            taxe_habitation = st.selectbox("Taxe habitation", ["Régulier", "Quelques retards", "Retards fréquents"])
            tsc = st.selectbox("Taxe services communaux", ["Régulier", "Quelques retards", "Retards fréquents"])

        predire = st.button("Évaluer le risque de crédit", use_container_width=True)

    with right:
        st.markdown('<div class="section-title">📊 Résultat du scoring</div>', unsafe_allow_html=True)

        if predire:
            pack = objets_modeles[scenario_choisi]

            client_enrichi, client_trad = creer_client(
                pack["colonnes_enrichies"],
                pack["colonnes_trad"],
                age, situation, zone, emploi, anciennete, revenu, encours, charges,
                nb_credits, nb_demandes, historique, incidents, decouverts, souffrance,
                eau, electricite, telecom, loyer, taxe_habitation, tsc
            )

            modele_enrichi = pack["modeles_enrichis"][modele_choisi]
            modele_trad = pack["modeles_trad"][modele_choisi]

            pred_e = modele_enrichi.predict(client_enrichi)[0]
            pred_t = modele_trad.predict(client_trad)[0]

            classe_e = label_classe(pred_e)
            classe_t = label_classe(pred_t)

            risque_e, decision_e, couleur_e, css_e = infos_risque(classe_e)
            risque_t, decision_t, couleur_t, css_t = infos_risque(classe_t)

            proba_e = proba_par_classe(modele_enrichi, client_enrichi)
            proba_t = proba_par_classe(modele_trad, client_trad)

            confiance_e = max(proba_e.values()) * 100
            confiance_t = max(proba_t.values()) * 100

            st.markdown(f"""
            <div class='result-card'>
                <h3>Modèle enrichi</h3>
                <div class='{css_e}'>{classe_e}</div>
                <p><b>Risque :</b> {risque_e}</p>
                <p><b>Décision :</b> {decision_e}</p>
                <p><b>Confiance :</b> {confiance_e:.1f} %</p>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confiance_e,
                title={"text": "Confiance du modèle enrichi"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": couleur_e},
                    "steps": [
                        {"range": [0, 60], "color": "#fee2e2"},
                        {"range": [60, 80], "color": "#fef3c7"},
                        {"range": [80, 100], "color": "#dcfce7"}
                    ]
                }
            ))
            fig.update_layout(paper_bgcolor="white", font={"color": "#1e293b"})
            st.plotly_chart(fig, use_container_width=True)

            comparaison = pd.DataFrame({
                "Approche": ["Traditionnelle", "Enrichie"],
                "Classe": [classe_t, classe_e],
                "Risque": [risque_t, risque_e],
                "Confiance": [f"{confiance_t:.1f} %", f"{confiance_e:.1f} %"],
                "Décision": [decision_t, decision_e]
            })

            st.write("### Comparaison client")
            st.dataframe(comparaison, use_container_width=True, hide_index=True)

            proba_df = pd.DataFrame({
                "Classe": ["S1", "S2", "S3"],
                "Modèle enrichi": [proba_e["S1"], proba_e["S2"], proba_e["S3"]],
                "Modèle traditionnel": [proba_t["S1"], proba_t["S2"], proba_t["S3"]]
            })

            st.write("### Probabilités")
            st.bar_chart(proba_df.set_index("Classe"))

            pdf = creer_pdf(
                {
                    "classe": classe_t,
                    "risque": risque_t,
                    "decision": decision_t,
                    "confiance": confiance_t
                },
                {
                    "classe": classe_e,
                    "risque": risque_e,
                    "decision": decision_e,
                    "confiance": confiance_e
                },
                scenario_choisi,
                modele_choisi
            )

            st.download_button(
                "Exporter le rapport PDF",
                data=pdf,
                file_name="rapport_scoring_credit.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        else:
            st.info("Renseignez les informations du client puis cliquez sur « Évaluer le risque de crédit ».")


# ==========================================================
# PAGE 2 : ANALYSE DES MODELES
# ==========================================================

def page_analyse_modeles(resultats_df, objets_modeles):
    st.markdown('<div class="section-title">📊 Analyse des modèles</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        scenario = st.selectbox("Scénario", list(SCENARIOS.keys()), key="analyse_scenario")
    with c2:
        algo = st.selectbox("Algorithme", ALGORITHMES, key="analyse_algo")

    ligne = resultats_df[
        (resultats_df["Scénario"] == scenario)
        & (resultats_df["Algorithme"] == algo)
        ].iloc[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy traditionnel", f"{ligne['Accuracy traditionnel']:.2%}")
    m2.metric("Accuracy enrichi", f"{ligne['Accuracy enrichi']:.2%}")
    m3.metric("Gain", f"{ligne['Gain']:.2%}")

    st.write("### Indicateurs détaillés")

    details = pd.DataFrame({
        "Indicateur": ["Precision", "Recall", "F1-score"],
        "Traditionnel": [
            ligne["Precision traditionnel"],
            ligne["Recall traditionnel"],
            ligne["F1 traditionnel"]
        ],
        "Enrichi": [
            ligne["Precision enrichi"],
            ligne["Recall enrichi"],
            ligne["F1 enrichi"]
        ]
    })

    st.dataframe(details, use_container_width=True, hide_index=True)

    pack = objets_modeles[scenario]
    tests = pack["tests"][algo]

    cm_e = confusion_matrix(tests["y_test_e"], tests["pred_e"])
    cm_t = confusion_matrix(tests["y_test_t"], tests["pred_t"])

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Matrice de confusion - Traditionnel")
        fig_t = px.imshow(
            cm_t,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Classe prédite", y="Classe réelle")
        )
        st.plotly_chart(fig_t, use_container_width=True)

    with col2:
        st.write("### Matrice de confusion - Enrichi")
        fig_e = px.imshow(
            cm_e,
            text_auto=True,
            color_continuous_scale="Greens",
            labels=dict(x="Classe prédite", y="Classe réelle")
        )
        st.plotly_chart(fig_e, use_container_width=True)


# ==========================================================
# PAGE 3 : COMPARAISON DES SCENARIOS
# ==========================================================

def page_comparaison_scenarios(resultats_df):
    st.markdown('<div class="section-title">📈 Comparaison des scénarios de pondération</div>', unsafe_allow_html=True)

    st.info(
        "Cette page ne concerne pas un client individuel. "
        "Elle compare les performances globales des modèles sur l'échantillon de test."
    )

    algo = st.selectbox("Choisir l'algorithme", ALGORITHMES, key="scenario_algo")
    df_algo = resultats_df[resultats_df["Algorithme"] == algo].copy()

    st.write("### Résultats par scénario")
    st.dataframe(df_algo, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_algo["Scénario"],
        y=df_algo["Accuracy traditionnel"],
        name="Modèle traditionnel",
        marker_color="#0f4c81"
    ))
    fig.add_trace(go.Bar(
        x=df_algo["Scénario"],
        y=df_algo["Accuracy enrichi"],
        name="Modèle enrichi",
        marker_color="#f59e0b"
    ))

    fig.update_layout(
        barmode="group",
        title=f"Comparaison des accuracies - {algo}",
        yaxis_title="Accuracy",
        xaxis_title="Scénario",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#1e293b"}
    )

    st.plotly_chart(fig, use_container_width=True)

    fig_gain = go.Figure(go.Bar(
        x=df_algo["Scénario"],
        y=df_algo["Gain"],
        marker_color=["#16a34a" if g >= 0 else "#dc2626" for g in df_algo["Gain"]]
    ))

    fig_gain.update_layout(
        title="Gain du modèle enrichi par rapport au modèle traditionnel",
        yaxis_title="Gain",
        xaxis_title="Scénario",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#1e293b"}
    )

    st.plotly_chart(fig_gain, use_container_width=True)


# ==========================================================
# PAGE 4 : A PROPOS
# ==========================================================

def page_a_propos():
    st.markdown('<div class="section-title">ℹ️ À propos du prototype</div>', unsafe_allow_html=True)

    st.write("""
    Ce prototype académique illustre l'intégration de données alternatives dans un système de scoring du risque de crédit.

    L'application distingue deux approches :

    - **Modèle traditionnel** : utilise uniquement les variables bancaires et socio-économiques classiques.
    - **Modèle enrichi** : ajoute des données alternatives telles que le paiement de l'eau, de l'électricité, du téléphone, du loyer et des taxes locales.

    Les scénarios de pondération permettent d'étudier l'effet de différents niveaux d'intégration des données alternatives :

    - Alternatives faibles : 10 %
    - Alternatives moyennes : 20 %
    - Alternatives renforcées : 30 %

    Les résultats sont basés sur une base de données simulée. Ils ne constituent donc pas une décision bancaire réelle.
    L'objectif est de démontrer une méthodologie et d'analyser le potentiel des données alternatives dans le contexte marocain.
    """)


# ==========================================================
# APPLICATION PRINCIPALE
# ==========================================================

afficher_header()

with st.spinner("Chargement et entraînement des modèles..."):
    resultats_df, objets_modeles = entrainer_tous_les_scenarios()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisir une page",
    [
        "🏠 Nouvelle simulation",
        "📊 Analyse des modèles",
        "📈 Comparaison des scénarios",
        "ℹ️ À propos"
    ]
)

if page == "🏠 Nouvelle simulation":
    page_simulation(resultats_df, objets_modeles)

elif page == "📊 Analyse des modèles":
    page_analyse_modeles(resultats_df, objets_modeles)

elif page == "📈 Comparaison des scénarios":
    page_comparaison_scenarios(resultats_df)

else:
    page_a_propos()

st.markdown("""
<div class="footer">
Prototype académique développé dans le cadre d'un mémoire de recherche — ISCAE / Banque Populaire
</div>
""", unsafe_allow_html=True)
