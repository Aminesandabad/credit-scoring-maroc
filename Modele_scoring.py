import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. CHARGEMENT ET CODIFICATION DE LA BASE
# ==========================================

df_original = pd.read_excel("base_scoring_credit_maroc_classe.xlsx", engine="openpyxl")

df_original["Historique_remboursement"] = df_original["Historique_remboursement"].map({
    "Mauvais": 1,
    "Moyen": 2,
    "Bon": 3
})

mapping_paiement = {
    "Retards frequents": 1,
    "Quelques retards": 2,
    "Regulier": 3
}

colonnes_alternatives = [
    "Paiement_eau",
    "Paiement_electricite",
    "Paiement_telecom",
    "Paiement_loyer",
    "Paiement_taxe_habitation",
    "Paiement_TSC"
]

for col in colonnes_alternatives:
    df_original[col] = df_original[col].map(mapping_paiement)


# ==========================================
# 2. FONCTION : CREATION DE CLASSE_SCORING
# ==========================================

def creer_classe_scoring(df_base, poids_alternatif):
    """
    Crée un score enrichi selon un scénario de pondération.
    Exemple :
    poids_alternatif = 0.10 signifie alternatives = 10 %.
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


# ==========================================
# 3. FONCTION : PREPARATION DE LA BASE
# ==========================================

def preparer_base_modele(df_scenario):
    """
    Prépare la base pour l'entraînement :
    - suppression des colonnes inutiles
    - transformation des variables nominales
    - création de X enrichi et X traditionnel
    """

    df = df_scenario.copy()

    df = df.drop(columns=["Client_ID", "Score_Enrichi", "Defaut_paiement"], errors="ignore")

    df = pd.get_dummies(
        df,
        columns=["Situation_familiale", "Adresse_zone", "Type_emploi"],
        drop_first=False
    )

    X_enrichi = df.drop(columns=["Classe_Scoring"])

    X_traditionnel = df.drop(columns=[
        "Classe_Scoring",
        "Paiement_eau",
        "Paiement_electricite",
        "Paiement_telecom",
        "Paiement_loyer",
        "Paiement_taxe_habitation",
        "Paiement_TSC"
    ], errors="ignore")

    y = df["Classe_Scoring"]

    return X_enrichi, X_traditionnel, y, df


# ==========================================
# 4. FONCTION : ENTRAINER ET EVALUER
# ==========================================

def entrainer_evaluer(nom_modele, modele, X_train, X_test, y_train, y_test, afficher_details=False):
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    if afficher_details:
        print("\n===================================")
        print(nom_modele)
        print("===================================")
        print("Accuracy :", accuracy)
        print("\nMatrice de confusion :")
        print(confusion_matrix(y_test, y_pred))
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred))

    return accuracy


# ==========================================
# 5. SCENARIOS DE PONDERATION
# ==========================================

scenarios = {
    "Alternatives faibles (10%)": 0.10,
    "Alternatives moyennes (20%)": 0.20,
    "Alternatives renforcées (30%)": 0.30
}

modeles = {
    "Arbre de décision": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    "Régression Logistique": Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42))
    ])
}

tous_resultats = []


# ==========================================
# 6. BOUCLE SUR LES SCENARIOS
# ==========================================

for nom_scenario, poids_alt in scenarios.items():

    df_scenario = creer_classe_scoring(df_original, poids_alt)
    X_enrichi, X_traditionnel, y, df_final = preparer_base_modele(df_scenario)

    X_train_e, X_test_e, y_train, y_test = train_test_split(
        X_enrichi,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_traditionnel,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n===================================")
    print(f"SCENARIO : {nom_scenario}")
    print("===================================")
    print("Répartition des classes :")
    print(y.value_counts(normalize=True) * 100)

    for nom_modele, modele in modeles.items():

        acc_enrichi = entrainer_evaluer(
            f"{nom_modele} - Modèle enrichi - {nom_scenario}",
            modele,
            X_train_e,
            X_test_e,
            y_train,
            y_test,
            afficher_details=False
        )

        modele_trad = modeles[nom_modele]

        acc_trad = entrainer_evaluer(
            f"{nom_modele} - Modèle traditionnel - {nom_scenario}",
            modele_trad,
            X_train_t,
            X_test_t,
            y_train_t,
            y_test_t,
            afficher_details=False
        )

        tous_resultats.append({
            "Scénario": nom_scenario,
            "Poids alternatives": poids_alt,
            "Algorithme": nom_modele,
            "Accuracy traditionnel": acc_trad,
            "Accuracy enrichi": acc_enrichi,
            "Gain": acc_enrichi - acc_trad
        })


# ==========================================
# 7. TABLEAU FINAL
# ==========================================

resultats_scenarios = pd.DataFrame(tous_resultats)

print("\n===================================")
print("COMPARAISON FINALE DES SCENARIOS")
print("===================================")
print(resultats_scenarios)

resultats_scenarios.to_excel("comparaison_scenarios_modeles.xlsx", index=False)

print("\nFichier comparaison_scenarios_modeles.xlsx créé avec succès.")