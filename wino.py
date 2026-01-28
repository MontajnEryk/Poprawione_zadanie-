import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------
# Custom CSS for modern styling and rounded corners
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Style the sidebar */
    .css-1d391kg {
        border-radius: 15px;
        background-color: #f0f0f5;
    }
    .css-1omlzdg {
        font-size: 18px;
        font-weight: bold;
        color: #333333;
    }
    /* Style the main content */
    .css-ffhzg2 {
        border-radius: 15px;
        background-color: #ffffff;
        padding: 20px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    .css-1j1w6p3.edgvbvh3 {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
    }
    .css-19yvsms {
        border-radius: 10px;
        border: 1px solid #dcdcdc;
    }
    .css-1kptt2d {
        border-radius: 10px;
        padding: 10px;
        background-color: #f9f9f9;
    }
    .css-2trqye {
        font-size: 24px;
        color: #333333;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Konfiguracja strony
# ---------------------------------------------------------
st.set_page_config(
    page_title="Wine Analytics & Food Pairings",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍷 Wine Analytics & Food Pairings")
st.markdown("Aplikacja do eksploracji jakości czerwonych win oraz parowania win z jedzeniem.")

# ---------------------------------------------------------
# Funkcje wczytywania danych
# ---------------------------------------------------------
@st.cache_data
def load_wine_quality(path: str = "winequality-red_filled.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_wine_food_pairings(path: str = "wine_food_pairings_filled.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# ---------------------------------------------------------
# Próba wczytania danych + komunikaty błędów
# ---------------------------------------------------------
wine_quality_df, pairings_df = None, None
wine_quality_error, pairings_error = None, None

try:
    wine_quality_df = load_wine_quality()
except Exception as e:
    wine_quality_error = str(e)

try:
    pairings_df = load_wine_food_pairings()
except Exception as e:
    pairings_error = str(e)

# ---------------------------------------------------------
# Sidebar – wybór modułu
# ---------------------------------------------------------
st.sidebar.header("⚙️ Ustawienia")
module = st.sidebar.radio(
    "Wybierz moduł:",
    options=["Analiza jakości wina", "Parowanie wina z jedzeniem", "Rekomendacje"]
)

# =========================================================
# 1. ANALIZA JAKOŚCI WINA (winequality-red.csv)
# =========================================================
if module == "Analiza jakości wina":
    st.subheader("📊 Analiza jakości czerwonych win")

    if wine_quality_df is None:
        st.error(f"Nie udało się wczytać `winequality-red_filled.csv`.\n\n{wine_quality_error}")
        st.stop()

    df = wine_quality_df.copy()

    # -------------------------
    # Podstawowe informacje
    # -------------------------
    st.markdown("### Podstawowa eksploracja danych")
    st.dataframe(df.head())

    with st.expander("Informacje o datasetcie"):
        st.write(f"**Kształt:** {df.shape}")
        st.write(f"**Typy danych:**")
        st.write(df.dtypes)
        st.write(f"**Brakujące wartości:**")
        st.write(df.isnull().sum())
        st.write(f"**Duplikaty:** {df[df.duplicated()].shape[0]}")

    # -------------------------
    # Filtrowanie po jakości
    # -------------------------
    st.markdown("### Filtrowanie po ocenie jakości")
    quality_range = st.slider(
        "Zakres jakości (kolumna `quality`):",
        min_value=int(df["quality"].min()),
        max_value=int(df["quality"].max()),
        value=(int(df["quality"].min()), int(df["quality"].max())),
        step=1
    )

    filtered = df[(df["quality"] >= quality_range[0]) & (df["quality"] <= quality_range[1])]
    st.write(f"Liczba rekordów po filtrze: **{filtered.shape[0]}**")
    st.dataframe(filtered.head())

    # -------------------------
    # Wykresy
    # -------------------------
    feature_choice = st.selectbox("Wybierz cechę:", df.columns)

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df[feature_choice], bins=30, edgecolor="black")
    st.pyplot(fig)

    # Boxplot
    fig_box, ax_box = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=df[feature_choice], ax=ax_box)
    st.pyplot(fig_box)

    # -------------------------
    # Rekomendacje win
    # -------------------------
    st.markdown("### Rekomendacje win na podstawie cech")
    alcohol_range = st.slider("Zakres alkoholu:", min_value=int(df["alcohol"].min()), max_value=int(df["alcohol"].max()), value=(int(df["alcohol"].min()), int(df["alcohol"].max())))
    acidity_range = st.slider("Zakres kwasowości:", min_value=int(df["volatile acidity"].min()), max_value=int(df["volatile acidity"].max()), value=(int(df["volatile acidity"].min()), int(df["volatile acidity"].max())))

    filtered_wine = df[
        (df["alcohol"] >= alcohol_range[0]) & (df["alcohol"] <= alcohol_range[1]) &
        (df["volatile acidity"] >= acidity_range[0]) & (df["volatile acidity"] <= acidity_range[1])
    ]
    st.write(f"Liczba win spełniających kryteria: {filtered_wine.shape[0]}")
    st.dataframe(filtered_wine)

# =========================================================
# 2. PAROWANIE WINA Z JEDZENIEM (wine_food_pairings.csv)
# =========================================================
elif module == "Parowanie wina z jedzeniem":
    st.subheader("🍽️ Parowanie wina z jedzeniem")

    if pairings_df is None:
        st.error(f"Nie udało się wczytać `wine_food_pairings_filled.csv`.\n\n{pairings_error}")
        st.stop()

    dfp = pairings_df.copy()

    # -------------------------
    # Podstawowa eksploracja danych
    # -------------------------
    st.markdown("### Podstawowa eksploracja danych (parowanie wina z jedzeniem)")
    st.dataframe(dfp.head())

    # -------------------------
    # Filtrowanie rekomendacji
    # -------------------------
    wine_type_sel = st.multiselect("Wybierz typ wina:", options=dfp["wine_type"].unique())
    food_cat_sel = st.multiselect("Wybierz kategorię jedzenia:", options=dfp["food_category"].unique())
    cuisine_sel = st.multiselect("Wybierz kuchnię:", options=dfp["cuisine"].unique())
    min_pair_quality = st.slider("Minimalna ocena parowania:", min_value=int(dfp["pairing_quality"].min()), max_value=int(dfp["pairing_quality"].max()), value=int(dfp["pairing_quality"].min()))

    filtered_pairings = dfp[
        (dfp["pairing_quality"] >= min_pair_quality) &
        (dfp["wine_type"].isin(wine_type_sel) if wine_type_sel else True) &
        (dfp["food_category"].isin(food_cat_sel) if food_cat_sel else True) &
        (dfp["cuisine"].isin(cuisine_sel) if cuisine_sel else True)
    ]
    st.write(f"Liczba rekordów po filtrze: **{filtered_pairings.shape[0]}**")
    st.dataframe(filtered_pairings.head())

# =========================================================
# 3. Rekomendacje win na podstawie jakości i cech
# =========================================================
elif module == "Rekomendacje":
    st.subheader("🍇 Rekomendacje win")
    quality_input = st.slider("Wybierz jakość wina (1-10):", 1, 10, 5)
    alcohol_input = st.slider("Zakres alkoholu:", min_value=0, max_value=15, value=(5, 12))
    acidity_input = st.slider("Zakres kwasowości:", min_value=0.1, max_value=1.0, value=(0.2, 0.8))

    recommended_wines = wine_quality_df[
        (wine_quality_df["quality"] == quality_input) &
        (wine_quality_df["alcohol"] >= alcohol_input[0]) & (wine_quality_df["alcohol"] <= alcohol_input[1]) &
        (wine_quality_df["volatile acidity"] >= acidity_input[0]) & (wine_quality_df["volatile acidity"] <= acidity_input[1])
    ]
    st.write(f"Liczba rekomendowanych win: **{recommended_wines.shape[0]}**")
    st.dataframe(recommended_wines)

