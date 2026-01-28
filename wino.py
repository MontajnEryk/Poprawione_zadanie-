import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------
# Konfiguracja strony
# ---------------------------------------------------------
st.set_page_config(
    page_title="Wine Analytics & Food Pairings",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍷 Wine Analytics & Food Pairings")
st.markdown(
    "Aplikacja do eksploracji jakości czerwonych win oraz "
    "parowania win z jedzeniem."
)

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
    options=["Analiza jakości wina", "Parowanie wina z jedzeniem"]
)

# =========================================================
# 1. ANALIZA JAKOŚCI WINA (winequality-red.csv)
# =========================================================
if module == "Analiza jakości wina":
    st.subheader("📊 Analiza jakości czerwonych win")

    if wine_quality_df is None:
        st.error(
            "Nie udało się wczytać `winequality-red_filled.csv`.\n\n"
            f"Komunikat błędu:\n`{wine_quality_error}`\n\n"
            "Upewnij się, że plik znajduje się w tym samym katalogu co `app.py`."
        )
        st.stop()

    df = wine_quality_df.copy()

    # -------------------------
    # Podstawowe informacje
    # -------------------------
    st.markdown("### Podstawowa eksploracja danych")
    st.write("Pierwsze wiersze datasetu:")
    st.dataframe(df.head())

    with st.expander("Informacje o datasetcie"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Kształt (liczba rekordów, liczba kolumn):**")
            st.write(df.shape)
            st.write("**Typy danych:**")
            st.write(df.dtypes)
        with col2:
            st.write("**Brakujące wartości:**")
            missing = df.isnull().sum()
            st.write(missing[missing > 0])
            st.write("**Duplikaty:**")
            st.write(f"Liczba duplikatów: {df[df.duplicated()].shape[0]}")

    # -------------------------
    # Filtrowanie po jakości
    # -------------------------
    st.markdown("### Filtrowanie po ocenie jakości")
    min_q = int(df["quality"].min())
    max_q = int(df["quality"].max())

    quality_range = st.slider(
        "Zakres jakości (kolumna `quality`):",
        min_value=min_q,
        max_value=max_q,
        value=(min_q, max_q),
        step=1
    )

    feature = st.selectbox("Wybierz cechę do filtrowania:", df.columns)

    filtered = df[(df["quality"] >= quality_range[0]) & (df["quality"] <= quality_range[1])]
    filtered = filtered[(filtered[feature] >= st.slider(f"Zakres {feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean())))]
    st.write(f"Liczba rekordów po filtrze: **{filtered.shape[0]}**")
    st.dataframe(filtered.head())

    # Statystyki
    st.write(f"Średnia dla {feature}: {filtered[feature].mean():.2f}")
    st.write(f"Mediana dla {feature}: {filtered[feature].median():.2f}")
    st.write(f"Min/Max dla {feature}: {filtered[feature].min():.2f} / {filtered[feature].max():.2f}")

    # -------------------------
    # Rozkład cechy
    # -------------------------
    st.markdown("### Rozkład cechy")
    feature_choice = st.selectbox("Wybierz cechę:", df.columns)

    # Histogram
    fig, ax = plt.subplots()
    ax.hist(df[feature_choice], bins=30, edgecolor="black")
    ax.set_title(f"Histogram {feature_choice}")
    ax.set_xlabel(feature_choice)
    ax.set_ylabel("Liczba próbek")
    st.pyplot(fig)

    # Boxplot
    fig_box, ax_box = plt.subplots()
    sns.boxplot(x=df[feature_choice], ax=ax_box)
    ax_box.set_title(f"Boxplot {feature_choice}")
    st.pyplot(fig_box)

    # Porównanie rozkładu cechy
    quality_comparison = st.selectbox("Wybierz grupy jakości do porównania:", [5, 6])
    group1 = df[df["quality"] == quality_comparison]
    group2 = df[df["quality"] == (quality_comparison + 1)]

    fig_comp, ax_comp = plt.subplots()
    ax_comp.hist(group1[feature_choice], alpha=0.5, label=f"Quality {quality_comparison}")
    ax_comp.hist(group2[feature_choice], alpha=0.5, label=f"Quality {quality_comparison + 1}")
    ax_comp.legend()
    ax_comp.set_title(f"Porównanie {feature_choice} dla jakości {quality_comparison} vs {quality_comparison + 1}")
    st.pyplot(fig_comp)

    # -------------------------
    # Wykres 3D
    # -------------------------
    st.markdown("### Wykres 3D porównujący trzy cechy")
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    x = df['alcohol']
    y = df['volatile acidity']
    z = df['quality']
    ax_3d.scatter(x, y, z, c=z, cmap='viridis')
    ax_3d.set_xlabel('Alcohol')
    ax_3d.set_ylabel('Volatile Acidity')
    ax_3d.set_zlabel('Quality')
    st.pyplot(fig_3d)

# =========================================================
# 2. PAROWANIE WINA Z JEDZENIEM (wine_food_pairings.csv)
# =========================================================
elif module == "Parowanie wina z jedzeniem":
    st.subheader("🍽️ Parowanie wina z jedzeniem")

    if pairings_df is None:
        st.error(
            "Nie udało się wczytać `wine_food_pairings_filled.csv`.\n\n"
            f"Komunikat błędu:\n`{pairings_error}`\n\n"
            "Upewnij się, że plik znajduje się w tym samym katalogu co `app.py`."
        )
        st.stop()

    dfp = pairings_df.copy()

    # -------------------------
    # Podstawowa eksploracja danych
    # -------------------------
    st.markdown("### Podstawowa eksploracja danych (parowanie wina z jedzeniem)")
    st.write("Pierwsze wiersze datasetu:")
    st.dataframe(dfp.head())

    with st.expander("Informacje o datasetcie"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Kształt:**", dfp.shape)
            st.write("**Kolumny:**")
            st.write(list(dfp.columns))
        with col2:
            st.write("**Przykładowe wartości kategorii:**")
            st.write("wine_type:", dfp["wine_type"].unique()[:10])
            st.write("food_category:", dfp["food_category"].unique()[:10])
            st.write("cuisine:", dfp["cuisine"].unique()[:10])
            st.write("quality_label:", dfp["quality_label"].unique())

    # -------------------------
    # Filtrowanie rekomendacji
    # -------------------------
    st.markdown("### Filtrowanie rekomendacji")

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

    # Statystyki
    st.write(f"Średnia ocena parowania: {filtered_pairings['pairing_quality'].mean():.2f}")
    st.write(f"Mediana oceny parowania: {filtered_pairings['pairing_quality'].median():.2f}")
    st.write(f"Min/Max oceny parowania: {filtered_pairings['pairing_quality'].min():.2f} / {filtered_pairings['pairing_quality'].max():.2f}")
