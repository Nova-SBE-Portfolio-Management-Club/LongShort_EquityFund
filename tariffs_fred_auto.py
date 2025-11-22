"""
Análise automática dos efeitos das tarifas de 2018 em aço e alumínio
Usa dados da FRED, sem necessidade de CSVs locais.

Passos para correr:
1) Instalar dependências no venv:
   pip install fredapi pandas matplotlib

2) Meter a tua API KEY da FRED na variável FRED_API_KEY abaixo.

3) Correr:
   python tariffs_fred_auto.py
"""

import pandas as pd
import matplotlib.pyplot as plt

try:
    from fredapi import Fred
except ImportError:
    raise ImportError(
        "fredapi não está instalado. Dentro do teu venv corre: pip install fredapi"
    )

# ============================
# 0. CONFIGURAR API FRED
# ============================

# >>>>>> METE AQUI A TUA API KEY DA FRED <<<<<<
FRED_API_KEY = "e7eeea8e00afb54a74c357bd916b3d7e"

fred = Fred(api_key=FRED_API_KEY)

# ============================
# 1. PARÂMETROS DO ESTUDO
# ============================

START_DATE = "2010-01-01"
END_DATE = "2022-12-31"
TARIFF_START = pd.Timestamp("2018-03-01")

# Séries da FRED que vamos usar:
SERIES = {
    "primary_metal_ip": "IPG331S",         # Industrial Production: Primary Metal
    "aluminum_ppi": "PCU331315331315",     # PPI: Aluminum sheet/plate/foil
    "steel_ppi": "WPU10170502",            # PPI: Steel wire, stainless
    # Se quiseres, mais tarde podes adicionar um índice de import prices aqui:
    # "import_price_index": "IR142",  # Exemplo, confirma o código certo na FRED
}

# ============================
# 2. FUNÇÃO PARA IR BUSCAR SÉRIES
# ============================

def fetch_series(series_dict, start, end):
    """
    Vai buscar cada série ao FRED e devolve um DataFrame com:
    - coluna 'date'
    - uma coluna por indicador
    """
    from functools import reduce

    dfs = []
    for col_name, fred_code in series_dict.items():
        print(f"[INFO] Baixando série {fred_code} -> coluna '{col_name}'...")
        s = fred.get_series(fred_code, observation_start=start, observation_end=end)
        if s is None or len(s) == 0:
            raise ValueError(f"Série {fred_code} veio vazia. Verifica o código ou a API key.")
        df = s.to_frame(name=col_name)
        df["date"] = df.index
        df = df.reset_index(drop=True)
        df = df[["date", col_name]]
        dfs.append(df)

    merged = reduce(lambda left, right: pd.merge(left, right, on="date", how="inner"), dfs)
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged

# ============================
# 3. BUSCAR E PREPARAR DADOS
# ============================

df = fetch_series(SERIES, START_DATE, END_DATE)

# Criar dummy pré-pós tarifa
df["post_tariff"] = (df["date"] >= TARIFF_START).astype(int)

print("\n[INFO] Primeiras linhas do DataFrame combinado:")
print(df.head(), "\n")

# ============================
# 4. TABELA PRÉ vs PÓS TARIFA
# ============================

cols_for_summary = [c for c in df.columns if c not in ["date", "post_tariff"]]

summary = df.groupby("post_tariff")[cols_for_summary].mean().T
summary.columns = ["pre_tariff", "post_tariff"]
summary["pct_change_%"] = (summary["post_tariff"] / summary["pre_tariff"] - 1) * 100

print("[INFO] MÉDIAS PRÉ vs PÓS TARIFA (2010–2022):")
print(summary.round(2))

# Guardar para análise no Word/Excel
summary_outfile = "summary_pre_post_tariff_fred.csv"
summary.to_csv(summary_outfile)
print(f"\n[INFO] Tabela guardada em: {summary_outfile}")

# ============================
# 5. GRÁFICOS
# ============================

def add_tariff_line():
    """Desenha linha vertical na data da tarifa."""
    plt.axvline(TARIFF_START, linestyle="--")

# 5.1 Produção – Primary Metals
plt.figure()
plt.plot(df["date"], df["primary_metal_ip"], label="Industrial Production – Primary Metals (IPG331S)")
add_tariff_line()
plt.title("Industrial Production: Primary Metals (Index)")
plt.xlabel("Date")
plt.ylabel("Index")
plt.legend()
plt.tight_layout()
prod_plot = "fred_production_primary_metals.png"
plt.savefig(prod_plot, dpi=300)

# 5.2 Preços – Aluminum vs Steel
plt.figure()
plt.plot(df["date"], df["aluminum_ppi"], label="PPI – Aluminum sheet/plate/foil (PCU331315331315)")
plt.plot(df["date"], df["steel_ppi"], label="PPI – Steel wire, stainless (WPU10170502)")
add_tariff_line()
plt.title("Domestic Prices: Aluminum vs Steel (PPI)")
plt.xlabel("Date")
plt.ylabel("Index")
plt.legend()
plt.tight_layout()
price_plot = "fred_prices_aluminum_steel.png"
plt.savefig(price_plot, dpi=300)

print("\n[INFO] Gráficos guardados como:")
print(f"  - {prod_plot}")
print(f"  - {price_plot}")
print("\n[INFO] Done.")
