import pandas as pd


def parse_range(value):
    """Преобразует '4.5–7.5' в среднее float"""
    if isinstance(value, str) and "–" in value:
        a, b = value.split("–")
        return (float(a) + float(b)) / 2
    try:
        return float(value)
    except:
        return None


def process_soil(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "%Гумус (0–30 см)",
        "pH (вода)",
        "CEC (смоль(+)/кг)",
        "Текстура (% глина)",
        "%Каменистость",
        "Потенциал урожая пшеницы (ц/га)",
        "Неопределенность (±%)"
    ]

    for col in numeric_cols:
        df[col] = df[col].apply(parse_range)

    df["Год"] = df["Год"].astype(int)

    df = pd.get_dummies(df, columns=["Регион / Тип почвы"], drop_first=True)

    df = df.fillna(df.mean())

    return df
