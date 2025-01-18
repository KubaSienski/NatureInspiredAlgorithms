import pandas as pd


def load_data(file_path):
    """
    Wczytaj dane z pliku Excel ze wszystkich zakładek.
    """
    return pd.read_excel(file_path, sheet_name=None,
                         header=None)  # Wczytaj wszystkie zakładki jako słownik DataFrame'ów


def extract_dimension_from_sheet_name(sheet_name):
    """Extract dimension (n) from the sheet name."""
    # Find the first numeric sequence in the sheet name
    import re
    match = re.search(r'\d+', sheet_name)
    return int(match.group()) if match else 10  # Default to 10 if no match found


def calculate_ecdf(data, quality_thresholds):
    """
    Oblicz punkty do wykresu ECDF na podstawie danych.

    Parameters:
        data (dict): Słownik zawierający DataFrame'y z każdej zakładki pliku Excel.
        quality_thresholds (list): Lista progów jakości (PJ).

    Returns:
        dict: Słownik z wartościami ECDF dla każdego progu jakości, z każdej zakładki.
    """
    combined_ecdf_values = {}

    for sheet_name, sheet_data in data.items():
        n = extract_dimension_from_sheet_name(sheet_name)

        print(f"n = {n}, for sheet - {sheet_name}")

        # Definicja progów budżetu (41 progów)
        budget_thresholds = [int(n * (10 ** (0.1 * i))) for i in range(41)]

        ecdf_values = {quality: [] for quality in quality_thresholds}

        # Iteracja po progach budżetu
        for budget_idx in budget_thresholds:
            current_budget_data = sheet_data.iloc[
                min(budget_idx - 1, len(sheet_data) - 1)]  # Budżet odpowiada wierszowi (iteracja)

            # Dla każdego progu jakości liczymy liczbę testów poniżej tego progu
            for quality in quality_thresholds:
                count_below_threshold = (current_budget_data < quality).sum()
                ecdf_values[quality].append(count_below_threshold)

        combined_ecdf_values[sheet_name] = ecdf_values

    return combined_ecdf_values


def save_ecdf_to_excel(combined_ecdf_values, output_path):
    """
    Zapisz dane ECDF do pliku Excel.

    Parameters:
        combined_ecdf_values (dict): Wartości ECDF dla każdego progu jakości z każdej zakładki.
        output_path (str): Ścieżka do pliku wyjściowego Excel.
    """
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, ecdf_values in combined_ecdf_values.items():
            df = pd.DataFrame(ecdf_values)
            df.index.name = 'Progi budżetu'
            df.to_excel(writer, sheet_name=sheet_name)


def main():
    # Ścieżka do pliku Excel
    file_path = 'results_30.xlsx'

    # Ścieżka do pliku wyjściowego
    output_path = 'ecdf_30.xlsx'

    # Wczytaj dane
    data = load_data(file_path)

    # Definicje progów jakości (51 progów)
    quality_thresholds = [10 ** (-8 + 0.2 * i) for i in range(51)][::-1]  # Progi jakości w przedziale [10^-8, 10^2]

    # Oblicz ECDF
    combined_ecdf_values = calculate_ecdf(data, quality_thresholds)

    # Zapisz ECDF do pliku Excel
    save_ecdf_to_excel(combined_ecdf_values, output_path)


if __name__ == '__main__':
    main()
