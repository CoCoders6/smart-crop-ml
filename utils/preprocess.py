# Minimal preprocessing (for future use)

def prepare_features(soil, rain=100):
    """
    Convert soil dict to ML features.
    soil: dict with keys N, P, K, pH
    rain: estimated rainfall (optional)
    """
    return [[soil.get('pH', 6.0), soil.get('N', 100), rain]]
