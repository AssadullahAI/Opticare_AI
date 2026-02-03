def load_description(disease_name):
    txt_path = f"data/descriptions/{disease_name}.txt"

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Description not available."
