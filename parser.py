import re
from datetime import datetime

def parse_posturography_data(content: str, file_name: str) -> dict:
    """Parse a posturography export into structured data."""
    data = {
        "fileName": file_name,
        "date": datetime.now(),
        "tests": {},
    }
    date_set = False

    try:
        test_blocks = re.split(r"Test \d+: ", content)[1:]
        if not test_blocks:
            return None
    except Exception:
        return None

    for block in test_blocks:
        lines = block.split("\n")
        test_name_line = lines[0].strip()

        test_key = ""
        key_match = re.search(r"([\w\s]+) \(([A-Z]{4})\)", test_name_line)

        if key_match and key_match.group(1) and key_match.group(2):
            description = key_match.group(1).strip()
            acronym = key_match.group(2).strip()

            if description.startswith("Foam Eyes Closed"):
                variant = description.split(" ")[-1]
                test_key = f"{acronym} {variant}"
            else:
                test_key = acronym
        else:
            short_name_match = re.search(r"([A-Z]{4}):?$", test_name_line)
            if short_name_match and short_name_match.group(1):
                test_key = short_name_match.group(1)
            else:
                test_key = test_name_line.split(":")[0].strip()

        if not test_key:
            continue

        test_data = {
            "fullName": re.sub(r":$", "", test_name_line),
            "metrics": {},
        }

        for line in lines[1:]:
            line = line.strip()
            if line.startswith("* Test Date:"):
                date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", line)
                if date_match and not date_set:
                    try:
                        data["date"] = datetime.strptime(date_match.group(1), "%m/%d/%Y")
                        date_set = True
                    except ValueError:
                        pass

            elif line.startswith("*"):
                parts = line[1:].split(":", 1)
                if len(parts) < 2:
                    continue

                key = parts[0].strip()
                value = parts[1].strip()

                numeric_match = re.search(r"^(-?[\d\.]+)", value)

                if numeric_match:
                    if "Sway Center" in key:
                        coords = re.search(r"\[(-?[\d\.]+)\s*m, (-?[\d\.]+)\s*m\]", value)
                        if coords:
                            test_data["metrics"][f"{key} Xo"] = float(coords.group(1))
                            test_data["metrics"][f"{key} Yo"] = float(coords.group(2))

                    elif re.search(r"\((Max & 95%|Max & Ave|Max & Min)\)", key):
                        values = re.search(
                            r"(-?[\d\.]+)\s*m(?:²?\/s?²?)? & (-?[\d\.]+)\s*m(?:²?\/s?²?)?",
                            value,
                        )
                        if values:
                            key_root = key.split("(")[0].strip()
                            key_type_match = re.search(r"\((.*?)\)", key)
                            if key_type_match:
                                val1_name, val2_name = key_type_match.group(1).split(" & ")
                                test_data["metrics"][f"{key_root} ({val1_name})"] = float(
                                    values.group(1)
                                )
                                test_data["metrics"][f"{key_root} ({val2_name})"] = float(
                                    values.group(2)
                                )

                    else:
                        test_data["metrics"][key] = float(numeric_match.group(1))

                elif key == "Stability Class":
                    test_data["metrics"][key] = value.split("|")[0].strip()

        if test_data["metrics"]:
            data["tests"][test_key] = test_data

    return data if data["tests"] else None
