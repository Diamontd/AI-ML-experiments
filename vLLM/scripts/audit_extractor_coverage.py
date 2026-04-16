import json
import re
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional

DATASET_PATH = Path("c:/Users/User/Desktop/Projects/SBER/validation_dataset_stage2.jsonl")


def parse_bool_ru(text: str, positive_words: List[str], negative_words: List[str]) -> Optional[bool]:
    t = text.lower()
    if any(w in t for w in positive_words):
        return True
    if any(w in t for w in negative_words):
        return False
    return None


def extract_osago_params(text: str) -> Dict[str, Any]:
    t = text.lower()
    params: Dict[str, Any] = {}

    patterns = {
        "engine_power_hp": r"(мощн(?:ость)?\s*[:=]?\s*(\d{2,3}))|((\d{2,3})\s*л\.с)",
        "driver_age": r"возраст\s*[:=]?\s*(\d{2})",
        "driving_experience_years": r"стаж\s*[:=]?\s*(\d{1,2})",
        "vehicle_market_value": r"стоим(?:ость)?\s*(?:авто)?\s*[:=]?\s*(\d{5,9})",
        "vehicle_year": r"год(?:\s*выпуска)?\s*[:=]?\s*(20\d{2}|19\d{2})",
        "accidents_last_3y": r"дтп(?:\s*за\s*3\s*года)?\s*[:=]?\s*(\d{1,2})",
        "yearly_mileage_km": r"пробег\s*[:=]?\s*(\d{3,6})",
        "additional_drivers_count": r"доп\s*(?:водител(?:ей|я)|)\s*[:=]?\s*(\d{1,2})",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, t)
        if match:
            nums = [g for g in match.groups() if g and g.isdigit()]
            if nums:
                value = int(nums[-1])
                params[key] = float(value) if key == "vehicle_market_value" else value

    city_match = re.search(r"регион\s*[:=]?\s*([а-яa-z\- ]{3,30})", t)
    if city_match:
        params["region"] = city_match.group(1).strip().title()
    else:
        for city in ["москва", "санкт-петербург", "спб", "казань", "екатеринбург", "новосибирск", "самара", "уфа", "краснодар"]:
            if city in t:
                params["region"] = city.title()
                break

    parking = parse_bool_ru(
        t,
        ["парковка: да", "парковка да", "охраняемая парковка: да", "охраняемая парковка да"],
        ["парковка: нет", "парковка нет", "охраняемая парковка: нет", "охраняемая парковка нет"],
    )
    telematics = parse_bool_ru(t, ["телематика: да", "телематика да"], ["телематика: нет", "телематика нет"])
    deductible = parse_bool_ru(t, ["франшиза: да", "франшиза да"], ["франшиза: нет", "франшиза нет"])

    if parking is not None:
        params["has_secured_parking"] = parking
    if telematics is not None:
        params["has_telematics"] = telematics
    if deductible is not None:
        params["has_deductible"] = deductible

    return params


def extract_property_params(text: str) -> Dict[str, Any]:
    t = text.lower()
    params: Dict[str, Any] = {}

    type_map = {"квартира": "квартира", "дом": "дом", "апартамент": "апартаменты"}
    for token, normalized in type_map.items():
        if token in t:
            params["property_type"] = normalized
            break

    city_match = re.search(r"город\s*[:=]?\s*([а-яa-z\- ]{3,30})", t)
    if city_match:
        params["city"] = city_match.group(1).strip().title()

    num_patterns = {
        "area_m2": r"площад[ьи]\s*[:=]?\s*(\d{2,4})",
        "year_built": r"год\s*постройки\s*[:=]?\s*(20\d{2}|19\d{2})",
        "market_value_rub": r"стоим(?:ость)?\s*[:=]?\s*(\d{5,10})",
        "floor": r"этаж\s*[:=]?\s*(-?\d{1,2})",
        "insured_risks_count": r"риск(?:ов|и)?\s*[:=]?\s*(\d{1,2})",
        "deductible_rub": r"франшиза\s*[:=]?\s*(\d{3,7})",
    }

    for key, pattern in num_patterns.items():
        match = re.search(pattern, t)
        if match:
            value = int(match.group(1))
            params[key] = float(value) if key in ["area_m2", "market_value_rub", "deductible_rub"] else value

    fire = parse_bool_ru(
        t,
        ["пожарная сигнализация: да", "пожарная сигнализация да", "сигнализация: да"],
        ["пожарная сигнализация: нет", "пожарная сигнализация нет", "сигнализация: нет"],
    )
    leak = parse_bool_ru(
        t,
        ["датчик протечки: да", "протечка: да", "датчик протечки да"],
        ["датчик протечки: нет", "протечка: нет", "датчик протечки нет"],
    )
    security = parse_bool_ru(
        t,
        ["охрана: да", "охранная система: да", "охранная система да"],
        ["охрана: нет", "охранная система: нет", "охранная система нет"],
    )
    rentals = parse_bool_ru(
        t,
        ["посуточно: да", "посуточная аренда: да", "сдается посуточно"],
        ["посуточно: нет", "посуточная аренда: нет", "не сдается посуточно"],
    )

    if fire is not None:
        params["has_fire_alarm"] = fire
    if leak is not None:
        params["has_water_leak_sensor"] = leak
    if security is not None:
        params["has_security_system"] = security
    if rentals is not None:
        params["short_term_rentals"] = rentals

    return params


OSAGO_REQUIRED = [
    "engine_power_hp",
    "driver_age",
    "driving_experience_years",
    "region",
    "vehicle_market_value",
    "vehicle_year",
    "accidents_last_3y",
    "yearly_mileage_km",
    "additional_drivers_count",
    "has_secured_parking",
    "has_telematics",
    "has_deductible",
]

PROPERTY_REQUIRED = [
    "property_type",
    "city",
    "area_m2",
    "year_built",
    "market_value_rub",
    "floor",
    "has_fire_alarm",
    "has_water_leak_sensor",
    "has_security_system",
    "insured_risks_count",
    "deductible_rub",
    "short_term_rentals",
]


def main() -> None:
    rows = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    call_rows = [r for r in rows if r.get("expected_state") == "call_tool"]
    by_tool = Counter(r.get("expected_tool") for r in call_rows)

    complete_by_tool = Counter()
    missing_stats = Counter()
    samples = []

    for row in call_rows:
        tool_name = row.get("expected_tool")
        query = row.get("user_query", "")

        if tool_name == "calculate_osago_quote":
            parsed = extract_osago_params(query)
            missing = [k for k in OSAGO_REQUIRED if k not in parsed]
        elif tool_name == "calculate_property_insurance_quote":
            parsed = extract_property_params(query)
            missing = [k for k in PROPERTY_REQUIRED if k not in parsed]
        elif tool_name == "create_support_ticket":
            # deterministic path does not parse this tool; it goes through the model
            parsed = {}
            missing = ["routed_via_llm"]
        else:
            parsed = {}
            missing = ["unsupported_expected_tool"]

        if not missing:
            complete_by_tool[tool_name] += 1
        else:
            for m in missing:
                missing_stats[(tool_name, m)] += 1
            if len(samples) < 30:
                samples.append(
                    {
                        "id": row.get("id"),
                        "tool": tool_name,
                        "missing": missing,
                        "query": query[:220],
                    }
                )

    print("call_rows", len(call_rows))
    print("by_tool", dict(by_tool))
    print("complete_by_tool", dict(complete_by_tool))

    print("\nmissing_top")
    for (tool_name, field), cnt in missing_stats.most_common(20):
        print(tool_name, field, cnt)

    print("\nexamples")
    for item in samples:
        print(item)


if __name__ == "__main__":
    main()
