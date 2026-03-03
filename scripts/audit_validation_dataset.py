import json
from pathlib import Path
from collections import Counter

DATASET_PATH = Path("c:/Users/User/Desktop/Projects/SBER/validation_dataset_stage2.jsonl")

REQ = {
    "calculate_osago_quote": [
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
    ],
    "calculate_property_insurance_quote": [
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
    ],
    "calculate_travel_insurance": ["days", "country", "traveler_age", "coverage_eur"],
    "get_document_checklist": ["product"],
    "get_claim_steps": ["event_type"],
    "create_callback_request": ["full_name", "phone", "topic"],
    "create_support_ticket": ["customer_name", "contact_phone", "topic", "urgency", "description"],
}


def main() -> None:
    rows = []
    json_errors = []

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                json_errors.append((i, str(e)))

    state_dist = Counter(r.get("expected_state") for r in rows)
    call_tool_dist = Counter()

    schema_issues = []
    unknown_tools = []

    for row in rows:
        if row.get("expected_state") != "call_tool":
            continue

        tool_name = row.get("expected_tool")
        call_tool_dist[tool_name] += 1

        if tool_name not in REQ:
            unknown_tools.append((row.get("id"), tool_name, row.get("user_query", "")[:160]))
            continue

        required_params = row.get("required_params") or []
        if isinstance(required_params, dict):
            required_params = list(required_params.keys())

        missing = [param for param in REQ[tool_name] if param not in required_params]
        if missing:
            schema_issues.append(
                {
                    "id": row.get("id"),
                    "tool": tool_name,
                    "missing": missing,
                    "query": row.get("user_query", "")[:220],
                }
            )

    print("rows_total", len(rows))
    print("json_errors", len(json_errors))
    print("state_dist", dict(state_dist))
    print("call_tool_dist", dict(call_tool_dist))
    print("unknown_tools", len(unknown_tools))
    print("schema_issues_count", len(schema_issues))

    print("\n-- first 20 schema issues --")
    for item in schema_issues[:20]:
        print(item)


if __name__ == "__main__":
    main()
