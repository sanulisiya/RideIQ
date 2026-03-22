
import re
from datetime import datetime

# Entity dictionaries

CITIES = {
    "colombo":     "Colombo",
    "kandy":       "Kandy",
    "galle":       "Galle",
    "negombo":     "Negombo",
    "jaffna":      "Jaffna",
    "matara":      "Matara",
}

VEHICLE_ALIASES = {
    # Bike
    "bike":          "bike",
    "motorbike":     "bike",
    "motorcycle":    "bike",
    "moto":          "bike",
    "two wheeler":   "bike",
    # Tuk-tuk
    "tuk tuk":       "tuk_tuk",
    "tuktuk":        "tuk_tuk",
    "tuk-tuk":       "tuk_tuk",
    "three wheeler": "tuk_tuk",
    "threewheeler":  "tuk_tuk",
    "bajaj":         "tuk_tuk",
    "auto":          "tuk_tuk",
    "trishaw":       "tuk_tuk",
    # Car
    "car":           "car",
    "cab":           "car",
    "taxi":          "car",
    "vehicle":       "car",
    "uber":          "car",
    "sedan":         "car",
    "suv":           "car",
    "van":           "car",
}

# Weather keyword → estimated precip_mm
WEATHER_MAP = {
    "no rain":       0.0,
    "dry":           0.0,
    "sunny":         0.0,
    "clear":         0.0,
    "light rain":    3.0,
    "light drizzle": 2.0,
    "drizzle":       3.0,
    "little rain":   3.0,
    "slight rain":   2.0,
    "rain":          8.0,
    "raining":       8.0,
    "rainy":         8.0,
    "showers":       7.0,
    "heavy rain":    20.0,
    "heavy shower":  18.0,
    "storm":         30.0,
    "stormy":        30.0,
    "thunderstorm":  30.0,
    "thunder":       25.0,
    "monsoon":       35.0,
    "pouring":       25.0,
    "wet":           6.0,
    "humid":         4.0,
    "hot":           0.0,
}

# Time-of-day keywords → hour mapping
TIME_WORDS = {
    "midnight":  0,
    "dawn":      5,
    "early morning": 6,
    "morning":   8,
    "breakfast": 8,
    "noon":      12,
    "midday":    12,
    "lunch":     12,
    "afternoon": 14,
    "evening":   18,
    "sunset":    18,
    "night":     20,
    "tonight":   20,
    "late night": 22,
}

HOLIDAY_WORDS = [
    "holiday", "poya", "public holiday", "festival",
    "vesak", "new year", "christmas", "deepavali", "eid",
]

# Core parser

def parse_trip_query(text: str) -> dict:

    raw   = text
    text  = text.lower().strip()
    result = {
        "origin":       None,
        "destination":  None,
        "vehicle_type": None,
        "hour":         datetime.now().hour,
        "precip_mm":    0.0,
        "is_holiday":   0,
        "confidence":   0,
        "warnings":     [],
        "raw_text":     raw,
    }

    #  1. Extract vehicle type
    # Sort by length descending so "tuk tuk" matches before "tuk"
    for alias in sorted(VEHICLE_ALIASES.keys(), key=len, reverse=True):
        if alias in text:
            result["vehicle_type"] = VEHICLE_ALIASES[alias]
            break

    #  2. Extract cities
    found_cities = []
    for token, canonical in CITIES.items():
        if token in text:
            # Record position of match for origin/destination ordering
            pos = text.find(token)
            found_cities.append((pos, canonical))

    found_cities.sort(key=lambda x: x[0])  # order by position in sentence

    if len(found_cities) >= 2:
        result["origin"]      = found_cities[0][1]
        result["destination"] = found_cities[1][1]
    elif len(found_cities) == 1:
        result["warnings"].append(f"Only one city found: {found_cities[0][1]}. Please specify both origin and destination.")

    # ── 3. Extract "from X to Y" pattern explicitly
    from_to = re.search(
        r'from\s+(\w+)\s+to\s+(\w+)', text
    )
    if from_to:
        city1 = CITIES.get(from_to.group(1).lower())
        city2 = CITIES.get(from_to.group(2).lower())
        if city1: result["origin"]      = city1
        if city2: result["destination"] = city2

    # ── 4. Extract hour
    # Try explicit time: 8am, 6pm, 14:00, 8:30am
    time_match = re.search(
        r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', text
    )
    if time_match:
        hour   = int(time_match.group(1))
        period = time_match.group(3)
        if period == "pm" and hour != 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
        result["hour"] = min(hour, 23)
    else:
        # Try 24h format: 14:00 or 1400
        time24 = re.search(r'\b([01]?\d|2[0-3]):([0-5]\d)\b', text)
        if time24:
            result["hour"] = int(time24.group(1))
        else:
            # Try time-of-day words (longest match first)
            for phrase in sorted(TIME_WORDS.keys(), key=len, reverse=True):
                if phrase in text:
                    result["hour"] = TIME_WORDS[phrase]
                    break

    # ── 5. Extract weather
    for phrase in sorted(WEATHER_MAP.keys(), key=len, reverse=True):
        if phrase in text:
            result["precip_mm"] = WEATHER_MAP[phrase]
            break

    # ── 6. Holiday detection
    for word in HOLIDAY_WORDS:
        if word in text:
            result["is_holiday"] = 1
            break

    # ── 7. Calculate confidence score
    score = 0
    if result["origin"]:       score += 30
    if result["destination"]:  score += 30
    if result["vehicle_type"]: score += 25
    if result["hour"] != datetime.now().hour: score += 10
    if result["precip_mm"] > 0: score += 5
    result["confidence"] = score

    # ── 8. Generate warnings for missing fields
    if not result["origin"]:
        result["warnings"].append("Could not detect origin city. Try: 'from Colombo'")
    if not result["destination"]:
        result["warnings"].append("Could not detect destination city. Try: 'to Kandy'")
    if not result["vehicle_type"]:
        result["warnings"].append("No vehicle type detected. Defaulting to Tuk-Tuk. Try: 'bike', 'car', or 'tuk tuk'")
        result["vehicle_type"] = "tuk_tuk"   # sensible default

    return result


def format_parsed_result(result: dict) -> str:
    """Return a human-readable summary of what was extracted."""
    vehicle_labels = {"bike": "Motorbike", "tuk_tuk": "Tuk-Tuk", "car": "Car"}
    lines = [
        f"Origin      : {result['origin'] or 'Not detected'}",
        f"Destination : {result['destination'] or 'Not detected'}",
        f"Vehicle     : {vehicle_labels.get(result['vehicle_type'], 'Not detected')}",
        f"Hour        : {result['hour']:02d}:00",
        f"Rain        : {result['precip_mm']} mm",
        f"Holiday     : {'Yes' if result['is_holiday'] else 'No'}",
        f"Confidence  : {result['confidence']}%",
    ]
    return "\n".join(lines)


# ── Quick test
if __name__ == "__main__":
    test_queries = [
        "tuk tuk from Colombo to Kandy tomorrow 8am heavy rain",
        "I need a car from Negombo to Galle tonight",
        "bike ride colombo to matara morning",
        "cab from jaffna to colombo",
        "three wheeler kandy galle 6pm rainy",
        "colombo kandy car evening",
        "motorbike from matara to colombo at noon on a holiday",
        "how much is a taxi from Colombo to Galle",
    ]

    print("=" * 55)
    print("  RideIQ NLP Parser — Test Results")
    print("=" * 55)

    for q in test_queries:
        result = parse_trip_query(q)
        print(f"\nQuery    : {q}")
        print(format_parsed_result(result))
        if result["warnings"]:
            for w in result["warnings"]:
                print(f"Warning  : {w}")
        print("-" * 55)