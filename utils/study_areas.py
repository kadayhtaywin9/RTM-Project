"""Shared dashboard study-area definitions, independent of observed tower presence."""

YANGON_CITY_TOWNSHIPS = [
    "Latha", "Lanmadaw", "Pabedan", "Kyauktada", "Botahtaung", "Pazundaung",
    "Dagon", "Bahan", "Kamaryut", "Ahlone", "Kyeemyindaing", "Sanchaung",
    "Mingalartaungnyunt", "Tamwe", "Hlaing", "Thingangyun", "Yankin",
    "Dawbon", "Thaketa", "Insein", "Mayangone", "Dala", "Dagon Myothit (North)",
    "South Okkalapa", "North Okkalapa", "Hlaingtharya (East)", "Hlaingtharya (West)",
    "Shwepyithar", "Mingaladon", "Dagon Myothit (South)", "Dagon Myothit (East)",
    "Dagon Myothit (Seikkan)", "Seikgyikanaungto",
]
AREA_TOWNSHIPS = {
    "Yangon City": YANGON_CITY_TOWNSHIPS,
    "Hmawbi": ["Hmawbi"],
    "Thanlyin": ["Thanlyin"],
    "Kyauktan": ["Kyauktan"],
}
ANALYSIS_AREAS = list(AREA_TOWNSHIPS)
AREA_ID = {"Yangon City": 1, "Hmawbi": 2, "Thanlyin": 3, "Kyauktan": 4}
TOWNSHIP_TO_AREA = {township: area for area, towns in AREA_TOWNSHIPS.items() for township in towns}
