"""FIFA six confederation mapping — inferred from country field

Used by ConfederationPrior: when two teams have very few cross-confederation
pairings in the data, the confederation random effect prevents the model from
estimating them on incomparable scales.
"""
CONFEDERATION_MAP: dict[str, str] = {}

# UEFA (Europe) — 55 members
UEFA = {
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
    "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
    "Czech Republic", "Czechoslovakia", "Denmark", "England", "Estonia",
    "Faroe Islands", "Finland", "France", "Georgia", "German DR", "Germany",
    "Gibraltar", "Greece", "Hungary", "Iceland", "Israel", "Italy", "Kazakhstan",
    "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta",
    "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia",
    "Northern Ireland", "Norway", "Poland", "Portugal", "Republic of Ireland",
    "Romania", "Russia", "San Marino", "Scotland", "Serbia", "Slovakia",
    "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine",
    "Wales", "Yugoslavia",
}
for t in UEFA:
    CONFEDERATION_MAP[t] = "UEFA"

# CONMEBOL (South America) — 10 members
CONMEBOL = {
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
    "Paraguay", "Peru", "Uruguay", "Venezuela",
}
for t in CONMEBOL:
    CONFEDERATION_MAP[t] = "CONMEBOL"

# CONCACAF (North/Central America & Caribbean) — 41 members
CONCACAF = {
    "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados",
    "Belize", "Bermuda", "Bonaire", "British Virgin Islands", "Canada",
    "Cayman Islands", "Costa Rica", "Cuba", "Curaçao", "Dominica",
    "Dominican Republic", "El Salvador", "French Guiana", "Grenada",
    "Guadeloupe", "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica",
    "Martinique", "Mexico", "Montserrat", "Nicaragua", "Panama", "Puerto Rico",
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Martin",
    "Saint Vincent and the Grenadines", "Sint Maarten", "Suriname",
    "Trinidad and Tobago", "Turks and Caicos Islands", "United States",
    "United States Virgin Islands",
}
for t in CONCACAF:
    CONFEDERATION_MAP[t] = "CONCACAF"

# CAF (Africa) — 54 members
CAF = {
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros",
    "Congo", "DR Congo", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea",
    "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
    "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya",
    "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco",
    "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "São Tomé and Príncipe",
    "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa",
    "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda",
    "Zambia", "Zanzibar", "Zimbabwe",
}
for t in CAF:
    CONFEDERATION_MAP[t] = "CAF"

# AFC (Asia) — 47 members
AFC = {
    "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan", "Brunei",
    "Cambodia", "China", "Guam", "Hong Kong", "India", "Indonesia", "Iran",
    "Iraq", "Japan", "Jordan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon",
    "Macau", "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal",
    "North Korea", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar",
    "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Syria",
    "Taiwan", "Tajikistan", "Thailand", "Timor-Leste", "Turkmenistan",
    "United Arab Emirates", "Uzbekistan", "Vietnam", "Vietnam Republic",
    "Yemen",
}
for t in AFC:
    CONFEDERATION_MAP[t] = "AFC"

# OFC (Oceania) — 11 members
OFC = {
    "American Samoa", "Cook Islands", "Fiji", "Kiribati", "Micronesia",
    "New Caledonia", "New Zealand", "Niue", "Papua New Guinea", "Samoa",
    "Solomon Islands", "Tahiti", "Tonga", "Tuvalu", "Vanuatu",
}
for t in OFC:
    CONFEDERATION_MAP[t] = "OFC"

# Historical/non-FIFA teams default to "Other"
__all__ = ["CONFEDERATION_MAP"]
