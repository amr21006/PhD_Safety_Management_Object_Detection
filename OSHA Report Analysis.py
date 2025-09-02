import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load the OSHA data
osha_data = pd.read_csv("minidatabase.csv")  # Replace with your file path


# --- 1. Filter for Construction ---
def is_construction(employer):
    keywords = ["construction", "roofing", "excavating", "building", "concrete", "steel", "framing",
                "scaffold", "electrical", "crane", "HVAC", "builder", "contractor", "engineering",  # Expanded keywords
                "road", "bridge", "tunnel", "demolition", "foundation", "plumbing", "carpenter"]

    return any(keyword in employer.lower() for keyword in keywords)

construction_data = osha_data[osha_data["Employer"].apply(is_construction)]

# --- 2. Categorize by Your 28 Classes ---
class_mapping = { # Improved keyword matching for accurate categorization
    "Hardhat": ["hard hat", "helmet"],
    "Vest": ["vest", "high-visibility", "hi-vis", "reflective"],  # Added "reflective"
    "Safety Boots": ["safety boots", "steel toe", "work boots"],
    "Not_Wearing_Safety_Equipment": ["no ppe", "lack of ppe", "not wearing", "failed to wear", "inadequate ppe"],  # Expanded
    "Goggles": ["goggles", "eye protection", "safety glasses"],  # Added "safety glasses"
    "Gloves": ["gloves", "hand protection"],
    "Mask": ["mask", "respirator", "face shield"],  # Added "face shield"
    "Workers": ["worker", "employee", "laborer"],  # This might still need further refinement based on your needs.
    "Scaffold": ["scaffold", "staging", "scaffolding"],  # Added "scaffolding"
    "No_Scaffold": ["no scaffold", "lack of scaffold", "missing scaffold"],  # More specific
    "Fence": ["fence", "barrier", "railing", "guardrail"],
    "Safety_cone": ["cone", "traffic cone", "safety cone"],
    "Machinery": ["excavator", "forklift", "crane", "backhoe", "bulldozer", "loader",  # More machine types
                 "machinery", "equipment", "power tool"],
    "Board": ["board", "plywood", "lumber", "wood panel"],
    "Rebar": ["rebar", "reinforcing steel", "steel bar"],
    "Wood": ["wood", "lumber", "timber"],
    "Ebox": ["electrical box", "ebox", "junction box"],
    "Hopper": ["hopper"],
    "Hook": ["hook", "crane hook", "lifting hook"],
    "Brick": ["brick"],
    "Toeboard": ["toeboard"],
    "Cutter": ["cutter", "saw", "knife", "grinder", "blade"],  # More cutter types
    "Slogan": ["slogan", "sign", "warning sign", "safety sign", "poster"],  # Added sign types
    "Handcart": ["handcart", "wheelbarrow", "cart", "dolly"],
    "Fall-Detected": ["fall", "fell", "falling", "slipped", "tripped"],  # More fall-related keywords
    "Opening_hole": ["hole", "opening", "shaft", "excavation", "trench"],  # Added excavation/trench
    "Fire": ["fire"],
    "Smoke": ["smoke"],
}

def categorize_incident(narrative):
    categories = []
    narrative_lower = narrative.lower() # Lowercase once for efficiency
    for class_name, keywords in class_mapping.items():
        if any(re.search(r’\b’ + re.escape(keyword) + r’\b’, narrative_lower) for keyword in keywords):
            categories.append(class_name)

    return categories

construction_data[“Categories”] = construction_data[“Final Narrative”].apply(categorize_incident)

# --- 3. Quantify and Visualize ---
category_counts = {}
for categories in construction_data[“Categories”]:
    for category in categories:  # Iterate through the list of categories
        category_counts[category] = category_counts.get(category, 0) + 1

sorted_counts = dict(sorted(category_counts.items(), key=lambda item: item[1], reverse=True))

plt.figure(figsize=(14, 6))  # Increased Figure size for better readability
plt.bar(sorted_counts.keys(), sorted_counts.values())
plt.xlabel("Safety Categories", fontsize=12)
plt.ylabel("Number of Incidents", fontsize=12)
plt.title("Frequency of Construction Safety Incidents (OSHA Data)", fontsize=14)
plt.xticks(rotation=90, ha=’right’, fontsize=10)  # Rotate x-axis labels
plt.tight_layout()
plt.show()

# --- 4. Date and Time Analysis ---

# Convert “EventDate” to datetime
construction_data[“EventDate”] = pd.to_datetime(construction_data[“EventDate”])

# Extract day of the week and hour of the day
construction_data[“DayOfWeek”] = construction_data[“EventDate”].dt.dayofweek  # Monday=0, Sunday=6
construction_data[“HourOfDay”] = construction_data[“EventDate”].dt.hour

# Analyze incident frequency by day of the week
day_of_week_counts = construction_data[“DayOfWeek”].value_counts().sort_index()
plt.figure(figsize=(10, 5))
day_of_week_counts.plot(kind=’bar’)
plt.xlabel("Day of the Week (0=Monday, 6=Sunday)")
plt.ylabel("Number of Incidents")
plt.title("Incident Frequency by Day of the Week")
plt.xticks(rotation=0)  # Keep x-axis labels horizontal
plt.show()

# Analyze incident frequency by hour of the day
hour_of_day_counts = construction_data[“HourOfDay”].value_counts().sort_index()
plt.figure(figsize=(12, 5))
hour_of_day_counts.plot(kind=’bar’)
plt.xlabel("Hour of the Day (0-23)")
plt.ylabel("Number of Incidents")
plt.title("Incident Frequency by Hour of the Day")
plt.xticks(rotation=0)  # Keep x-axis labels horizontal
plt.show()
 
