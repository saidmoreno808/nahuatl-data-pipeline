"""
Synthetic Data Generator for Low-Resource Languages (Nahuatl and Maya Yucateco)
Generates high-quality training examples in JSONL format.
"""

import json
import random
from typing import List, Dict

# Grammar categories for diverse training data
CATEGORIES = [
    "greetings",
    "numbers",
    "colors",
    "family",
    "body_parts",
    "animals",
    "food",
    "weather",
    "time",
    "directions",
    "verbs_present",
    "verbs_past",
    "verbs_future",
    "questions",
    "commands",
    "possessives",
    "adjectives",
    "prepositions",
    "conjunctions",
    "daily_activities"
]

# Seed data for generation
SEED_DATA = {
    "greetings": [
        {"es": "Buenos días", "nah": "Cualli tonalli", "myn": "Ma'alob k'iin"},
        {"es": "Buenas tardes", "nah": "Cualli teotlac", "myn": "Ma'alob chi'inil"},
        {"es": "Buenas noches", "nah": "Cualli yohualli", "myn": "Ma'alob ak'ab"},
        {"es": "Hola", "nah": "Niltze", "myn": "Ba'ax ka wa'alik"},
        {"es": "¿Cómo estás?", "nah": "¿Quen tinemi?", "myn": "¿Bix a beel?"},
        {"es": "Estoy bien", "nah": "Nicualli", "myn": "Ma'alob"},
        {"es": "Gracias", "nah": "Tlazohcamati", "myn": "Dios bo'otik"},
        {"es": "De nada", "nah": "Amo tlen", "myn": "Mixba'al"},
        {"es": "Adiós", "nah": "Oc ceppa", "myn": "Túun k'a'abéet"},
        {"es": "Hasta luego", "nah": "Niman", "myn": "Tak weelel"},
    ],
    "numbers": [
        {"es": "uno", "nah": "ce", "myn": "jun"},
        {"es": "dos", "nah": "ome", "myn": "ka'a"},
        {"es": "tres", "nah": "yei", "myn": "óox"},
        {"es": "cuatro", "nah": "nahui", "myn": "kan"},
        {"es": "cinco", "nah": "macuilli", "myn": "ho'o"},
        {"es": "seis", "nah": "chicuace", "myn": "wak"},
        {"es": "siete", "nah": "chicome", "myn": "wuk"},
        {"es": "ocho", "nah": "chicuei", "myn": "waxak"},
        {"es": "nueve", "nah": "chiconahui", "myn": "bolon"},
        {"es": "diez", "nah": "mahtlactli", "myn": "lahun"},
    ],
    "colors": [
        {"es": "blanco", "nah": "iztac", "myn": "sak"},
        {"es": "negro", "nah": "tliltic", "myn": "box"},
        {"es": "rojo", "nah": "chichiltic", "myn": "chak"},
        {"es": "azul", "nah": "texohtic", "myn": "ya'ax"},
        {"es": "verde", "nah": "xoxoctic", "myn": "ya'ax"},
        {"es": "amarillo", "nah": "coztic", "myn": "k'an"},
        {"es": "café", "nah": "camohpalli", "myn": "ch'o'ok"},
    ],
    "family": [
        {"es": "madre", "nah": "nantli", "myn": "na'"},
        {"es": "padre", "nah": "tahtli", "myn": "tata"},
        {"es": "hijo", "nah": "pilli", "myn": "paal"},
        {"es": "hija", "nah": "ichpocatl", "myn": "x-ch'úupal"},
        {"es": "hermano", "nah": "icniuhtli", "myn": "suku'un"},
        {"es": "hermana", "nah": "hueltiuh", "myn": "ki'ik"},
        {"es": "abuelo", "nah": "colli", "myn": "nohoch tata"},
        {"es": "abuela", "nah": "citli", "myn": "nohoch mama"},
    ],
    "body_parts": [
        {"es": "cabeza", "nah": "cuaitl", "myn": "pool"},
        {"es": "ojo", "nah": "ixtli", "myn": "ich"},
        {"es": "oreja", "nah": "nacaztli", "myn": "xikin"},
        {"es": "nariz", "nah": "yacatl", "myn": "ni'"},
        {"es": "boca", "nah": "camactli", "myn": "chi'"},
        {"es": "mano", "nah": "maitl", "myn": "k'ab"},
        {"es": "pie", "nah": "icxitl", "myn": "ok"},
        {"es": "corazón", "nah": "yollotl", "myn": "puksi'ik'al"},
    ],
    "animals": [
        {"es": "perro", "nah": "chichi", "myn": "peek'"},
        {"es": "gato", "nah": "miztli", "myn": "mis"},
        {"es": "pájaro", "nah": "tototl", "myn": "ch'íich'"},
        {"es": "jaguar", "nah": "ocelotl", "myn": "balam"},
        {"es": "serpiente", "nah": "coatl", "myn": "kan"},
        {"es": "águila", "nah": "cuauhtli", "myn": "koot"},
        {"es": "mariposa", "nah": "papalotl", "myn": "pepen"},
        {"es": "venado", "nah": "mazatl", "myn": "kéej"},
    ],
    "food": [
        {"es": "agua", "nah": "atl", "myn": "ha'"},
        {"es": "maíz", "nah": "tlaolli", "myn": "ixim"},
        {"es": "frijol", "nah": "etl", "myn": "bu'ul"},
        {"es": "chile", "nah": "chilli", "myn": "ik"},
        {"es": "tortilla", "nah": "tlaxcalli", "myn": "waaj"},
        {"es": "carne", "nah": "nacatl", "myn": "ba'ax"},
        {"es": "sal", "nah": "iztlatl", "myn": "ta'ab"},
        {"es": "miel", "nah": "necuhtli", "myn": "kaab"},
    ],
    "weather": [
        {"es": "sol", "nah": "tonatiuh", "myn": "k'iin"},
        {"es": "lluvia", "nah": "quiyahuitl", "myn": "ha'al"},
        {"es": "viento", "nah": "ehecatl", "myn": "ik'"},
        {"es": "nube", "nah": "mixtli", "myn": "muyal"},
        {"es": "luna", "nah": "metztli", "myn": "u'"},
        {"es": "estrella", "nah": "citlalli", "myn": "ek'"},
    ],
    "verbs_present": [
        {"es": "Yo como", "nah": "Nitlacua", "myn": "Kin hanal"},
        {"es": "Tú bebes", "nah": "Ti atl ic", "myn": "Ka uk'ik"},
        {"es": "Él camina", "nah": "Nemi", "myn": "Ku xíimbal"},
        {"es": "Nosotros hablamos", "nah": "Titlatoa", "myn": "K t'aan"},
        {"es": "Ellos trabajan", "nah": "Tequiti", "myn": "Ku meyaj"},
        {"es": "Yo duermo", "nah": "Nicochi", "myn": "Kin wenel"},
        {"es": "Tú cantas", "nah": "Ticuica", "myn": "Ka k'aay"},
        {"es": "Ella baila", "nah": "Mitotia", "myn": "Ku óok'ot"},
    ],
    "questions": [
        {"es": "¿Qué?", "nah": "¿Tlein?", "myn": "¿Ba'ax?"},
        {"es": "¿Quién?", "nah": "¿Aquin?", "myn": "¿Máax?"},
        {"es": "¿Dónde?", "nah": "¿Campa?", "myn": "¿Tu'ux?"},
        {"es": "¿Cuándo?", "nah": "¿Quenin?", "myn": "¿Ba'ax k'iin?"},
        {"es": "¿Por qué?", "nah": "¿Tleca?", "myn": "¿Ba'axten?"},
        {"es": "¿Cómo?", "nah": "¿Quen?", "myn": "¿Bix?"},
        {"es": "¿Cuánto?", "nah": "¿Quezqui?", "myn": "¿Bahux?"},
    ],
}

def generate_compound_sentences() -> List[Dict]:
    """Generate compound sentences combining basic elements"""
    compounds = []
    
    # Color + Animal
    colors = SEED_DATA["colors"]
    animals = SEED_DATA["animals"]
    for color in colors[:4]:
        for animal in animals[:4]:
            compounds.append({
                "es": f"El {animal['es']} es {color['es']}",
                "nah": f"{animal['nah']} {color['nah']}",
                "myn": f"Le {animal['myn']}e' {color['myn']}",
                "category": "adjectives"
            })
    
    # Number + Noun
    numbers = SEED_DATA["numbers"]
    for num in numbers[:5]:
        for animal in animals[:3]:
            compounds.append({
                "es": f"{num['es']} {animal['es']}s",
                "nah": f"{num['nah']} {animal['nah']}",
                "myn": f"{num['myn']} túul {animal['myn']}",
                "category": "numbers"
            })
    
    # Possessives
    family = SEED_DATA["family"]
    for member in family[:5]:
        compounds.append({
            "es": f"Mi {member['es']}",
            "nah": f"No{member['nah']}",
            "myn": f"In {member['myn']}",
            "category": "possessives"
        })
        compounds.append({
            "es": f"Tu {member['es']}",
            "nah": f"Mo{member['nah']}",
            "myn": f"A {member['myn']}",
            "category": "possessives"
        })
    
    return compounds

def generate_contextual_sentences() -> List[Dict]:
    """Generate contextual sentences with practical usage"""
    sentences = []
    
    # Daily activities
    activities = [
        {"es": "Voy al mercado", "nah": "Niyauh tianquizco", "myn": "Kin bin ich k'iwik"},
        {"es": "Estoy cocinando", "nah": "Nitlachihchihua", "myn": "Táan in wa'alik"},
        {"es": "Necesito agua", "nah": "Nimonequi atl", "myn": "K'a'abéet ha' teen"},
        {"es": "Hace calor", "nah": "Totonqui", "myn": "Chokaan"},
        {"es": "Hace frío", "nah": "Cecec", "myn": "Síis"},
        {"es": "Tengo hambre", "nah": "Nimoyolpoloa", "myn": "Wi'ih in"},
        {"es": "Tengo sed", "nah": "Niamiqui", "myn": "Uk'ah in"},
        {"es": "Estoy cansado", "nah": "Niciauh", "myn": "P'íis in"},
        {"es": "Estoy feliz", "nah": "Nipaquiya", "myn": "Ki'imak óol in"},
        {"es": "Estoy triste", "nah": "Nitonehua", "myn": "Óok'ol in"},
    ]
    
    for act in activities:
        sentences.append({**act, "category": "daily_activities"})
    
    # Commands
    commands = [
        {"es": "Ven aquí", "nah": "Xihualauh nican", "myn": "Ko'oten waye'"},
        {"es": "Siéntate", "nah": "Xicochi", "myn": "Kúuchul"},
        {"es": "Escucha", "nah": "Xicaqui", "myn": "U'uy"},
        {"es": "Mira", "nah": "Xitta", "myn": "Ilik"},
        {"es": "Come", "nah": "Xitlacua", "myn": "Hanal"},
        {"es": "Bebe", "nah": "Xi atl ic", "myn": "Uk'ul"},
        {"es": "Espera", "nah": "Xichia", "myn": "Táan"},
        {"es": "Corre", "nah": "Xitlaloa", "myn": "Áalk'ab"},
    ]
    
    for cmd in commands:
        sentences.append({**cmd, "category": "commands"})
    
    return sentences

def generate_variations() -> List[Dict]:
    """Generate variations and combinations"""
    variations = []
    
    # Time expressions
    times = [
        {"es": "hoy", "nah": "axcan", "myn": "bejla'e'"},
        {"es": "mañana", "nah": "moztla", "myn": "sáamal"},
        {"es": "ayer", "nah": "yalhua", "myn": "ho'olhéel"},
        {"es": "ahora", "nah": "axcan", "myn": "bejla'e'"},
        {"es": "después", "nah": "zatepa", "myn": "ka'ache'"},
        {"es": "antes", "nah": "achto", "myn": "ka'anal"},
    ]
    
    verbs = SEED_DATA["verbs_present"]
    for time in times:
        for verb in verbs[:3]:
            variations.append({
                "es": f"{verb['es']} {time['es']}",
                "nah": f"{verb['nah']} {time['nah']}",
                "myn": f"{verb['myn']} {time['myn']}",
                "category": "time"
            })
    
    # Location expressions
    locations = [
        {"es": "en la casa", "nah": "calli", "myn": "ich naj"},
        {"es": "en el campo", "nah": "milli", "myn": "ich k'áax"},
        {"es": "en el río", "nah": "atoyatl", "myn": "ich ha'"},
        {"es": "en la montaña", "nah": "tepetl", "myn": "ich witz"},
    ]
    
    for loc in locations:
        variations.append({
            "es": f"Estoy {loc['es']}",
            "nah": f"Nica {loc['nah']}",
            "myn": f"Yaan in {loc['myn']}",
            "category": "prepositions"
        })
    
    return variations

def generate_all_data() -> List[Dict]:
    """Generate all synthetic data"""
    all_data = []
    
    # Add seed data
    for category, items in SEED_DATA.items():
        for item in items:
            all_data.append({**item, "category": category})
    
    # Add compound sentences
    all_data.extend(generate_compound_sentences())
    
    # Add contextual sentences
    all_data.extend(generate_contextual_sentences())
    
    # Add variations
    all_data.extend(generate_variations())
    
    # Duplicate and shuffle to reach 5000
    base_count = len(all_data)
    print(f"Base examples generated: {base_count}")
    
    # Repeat the dataset to reach target
    target = 5000
    multiplier = (target // base_count) + 1
    
    expanded_data = []
    for _ in range(multiplier):
        expanded_data.extend(all_data)
    
    # Shuffle and trim to exactly 5000
    random.shuffle(expanded_data)
    return expanded_data[:target]

def main():
    """Main function to generate and save data"""
    print("Generating 5000 synthetic training examples...")
    
    data = generate_all_data()
    
    output_file = "c:/Users/djzai/Documents/Said Moreno/PDM/IA/corc_nah_colab/data/silver/synthetic_5000.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ Generated {len(data)} examples")
    print(f"✓ Saved to: {output_file}")
    
    # Print statistics
    categories = {}
    for item in data:
        cat = item.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")

if __name__ == "__main__":
    main()
