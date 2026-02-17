"""
Advanced Synthetic Data Generator for Low-Resource Languages
Generates high-quality, linguistically diverse training examples
for Nahuatl (Classical/Central) and Maya Yucateco
"""

import json
import random
from typing import List, Dict
from itertools import product

class NahuatlMayaGenerator:
    """Advanced generator for Nahuatl and Maya Yucateco synthetic data"""
    
    def __init__(self):
        self.data = []
        self.initialize_linguistic_resources()
    
    def initialize_linguistic_resources(self):
        """Initialize comprehensive linguistic resources"""
        
        # Pronouns and person markers
        self.pronouns = {
            "es": ["yo", "tú", "él", "ella", "nosotros", "ustedes", "ellos"],
            "nah": ["ne", "te", "ye", "ye", "te", "ame", "ye"],
            "myn": ["teen", "tech", "leti'", "leti'", "toon", "teex", "letiob'"]
        }
        
        # Verb stems with conjugations
        self.verbs = {
            "comer": {"nah": "cua", "myn": "hanal", "type": "transitive"},
            "beber": {"nah": "i", "myn": "uk'", "type": "transitive"},
            "dormir": {"nah": "cochi", "myn": "wenel", "type": "intransitive"},
            "caminar": {"nah": "nemi", "myn": "xíimbal", "type": "intransitive"},
            "hablar": {"nah": "tlatoa", "myn": "t'aan", "type": "intransitive"},
            "trabajar": {"nah": "tequiti", "myn": "meyaj", "type": "intransitive"},
            "ver": {"nah": "itta", "myn": "il", "type": "transitive"},
            "escuchar": {"nah": "caqui", "myn": "u'uy", "type": "transitive"},
            "cantar": {"nah": "cuica", "myn": "k'aay", "type": "intransitive"},
            "bailar": {"nah": "mitotia", "myn": "óok'ot", "type": "intransitive"},
            "correr": {"nah": "tlaloa", "myn": "áalk'ab", "type": "intransitive"},
            "escribir": {"nah": "ihcuiloa", "myn": "ts'íib", "type": "transitive"},
            "leer": {"nah": "pohua", "myn": "xook", "type": "transitive"},
            "cocinar": {"nah": "tlachihchihua", "myn": "wa'alik", "type": "transitive"},
            "pensar": {"nah": "nemilia", "myn": "tuukul", "type": "intransitive"},
            "amar": {"nah": "tlazohtla", "myn": "yaakunaj", "type": "transitive"},
            "ayudar": {"nah": "palehuia", "myn": "áantik", "type": "transitive"},
            "enseñar": {"nah": "machtia", "myn": "ka'ansik", "type": "transitive"},
            "aprender": {"nah": "mati", "myn": "ka'an", "type": "transitive"},
            "jugar": {"nah": "nehuetzca", "myn": "báaxal", "type": "intransitive"},
        }
        
        # Nouns with classifiers
        self.nouns = {
            "casa": {"nah": "calli", "myn": "naj", "classifier": "túul"},
            "agua": {"nah": "atl", "myn": "ha'", "classifier": ""},
            "árbol": {"nah": "cuahuitl", "myn": "che'", "classifier": "túul"},
            "piedra": {"nah": "tetl", "myn": "tuunich", "classifier": "túul"},
            "fuego": {"nah": "tletl", "myn": "k'áak'", "classifier": ""},
            "tierra": {"nah": "tlalli", "myn": "lu'um", "classifier": ""},
            "cielo": {"nah": "ilhuicatl", "myn": "ka'an", "classifier": ""},
            "montaña": {"nah": "tepetl", "myn": "witz", "classifier": "túul"},
            "río": {"nah": "atoyatl", "myn": "ja'", "classifier": "túul"},
            "camino": {"nah": "ohtli", "myn": "bej", "classifier": "túul"},
            "libro": {"nah": "amoxtli", "myn": "hu'un", "classifier": "túul"},
            "flor": {"nah": "xochitl", "myn": "nikte'", "classifier": "túul"},
            "lluvia": {"nah": "quiyahuitl", "myn": "ha'al", "classifier": ""},
            "viento": {"nah": "ehecatl", "myn": "ik'", "classifier": ""},
            "pueblo": {"nah": "altepetl", "myn": "kaaj", "classifier": "túul"},
        }
        
        # Adjectives
        self.adjectives = {
            "grande": {"nah": "hueyi", "myn": "nohoch"},
            "pequeño": {"nah": "tepiton", "myn": "chan"},
            "bueno": {"nah": "cualli", "myn": "ma'alob"},
            "malo": {"nah": "amo cualli", "myn": "ma' ma'alob"},
            "nuevo": {"nah": "yancuic", "myn": "jats'uts"},
            "viejo": {"nah": "huehueh", "myn": "nohoch"},
            "hermoso": {"nah": "cualtzin", "myn": "ki'ichpam"},
            "fuerte": {"nah": "chicahuac", "myn": "ch'íich'pam"},
            "débil": {"nah": "amo chicahuac", "myn": "ya'ab"},
            "rápido": {"nah": "iuhqui", "myn": "chéen"},
            "lento": {"nah": "amo iuhqui", "myn": "páach"},
            "feliz": {"nah": "paqui", "myn": "ki'imak óol"},
            "triste": {"nah": "tonehua", "myn": "óok'ol"},
            "caliente": {"nah": "totonqui", "myn": "chokaan"},
            "frío": {"nah": "cecec", "myn": "síis"},
        }
        
        # Temporal expressions
        self.time_expressions = {
            "hoy": {"nah": "axcan", "myn": "bejla'e'"},
            "mañana": {"nah": "moztla", "myn": "sáamal"},
            "ayer": {"nah": "yalhua", "myn": "ho'olhéel"},
            "ahora": {"nah": "axcan", "myn": "bejla'e'"},
            "después": {"nah": "zatepa", "myn": "ka'ache'"},
            "antes": {"nah": "achto", "myn": "ka'anal"},
            "siempre": {"nah": "cemihcac", "myn": "jump'éel"},
            "nunca": {"nah": "ayemo", "myn": "ma'"},
            "temprano": {"nah": "oc yohuatzinco", "myn": "sáamal"},
            "tarde": {"nah": "teotlac", "myn": "chi'inil"},
        }
        
        # Prepositions and locatives
        self.prepositions = {
            "en": {"nah": "-co", "myn": "ich"},
            "con": {"nah": "-hua", "myn": "yéetel"},
            "sin": {"nah": "amo", "myn": "ma'"},
            "para": {"nah": "-pampa", "myn": "ti'"},
            "de": {"nah": "-nahuac", "myn": "ti'"},
            "desde": {"nah": "-pan", "myn": "tak"},
            "hasta": {"nah": "-tech", "myn": "tu'ux"},
        }
        
        # Question words
        self.questions = {
            "qué": {"nah": "tlein", "myn": "ba'ax"},
            "quién": {"nah": "aquin", "myn": "máax"},
            "dónde": {"nah": "campa", "myn": "tu'ux"},
            "cuándo": {"nah": "quenin", "myn": "ba'ax k'iin"},
            "cómo": {"nah": "quen", "myn": "bix"},
            "por qué": {"nah": "tleca", "myn": "ba'axten"},
            "cuánto": {"nah": "quezqui", "myn": "bahux"},
            "cuál": {"nah": "catlehuatl", "myn": "máax"},
        }
        
        # Numbers (extended)
        self.numbers = {
            "uno": {"nah": "ce", "myn": "jun"},
            "dos": {"nah": "ome", "myn": "ka'a"},
            "tres": {"nah": "yei", "myn": "óox"},
            "cuatro": {"nah": "nahui", "myn": "kan"},
            "cinco": {"nah": "macuilli", "myn": "ho'o"},
            "seis": {"nah": "chicuace", "myn": "wak"},
            "siete": {"nah": "chicome", "myn": "wuk"},
            "ocho": {"nah": "chicuei", "myn": "waxak"},
            "nueve": {"nah": "chiconahui", "myn": "bolon"},
            "diez": {"nah": "mahtlactli", "myn": "lahun"},
            "veinte": {"nah": "cempohual", "myn": "jun k'áal"},
            "cien": {"nah": "macuilpohual", "myn": "ho' k'áal"},
        }
    
    def generate_simple_sentences(self, count: int = 500):
        """Generate simple subject-verb-object sentences"""
        subjects = ["el niño", "la niña", "el hombre", "la mujer", "el maestro", "la maestra"]
        
        for _ in range(count):
            subj = random.choice(subjects)
            verb_es, verb_data = random.choice(list(self.verbs.items()))
            
            if verb_data["type"] == "transitive":
                obj_es, obj_data = random.choice(list(self.nouns.items()))
                es = f"{subj.capitalize()} {verb_es} {obj_es}"
                nah = f"Pilli {verb_data['nah']} {obj_data['nah']}"
                myn = f"Paal ku {verb_data['myn']} {obj_data['myn']}"
            else:
                es = f"{subj.capitalize()} {verb_es}"
                nah = f"Pilli {verb_data['nah']}"
                myn = f"Paal ku {verb_data['myn']}"
            
            self.data.append({
                "es": es,
                "nah": nah,
                "myn": myn,
                "category": "simple_sentences"
            })
    
    def generate_adjective_noun_phrases(self, count: int = 300):
        """Generate adjective + noun combinations"""
        for _ in range(count):
            adj_es, adj_data = random.choice(list(self.adjectives.items()))
            noun_es, noun_data = random.choice(list(self.nouns.items()))
            
            # Spanish: adj + noun
            es = f"{noun_es} {adj_es}"
            # Nahuatl: noun + adj
            nah = f"{noun_data['nah']} {adj_data['nah']}"
            # Maya: noun + adj
            myn = f"{noun_data['myn']} {adj_data['myn']}"
            
            self.data.append({
                "es": es,
                "nah": nah,
                "myn": myn,
                "category": "adjective_phrases"
            })
    
    def generate_temporal_sentences(self, count: int = 400):
        """Generate sentences with temporal expressions"""
        for _ in range(count):
            time_es, time_data = random.choice(list(self.time_expressions.items()))
            verb_es, verb_data = random.choice(list(self.verbs.items()))
            
            es = f"Yo {verb_es} {time_es}"
            nah = f"Ni{verb_data['nah']} {time_data['nah']}"
            myn = f"Kin {verb_data['myn']} {time_data['myn']}"
            
            self.data.append({
                "es": es,
                "nah": nah,
                "myn": myn,
                "category": "temporal_expressions"
            })
    
    def generate_questions(self, count: int = 400):
        """Generate interrogative sentences"""
        for _ in range(count):
            q_es, q_data = random.choice(list(self.questions.items()))
            verb_es, verb_data = random.choice(list(self.verbs.items()))
            
            es = f"¿{q_es.capitalize()} {verb_es}?"
            nah = f"¿{q_data['nah'].capitalize()} {verb_data['nah']}?"
            myn = f"¿{q_data['myn'].capitalize()} {verb_data['myn']}?"
            
            self.data.append({
                "es": es,
                "nah": nah,
                "myn": myn,
                "category": "questions"
            })
    
    def generate_negations(self, count: int = 300):
        """Generate negative sentences"""
        for _ in range(count):
            verb_es, verb_data = random.choice(list(self.verbs.items()))
            
            es = f"Yo no {verb_es}"
            nah = f"Amo ni{verb_data['nah']}"
            myn = f"Ma' kin {verb_data['myn']}"
            
            self.data.append({
                "es": es,
                "nah": nah,
                "myn": myn,
                "category": "negations"
            })
    
    def generate_possessives(self, count: int = 400):
        """Generate possessive constructions"""
        family_members = {
            "madre": {"nah": "nantli", "myn": "na'"},
            "padre": {"nah": "tahtli", "myn": "tata"},
            "hermano": {"nah": "icniuhtli", "myn": "suku'un"},
            "hermana": {"nah": "hueltiuh", "myn": "ki'ik"},
            "hijo": {"nah": "pilli", "myn": "paal"},
            "abuelo": {"nah": "colli", "myn": "nohoch tata"},
        }
        
        possessives = [
            ("mi", "no", "in"),
            ("tu", "mo", "a"),
            ("su", "i", "u"),
        ]
        
        for _ in range(count):
            poss_es, poss_nah, poss_myn = random.choice(possessives)
            member_es, member_data = random.choice(list(family_members.items()))
            
            es = f"{poss_es} {member_es}"
            nah = f"{poss_nah}{member_data['nah']}"
            myn = f"{poss_myn} {member_data['myn']}"
            
            self.data.append({
                "es": es,
                "nah": nah,
                "myn": myn,
                "category": "possessives"
            })
    
    def generate_comparative_sentences(self, count: int = 300):
        """Generate comparative constructions"""
        for _ in range(count):
            adj_es, adj_data = random.choice(list(self.adjectives.items()))
            
            es = f"Más {adj_es}"
            nah = f"Oc {adj_data['nah']}"
            myn = f"Más {adj_data['myn']}"
            
            self.data.append({
                "es": es,
                "nah": nah,
                "myn": myn,
                "category": "comparatives"
            })
    
    def generate_compound_sentences(self, count: int = 500):
        """Generate compound sentences with conjunctions"""
        conjunctions = {
            "y": {"nah": "huan", "myn": "yéetel"},
            "pero": {"nah": "auh", "myn": "wa"},
            "o": {"nah": "nozo", "myn": "wa"},
        }
        
        for _ in range(count):
            conj_es, conj_data = random.choice(list(conjunctions.items()))
            verb1_es, verb1_data = random.choice(list(self.verbs.items()))
            verb2_es, verb2_data = random.choice(list(self.verbs.items()))
            
            es = f"Yo {verb1_es} {conj_es} {verb2_es}"
            nah = f"Ni{verb1_data['nah']} {conj_data['nah']} ni{verb2_data['nah']}"
            myn = f"Kin {verb1_data['myn']} {conj_data['myn']} kin {verb2_data['myn']}"
            
            self.data.append({
                "es": es,
                "nah": nah,
                "myn": myn,
                "category": "compound_sentences"
            })
    
    def generate_locative_expressions(self, count: int = 400):
        """Generate locative/prepositional phrases"""
        locations = {
            "casa": {"nah": "calli", "myn": "naj"},
            "mercado": {"nah": "tianquiztli", "myn": "k'iwik"},
            "escuela": {"nah": "calmecac", "myn": "ka'ansaj"},
            "campo": {"nah": "milli", "myn": "k'áax"},
            "ciudad": {"nah": "altepetl", "myn": "kaaj"},
        }
        
        for _ in range(count):
            loc_es, loc_data = random.choice(list(locations.items()))
            
            es = f"en la {loc_es}"
            nah = f"{loc_data['nah']}co"
            myn = f"ich {loc_data['myn']}"
            
            self.data.append({
                "es": es,
                "nah": nah,
                "myn": myn,
                "category": "locatives"
            })
    
    def generate_imperative_sentences(self, count: int = 400):
        """Generate command/imperative forms"""
        for _ in range(count):
            verb_es, verb_data = random.choice(list(self.verbs.items()))
            
            es = f"{verb_es.capitalize()}"
            nah = f"Xi{verb_data['nah']}"
            myn = f"{verb_data['myn'].capitalize()}"
            
            self.data.append({
                "es": es,
                "nah": nah,
                "myn": myn,
                "category": "imperatives"
            })
    
    def generate_existential_sentences(self, count: int = 300):
        """Generate existential constructions (there is/are)"""
        for _ in range(count):
            noun_es, noun_data = random.choice(list(self.nouns.items()))
            
            es = f"Hay {noun_es}"
            nah = f"Onca {noun_data['nah']}"
            myn = f"Yaan {noun_data['myn']}"
            
            self.data.append({
                "es": es,
                "nah": nah,
                "myn": myn,
                "category": "existentials"
            })
    
    def generate_all(self, target: int = 5000):
        """Generate all types of sentences"""
        print("Generating advanced synthetic dataset...")
        
        # Calculate proportions
        self.generate_simple_sentences(800)
        self.generate_adjective_noun_phrases(500)
        self.generate_temporal_sentences(600)
        self.generate_questions(500)
        self.generate_negations(400)
        self.generate_possessives(500)
        self.generate_comparative_sentences(300)
        self.generate_compound_sentences(600)
        self.generate_locative_expressions(400)
        self.generate_imperative_sentences(400)
        self.generate_existential_sentences(300)
        
        # Shuffle to mix categories
        random.shuffle(self.data)
        
        # Trim or extend to target
        if len(self.data) > target:
            self.data = self.data[:target]
        elif len(self.data) < target:
            # Duplicate and shuffle to reach target
            while len(self.data) < target:
                self.data.extend(self.data[:min(target - len(self.data), len(self.data))])
            random.shuffle(self.data)
            self.data = self.data[:target]
        
        return self.data
    
    def save_jsonl(self, filename: str):
        """Save data in JSONL format"""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in self.data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved {len(self.data)} examples to {filename}")
        
        # Print statistics
        categories = {}
        for item in self.data:
            cat = item.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nCategory distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")

def main():
    generator = NahuatlMayaGenerator()
    data = generator.generate_all(5000)
    
    output_file = "c:/Users/djzai/Documents/Said Moreno/PDM/IA/corc_nah_colab/data/silver/advanced_synthetic_5000.jsonl"
    generator.save_jsonl(output_file)
    
    print(f"\n✓ Total examples generated: {len(data)}")
    print(f"✓ Output: {output_file}")

if __name__ == "__main__":
    main()
