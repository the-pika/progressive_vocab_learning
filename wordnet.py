"""
Citation for this work: 
Deepika Verma, Daison Darlan, Rammohan Mallipeddi. ''Progressive Vocabulary Learning via Pareto-Optimal Clustering''.
International Conference on ICT Convergence, South Korea (2025).

@author: Deepika Verma
"""


from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from readability import Readability
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

import pandas as pd
import numpy as np
import nltk
import os
import matplotlib.pyplot as plt

nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import wordnet

# ======================== Text & Global Vars ========================

text = '''
Christmas, a globally celebrated holiday rooted in both religious and secular traditions, evokes a tapestry of customs, symbolism, and emotional resonance. Although its origin lies in the commemoration of the birth of Jesus Christ, the holiday has evolved into a multifaceted cultural phenomenon. Adorned in twinkling lights and embellished with evergreen garlands, cityscapes are transformed into festive tableaux that captivate both young and old. For many, the holiday season ignites a nostalgic longing for familial warmth, childhood memories, and cherished rituals. Shopping districts swell with eager patrons seeking the perfect gift, while homes echo with the harmonious strains of carols and laughter. The exchange of presents, a gesture imbued with affection and thoughtfulness, symbolizes both generosity and reciprocity.

Culinary traditions vary widely across regions, yet often share an emphasis on abundance and indulgence. Tables are laden with sumptuous feasts—roasted poultry, candied yams, savory stuffing, and elaborate desserts—that foster conviviality and gratitude. Yet, amid the opulence, Christmas also inspires acts of charity and compassion toward the less fortunate. Many volunteer at shelters, donate to food banks, or participate in toy drives, reinforcing the holiday’s ethos of altruism. The iconic figure of Santa Claus, derived from Saint Nicholas and popularized by folklore and advertising, embodies the spirit of benevolence and joyful surprise.

In colder climates, the season is accompanied by snow-draped landscapes, crackling fireplaces, and the ceremonial adornment of Christmas trees—rituals that enhance the sensory allure of the holiday. From the perspective of sociolinguistics, Christmas discourse reflects idiomatic richness, emotional expressiveness, and cross-cultural diffusion. Phrases like “Yuletide cheer” and “season of goodwill” permeate vernacular speech, infusing language with poetic resonance. Despite its Christian origins, Christmas has been embraced by diverse cultures, often adapted to reflect local traditions, cuisines, and spiritual beliefs.

The commercial dimension of Christmas, while criticized for its material excess, also sustains entire industries in publishing, entertainment, retail, and tourism. Films, books, and songs themed around Christmas perpetuate seasonal narratives that range from whimsical to profound. Even in literature, Christmas frequently serves as a motif for redemption, reconciliation, or introspection—as seen in classics like Dickens’ A Christmas Carol. The emotional complexity of the holiday, oscillating between elation and melancholy, underscores its psychological depth.

In modern times, environmental awareness has prompted a reexamination of holiday consumption, leading some to adopt sustainable practices such as eco-friendly wrapping, LED lighting, or tree planting. Meanwhile, digital technologies have redefined how people celebrate, enabling virtual gatherings, e-greetings, and online gift exchanges across continents. Despite these innovations, the essence of Christmas remains tethered to timeless human values: love, unity, generosity, and hope. It is a season that transcends doctrinal boundaries and invites both reflection and celebration. In essence, Christmas offers not only a pause from quotidian routines but also an opportunity for renewal, connection, and the reaffirmation of shared humanity.

'''  

text_array = []
index_array = []

# ======================== Synonym Logic (WordNet) ========================

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lm in syn.lemmas():
            synonyms.append(lm.name())
    return list(set(synonyms))  # remove duplicates

def replace_with_definition(word, number):
    word_lower = word.lower()
    synonyms = get_synonyms(word_lower)
    if not synonyms:
        return -2, word
    elif number >= len(synonyms):
        return len(synonyms)-1, synonyms[-1]
    else:
        return int(number), synonyms[int(number-1)]

def listToString(s):
    str1 = " ".join(str(ele) for ele in s)
    str1 = str1.replace(' ,', ',').replace('_', ' ')
    return str1

def obtain_text(solution):
    res2 = text.split()
    text_converted = []
    index = 0
    solution_values = solution.variables if hasattr(solution, 'variables') else solution

    for i in res2:
        if solution_values[index] < 1:
            text_converted.append(i)
        elif solution_values[index] >= 1:
            number, word = replace_with_definition(i, solution_values[index])
            text_converted.append(word.upper())
        else:
            text_converted.append(i)
        index += 1

    result = listToString(text_converted)
    return result

# ======================== Preprocessing Input Text ========================

res = text.split()
for i in res:
    flag = 0
    if ',' in i:
        i = i.replace(',', '')
        flag = 1
    if '.' in i:
        i = i.replace('.', '')
        flag = 2   

    if (not i[0].isupper() and len(i) > 3 and i[-2:] != 'ed'):
        number, word = replace_with_definition(i, 6)
        text_array.append(word)
        index_array.append(number)
    else:
        text_array.append(i)
        index_array.append(0)
        
    if flag == 1:
        cad = str(text_array[-1]) + ','
        text_array[-1] = cad
    elif flag == 2:
        cad = str(text_array[-1]) + '.'
        text_array[-1] = cad

# ======================== Fitness Function ========================

def fitness_func1(solution):
    # clean inputs
    for i in range(len(solution)):
        if index_array[i] <= 0:
            solution[i] = 0

    res2 = text.split()
    text_converted = []
    for index, i in enumerate(res2):
        if solution[index] < 1:
            text_converted.append(i)
        elif solution[index] >= 1:
            number, word = replace_with_definition(i, solution[index])
            text_converted.append(word.upper())
        else:
            text_converted.append(i)

    result = listToString(text_converted)
    r = Readability(result)
    return r.flesch_kincaid().score  # using FKGL for consistency

# ======================== Optimization Class ========================

class OptiMDS(FloatProblem):

    def __init__(self):
        super(OptiMDS, self).__init__()
        self.number_of_objectives = 2
        self.number_of_variables = len(index_array)
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MAXIMIZE]
        self.obj_labels = ['Words Replaced', 'Readability Score']

        self.lower_bound = self.number_of_variables * [-4]
        self.upper_bound = self.number_of_variables * [4]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[0] = len([1 for i in solution.variables if i >= 1])
        solution.objectives[1] = fitness_func1(solution.variables)
        return solution

    def get_name(self):
        return 'OptiMDS-WordNet'

# ======================== Main GA Execution ========================

from jmetal.util.solution import get_non_dominated_solutions
output_folder = "pareto_fronts_wordnet"
solutions_folder = "pareto_solutions"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(solutions_folder, exist_ok=True)

data_records = []
solution_texts = []
num_runs = 30

problem = OptiMDS()
algorithm = NSGAII(
    problem=problem,
    population_size=50,
    offspring_population_size=50,
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    termination_criterion=StoppingByEvaluations(max_evaluations=800)
)

for run in range(1, num_runs + 1):
    print(f"\n▶ Run {run}/{num_runs}")
    algorithm.run()
    front = get_non_dominated_solutions(algorithm.get_result())

    for solution in front:
        data_records.append([run] + solution.objectives)

    for idx, solution in enumerate(front):
        try:
            text_output = obtain_text(solution)
            solution_texts.append(text_output)
            fname = f"run{run}_pareto_{idx+1}.txt"
            with open(os.path.join(solutions_folder, fname), "w", encoding="utf-8") as f:
                f.write(f"Solution #{idx+1} from Run {run}\n\n{text_output.strip()}\n")
        except Exception as e:
            print(f"⚠️ Error in run {run}, solution {idx+1}: {e}")

    # Plot pareto front
    plt.figure(figsize=(8, 6))
    plt.scatter([s.objectives[0] for s in front], [s.objectives[1] for s in front], color='green', s=100)
    plt.xlabel('Words Replaced', fontsize=14)
    plt.ylabel('FKGL Readability Score', fontsize=14)
    plt.title(f"Pareto Front (Run {run})", fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"pareto_front_run_{run}.png"))
    plt.close()

# Save CSV and Combined Texts
df_results = pd.DataFrame(data_records, columns=["Run", "Words Replaced", "Readability Score"])
df_results.to_csv("nsga2_wordnet_results.csv", index=False)

with open("all_pareto_texts_wordnet.txt", "w", encoding="utf-8") as f:
    for idx, txt in enumerate(solution_texts, 1):
        f.write(f"\n================ Solution #{idx} ================\n")
        f.write(txt.strip() + "\n")

print("\n✅ Optimization completed and saved.")
