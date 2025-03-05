from itertools import combinations
from collections import Counter, defaultdict


class MonsterDiagnosisAgent:
    def __init__(self):
        # If you want to do any initial processing, add it here.
        pass

    def solve(self, diseases, patient):
        # Add your code here!
        #
        # The first parameter to this method is a list of diseases, represented as a
        # dictionary. The key in each dictionary entry is the name of a disease. The
        # value for the key item is another dictionary of symptoms of that disease, where
        # the keys are letters representing vitamin names ("A" through "Z") and the values
        # are "+" (for elevated), "-" (for reduced), or "0" (for normal).
        #
        # The second parameter to this method is a particular patient's symptoms, again
        # represented as a dictionary where the keys are letters and the values are
        # "+", "-", or "0".
        #
        # This method should return a list of names of diseases that together explain the
        # observed symptoms. If multiple lists of diseases can explain the symptoms, you
        # should return the smallest list. If multiple smallest lists are possible, you
        # may return any sufficiently explanatory list.
        #
        # The solve method will be called multiple times, each of which will have a new set
        # of diseases and a new patient to diagnose.
        # pass
        disease_symptoms = {disease: symptoms for disease, symptoms in diseases.items()}

        for size in range(1, len(disease_symptoms) + 1):
            for disease_combo in combinations(disease_symptoms.keys(), size):
                if self.matches_patient_symptoms(disease_combo, disease_symptoms, patient):
                    return list(disease_combo)
    def matches_patient_symptoms(self, disease_combo, disease_symptoms, patient):
        combined_effects = defaultdict(int)

        for disease in disease_combo:
            for vitamin, effect in disease_symptoms[disease].items():
                if effect == "+":
                    combined_effects[vitamin] += 1
                elif effect == "-":
                    combined_effects[vitamin] -= 1

        for vitamin, patient_symptom in patient.items():
            net_effect = combined_effects[vitamin]
            if patient_symptom == "+" and net_effect <= 0:
                return False
            elif patient_symptom == "-" and net_effect >= 0:
                return False
            elif patient_symptom == "0" and net_effect != 0:
                return False

        return True

