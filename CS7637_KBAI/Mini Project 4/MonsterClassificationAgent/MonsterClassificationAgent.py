class MonsterClassificationAgent:
    def __init__(self):
        #If you want to do any initial processing, add it here.
        pass

    def solve(self, samples, new_monster):
        #Add your code here!
        #
        #The first parameter to this method will be a labeled list of samples in the form of
        #a list of 2-tuples. The first item in each 2-tuple will be a dictionary representing
        #the parameters of a particular monster. The second item in each 2-tuple will be a
        #boolean indicating whether this is an example of this species or not.
        #
        #The second parameter will be a dictionary representing a newly observed monster.
        #
        #Your function should return True or False as a guess as to whether or not this new
        #monster is an instance of the same species as that represented by the list.
        # pass

        positive_features = {}
        negative_features = {}

        for monster, is_positive in samples:
            target_dict = positive_features if is_positive else negative_features

            for key, value in monster.items():
                if key not in target_dict:
                    target_dict[key] = set()
                target_dict[key].add(value)

        positive_matches = 0
        negative_matches = 0

        for key, value in new_monster.items():
            if key in positive_features and value in positive_features[key]:
                positive_matches += 1
            if key in negative_features and value in negative_features[key]:
                negative_matches += 1

        if positive_matches >= negative_matches:
            return True
        else:
            return False