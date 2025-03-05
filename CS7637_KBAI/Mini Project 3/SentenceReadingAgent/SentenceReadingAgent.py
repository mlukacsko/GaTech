# import spacy
import string
from collections import defaultdict


class SentenceReadingAgent:
    def __init__(self):
        # If you want to do any initial processing, add it here.
        pass

    def clean_text(self, text):
        return text.translate(str.maketrans('', '', string.punctuation)).lower()

    def tag_words(self, words, words_dict):
        tagged_words = {}
        for word in words:
            pos_tag = words_dict.get(word)
            if pos_tag in tagged_words:
                tagged_words[pos_tag].append(word)
            else:
                tagged_words[pos_tag] = [word]
        return tagged_words

    def solve(self, sentence, question):
        words_dict = self.processed_words()
        sentence_split = sentence.split()

        sentence = self.clean_text(sentence)
        question = self.clean_text(question)

        sentence_words = sentence.split()
        question_words = question.split()

        sentence_tags = defaultdict(list)
        question_tags = defaultdict(list)

        for word in sentence_words:
            sentence_tags[words_dict.get(word)].append(word)

        for word in question_words:
            question_tags[words_dict.get(word)].append(word)

        question_type = question_words[0] if question_words else ""

        match question_type:


            case "who":
                return self.handle_who(sentence_tags, question_tags)

            case "what":
                return self.handle_what(sentence_tags, question_tags, question_words, sentence_split)

            case "when":
                return self.handle_when(sentence_tags, sentence_words)

            case "where":
                return self.handle_where(sentence_tags, sentence_words, question_words, question_tags)

            case "how":
                return self.handle_how(sentence_tags, question_words, sentence_words, words_dict)

            case _:
                return self.handle_check_for_time(question_words, sentence_split)


    def handle_who(self, sentence_tags, question_tags):
        def find_recipient_or_noun(sentence_tags, question_tags):
            sentence_propn = sentence_tags.get('PROPN', [])
            question_propn = question_tags.get('PROPN', [])
            sentence_pronouns = sentence_tags.get('PRON', [])
            question_pronouns = question_tags.get('PRON', [])
            common_nouns = sentence_tags.get('NOUN', [])

            indirect_object_pronouns = {"him", "her", "them"}
            for pronoun in sentence_pronouns:
                if pronoun in indirect_object_pronouns:
                    return pronoun

            if not question_propn and sentence_propn:
                return sentence_propn[0].capitalize()

            if len(question_propn) == 1 and question_propn[0] in sentence_propn:
                return next((p.capitalize() for p in sentence_propn if p != question_propn[0]), None)

            if sentence_propn:
                return sentence_propn[0]

            if sentence_pronouns:
                return sentence_pronouns[0]

            if common_nouns:
                return common_nouns[0]

            return None

        result = find_recipient_or_noun(sentence_tags, question_tags)
        return result

    def handle_what(self, sentence_tags, question_tags, question_words, sentence_split):
        def handle_name_case(sentence_split):
            for word in sentence_split[1:]:
                if word[0].isupper():
                    return word
            return None

        def handle_color_case(sentence_tags, question_tags, sentence_split):
            colors = sentence_tags.get('ADJ', [])
            subjects = question_tags.get('NOUN', [])
            if len(colors) == 1:
                return colors[0]
            if subjects:
                subject = subjects[1] if len(subjects) > 1 else subjects[0]
                index = sentence_split.index(subject)
                return sentence_split[index - 1] if index > 0 else colors[0]
            return colors[0] if colors else None

        def find_object_of_verb(sentence_tags, sentence_split):
            verbs = sentence_tags.get('VERB', [])
            nouns = sentence_tags.get('NOUN', [])

            if "saw" in verbs:
                see_index = sentence_split.index("saw")
                for i in range(see_index + 1, len(sentence_split)):
                    if sentence_split[i] in nouns:
                        return sentence_split[i]
            return None

        def find_first_difference(s_list, q_list):
            if q_list and s_list:
                differences = set(s_list) - set(q_list)
                return next(iter(differences), None)
            return None

        if 'name' in question_words:
            result = handle_name_case(sentence_split)
            if result:
                return result

        if 'do' in question_words and sentence_tags.get('VERB'):
            return sentence_tags.get('VERB', [None])[0]

        if 'color' in question_words:
            result = handle_color_case(sentence_tags, question_tags, sentence_split)
            if result:
                return result

        result = find_object_of_verb(sentence_tags, sentence_split)
        if result:
            return result

        sentence_tags_noun = sentence_tags.get('NOUN', [])
        question_tags_noun = question_tags.get('NOUN', [])
        result = find_first_difference(sentence_tags_noun, question_tags_noun)
        if result:
            return result

        n = sentence_tags.get('NOUN', [None])[0]
        a = sentence_tags.get('ADJ', [None])[0]
        v = sentence_tags.get('VERB', [None])[0]
        return n or a or v

    def handle_when(self, sentence_tags, sentence_words):
        sentence_adv = sentence_tags.get('ADV', [])
        if sentence_adv:
            return sentence_adv[0]

        sentence_noun = sentence_tags.get('NOUN', [])
        if sentence_noun:
            return sentence_noun[0]
        return None

    def handle_how(self, sentence_tags, question_words, sentence_words, words_dict):
        if "many" in question_words:
            sentence_list_numbers = sentence_tags.get('NUM', [])
            if sentence_list_numbers:
                return sentence_list_numbers[0]

        if "much" in question_words:
            sentence_determiner = sentence_tags.get('DET', [])
            if sentence_determiner:
                return sentence_determiner[0]

        mod = question_words[1]
        tag_type = words_dict.get(mod)

        match tag_type:
            case 'ADJ':
                sentence_list_adj = sentence_tags.get('ADJ', [])
                sentence_list_adv = sentence_tags.get('ADV', [])

                if sentence_list_adj:
                    return sentence_list_adj[0]

                if sentence_list_adv:
                    location_index = sentence_words.index(sentence_list_adv[0]) + 1
                    if location_index < len(sentence_words):
                        return sentence_words[location_index]

                sentence_list_noun = sentence_tags.get('NOUN', [])
                if sentence_list_noun:
                    return sentence_list_noun[0]

            case 'VERB':
                sentence_list_verb = sentence_tags.get('VERB', [])
                if sentence_list_verb:
                    return sentence_list_verb[0]

            case _:
                return None

    def handle_where(self, sentence_tags, sentence_words, question_words, question_tags):
        sentence_nouns = sentence_tags.get('NOUN', [])
        question_nouns = question_tags.get('NOUN', [])
        match next((word for word in sentence_words if word in {"go", "in"}), None):
            case "go":
                go_index = sentence_words.index('go') + 2
                if go_index < len(sentence_words):
                    x = sentence_words[go_index]
                    return x
            case "in":
                in_index = sentence_words.index('in') + 2
                if in_index < len(sentence_words):
                    x = sentence_words[in_index]
                    return x

        if question_nouns:
            sentence_nouns = [noun for noun in sentence_nouns if noun != question_nouns[0]]
        return sentence_nouns[0] if sentence_nouns else None

    def handle_check_for_time(self, question_words, sentence_split):
        return next((word for word in sentence_split if ":" in word), None)

    def processed_words(self):
        word_dict = {
            'Serena': 'PROPN',
             'Andrew': 'PROPN',
             'Bobbie': 'PROPN',
             'Cason': 'PROPN',
             'David': 'PROPN',
             'Farzana': 'PROPN',
             'Frank': 'PROPN',
             'Hannah': 'PROPN',
             'Ida': 'PROPN',
             'Irene': 'PROPN',
             'Jim': 'PROPN',
             'Jose': 'PROPN',
             'Keith': 'PROPN',
             'Laura': 'PROPN',
             'Lucy': 'PROPN',
             'Meredith': 'PROPN',
             'Nick': 'PROPN',
             'Ada': 'PROPN',
             'Yeeling': 'PROPN',
             'Yan': 'PROPN',
             'the': 'DET',
             'of': 'ADP',
             'to': 'ADP',
             'and': 'CCONJ',
             'a': 'DET',
             'in': 'NOUN',
             'is': 'AUX',
             'it': 'PRON',
             'you': 'PRON',
             'that': 'SCONJ',
             'he': 'PRON',
             'was': 'AUX',
             'for': 'ADP',
             'on': 'ADV',
             'are': 'AUX',
             'with': 'ADP',
             'as': 'ADP',
             'I': 'PRON',
             'his': 'PRON',
             'they': 'PRON',
             'be': 'VERB',
             'at': 'ADP',
             'one': 'NUM',
             'have': 'VERB',
             'this': 'DET',
             'from': 'ADP',
             'or': 'CCONJ',
             'had': 'VERB',
             'by': 'ADP',
             'hot': 'ADJ',
             'but': 'CCONJ',
             'some': 'DET',
             'what': 'PRON',
             'there': 'ADV',
             'we': 'PRON',
             'can': 'AUX',
             'out': 'ADV',
             'other': 'ADJ',
             'were': 'AUX',
             'all': 'DET',
             'your': 'PRON',
             'when': 'ADV',
             'up': 'ADP',
             'use': 'VERB',
             'word': 'NOUN',
             'how': 'ADV',
             'said': 'VERB',
             'an': 'DET',
             'each': 'DET',
             'she': 'PRON',
             'which': 'DET',
             'do': 'VERB',
             'their': 'PRON',
             'time': 'NOUN',
             'if': 'SCONJ',
             'will': 'AUX',
             'way': 'VERB',
             'about': 'ADV',
             'many': 'ADJ',
             'then': 'ADV',
             'them': 'PRON',
             'would': 'AUX',
             'write': 'VERB',
             'like': 'ADP',
             'so': 'ADV',
             'these': 'DET',
             'her': 'PRON',
             'long': 'ADJ',
             'make': 'NOUN',
             'thing': 'NOUN',
             'see': 'VERB',
             'him': 'PRON',
             'two': 'NUM',
             'has': 'AUX',
             'look': 'VERB',
             'more': 'ADJ',
             'day': 'NOUN',
             'could': 'AUX',
             'go': 'VERB',
             'come': 'VERB',
             'did': 'VERB',
             'my': 'PRON',
             'sound': 'NOUN',
             'no': 'DET',
             'most': 'ADJ',
             'number': 'NOUN',
             'who': 'PRON',
             'over': 'ADP',
             'know': 'VERB',
             'water': 'NOUN',
             'than': 'SCONJ',
             'call': 'VERB',
             'first': 'ADJ',
             'people': 'NOUN',
             'may': 'AUX',
             'down': 'ADV',
             'side': 'NOUN',
             'been': 'AUX',
             'now': 'ADV',
             'find': 'VERB',
             'any': 'DET',
             'new': 'ADJ',
             'work': 'NOUN',
             'part': 'NOUN',
             'take': 'VERB',
             'get': 'VERB',
             'place': 'NOUN',
             'made': 'VERB',
             'live': 'ADJ',
             'where': 'ADV',
             'after': 'ADP',
             'back': 'ADV',
             'little': 'ADJ',
             'only': 'ADJ',
             'round': 'ADJ',
             'man': 'NOUN',
             'year': 'NOUN',
             'came': 'VERB',
             'show': 'VERB',
             'every': 'DET',
             'good': 'ADJ',
             'me': 'PRON',
             'give': 'VERB',
             'our': 'PRON',
             'under': 'ADP',
             'name': 'NOUN',
             'very': 'ADV',
             'through': 'ADP',
             'just': 'ADV',
             'form': 'VERB',
             'much': 'ADV',
             'great': 'ADJ',
             'think': 'NOUN',
             'say': 'VERB',
             'help': 'VERB',
             'low': 'ADJ',
             'line': 'NOUN',
             'before': 'ADP',
             'turn': 'NOUN',
             'cause': 'VERB',
             'same': 'ADJ',
             'mean': 'NOUN',
             'differ': 'VERB',
             'move': 'NOUN',
             'right': 'ADJ',
             'boy': 'INTJ',
             'old': 'ADJ',
             'too': 'ADV',
             'does': 'AUX',
             'tell': 'VERB',
             'sentence': 'NOUN',
             'set': 'VERB',
             'three': 'NUM',
             'want': 'ADJ',
             'air': 'NOUN',
             'well': 'ADV',
             'also': 'ADV',
             'play': 'VERB',
             'small': 'ADJ',
             'end': 'NOUN',
             'put': 'VERB',
             'home': 'NOUN',
             'read': 'VERB',
             'hand': 'NOUN',
             'port': 'NOUN',
             'large': 'ADJ',
             'spell': 'NOUN',
             'add': 'VERB',
             'even': 'ADV',
             'land': 'NOUN',
             'here': 'ADV',
             'must': 'AUX',
             'big': 'ADJ',
             'high': 'ADJ',
             'such': 'ADJ',
             'follow': 'NOUN',
             'act': 'NOUN',
             'why': 'SCONJ',
             'ask': 'VERB',
             'men': 'NOUN',
             'change': 'NOUN',
             'went': 'VERB',
             'light': 'ADJ',
             'kind': 'ADV',
             'off': 'ADP',
             'need': 'PROPN',
             'house': 'NOUN',
             'picture': 'NOUN',
             'try': 'VERB',
             'us': 'PRON',
             'again': 'ADV',
             'animal': 'NOUN',
             'point': 'NOUN',
             'mother': 'NOUN',
             'world': 'NOUN',
             'near': 'ADP',
             'build': 'VERB',
             'self': 'NOUN',
             'earth': 'PROPN',
             'father': 'NOUN',
             'head': 'NOUN',
             'stand': 'VERB',
             'own': 'ADJ',
             'page': 'NOUN',
             'should': 'AUX',
             'country': 'NOUN',
             'found': 'VERB',
             'answer': 'NOUN',
             'school': 'NOUN',
             'grow': 'NOUN',
             'study': 'NOUN',
             'still': 'ADV',
             'learn': 'VERB',
             'plant': 'NOUN',
             'cover': 'NOUN',
             'food': 'NOUN',
             'sun': 'NOUN',
             'four': 'NUM',
             'thought': 'NOUN',
             'let': 'VERB',
             'keep': 'VERB',
             'eye': 'NOUN',
             'never': 'ADV',
             'last': 'ADJ',
             'door': 'NOUN',
             'between': 'ADP',
             'city': 'NOUN',
             'tree': 'NOUN',
             'cross': 'NOUN',
             'since': 'SCONJ',
             'hard': 'ADJ',
             'start': 'NOUN',
             'might': 'AUX',
             'story': 'NOUN',
             'saw': 'VERB',
             'far': 'ADV',
             'sea': 'NOUN',
             'draw': 'NOUN',
             'left': 'VERB',
             'late': 'ADJ',
             'run': 'NOUN',
             "don't": 'PART',
             'while': 'SCONJ',
             'press': 'VERB',
             'close': 'ADJ',
             'night': 'NOUN',
             'real': 'ADJ',
             'life': 'NOUN',
             'few': 'ADJ',
             'stop': 'VERB',
             'open': 'ADJ',
             'seem': 'VERB',
             'together': 'ADV',
             'next': 'ADJ',
             'white': 'ADJ',
             'children': 'NOUN',
             'begin': 'VERB',
             'got': 'VERB',
             'walk': 'VERB',
             'example': 'NOUN',
             'ease': 'NOUN',
             'paper': 'NOUN',
             'often': 'ADV',
             'always': 'ADV',
             'music': 'VERB',
             'those': 'PRON',
             'both': 'DET',
             'mark': 'PROPN',
             'book': 'NOUN',
             'letter': 'NOUN',
             'until': 'SCONJ',
             'mile': 'NOUN',
             'river': 'NOUN',
             'car': 'NOUN',
             'feet': 'NOUN',
             'care': 'VERB',
             'second': 'ADJ',
             'group': 'NOUN',
             'carry': 'VERB',
             'took': 'VERB',
             'rain': 'NOUN',
             'eat': 'NOUN',
             'room': 'NOUN',
             'friend': 'NOUN',
             'began': 'VERB',
             'idea': 'NOUN',
             'fish': 'NOUN',
             'mountain': 'NOUN',
             'north': 'NOUN',
             'once': 'ADV',
             'base': 'NOUN',
             'hear': 'VERB',
             'horse': 'NOUN',
             'cut': 'VERB',
             'sure': 'ADV',
             'watch': 'VERB',
             'color': 'NOUN',
             'face': 'VERB',
             'wood': 'NOUN',
             'main': 'ADJ',
             'enough': 'ADJ',
             'plain': 'ADJ',
             'girl': 'NOUN',
             'usual': 'ADJ',
             'young': 'ADJ',
             'ready': 'ADJ',
             'above': 'ADP',
             'ever': 'ADV',
             'red': 'NOUN',
             'list': 'NOUN',
             'though': 'SCONJ',
             'feel': 'VERB',
             'talk': 'NOUN',
             'bird': 'NOUN',
             'soon': 'ADV',
             'body': 'NOUN',
             'dog': 'NOUN',
             'family': 'PROPN',
             'direct': 'ADJ',
             'pose': 'NOUN',
             'leave': 'VERB',
             'song': 'NOUN',
             'measure': 'NOUN',
             'state': 'NOUN',
             'product': 'NOUN',
             'black': 'ADJ',
             'short': 'ADJ',
             'numeral': 'ADJ',
             'class': 'NOUN',
             'wind': 'NOUN',
             'question': 'NOUN',
             'happen': 'VERB',
             'complete': 'ADJ',
             'ship': 'NOUN',
             'area': 'NOUN',
             'half': 'PRON',
             'rock': 'NOUN',
             'order': 'NOUN',
             'fire': 'NOUN',
             'south': 'ADJ',
             'problem': 'NOUN',
             'piece': 'NOUN',
             'told': 'VERB',
             'knew': 'VERB',
             'pass': 'NOUN',
             'farm': 'NOUN',
             'top': 'ADJ',
             'whole': 'ADJ',
             'king': 'NOUN',
             'size': 'NOUN',
             'heard': 'VERB',
             'best': 'ADJ',
             'hour': 'NOUN',
             'better': 'ADV',
             'true': 'ADJ',
             'during': 'ADP',
             'hundred': 'NUM',
             'am': 'NOUN',
             'remember': 'VERB',
             'sings': 'NOUN',
             'step': 'NOUN',
             'early': 'ADV',
             'hold': 'VERB',
             'west': 'ADJ',
             'ground': 'NOUN',
             'interest': 'NOUN',
             'reach': 'VERB',
             'fast': 'ADV',
             'five': 'NUM',
             'sing': 'VERB',
             'listen': 'VERB',
             'six': 'NUM',
             'table': 'NOUN',
             'travel': 'NOUN',
             'less': 'ADJ',
             'morning': 'NOUN',
             'ten': 'NUM',
             'simple': 'ADJ',
             'several': 'ADJ',
             'vowel': 'NOUN',
             'toward': 'ADP',
             'war': 'NOUN',
             'lay': 'VERB',
             'against': 'ADP',
             'pattern': 'NOUN',
             'slow': 'ADJ',
             'center': 'NOUN',
             'love': 'NOUN',
             'person': 'NOUN',
             'money': 'NOUN',
             'serve': 'VERB',
             'appear': 'VERB',
             'road': 'NOUN',
             'map': 'NOUN',
             'science': 'NOUN',
             'rule': 'NOUN',
             'govern': 'NOUN',
             'pull': 'VERB',
             'cold': 'ADJ',
             'notice': 'NOUN',
             'voice': 'NOUN',
             'fall': 'NOUN',
             'power': 'NOUN',
             'town': 'NOUN',
             'fine': 'ADJ',
             'certain': 'ADJ',
             'fly': 'NOUN',
             'unit': 'NOUN',
             'lead': 'VERB',
             'cry': 'VERB',
             'dark': 'ADJ',
             'machine': 'NOUN',
             'note': 'NOUN',
             'wait': 'VERB',
             'plan': 'NOUN',
             'figure': 'NOUN',
             'star': 'PROPN',
             'box': 'PROPN',
             'noun': 'PROPN',
             'field': 'NOUN',
             'rest': 'NOUN',
             'correct': 'ADJ',
             'able': 'ADJ',
             'pound': 'NOUN',
             'done': 'VERB',
             'beauty': 'NOUN',
             'drive': 'NOUN',
             'stood': 'VERB',
             'contain': 'VERB',
             'front': 'ADJ',
             'teach': 'NOUN',
             'week': 'NOUN',
             'final': 'NOUN',
             'gave': 'VERB',
             'green': 'ADJ',
             'oh': 'INTJ',
             'quick': 'ADJ',
             'develop': 'VERB',
             'sleep': 'NOUN',
             'warm': 'ADJ',
             'free': 'ADJ',
             'minute': 'NOUN',
             'strong': 'ADJ',
             'special': 'ADJ',
             'mind': 'NOUN',
             'behind': 'ADP',
             'clear': 'ADJ',
             'tail': 'NOUN',
             'produce': 'VERB',
             'fact': 'NOUN',
             'street': 'NOUN',
             'inch': 'NOUN',
             'lot': 'NOUN',
             'nothing': 'PRON',
             'course': 'NOUN',
             'stay': 'VERB',
             'wheel': 'ADJ',
             'full': 'ADJ',
             'force': 'NOUN',
             'blue': 'ADJ',
             'object': 'NOUN',
             'decide': 'VERB',
             'surface': 'NOUN',
             'deep': 'ADJ',
             'moon': 'NOUN',
             'island': 'NOUN',
             'foot': 'NOUN',
             'yet': 'ADV',
             'busy': 'ADJ',
             'test': 'NOUN',
             'record': 'NOUN',
             'boat': 'NOUN',
             'common': 'ADJ',
             'gold': 'NOUN',
             'possible': 'ADJ',
             'plane': 'NOUN',
             'age': 'NOUN',
             'dry': 'ADJ',
             'wonder': 'NOUN',
             'laugh': 'NOUN',
             'thousand': 'NUM',
             'ago': 'ADV',
             'ran': 'VERB',
             'check': 'NOUN',
             'game': 'NOUN',
             'shape': 'NOUN',
             'yes': 'INTJ',
             'cool': 'ADJ',
             'miss': 'NOUN',
             'brought': 'VERB',
             'heat': 'NOUN',
             'snow': 'NOUN',
             'bed': 'NOUN',
             'bring': 'VERB',
             'sit': 'NOUN',
             'perhaps': 'ADV',
             'fill': 'VERB',
             'east': 'ADJ',
             'weight': 'NOUN',
             'language': 'NOUN',
             'among': 'ADP',
             'wrote': 'VERB',
             'Red': 'NOUN',
             'dogs': 'NOUN',
             "'s": 'PART',
             'adult': 'NOUN',
             'adults': 'NOUN'
        }
        lower = {}
        for key, value in word_dict.items():
            lower[str(key).lower()] = value
        return lower