from abc import ABC, abstractmethod
from text2phonemesequence import Text2PhonemeSequence

class RuleBasedMisspeller(ABC):
    
    @abstractmethod
    def misspell_word(self, word, phonetic_word):
        pass
    
class GDropping(RuleBasedMisspeller):
    
    def misspell_word(self, word, phonetic_word = ''):
        # Check if the word ends with "ing"
        if word.endswith("ing"):
            # If so, remove the last character ('g')
            processed_word = word[:-1]
            return processed_word
        else:
            # If not, return the original word
            return word
        
class DoubleLetterElimination(RuleBasedMisspeller):
    
    def misspell_word(self, word, phonetic_word = ''):
        
        result = []
        vowels = set('aeiouy')
        length = len(word)

        for i in range(length - 1):
            current_char = word[i]
            next_char = word[i + 1]

            # Check if characters are the same and meet the specified conditions
            if current_char == next_char and current_char not in vowels and i != length - 2:
                continue

            result.append(current_char)

        # Add the last character to the result
        result.append(word[length - 1])

        return ''.join(result)
    
class ThToT(RuleBasedMisspeller):
    
    def misspell_word(self, word, phonetic_word):
        # If "th" in word and "θ" in phonetic_word, replace "th" with "t"
        if 'th' in word and 'θ' in phonetic_word:
            processed_word = word.replace('th', 't')
            return processed_word
        else:
            return word
        
class ThToF(RuleBasedMisspeller):
    
    def misspell_word(self, word, phonetic_word):
        # If "th" in word and "θ" in phonetic_word, replace "th" with "t"
        if 'th' in word and 'θ' in phonetic_word:
            processed_word = word.replace('th', 'f')
            return processed_word
        else:
            return word
        
class ThToD(RuleBasedMisspeller):
    
    def misspell_word(self, word, phonetic_word):
        # If "th" in word and "θ" in phonetic_word, replace "th" with "t"
        if 'th' in word and 'ð' in phonetic_word:
            processed_word = word.replace('th', 'd')
            return processed_word
        else:
            return word
        
class KsToX(RuleBasedMisspeller):
    
    def misspell_word(self, word, phonetic_word):
        # If "th" in word and "θ" in phonetic_word, replace "th" with "t"
        if 'ks' in word and 'k s' in phonetic_word:
            processed_word = word.replace('ks', 'x')
            return processed_word
        else:
            return word
        
class auToo(RuleBasedMisspeller):
    
    def misspell_word(self, word, phonetic_word):
        # If "th" in word and "θ" in phonetic_word, replace "th" with "t"
        if 'au' in word and 'ɑ u̯' in phonetic_word or 'ɔ' in phonetic_word:
            processed_word = word.replace('au', 'o')
            return processed_word
        else:
            return word

class auTou(RuleBasedMisspeller):
    
    def misspell_word(self, word, phonetic_word):
        # If "th" in word and "θ" in phonetic_word, replace "th" with "t"
        if 'au' in word and 'ɑ u̯' in phonetic_word or 'ɔ' in phonetic_word:
            processed_word = word.replace('au', 'u')
            return processed_word
        else:
            return word
        
class ooTou(RuleBasedMisspeller):
    
    def misspell_word(self, word, phonetic_word):
        # If "th" in word and "θ" in phonetic_word, replace "th" with "t"
        if 'oo' in word and 'ʊ' in phonetic_word:
            processed_word = word.replace('oo', 'u')
            return processed_word
        else:
            return word

class LastZRuleWithVowelEnd(RuleBasedMisspeller):
    
    def misspell_word(self, word, phonetic_word):
        # Check if "z" is in the last place of the phonetic transcription
        if phonetic_word.endswith('z') and word[-1] in 'aeiouAEIOU':
            # Change "s" to "z"
            processed_word = word.replace('s', 'z')
            # Cancel every letter after "z"
            index_of_z = processed_word.rfind('z')
            processed_word = processed_word[:index_of_z + 1]
            return processed_word
        else:
            return word
        
class SpellingTransform:
    def __init__(self, transformers):
        self.transformers = transformers

    def apply_transformations(self, word, phonetic_word):
        transformed_words = []
        for transformer in self.transformers:
            tr_name = str(transformer.__class__).split('rule_based_misspeller.')[1].strip(">'")
            transformed_word = transformer.misspell_word(word, phonetic_word)
            if transformed_word != word:
                transformed_words.append((transformed_word, tr_name))
        return transformed_words
    
class AllInOneSpellingTransform(SpellingTransform):
    def __init__(self):
        super().__init__(transformers = [
                                        GDropping(),
                                        DoubleLetterElimination(),
                                        ThToT(),
                                        ThToF(),
                                        ThToD(),
                                        KsToX(),
                                        auToo(),
                                        auTou(),
                                        ooTou(),
                                        LastZRuleWithVowelEnd(),
                                ])
        
                                    
    
if __name__ == '__main__':
    

    phonemizer = Text2PhonemeSequence(language='en-us')
    spelling_transform = AllInOneSpellingTransform()

    word = 'cause'
    phonetic_word = phonemizer.infer_sentence(word)

    spelling_transform_list = spelling_transform.apply_transformations(word, phonetic_word)

    print(spelling_transform_list)

