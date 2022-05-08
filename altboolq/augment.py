
import re

def same_case(a, b):
    if a.lower() == a:
        return a.lower()
    if a.upper() == a:
        return b.upper()
    if a[0].upper() == a[0]:
        return b.capitalize()
    return b.lower()
    

def replace_words(sentence, word_map, token_pattern="(?u)\\b\\w\\w+\\b"):
    pattern = re.compile(token_pattern)
    chars = []
    index = 0
    for match in pattern.finditer(sentence):
        word = match.group()
        start = match.start()
        end = match.end()
        while index < start:
            chars.append(sentence[index])
            index += 1
        
        token = word.lower()
        if token in word_map:
            new_word = word_map[token]
            chars += same_case(word, new_word)
        else:
            chars += word
        
        index = end

    while index < len(sentence):
        chars.append(sentence[index])
        index += 1

    return "".join(chars)