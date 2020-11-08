#GA for guessing a word: knowing a length, not knowing a length, guess a city by coordinates, build a picture from a corrupted one

import random

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# ALPHABET = ['a', 'b', 'c', 'd', 'e']

word_to_guess = 'hello'
generation_pool = 5
mode = 'one_char'
debug = True

def get_distance(new_char, ethalone_char, alphabet, to_abs=True):
    char_index, ethalone_index = alphabet.index(new_char), alphabet.index(ethalone_char)
    distance = ethalone_index - char_index
    if to_abs: distance = abs(distance)
    return distance


def create_chars(amount, alphabet):
    res = random.choices(alphabet, k=amount)
    return res


def get_exact_char(char, distance, alphabet):
    index = alphabet.index(char) + distance
    return alphabet[index]





def make_element():
    global mode
    if mode == 'one_char':
        return create_chars(1, ALPHABET)

def create_generation(unique_entities=False):
    global generation_pool
    offspring = []
    entity_values = []
    for x in range(generation_pool):
        entity = {'index': x, 'element': make_element(), 'fit_score': None}

        if unique_entities:
            retake_counter = 0
        if entity['element'] not in entity_values:
            entity_values.append(entity['element'])
        else:
            while entity['element'] in entity_values:
                previous_elem = entity['element']
                entity['element'] = make_element()
                retake_counter += 1
                if retake_counter > 100:
                    raise RecursionError("Cannot generate non-unique values")
                if debug: print(f"Changed {previous_elem} to {entity['element']}")
            entity_values.append(entity['element'])

        #можно копировать алфавит и перед генерацией буквы удалять из копии элементы, которые уже повторились, чтобы не дублировать
        offspring.append(entity)
    return offspring

def history_flow():
    # make least fitting elements vanish and new random ones take their place
    global history_flow_coeffs
    pass



def guess_by_one():
    pass

def guess_by_length():
    pass

def guess_blindly():
    pass




def run_tests():
    assert get_distance('a', 'f', ALPHABET) == 5
    assert len(create_chars(5, ALPHABET)) == 5
    assert get_exact_char('a', 5, ALPHABET) == 'f'
    assert get_exact_char('f', get_distance('f', 'a', ALPHABET, to_abs=False), ALPHABET) == 'a'

if __name__ == '__main__':
    run_tests()
    print(create_generation(unique_entities=True))