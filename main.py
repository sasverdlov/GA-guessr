#GA for guessing a word: knowing a length, not knowing a length, guess a city by coordinates, build a picture from a corrupted one

import random
import time

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f']
# ToDo: processing

word_to_guess = 'hello'
generation_pool = 5
mode = 'known_len' # ToDo: 'unknown_len'
optimization = 'minimize'
sort_children_before_mating = True
debug = False

word_to_guess = [char for char in word_to_guess]

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

def remake_if_non_unique(element, all_elements, remake_function, *args, modify_function=None):
        retake_counter = 0
        while element in all_elements:
            previous_elem = element
            element = remake_function(*args)
            retake_counter += 1
            if retake_counter > 100:
                if not modify_function:
                    raise RecursionError("Cannot generate non-unique values")
                else:
                    modify_function(element)
                    if debug: print(f"Used {modify_function} to modify element")
            if debug: print(f"Changed {previous_elem} to {element}")
        return element

def random_subelement(element, amount=None, chance=False, twisted_chance=True):

    if not amount: amount = random.randint(0, len(element)-1)
    if chance and amount < 1:
        coeffs = [1-amount, amount] if twisted_chance else [amount, 1-amount]
        amount = random.choices([0,1], weights=coeffs)[0]
    else: amount = round(amount)
    for _ in range(amount):
        element[random.randint(0, len(element)-1)] = random.choices(ALPHABET)[0]
        if debug: print(f"Randomly chosen {element}")
    return element




def make_element(*args):
    if mode == 'one_char':
        return create_chars(1, ALPHABET)
    elif mode == 'known_len':
        return create_chars(len(word_to_guess), ALPHABET)

def create_generation(unique_entities=True):
    global generation_pool
    offspring = []
    entity_values = []
    for x in range(generation_pool):
        entity = {'index': x, 'element': make_element(), 'fit_score': None}

        if unique_entities:
            entity['element'] = remake_if_non_unique(entity['element'], entity_values, make_element)
        entity_values.append(entity['element'])

        #можно копировать алфавит и перед генерацией буквы удалять из копии элементы, которые уже повторились, чтобы не дублировать
        offspring.append(entity)
    return offspring

def history_flow():
    # make least fitting elements vanish and new random ones take their place
    pass


def compare(element, ethalone):
    res = {'score': 0, 'chars_imbalance': len(element) - len(ethalone)}
    for index, char in enumerate(element):
        dist = get_distance(char, ethalone[index], ALPHABET)
        res['score'] += dist
    return res

def verify_entities(pool, ethalone):
    for entity in pool:
        entity['fit_score'] = compare(entity['element'], ethalone)
    return pool

def process_generation():
    pool = create_generation()
    generation = verify_entities(pool, word_to_guess)
    pass

def mate(parents, family_size=2, unique_entities=True):

    def extract_genes(parents):
        genes = []
        for parent in parents:
            genes.append(parent['element']) #также нужно формировать пустые гены, если длина различается. Через сет длин, мб
        return genes

    def prepare_coeffs(weights):
        if debug: print("Coefficients: ", weights)
        if optimization == 'maximize':
            coeffs = weights
        elif optimization == 'minimize':
            coeffs = [1/coeff for coeff in weights]
        if debug: print("Coefficients: ", coeffs)
        return coeffs

    def select_genes(genes, coeffs, alphabet):
        start, end = ALPHABET.index(genes[0]), ALPHABET.index(genes[1])
        if end < start:
            start, end = end, start
            coeffs = coeffs[::-1]
        # print(start, end)
        # print(coeffs)
        chain = alphabet[start:end+1]
        div = len(chain)
        weights = []
        if optimization == 'minimize':
            coeffs = [1/coeff for coeff in coeffs]
        for i in range(div):
            w = ((i+1)/div)*coeffs[0] + ((div-(i+1))/div)*coeffs[1]
            weights.append(w)
        # print(chain, weights)
        res = random.choices(chain, weights=weights)[0]
        if debug: print(f'Selected {res} from {chain} with coeffs {weights}')
        return res

    def choose_genes(genes, coeffs):

        def choose_single(single_genes, single_coeffs):
            local_coeffs = [x for x in single_coeffs]
            while [] in single_genes:
                # print(single_genes)
                for item_index, item in enumerate(single_genes):
                    if item == []:
                        local_coeffs.remove(local_coeffs[item_index])
                        single_genes.remove(item)

            for gene in single_genes:
                chosen_gene = select_genes(single_genes, local_coeffs, ALPHABET)
                res = chosen_gene
                if debug: print("Appended ", chosen_gene)
            if debug: print("Result of chose_single():", res)
            return res

        def choose_multiple(multiple_genes, local_coeffs):
            res = []
            for gene_index in range(len(multiple_genes[0])):
                local_genes = [sequence[gene_index] for sequence in multiple_genes]
                chosen_gene = choose_single(local_genes, local_coeffs)
                res.append(chosen_gene[0])
                if debug: print("Appended to genes", chosen_gene)
            if debug: print("Result of chose_multiple():", res)
            return res

        def make_same_length(unequal_genes, lenghts):
            needed_length = max(lenghts)
            for sequence in unequal_genes:
                while len(sequence) < needed_length:
                    sequence.append([])

        lenghts = set([len(sequence) for sequence in genes])
        if lenghts == (1):
            res = choose_single(genes, coeffs)
        else:
            if len(lenghts) >= 1:
                make_same_length(genes, lenghts)
            res = choose_multiple(genes, coeffs)
        return res

    assert family_size <= len(parents), "Family size cannot exceed full generation"

    families = []
    children = []

    if sort_children_before_mating:
        if optimization == 'minimize': parents = sorted(parents, key=lambda generation: generation['fit_score']['score'])
        elif optimization == 'maximize': parents = sorted(parents, key=lambda generation: generation['fit_score']['score'], reverse=True)

    if family_size > 0:
        for index in range(len(parents)):
            start = index
            end = index + family_size
            family = parents[start:end]
            if len(family) == family_size and family[0] != family[1]: families.append(family)# ToDo: костыль, нужно сделать правильную индексацию, чтобы всегда было по N членов семьи
            # families.append(family) # ToDo: смотри, чтобы родители не были одинаковыми
            if debug: print("Family created: ", family)
    else:
        families = [parents]

    if debug: print("Resulting families: ", families)
    for family in families:
        genes = extract_genes(family)
        weights = [parent['fit_score']['score'] for parent in family]
        coeffs = prepare_coeffs(weights)
        # print(coeffs)

        rand_mutations = True
        if rand_mutations:
            for index, gene in enumerate(genes):
                coeff = 1/coeffs[index] if optimization == 'minimize' else coeffs[index]
                # print(gene, coeff)
                # print(len(gene), (coeff, len(ALPHABET)*len(gene)))
                amount_to_randomize = len(gene)*(coeff/(len(ALPHABET)*len(gene)))
                # print(amount_to_randomize)
                gene = random_subelement(gene, amount_to_randomize, chance=True)
                genes[index] = gene
                # print('1', gene, coeff)

        child = choose_genes(genes, coeffs)
        if unique_entities:
            child = remake_if_non_unique(child, children, choose_genes, genes, coeffs, modify_function=random_subelement)


        # print(child)

        children.append(child)

    # print(children)

    return children

def make_children(generation, make_mode, min_generation_amount=10):
    generation = sorted(generation, key=lambda generation: generation['fit_score']['score'])
    new_generation = []
    # if max_generation_amount == 0:
    if make_mode == 'random':
        while len(new_generation) < min_generation_amount:
            children = mate(generation)
            index = 0
            for child in children:
                new_generation.append({'index': index, 'element': child, 'fit_score': None})
                index += 1
        return new_generation


def guess():
    generation_new = create_generation()
    generation = verify_entities(generation_new, word_to_guess)

    start_time = time.time()
    i = 0

    while 0 not in [entity['fit_score']['score'] for entity in generation]:
        offspring = make_children(generation, 'random', 10)
        generation = verify_entities(offspring, word_to_guess)
        # print(len(generation))
        # print(sorted([entity['fit_score']['score'] for entity in generation]))
        i += 1
    overall_time = time.time() - start_time
    print(f'Solved on iteration {i}')
    print(f'Time taken: {overall_time}')
    print(f'Resulting generation: {generation}')


def guess_blindly():
    pass

def run_tests():
    assert get_distance('a', 'f', ALPHABET) == 5
    assert len(create_chars(5, ALPHABET)) == 5
    assert get_exact_char('a', 5, ALPHABET) == 'f'
    assert get_exact_char('f', get_distance('f', 'a', ALPHABET, to_abs=False), ALPHABET) == 'a'
    # print(mate(parents, family_size=0, unique_entities=True)[['a', 'a'], ['b', 'b']], [0, 1]))

if __name__ == '__main__':
    run_tests()
    # print(create_generation(unique_entities=True))
    #
    #
    #
    # pool = create_generation()
    #
    # print(random.choices(ALPHABET, k=5))
    # print(pool)
    # generation_1 = verify_entities(pool, word_to_guess)
    # print(generation_1)
    # children_1 = make_children(generation_1, 'random', 10)
    # print(children_1)
    # generation_2 = verify_entities(children_1, word_to_guess)
    # print(sorted(generation_1, key=lambda generation: generation['fit_score']['score']))
    # print(sorted(generation_2, key=lambda generation: generation['fit_score']['score']))
    guess()