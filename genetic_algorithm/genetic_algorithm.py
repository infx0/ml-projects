import random
from pprint import pprint
ALPHABET = "abcdefghijklmnopqrstuvwxyz "

random.seed(42)

def crossover(parent1: list[str], parent2: list[str], crossover_idx: int) -> tuple[list[str], list[str]]:
    """
    `crossover` performs crossover from the genetic algorithm for a given crossover index. It assumes two parents and creates two children, and assumes that the provided crossover index is valid given the list lengths. **Used by**: [reproduce](#reproduce)

* **parent1** list[str]: the first parent for the crossover operation.
* **parent2** list[str]: the second parent for the crossover operation.
* **crossover_idx** int: the split point at which to do crossover.

**returns** tuple[list[str], list[str]]: the two children that have had the crossover operation completed on them.
    """
    child1 = parent1[:crossover_idx] + parent2[crossover_idx:]
    child2 = parent2[:crossover_idx] + parent1[crossover_idx:]
    return child1, child2

def eval_random_draw(draw_result: float, threshold: float) -> bool:
    """
    `eval_random_draw` evaluates the results of a random draw, and returns a boolean indicating whether to do an action if the random draw is under the threshold. E.g. if the threshold is 0.7, an action should be taken if the draw result is < 0.7. Intended for use on random numbers in the interval [0, 1]. **Used by**: [reproduce](#reproduce)

* **draw_result** float: the result of the random draw.
* **threshold** float: the threshold value.

**returns** bool: whether to perform an action.
    """
    if draw_result < threshold:
        return True
    return False

def mutate(genotype: list[str], position: int, symbol: str) -> list[str]:
    """
    `mutate` applies a mutation to an individual as part of the genetic algorithm. It replaces a symbol within the genotype at the indicated position, with the indicated symbol. It assumes that the provided mutation index is valid given the genotype length, and that the symbol input is a valid part of the encoding scheme. **Used by**: [reproduce](#reproduce)

* **genotype** list[str]: the individual to perform the mutation on.
* **position** int: the position in the genotype that mutates.
* **symbol** str: the new symbol in the encoding that replaces the original symbol.

**returns** list[str]: the individual with the mutation applied.
    """
    new_genotype = list(genotype)
    new_genotype[position] = symbol
    return new_genotype

def reproduce(parent1: list[str], parent2: list[str], crossover_prob: float, mutate_prob: float, encoding: str) -> tuple[list[str], list[str]]:
    """
    `reproduce` performs reproduction using the given parents as part of the genetic algorithm. It checks whether crossover will occur per the random draw, and if not, returns the parents. If there's crossover, it randomly selects a crossover index, does the crossover, and then checks for mutation. If mutation occurs per random draw, a random position and symbol is selected, and the mutation is done before returning the children. **Uses:** [crossover](#crossover), [eval_random_draw](#eval_random_draw), [mutate](#mutate) **Used by**: [genetic_algorithm](#genetic_algorithm)

* **parent1** list[str]: the first parent to use for reproduction.
* **parent2** list[str]: the second parent to use for reproduction.
* **crossover_prob** float: the probability that crossover occurs between the parents and children.
* **mutate_prob** float: the probability that mutation occurs for each child.
* **encoding** str: the string representing the valid characters that can be used for replacement during mutation.

**returns** tuple[list[str], list[str]]: returns either the parents if there's no crossover, or the children if there's crossover and possible mutation.
    """
    do_crossover = eval_random_draw(random.random(), crossover_prob)
    if not do_crossover:
        return parent1, parent2
    crossover_idx = random.randint(1, len(parent1) - 1) # the 1 and -1 prevents a no-op crossover
    child1, child2 = crossover(parent1, parent2, crossover_idx)
    mutate_child1 = eval_random_draw(random.random(), mutate_prob)
    mutate_child2 = eval_random_draw(random.random(), mutate_prob)
    if mutate_child1:
        position = random.randint(0, len(child1) - 1)
        symbol_idx = random.randint(0, len(encoding) - 1)
        symbol = encoding[symbol_idx]
        child1 = mutate(child1, position, symbol)
    if mutate_child2:
        position = random.randint(0, len(child2) - 1)
        symbol_idx = random.randint(0, len(encoding) - 1)
        symbol=encoding[symbol_idx]
        child2 = mutate(child2, position, symbol)
    return child1, child2

def get_ascii(tgt_char: str, rotate: int) -> int:
    """
    `get_ascii` takes a lowercase character, converts it to its ASCII number representation, and then optionally "rotates" by a certain number of letters, e.g. like the Caesar Cipher, and returns an integer that's bounded within the lowercase ASCII representation. **Used by:** [evaluate](#evaluate) 

* **tgt_char** str: the character to be converted and modulated.
* **rotate** int: the number of rotations in the alphabet to perform.

**returns** int: ASCII representation of the converted and rotated character.
    """
    if rotate > 0:
        tgt_ascii = (ord(tgt_char) - ord("a") + rotate) % 26 + ord("a")
    else:
        tgt_ascii = ord(tgt_char)
    return tgt_ascii

def evaluate(population: list[list[str]], target_str: str, flip: bool = False, rotate: int = 0) -> dict:
    """
    `evaluate` evaluates the fitness score across all individuals in the population and returns the best individual with corresponding fitness score. Fitness is defined by converting each character in the target phenotype and each genotype to ASCII using ord(), summing the absolute value of the difference over all characters, and finally evaluating 1 / (1 + f). This function assumes all population characters and the target string are lowercase, and the the length of each genotype list is the same as the number of characters in the target string. There are also optional flip and rotate operations that can be done depending on the application. **Uses:** [get_ascii](#get_ascii)  **Used by**: [genetic_algorithm](#genetic_algorithm), [pick_parents](#pick_parents)

* **population** list[list[str]]: the population to evaluate for the best fitness.
* **target_str** str: the goal string, i.e. the further the genotype is from the goal string (including any flip or rotate operations), the lower the fitness.
* **flip** bool: evaluates against a reversed version of the target string.
* **rotate** int: evaluates against a position of 'rotate' letters down the alphabet.

**returns** dict: the genotype, phenotype, and fitness score of the individual with best fitness.
    """
    best_fitness = float("-inf")
    best_geno = []
    best_pheno = ""
    for p in population:
        p_fitness = 0
        for idx in range(len(p)):
            tgt_idx = -(idx + 1) if flip else idx
            tgt_ascii = get_ascii(target_str[tgt_idx], rotate)
            p_fitness += abs(ord(p[idx]) - tgt_ascii)
        p_fitness  = 1 / (1 + p_fitness)
        if p_fitness > best_fitness:
            best_fitness = p_fitness
            best_geno = p
            best_pheno = "".join(p)
    best = {"genotype": best_geno, "fitness": best_fitness, "phenotype": best_pheno}
    return best

def pick_parents(population: list[list[str]], subsample_size: int, target_str: str, num_parents: int, flip: bool, rotate: int) -> tuple:
        """
        `pick_parents` selects parents as part of the genetic algorithm using tournament-style selection. It uses the subsample size to uniformly select genotypes from the population, pick the genotype with the best fitness, and repeat for as many parents as needed. It assumes that subsample_size is less than or equal to the population size. **Uses:** [evaluate](#evaluate)  **Used by**: [genetic_algorithm](#genetic_algorithm) 

* **population** list[list[str]]: the population of genotypes.
* **subsample_size** int: the number of genotypes to select from the population for evaluation.
* **target_str** str: the phenotype to use in evaluation of the subsample.
* **num_parents** int: the number of parents to create.
* **flip** bool: evaluates against a reversed version of the target string.
* **rotate** int: evaluates against a position of 'rotate' letters down the alphabet.

**returns** tuple: a tuple with length equal to the number of parents. Each tuple element is a list of strings correponding to a genotype.
        """
        parents = []
        for _ in range(num_parents):
            subgroup = [random.choice(population) for _ in range(subsample_size)]
            best_of_subgroup = evaluate(subgroup, target_str, flip=flip, rotate=rotate)
            parents.append(best_of_subgroup["genotype"])
        return tuple(parents)

def generate_random_population(pop_size: int, genotype_len: int, encoding: str = ALPHABET) -> list[list[str]]:
    """
    `generate_random_population` creates a population of genotypes given the encoding and phenotype length. Used in seeding the population for the genetic algorithm. **Used by**: [genetic_algorithm](#genetic_algorithm)

* **pop_size** int: the number of genotypes to generate.
* **genotype_len** int: the length, in characters of each genotype.
* **encoding** str: the encoding to use for creating each genotype, i.e. the possible choices for character.

**returns** list[list[str]]: a population (list) of genotypes.
    """
    population = []
    for _ in range(pop_size):
        genotype = []
        for _ in range(genotype_len):
            idx = random.randint(0, len(encoding) - 1)
            genotype.append(encoding[idx])
        population.append(genotype)
    return population

def genetic_algorithm(pop_size: int, genotype_len: int, encoding: str, limit: int, target_phenotype: str, flip: bool, rotate: int) -> dict:
    """
    `genetic_algorithm` performs a genetic algorithm by seeding a population witih pop_size genotypes, performs evalution using an ASCII string distance measure, and selects parents using tournament-style selection. Children are subject to possible crossover from parents as well as mutation. The algorithm terminates after 'limit' number of generations and returns the best genotype found. **Uses:** [generate_random_population](#generate_random_population), [evaluate](#evaluate), [pick_parents](#pick_parents), [reproduce](#reproduce) 

* **pop_size** int: the number of genotypes to produce from which to seed the algorithm.
* **genotype_len** int: the number of characters in each genotype, which for this problem is a list of character strings.
* **encoding** str: the string with the valid characters for each genotype element.
* **limit** int: the max generations for which to run the algorithm.
* **target_phenotype** str: the desired phenotype for use in fitness function evaluation.
* **flip** bool: evaluate against a reversed version of the target string.
* **rotate** int: evaluate against a position of 'rotate' letters down the alphabet.

**returns** dict: returns the best genotype, phenotype, and fitness score after 'limit' generations of the algorithm.
    """
    generations = 0
    population = generate_random_population(pop_size, genotype_len, encoding)
    best_individual = evaluate(population=[population[0]], target_str=target_phenotype, flip=flip, rotate=rotate)
    while generations < limit:
        next_population = []
        best_individual = evaluate(population, target_phenotype, flip=flip, rotate=rotate)
        for _ in range(int(len(population) / 2)):
            parent1, parent2 = pick_parents(population=population, subsample_size=7, target_str=target_phenotype, num_parents=2, flip=flip, rotate=rotate)
            child1, child2 = reproduce(parent1=parent1, parent2=parent2, crossover_prob=0.9, mutate_prob=0.05, encoding=encoding)
            next_population.append(child1)
            next_population.append(child2)
        population = next_population
        if generations % 10 == 0:
            print("generation: ", generations, "best genotype ", best_individual)
        generations += 1
    
    return best_individual

target1 = "this is so much fun"
pop_size = 100
genotype_len = len(target1)
encoding=ALPHABET
limit = 600

result1 = genetic_algorithm(pop_size=pop_size,
                            genotype_len=genotype_len,
                            encoding=encoding,
                            limit=limit,
                            target_phenotype=target1,
                            flip=False,
                            rotate=0) 

pprint(result1, compact=True)

target2 = "nuf hcum os si siht"
pop_size = 100
genotype_len = len(target2)
encoding=ALPHABET
limit = 500

result2 = genetic_algorithm(pop_size=pop_size,
                            genotype_len=genotype_len,
                            encoding=encoding,
                            limit=limit,
                            target_phenotype=target2,
                            flip=True,
                            rotate=0)

pprint(result2, compact=True)

ALPHABET3 = "abcdefghijklmnopqrstuvwxyz"
target3 = "guvfvffbzhpusha"

pop_size = 100
genotype_len = len(target3)
encoding=ALPHABET3
limit = 500

result3 = genetic_algorithm(pop_size=pop_size,
                            genotype_len=genotype_len,
                            encoding=encoding,
                            limit=limit,
                            target_phenotype=target3,
                            flip=False,
                            rotate=13)

pprint(result3, compact=True)