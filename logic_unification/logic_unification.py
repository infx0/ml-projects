"""
This script parses simple parenthesized logic expressions and attempts to unify two
expressions by finding variable substitutions that make them syntactically equal.
Variables are strings beginning with "?", constants are ordinary strings, and nested
parenthesized expressions are represented as Python lists.

The main workflow starts in unify(), which parses each input string into list form,
normalizes single atoms into one-item lists, and then calls unification(). The
recursive unification algorithm compares constants directly, binds variables when
possible, rejects circular bindings, applies discovered substitutions to the remaining
subexpressions, and merges compatible substitution dictionaries. Helper functions
format nested substitution results into readable strings and apply substitutions
in-place across expression lists.

When run as a script, the file executes a set of self-checking examples that print the
actual and expected unification results, then assert that the implementation behaves as
expected for matching constants, variable bindings, nested expressions, failures, and
empty expressions.
"""

import tokenize
from io import StringIO


def atom(next, token):
    """
    Parses a single token into an atom, variable, or nested logic expression.

    Args:
        next (callable): The token generator's next-token function.
        token (tuple): The current token being evaluated.

    Returns:
        str or list: Returns a parsed string atom, variable string, or nested list.
    """
    if token[1] == "(":
        out = []
        token = next()
        while token[1] != ")":
            out.append(atom(next, token))
            token = next()
            if token[1] == " ":
                token = next()
        return out
    elif token[1] == "?":
        token = next()
        return "?" + token[1]
    else:
        return token[1]


def parse(exp):
    """
    Converts a string expression into its parsed atom or nested list representation.

    Args:
        exp (str): The logic expression string to parse.

    Returns:
        str or list: Returns the parsed expression.
    """
    src = StringIO(exp).readline
    tokens = tokenize.generate_tokens(src)
    return atom(tokens.__next__, tokens.__next__())


def is_variable(exp):
    """
    Checks whether an expression is a variable.

    Args:
        exp (str or list): The expression to evaluate.

    Returns:
        bool: Returns True when the expression is a string beginning with "?".
    """
    return isinstance(exp, str) and exp[0] == "?"


def is_constant(exp):
    """
    Checks whether an expression is a constant.

    Args:
        exp (str or list): The expression to evaluate.

    Returns:
        bool: Returns True when the expression is a string and not a variable.
    """
    return isinstance(exp, str) and not is_variable(exp)


def format_substitution_list(subst: dict) -> dict:
    """
    Is a helper function that converts dictionary values comprised of string lists into
    single strings for use in checking the substition lists during unit tests.

    Args:
        subst (dict): The dictionary to evaluate for modification.

    Returns:
        dict: Return the altered dictionary.
    """
    subst = dict(sorted(subst.items()))
    for k, v in subst.items():
        if isinstance(v, list):
            if len(v) > 1:
                subst[k] = " ".join(v)
            else:
                subst[k] = v[0]
    return subst


def apply_result(result: dict, exp: list) -> None:
    """
    Is a helper function that recursively makes in-place variable substitutions in logic
    expressions.

    Args:
        result (dict): The resultant variable to constant mapping from recursive
            unification.
        exp (list): The logic expression. Modified in-place.

    Returns:
        None: Does not return anything and modifies in-place.
    """
    for i, e in enumerate(exp):
        if isinstance(e, list):
            apply_result(result, e)
        elif e in result:
            exp[i] = result[e]


def unification(list_expression1, list_expression2) -> dict:
    """
    Takes two lists of strings that have been parsed by the parse function, and runs the
    unification algorithm on them, evaluating what substitution can be made, if any,
    that make the expressions syntactically equal.

    Args:
        list_expression1 (list): The first parsed logic expression.
        list_expression2 (list): The second parsed logic expression.

    Returns:
        dict: Returns the substitution list as a result of the unification algorithm.
    """
    if (is_constant(list_expression1) or not list_expression1) and (
        is_constant(list_expression2) or not list_expression2
    ):
        if list_expression1 == list_expression2:
            return {}
        else:
            return None
    if is_variable(list_expression1):
        if list_expression1 in list_expression2:
            return None
        else:
            return {list_expression1: list_expression2}
    if is_variable(list_expression2):
        if list_expression2 in list_expression1:
            return None
        else:
            return {list_expression2: list_expression1}

    if isinstance(list_expression1, list):
        first1 = list_expression1[0]
        rest1 = list_expression1[1:]
    else:
        first1 = list_expression1
    if isinstance(list_expression2, list):
        first2 = list_expression2[0]
        rest2 = list_expression2[1:]
    else:
        first2 = list_expression2

    result1 = unification(first1, first2)
    if result1 is None:
        return None
    if isinstance(rest1, list) and isinstance(rest2, list):
        apply_result(result1, rest1)
        apply_result(result1, rest2)
    result2 = unification(rest1, rest2)
    if result2 is None:
        return None
    for k in result1.keys() & result2.keys():
        if result1[k] != result2[k]:
            return None
    result_out = result1 | result2
    result_out = dict(sorted(result_out.items()))
    result_out = format_substitution_list(result_out)

    return result_out


def list_check(parsed_expression):
    """
    Ensures a parsed expression is represented as a list.

    Args:
        parsed_expression (str or list): The parsed expression to evaluate.

    Returns:
        list: Returns the original list or wraps a single atom in a list.
    """
    if isinstance(parsed_expression, list):
        return parsed_expression
    return [parsed_expression]


def unify(s_expression1, s_expression2):
    """
    Parses two string expressions and runs the unification algorithm on them.

    Args:
        s_expression1 (str): The first logic expression string.
        s_expression2 (str): The second logic expression string.

    Returns:
        dict: Returns the substitution list, or None when unification fails.
    """
    list_expression1 = list_check(parse(s_expression1))
    list_expression2 = list_check(parse(s_expression2))
    return unification(list_expression1, list_expression2)


if __name__ == "__main__":
    self_check_test_cases = [
        ["(son Barney Barney)", "(daughter Wilma Pebbles)", None],
        ["Fred", "Barney", None],
        ["Pebbles", "Pebbles", {}],
        ["(quarry_worker Fred)", "(quarry_worker ?x)", {"?x": "Fred"}],
        ["(son Barney ?x)", "(son ?y Bam_Bam)", {"?x": "Bam_Bam", "?y": "Barney"}],
        ["(married ?x ?y)", "(married Barney Wilma)", {"?x": "Barney", "?y": "Wilma"}],
        [
            "(son Barney ?x)",
            "(son ?y (son Barney))",
            {"?x": "son Barney", "?y": "Barney"},
        ],
        ["(son Barney ?x)", "(son ?y (son ?y))", {"?x": "son Barney", "?y": "Barney"}],
        ["(son Barney Bam_Bam)", "(son ?y (son Barney))", None],
        ["(loves Fred Fred)", "(loves ?x ?x)", {"?x": "Fred"}],
        ["(future George Fred)", "(future ?y ?y)", None],
    ]
    for case in self_check_test_cases:
        exp1, exp2, expected = case
        actual = unify(exp1, exp2)
        print(f"actual = {actual}")
        print(f"expected = {expected}")
        print("\n")
        assert expected == actual

    new_test_cases = [
        [
            "(son Barney Barney)",
            "(daughter Wilma Pebbles)",
            None,
            "non-equal constants",
        ],
        ["(?x)", "(?x)", None, "equal variables"],
        ["(?x)", "(?y)", {"?x": "?y"}, "non-equal variables"],
        ["(?x)", "(?x (?x))", None, "nested equal variables"],
        ["(?x ?y ?z)", "(a b c)", {"?x": "a", "?y": "b", "?z": "c"}, "three variables"],
        ["()", "()", {}, "no inputs, can still unify"],
    ]
    for case in new_test_cases:
        exp1, exp2, expected, message = case
        actual = unify(exp1, exp2)
        print(f"Testing {message}...")
        print(f"actual = {actual}")
        print(f"expected = {expected}")
        print("\n")
        assert expected == actual
