import ast
from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple
import hashlib
import json

try:
    from ast import unparse  # Available in Python 3.9+
except ImportError:
    import astunparse
    def unparse(node):
        return astunparse.unparse(node)

def add_parent_pointers(node):
    for child in ast.iter_child_nodes(node):
        child.parent = node
        add_parent_pointers(child)

class HierarchicalNormalizer(ast.NodeTransformer):
    def __init__(self, level: int):
        self.level = level
        self.var_placeholder = 'VAR'
        self.num_placeholder = 'NUM'
        self.str_placeholder = 'STR'
        self.func_name_mapping = {}
        self.func_placeholder_counter = 1
        self.function_names = set()

    def visit_Call(self, node):
        if self.level >= 2:
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                self.function_names.add(func_name)
                if func_name not in self.func_name_mapping:
                    placeholder = f'FUNC_{self.func_placeholder_counter}'
                    self.func_name_mapping[func_name] = placeholder
                    self.func_placeholder_counter += 1
                else:
                    placeholder = self.func_name_mapping[func_name]
                node.func.id = placeholder
            elif isinstance(node.func, ast.Attribute):
                self.visit(node.func)
            # Visit arguments to capture function names used as arguments
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)
        else:
            self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if self.level >= 1:
            if node.id not in self.function_names:
                node.id = self.var_placeholder
        return node

    def visit_Num(self, node):
        if self.level >= 3:
            node.n = self.num_placeholder
        return node

    def visit_Str(self, node):
        if self.level >= 3:
            node.s = self.str_placeholder
        return node

    def visit_Constant(self, node):
        if self.level >= 3:
            if isinstance(node.value, (int, float)):
                node.value = self.num_placeholder
            elif isinstance(node.value, str):
                node.value = self.str_placeholder
        return node

def parse_program(code_str: str) -> ast.AST:
    try:
        return ast.parse(code_str)
    except SyntaxError as e:
        print(f"SyntaxError while parsing code:\n{code_str}\n{e}")
        return None

def normalize_ast(tree: ast.AST, level: int) -> ast.AST:
    normalizer = HierarchicalNormalizer(level)
    normalized_tree = normalizer.visit(tree)
    ast.fix_missing_locations(normalized_tree)
    return normalized_tree

def serialize_node(node: ast.AST) -> str:
    node_json = ast_to_json(node)
    node_str = json.dumps(node_json, sort_keys=True)
    node_hash = hashlib.md5(node_str.encode('utf-8')).hexdigest()
    return node_hash

def ast_to_json(node: ast.AST) -> Any:
    if isinstance(node, ast.AST):
        result = {'_type': type(node).__name__}
        for field, value in ast.iter_fields(node):
            result[field] = ast_to_json(value)
        return result
    elif isinstance(node, list):
        return [ast_to_json(item) for item in node]
    else:
        return node

def extract_subtrees(node: ast.AST) -> List[ast.AST]:
    subtrees = []

    def _extract(node):
        subtrees.append(node)
        for child in ast.iter_child_nodes(node):
            _extract(child)

    _extract(node)
    return subtrees

def process_programs(programs: List[str], max_level: int = 2):
    substructure_maps: Dict[int, Dict[str, List[Dict[str, Any]]]] = {
        level: defaultdict(list) for level in range(max_level + 1)
    }

    for idx, code_str in enumerate(programs):
        print(f"Processing Program {idx + 1}/{len(programs)}")
        tree = parse_program(code_str)
        if tree is None:
            continue
        add_parent_pointers(tree)
        for level in range(max_level + 1):
            normalized_tree = normalize_ast(tree, level)
            subtrees = extract_subtrees(normalized_tree)
            for subtree in subtrees:
                serialized = serialize_node(subtree)
                substructure_maps[level][serialized].append({
                    'program_index': idx,
                    'subtree': subtree,
                    'level': level,
                })
    return substructure_maps

def detect_cross_level_matches(substructure_maps: Dict[int, Dict[str, List[Dict[str, Any]]]], levels: List[int]):
    matches = []
    levels = sorted(levels)
    for level in levels:
        current_level_subs = substructure_maps[level]
        for serialized_sub, occurrences in current_level_subs.items():
            for other_level in levels:
                if other_level <= level:
                    continue
                other_level_subs = substructure_maps[other_level]
                if serialized_sub in other_level_subs:
                    combined_occurrences = occurrences + other_level_subs[serialized_sub]
                    programs_involved = set([occ['program_index'] for occ in combined_occurrences])
                    if len(programs_involved) < 2:
                        continue
                    match_info = {
                        'substructure_hash': serialized_sub,
                        'substructure': occurrences[0]['subtree'],
                        'levels': [level, other_level],
                        'occurrences': combined_occurrences,
                        'programs_involved': programs_involved,
                    }
                    matches.append(match_info)
    print(f"Total matches found: {len(matches)}")
    return matches

def compute_abstraction_metrics(subtree: ast.AST) -> Tuple[int, int]:
    node_count = 0
    placeholders: Set[str] = set()

    def _count_nodes(node):
        nonlocal node_count
        node_count += 1
        if isinstance(node, ast.Name):
            if node.id.startswith('VAR') or node.id.startswith('FUNC'):
                placeholders.add(node.id)
        elif isinstance(node, ast.Constant):
            if node.value in ['NUM', 'STR']:
                placeholders.add(str(node.value))
        for child in ast.iter_child_nodes(node):
            _count_nodes(child)

    _count_nodes(subtree)

    number_of_placeholders = len(placeholders)
    return node_count, number_of_placeholders

def generate_abstraction_candidates(matches: List[Dict[str, Any]]):
    abstractions = []
    for match in matches:
        programs_involved = match['programs_involved']
        if len(programs_involved) < 2:
            continue

        subtree = match['substructure']
        node_count, number_of_placeholders = compute_abstraction_metrics(subtree)
        if node_count < 5:
            continue

        abstraction_code, params = create_abstraction_function(subtree)
        abstraction = {
            'substructure_hash': match['substructure_hash'],
            'levels': match['levels'],
            'programs': list(programs_involved),
            'node_count': node_count,
            'placeholders': number_of_placeholders,
            'function_code': abstraction_code,
            'parameters': params,
        }
        abstractions.append(abstraction)
    return abstractions

def create_abstraction_function(subtree: ast.AST) -> Tuple[str, List[str]]:
    placeholders = set()

    class PlaceholderReplacer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id.startswith('VAR') or node.id.startswith('FUNC'):
                placeholders.add(node.id)
            return node

        def visit_Constant(self, node):
            if node.value in ['NUM', 'STR']:
                placeholders.add(str(node.value))
            return node

    replacer = PlaceholderReplacer()
    subtree_copy = replacer.visit(subtree)

    params = sorted(placeholders)

    body = [subtree_copy] if isinstance(subtree_copy, ast.stmt) else [ast.Expr(value=subtree_copy)]

    func_def = ast.FunctionDef(
        name='abstraction',
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=param, annotation=None) for param in params],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=body,
        decorator_list=[]
    )
    module = ast.Module(body=[func_def], type_ignores=[])

    ast.fix_missing_locations(module)

    code = unparse(module).strip()
    return code, params

if __name__ == '__main__':
    programs = [
        """
for i in range(8):
    forward(1*i)
    left(90.0)
""",
        """
for i in range(7):
    forward(16)
    left(180.0 - 51.42857142857143)
""",
        """
for i in range(7):
    embed(\"\"\"for j in range(4):
    forward(2*i)
    left(90.0)\"\"\", locals())
""",
        """
for i in range(6):
    embed(\"\"\"for j in range(4):
    forward(2*i)
    left(90.0)\"\"\", locals())
""",
        """
for j in range(6):
    for i in range(HALF_INF):
        forward(EPS_DIST*j)
        left(EPS_ANGLE)
    for i in range(HALF_INF):
        forward(EPS_DIST*j)
        left(EPS_ANGLE)
""",
        """
for j in range(5):
    for i in range(HALF_INF):
        forward(EPS_DIST*j)
        left(EPS_ANGLE)
    for i in range(HALF_INF):
        forward(EPS_DIST*j)
        left(EPS_ANGLE)
""",
    ]

    max_level = 4
    substructure_maps = process_programs(programs, max_level)

    levels = list(range(max_level + 1))
    matches = detect_cross_level_matches(substructure_maps, levels)

    abstractions = generate_abstraction_candidates(matches)

    abstractions.sort(key=lambda x: (x['placeholders'], -x['node_count']))

    print("\nAbstraction Candidates:")
    for idx, abstraction in enumerate(abstractions):
        print(f"\nAbstraction {idx + 1}:")
        print(f"Substructure Hash: {abstraction['substructure_hash']}")
        print(f"Levels: {abstraction['levels']}")
        print(f"Programs Involved: {abstraction['programs']}")
        print(f"Node Count: {abstraction['node_count']}")
        print(f"Placeholders: {abstraction['placeholders']}")
        print("Function Definition:")
        print(abstraction['function_code'])