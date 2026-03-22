import ast
import random
import math

# ==========================================
# METAKLASA: pilnuje kontraktu klasy
# ==========================================
class GPAgentMeta(type):
    def __new__(mcls, name, bases, namespace):
        if "decide" not in namespace:
            raise TypeError(f"Class {name} must define decide(self, x)")
        cls = super().__new__(mcls, name, bases, namespace)
        cls.created_by = "GPAgentMeta"
        return cls


# ==========================================
# GENEROWANIE DRZEW AST
# ==========================================
OPS = [ast.Add, ast.Sub, ast.Mult]
TERMINALS = ["x", 1, 2, 3, 5, 7]


def random_terminal():
    t = random.choice(TERMINALS)
    if t == "x":
        return ast.Name(id="x", ctx=ast.Load())
    else:
        return ast.Constant(value=t)


def random_expr(depth=3):
    if depth == 0 or random.random() < 0.35:
        return random_terminal()

    left = random_expr(depth - 1)
    right = random_expr(depth - 1)
    op = random.choice(OPS)()
    return ast.BinOp(left=left, op=op, right=right)


# ==========================================
# AST -> kod funkcji decide(self, x)
# ==========================================
def build_decide_function(expr_ast):
    func_def = ast.FunctionDef(
        name="decide",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self"), ast.arg(arg="x")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[
            ast.Return(value=expr_ast)
        ],
        decorator_list=[]
    )

    module = ast.Module(body=[func_def], type_ignores=[])
    ast.fix_missing_locations(module)

    code = compile(module, filename="<gp_agent>", mode="exec")
    namespace = {}
    exec(code, {}, namespace)
    return namespace["decide"], module


# ==========================================
# TWORZENIE KLASY AGENTA Z AST
# ==========================================
def make_agent_class(class_name, expr_ast):
    decide_func, module = build_decide_function(expr_ast)

    namespace = {
        "decide": decide_func,
        "_ast_module": module,
        "_expr_ast": expr_ast
    }

    cls = GPAgentMeta(class_name, (), namespace)
    return cls


# ==========================================
# FITNESS
# Cel: przybliżenie funkcji target(x) = x*x + 2*x + 1
# ==========================================
def target(x):
    return x * x + 2 * x + 1


TEST_POINTS = [-3, -2, -1, 0, 1, 2, 3]


def fitness(agent_class):
    agent = agent_class()
    error = 0.0

    for x in TEST_POINTS:
        try:
            y_pred = agent.decide(x)
            y_true = target(x)

            # kara za dziwne typy
            if not isinstance(y_pred, (int, float)):
                return -1e9

            error += abs(y_true - y_pred)

        except Exception:
            return -1e9

    # im mniejszy błąd, tym lepiej
    return -error


# ==========================================
# MUTACJA AST
# Zmieniamy losowy fragment drzewa
# ==========================================
def mutate_expr(expr_ast, max_depth=3, mutation_prob=0.2):
    expr_copy = ast.fix_missing_locations(ast.parse(ast.unparse(expr_ast), mode="eval").body)

    def mutate_node(node, depth):
        if random.random() < mutation_prob:
            return random_expr(depth=max(0, depth))

        if isinstance(node, ast.BinOp):
            node.left = mutate_node(node.left, depth - 1)
            node.right = mutate_node(node.right, depth - 1)

            if random.random() < 0.15:
                node.op = random.choice(OPS)()

        return node

    return mutate_node(expr_copy, max_depth)


# ==========================================
# KRZYŻOWANIE
# Bierzemy lewe poddrzewo z jednego, prawe z drugiego
# ==========================================
def crossover_expr(expr1, expr2):
    if isinstance(expr1, ast.BinOp) and isinstance(expr2, ast.BinOp):
        return ast.BinOp(
            left=expr1.left,
            op=random.choice(OPS)(),
            right=expr2.right
        )
    return random.choice([expr1, expr2])


# ==========================================
# POMOCNICZO: ładny wydruk AST jako kod
# ==========================================
def expr_to_source(expr_ast):
    return ast.unparse(expr_ast)


# ==========================================
# INICJALIZACJA POPULACJI
# ==========================================
POP_SIZE = 8
GENERATIONS = 8
ELITE_SIZE = 3

population = []
for i in range(POP_SIZE):
    expr = random_expr(depth=3)
    cls = make_agent_class(f"Agent_{i}", expr)
    population.append(cls)


# ==========================================
# PĘTLA GP
# ==========================================
for gen in range(GENERATIONS):
    scored = [(cls, fitness(cls)) for cls in population]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_cls, best_fit = scored[0]

    print(f"\n=== GENERATION {gen} ===")
    print("Najlepszy fitness:", best_fit)
    print("Najlepsza metoda decide(x):")
    print("   ", expr_to_source(best_cls._expr_ast))

    elites = [cls for cls, _ in scored[:ELITE_SIZE]]
    new_population = elites[:]

    while len(new_population) < POP_SIZE:
        p1 = random.choice(elites)
        p2 = random.choice(elites)

        child_expr = crossover_expr(p1._expr_ast, p2._expr_ast)
        child_expr = mutate_expr(child_expr, max_depth=3, mutation_prob=0.25)

        child_cls = make_agent_class(
            f"Agent_g{gen}_{len(new_population)}",
            child_expr
        )
        new_population.append(child_cls)

    population = new_population


# ==========================================
# WYNIK KOŃCOWY
# ==========================================
final_scored = [(cls, fitness(cls)) for cls in population]
final_scored.sort(key=lambda x: x[1], reverse=True)

best_cls, best_fit = final_scored[0]

print("\n" + "=" * 50)
print("NAJLEPSZY OSOBNIK KOŃCOWY")
print("=" * 50)
print("Fitness:", best_fit)
print("Kod decide(x):", expr_to_source(best_cls._expr_ast))

agent = best_cls()
print("\nPorównanie z target(x)=x*x+2*x+1")
for x in TEST_POINTS:
    print(f"x={x:2d} | target={target(x):3d} | decide={agent.decide(x)}")
