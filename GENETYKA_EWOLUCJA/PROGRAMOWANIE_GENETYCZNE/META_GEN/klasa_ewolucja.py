import random


class EvolvableMeta(type):
    def __new__(mcls, name, bases, namespace):
        if "move_cost" in namespace and namespace["move_cost"] <= 0:
            raise ValueError(f"{name}: move_cost must be > 0")

        cls = super().__new__(mcls, name, bases, namespace)
        cls.created_by = "EvolvableMeta"
        return cls


def make_agent_class(class_name, move_cost, reward):
    def __init__(self, name, energy=100):
        self.name = name
        self.energy = energy
        self.points = 0

    def move(self):
        if self.energy >= move_cost:
            self.energy -= move_cost
            self.points += reward
        return self.points, self.energy

    def fitness(self):
        return self.points + self.energy

    namespace = {
        "move_cost": move_cost,
        "reward": reward,
        "__init__": __init__,
        "move": move,
        "fitness": fitness
    }

    return EvolvableMeta(class_name, (), namespace)


population = []

for i in range(8):
    move_cost = random.randint(5, 30)
    reward = random.randint(5, 25)
    AgentClass = make_agent_class(f"Agent_{i}", move_cost, reward)
    population.append(AgentClass)

results = []

for AgentClass in population:
    agent = AgentClass(AgentClass.__name__)
    for _ in range(5):
        agent.move()

    score = agent.fitness()
    results.append((AgentClass.__name__, AgentClass.move_cost, AgentClass.reward, score))

results.sort(key=lambda x: x[3], reverse=True)

print("=== EVOLVED CLASSES RANKING ===")
for name, move_cost, reward, score in results:
    print(f"{name:10s} | move_cost={move_cost:2d} | reward={reward:2d} | fitness={score}")

best = results[0]
print("\nBest class:")
print(best)
