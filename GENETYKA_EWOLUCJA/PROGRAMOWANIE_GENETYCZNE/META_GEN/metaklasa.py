class AgentMeta(type):
    def __new__(mcls, name, bases, namespace):
        if "move" not in namespace:
            raise TypeError(f"Class {name} must define a 'move' method")

        original_move = namespace["move"]

        def wrapped_move(self, *args, **kwargs):
            print(f"[META] Before move in class {name}")
            result = original_move(self, *args, **kwargs)
            print(f"[META] After move in class {name}")
            return result

        namespace["move"] = wrapped_move
        namespace["category"] = "meta-controlled"

        cls = super().__new__(mcls, name, bases, namespace)
        return cls


class MetaAgent(metaclass=AgentMeta):
    def __init__(self, name, energy):
        self.name = name
        self.energy = energy

    def move(self):
        self.energy -= 20
        return f"{self.name} moved under metaclass control. Energy={self.energy}"

    def status(self):
        return f"MetaAgent(name={self.name}, energy={self.energy}, category={self.category})"


a3 = MetaAgent("Gamma", 150)

print(a3.status())
print(a3.move())
print(a3.move())
print(a3.status())
