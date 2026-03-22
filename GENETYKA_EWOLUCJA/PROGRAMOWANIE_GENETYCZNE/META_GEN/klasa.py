class Agent:
    def __init__(self, name, energy):
        self.name = name
        self.energy = energy

    def move(self):
        self.energy -= 10
        return f"{self.name} moved. Energy left: {self.energy}"

    def status(self):
        return f"Agent(name={self.name}, energy={self.energy})"


a1 = Agent("Alpha", 100)

print(a1.status())
print(a1.move())
print(a1.move())
print(a1.status())
