def move(self):
    self.energy -= 15
    return f"{self.name} dynamically moved. Energy left: {self.energy}"

def status(self):
    return f"[DynamicAgent] name={self.name}, energy={self.energy}"

def init(self, name, energy):
    self.name = name
    self.energy = energy


DynamicAgent = type(
    "DynamicAgent",
    (),
    {
        "__init__": init,
        "move": move,
        "status": status
    }
)

a2 = DynamicAgent("Beta", 120)

print(type(DynamicAgent))
print(a2.status())
print(a2.move())
print(a2.move())
print(a2.status())
