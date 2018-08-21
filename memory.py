
class MemoryState:
   def __init__(self, state):
      self.state = state 
      # Starts as a losing position
      self.win = -1
      
   def set_state_value(self, value):
      self.win = value

class Memory:
   def __init__(self):
      self.memories = list()

   def add_to_memory(self, state):
      self.memories.append(state)

   def join_memories(self, other):
      self.memories.extend(other)

   def declare_win_or_loss(self, value):
      for memory in self.memories:
         memory.set_state_value(value)

   
