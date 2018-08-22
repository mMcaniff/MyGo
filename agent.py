import math
import copy as cp
import numpy as np
import random
import config 
import board as gb
import time
import debugger

class Agent: 
   
   def __init__(self, treeSearch, model):
      self.mcts = treeSearch 
      self.model = model

   def act(self):
      #Simulate
                
      start = time.time()
      for i in range(config.MCTS_SIMS):
         next_state = self.mcts.select()
         if next_state == None: 
            value, probs = self.mcts.expand_and_evaluate()
            self.mcts.backup(value)
         else: 
            self.mcts.current_node = next_state
         #debugger.log("------ Act Loop Time elapsed: ", end - start)
      new_state = self.mcts.play()
      value, preds = self.mcts.get_predictions(new_state.state) 

      end = time.time()
      debugger.log("------ Act Time elapsed: " + str(end - start))

      return (new_state, value, preds)
      

   def oppAct(self, new_state):
      self.mcts.root = new_state 
      return 


class TreeNode:
    previous_tree_node = None

    def __init__(self, previous, state, color, level):
        self.previous_tree_node = previous
        self.state = state
        self.next_states = []
        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = 0
        self.probability = 0
        self.color = color
        self.level = level

    def add_state(self, state):
        self.next_states.append(state)

    def visit(self):
        self.visit_count += 1

class MCTS:
    root = TreeNode(None, gb.GoBoard(), "w", 0)
    current_node = root
    exploration = 1
    total_visits = 0

    def __init__(self, model):
        self.model = model

    def select(self):
        if self.current_node == None: 
           return

        start = time.time()
        next_state = None
        best_value = 0
        for state in self.current_node.next_states:
            q_value = state.mean_action_value
            explore_value = random.random() * self.exploration
            u_value = explore_value + math.sqrt(self.current_node.visit_count / (1 + state.visit_count))
            
            value = q_value + u_value
            #print("values", q_value, u_value, value)
            if value > best_value:
                next_state = state
                best_value = value

        end = time.time()
        #print("------ Select Time elapsed: ", end - start)
        return next_state

    def expand_and_evaluate(self):
        if self.current_node.color == "b":
            color = "w"
        else:
            color = "b"
        moves = self.current_node.state.get_all_legal_moves(color)
        for move in moves:
            new_state = cp.deepcopy(self.current_node.state)
            new_state.apply_move(color, move)
            temp_state = TreeNode(self.current_node, new_state, color, self.current_node.level + 1)
            self.current_node.add_state(temp_state)

        #Todo: Pass the state to the CNN for evaluation
        value, probs = self.get_predictions(self.current_node.state)

        #print("Predictions")
        #print("Value")
        #print(value)
        #print("Probs")
        #print(probs)

        return (value, probs)

    def backup(self, value):
        start = time.time()
        tree_node = self.current_node

        temp_node = self.update_node(tree_node, value[0])
        while temp_node:
            tree_node = temp_node
            temp_node = self.update_node(temp_node, value[0])
   
        if not (tree_node == self.root):
            print("Looks like we took a wrong turn")
            return
        self.current_node = tree_node

        end = time.time()
        #print("------ Backup Time elapsed: ", end - start)


    def get_predictions(self, state):
        start = time.time()
        image = np.expand_dims(self.model.convert_to_input(state), axis=0)
        predictions = self.model.predict(image)

        value_array = predictions[0]
        logits_array = predictions[1]
        value = value_array[0]
        logits = logits_array[0]

        odds = np.exp(logits)
        probs = odds / np.sum(odds)

        end = time.time()
        #print("------ Predictions Time elapsed: ", end - start)

        return (value, probs) 

    # Returns next node or None if there is not one
    def update_node(self, tree_node, value):
        tree_node.visit_count += 1
        tree_node.total_action_value += value
        tree_node.mean_action_value = tree_node.total_action_value / tree_node.visit_count
        return tree_node.previous_tree_node

    def play(self):
        best_state = None
        best_value = 0
        for state in self.root.next_states:
            if state.mean_action_value >= best_value:
                best_state = state
                best_value = state.mean_action_value
        best_state.previous_tree_node = None
        self.root = self.current_node = best_state
        return best_state

    def print_tree(self):
        return


