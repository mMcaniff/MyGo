import agent
import config
import model
import board
import random
from memory import Memory

from agent import Agent

def main():
    print("Beginning BetaGo")

    isExistingPlayer = False


    ######## Create Neural Network Models ########
    current_model = model.Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2, 19, 19), 19*19*2, config.HIDDEN_CNN_LAYERS)
    best_model = model.Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2, 19, 19), 19*19*2, config.HIDDEN_CNN_LAYERS)

    best_model.model.set_weights(current_model.model.get_weights())

    ######## Load Existing Neural Network ########
    if (isExistingPlayer):
        m_tmp = best_model.read("BestNeuralNetwork", initialise.INITIAL_RUN_NUMBER, best_player_version)
        current_model.model.set_weights(m_tmp.get_weights())
        best_model.model.set_weights(m_tmp.get_weights())

    ######## Create Players ########
    current_player = Agent(agent.MCTS(current_model), current_model)
    best_player = Agent(agent.MCTS(best_model), best_model)

    while True:
        ######## Self-play ########
        memory = play_matches(best_player, best_player, games=10)          

        ######## Retrain ########

        ######## Tournament ########
        #print("")

def play_matches(player1, player2, games):
    game = board.GoBoard()
    currentGameCount = 0
    
    scores = {"player1":0, "player2":0}

    switchTurn = {-1:1, 1:-1} 

    main_memory = Memory()   

    while currentGameCount < games: 
        game = board.GoBoard()
        rand = random.randint(0, 1) * 2 - 1
        if rand == -1:
            playerTurn = -1
            players = {1:{"agent": player1, "name":"Player 1"}, -1: {"agent": player2, "name": "Player 2"}}
        else: 
            playerTurn = 1
            players = {1:{"agent": player2, "name":"Player 2"}, -1: {"agent": player1, "name":"Player1"}}
              
        game_running = 0
        turn = 0 
   
        while game_running == 0:
            memory_1 = Memory()
            memory_2 = Memory()
   
            turn += 1 
  
            # Make an action
            #action ,pi, mcts_values, nn_value = players[playerTurn]["agent"].mcts.act()
            new_state, value, preds = players[playerTurn]["agent"].act()                
            print(playerTurn)
            print(players[playerTurn])
            print("Value: " + str(value))
            print(new_state.state) 

            if playerTurn == -1:
                memory_1.add_to_memory(new_state.state)
            else:
                memory_2.add_to_memory(new_state.state)    

            # Switch the current player
            playerTurn = switchTurn[playerTurn]
            
            # Update the new current players game state
            players[playerTurn]["agent"].oppAct(new_state)
         
            if new_state.state.is_game_over():
               print("Game " + str(currentGameCount) + " is over!")
               winner = new_state.state.get_winner()

               print(winner + " Won!")
               if winner == "Player 1":
                  memory_1.declare_win_or_loss(1)
                  memory_2.declare_win_or_loss(-1)
               else: 
                  memory_2.declare_win_or_loss(1)
                  memory_1.declare_win_or_loss(-1)

               main_memory.join_memories(memory_1)
               main_memory.join_memories(memory_2)

               game_running = 1

        currentGameCount += 1

if __name__ == "__main__":
    main()
