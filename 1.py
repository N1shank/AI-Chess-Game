import math
import itertools
import copy
from collections import defaultdict

class TreeNode(object):
    def _init_(self, data, parent=None):
        self.data = data
        self.children = []
        
    def add_child(self, data):
        self.children.append(TreeNode(data))


class ChessAi(object):
    #set the value of each of the pieces to be used in the hard coded heuristic algorithm
    piece_values = {'p':1,'r':5,'n':3,'b':3,'q':9,'k':15}
    
    def _init_(self, ai_depth = 3):
       
       # input: ai_depth is amount of moves to search into the future.
        self.depth = ai_depth
        self.current_game_state = None
    

    def position_evaluator(chess_position):
        """
        Heuristic algorithm that evaluates a chess position
        input: a chess_position (8 x 8 2d array), such as self.chessboard
        output: a float representing how good the position is 
        (positive score means white is winning, negative score means that black is winning)
        
        Example of a possible heuristic algorithm:
        position_score = sum_of_pieces + king_castled + pawn_islands + free_bishops + forward_knights
        """

        #sum up all material by iterating through the entire chessboard
        final_position_score = 0
        
        #iterate through every row on the chessboard and calculate heuristics adjustment 
        #for the first iteration of this, I will look for developed pieces
        for x, row in enumerate(chess_position):
            for y, j in enumerate(row):
                color = j.split('-')[0]
                piece = j.split('-')[1]
                
                #sum up the value of all the pieces
                if color == 'w':
                    final_position_score += ChessAi.piece_values[piece]
                elif color == 'b':
                    final_position_score -= ChessAi.piece_values[piece]

                #score adjustment for forward pawn (plus .1 for each square that the pawn is advanced)
                if piece == 'p' and color == 'w':
                    final_position_score += (8 - x - 2)*.02 
                if piece == 'p' and color == 'b':
                    final_position_score -= (x - 1)*.02
                    
                #score adjustment for developed knights (plus .1 for each square that the knight is advanced)
                if piece == 'n' and color == 'w':
                    final_position_score += math.pow(1 + (8 - x - 1)*.05, 2)
                    #penalize for knights on the sides of the board
                    if j == 0 or j == 7:
                        final_position_score -= .1

                if piece == 'n' and color == 'b':
                    final_position_score -= math.pow(1 + (x)*.05, 2)
                    #penalize for knights on the sides of the board
                    if j == 0 or j == 7:
                        final_position_score -= .1

                #score adjustment for bishops with open diagonals
                if piece == 'b' and color == 'w':
                    #check for open diagonals
                    new_row = x - 1
                    new_col = y + 1
                    while new_row >= 0 and new_col <= 7:
                        square = chess_position[x][y]
                        #add points for open diagonals
                        if square == '0-0':
                            final_position_score += .05
                            new_row -= 1
                            new_col += 1
                            continue
                        elif piece.split('-')[0] == 'b':
                            final_position_score += .25
                            break
                        elif piece.split('-')[0] == 'w':
                            break

                if piece == 'b' and color == 'w':
                    #check for open diagonals
                    new_row = x - 1
                    new_col = y - 1
                    while new_row >= 0 and new_col >= 0:
                        square = chess_position[x][y]
                        #add points for open diagonals
                        if square == '0-0':
                            final_position_score += .05
                            new_row -= 1
                            new_col -= 1
                            continue
                        elif piece.split('-')[0] == 'b':
                            final_position_score += .25
                            break
                        elif piece.split('-')[0] == 'w':
                            break

                #score adjustment for bishops with open diagonals
                if piece == 'b' and color == 'b':
                    #check for open diagonals
                    new_row = x + 1
                    new_col = y + 1
                    while new_row <= 7 and new_col <= 7:
                        square = chess_position[x][y]
                        #add points for open diagonals
                        if square == '0-0':
                            final_position_score += .05
                            new_row -= 1
                            new_col += 1
                            continue
                        elif piece.split('-')[0] == 'w':
                            final_position_score += .25
                            break
                        elif piece.split('-')[0] == 'b':
                            break

                if piece == 'b' and color == 'b':
                    #check for open diagonals
                    new_row = x + 1
                    new_col = y - 1
                    while new_row <= 7 and new_col >= 0:
                        square = chess_position[x][y]
                        #add points for open diagonals
                        if square == '0-0':
                            final_position_score += .05
                            new_row -= 1
                            new_col -= 1
                            continue
                        elif piece.split('-')[0] == 'w':
                            final_position_score += .25
                            break
                        elif piece.split('-')[0] == 'b':
                            break
        return round(final_position_score, 4)
       

    def tree_generator(self, depth_override = None):
        """
        input: current chess position (8 x 8 2d array)
        output: returns nothing but sets the current game state at self.current_game_state
        For the first iteration, just calculate three moves into the future
        """

        #first, lets try to look one move into the future.  Then we will expand the AI to look more moves into the future 
        #initialize the tree by putting the current state into the parent node of the chessboard. 
        self.current_game_state = TreeNode([copy.deepcopy(self.chessboard),0])
        current_positions = [self.current_game_state]

        current_depth = 1
        target_depth = depth_override or self.depth

        #get the current turn
        current_turn = copy.copy(self.current_turn)
        while current_depth <= target_depth:
            for position in current_positions:
                #returns a dictionary of possible chess moves
                pos_moves = RulesEnforcer.all_possible_moves(position.data[0], current_turn)

                #now we need to generate all possible moves in the future...
                for start, moves in pos_moves.items():
                    for move in moves:
                        current_pos = position.data[0]
                        new_pos = ChessAi.make_hypothetical_move(start, move, current_pos)

                        if current_turn == 'w':
                            score = ChessAi.position_evaluator(new_pos)
                        else:
                            #if black, store the negative score because black wants to play the best move
                            score = -ChessAi.position_evaluator(new_pos)

                        if current_depth > 1:
                            position.add_child([new_pos, score])
                        else:
                            position.add_child([new_pos, score, start, move])

            current_depth += 1

            #now, populate the new current positions list
            new_positions = []
            for position in current_positions:
                new_positions += position.children
            current_positions = new_positions

            #now, switch the turn
            if current_turn == 'w':
                current_turn = 'b' 
            else:
                current_turn = 'w' 


    def minimax(self, node, depth = 0):
        """Minimax algorithm to find the best moves at each layer of the tree
        
        Takes as input a tree of moves and uses minimax to find the best within in that tree.  
        Will use the heuristic algorithm to evaluate the chess moves.   
        
        Basically, use recursion to get to the leaf note of th tree
        Then use the minimax algorithm to backwards compute back to the original value.
        
        input: root/starting node of the possible move tree (created by the tree generator function)
        output: the best move to make at the current state (str)
        """

        #current player wants to maximize the score.
        #opponent wants to minimize the score.

        #current_turn = copy.copy(self.current_turn)
        scores = []

        #if children of children exist, that means you need to go one level deeper
        if node.children[0].children:
            for child in node.children:
                scores.append(self.minimax(child, depth + 1))
            #if its your turn, do max
            if depth % 2 == 0:
                #finally, when you get back to the root node, output the optimal move
                if depth == 0:
                    if self.current_turn == 'w':
                        return node.children[scores.index(max(scores))].data + [max(scores)]
                    elif self.current_turn == 'b':
                        #show the negative score for black to avoid confusion
                        myresult = node.children[scores.index(max(scores))].data
                        myresult[1] = -myresult[1]
                        return myresult + [-max(scores)]

                else:
                    return max(scores)          
            #if its the opponents turn, do min
            else:
                return min(scores)

        else:
            #if no children of children exist, then it is time to apply the minimax algorithm
            for child in node.children:
                scores.append(child.data[1])

            #if its your turn, do max
            if depth % 2 == 0:
                max_scores = max(scores)
                #store in the max or min score so you know how the chess engine evaluates the position
                self.future_position_score = max(scores)
                return max_scores
            #if its the opponents turn, do min
            else:
                min_scores = min(scores)
                #store in the max or min score so you know how the chess engine evaluates the position
                self.future_position_score = min(scores)
                return min_scores


    def make_hypothetical_move(start, finish, chessboard):
        """
        Make a hypothetical move, this will be used to generate the possibilities to be
        stored in the chess tree
        
        """
        #deepcopy the chessboard so that it does not affect the original
        mychessboard = copy.deepcopy(chessboard[:])
        
        #map start and finish to gameboard coordinates
        start  = RulesEnforcer.coordinate_map(start)
        finish = RulesEnforcer.coordinate_map(finish)
        
        #need to move alot of this logic to the rules enforcer
        start_cor0  = start[0]
        start_cor1  = start[1]
        
        finish_cor0 = finish[0]
        finish_cor1 = finish[1]
        
        start_color = mychessboard[start_cor0][start_cor1].split('-')[0]
        start_piece = mychessboard[start_cor0][start_cor1].split('-')[1]
        
        destination_color = mychessboard[finish_cor0][finish_cor1].split('-')[0]
        destination_piece = mychessboard[finish_cor0][finish_cor1].split('-')[1]
        
        #cannot move if starting square is empty
        if start_color == '0':
            return "Starting square is empty!"
        
        mypiece = mychessboard[start_cor0][start_cor1]
        mychessboard[start_cor0][start_cor1] = '0-0'
        mychessboard[finish_cor0][finish_cor1] = mypiece
        
        return mychessboard


class Pawn(object):
   
   
    def moves(cords, color, chessboard = None):
        """
        takes as input the coordinate and color of the piece, outputs the possible moves
        Pawns can attack adjacent diagonals
        """
        
        if not chessboard:
            if color == 'w':
                if int(cords[1]) == 2:
                    #if the pawn is at the starting position then it can move either one or two squares
                    pos_moves = [[cords[0], int(cords[1]) + 1], [cords[0], int(cords[1]) + 2]]
                else:
                    pos_moves = [[cords[0], int(cords[1]) + 1]]
            if color == 'b':
                if int(cords[1]) == 7:
                    pos_moves = [[cords[0], int(cords[1]) - 1], [cords[0], int(cords[1]) - 2]]
                else:
                    pos_moves = [[cords[0], int(cords[1]) - 1]]

        
        temp = False
        if chessboard:
            pos_moves = []
            if color == 'w':
                move1 = [cords[0], int(cords[1]) + 1]
                if RulesEnforcer.collision_detection(move1, color, chessboard) not in ["friend","enemy"]:
                    pos_moves.append(move1)
                else:
                    temp = True

                if int(cords[1]) == 2 and temp == False:
                    move2 = [cords[0], int(cords[1]) + 2]
                    if RulesEnforcer.collision_detection(move2, color, chessboard) not in ["friend","enemy"]:
                        pos_moves.append(move2)
            if color == 'b':
                move1 = [cords[0], int(cords[1]) - 1]
                if RulesEnforcer.collision_detection(move1, color, chessboard) not in ["friend","enemy"]:
                    pos_moves.append(move1)
                else:
                    temp = True

                if int(cords[1]) == 7 and temp == False:
                    move2 = [cords[0], int(cords[1]) - 2]
                    if RulesEnforcer.collision_detection(move2, color, chessboard) not in ["friend","enemy"]:
                        pos_moves.append(move2)

        #check adjacent diagonals for piece taking opportunities
        if chessboard:
            board_cords = RulesEnforcer.coordinate_map(cords)
            if color == 'w':
                #check diagonal left
                if board_cords[0] > 0 and board_cords[1] > 0:
                    row    = board_cords[0] - 1
                    column = board_cords[1] - 1
                    square = chessboard[row][column]
                    if square.split('-')[0] == 'b':
                        final_cord = RulesEnforcer.coordinate_map_reverser([row,column])
                        pos_moves.append([final_cord[0],int(final_cord[1])])

                #check diagonal right
                if board_cords[0] > 0 and board_cords[1] < 7:
                    row    = board_cords[0] - 1
                    column = board_cords[1] + 1
                    square = chessboard[row][column]
                    if square.split('-')[0] == 'b':
                        final_cord = RulesEnforcer.coordinate_map_reverser([row,column])
                        pos_moves.append([final_cord[0],int(final_cord[1])])
                
            if color == 'b':
                #check diagonal left
                if board_cords[0] < 7 and board_cords[1] > 0:
                    row    = board_cords[0] + 1
                    column = board_cords[1] - 1
                    square = chessboard[row][column] 
                    if square.split('-')[0] == 'w':
                        final_cord = RulesEnforcer.coordinate_map_reverser([row,column])
                        pos_moves.append([final_cord[0],int(final_cord[1])])

                #check diagonal right
                if board_cords[0] < 7 and board_cords[1] < 7:
                    row    = board_cords[0] + 1
                    column = board_cords[1] + 1
                    square = chessboard[row][column]
                    if square.split('-')[0] == 'w':
                        final_cord = RulesEnforcer.coordinate_map_reverser([row,column])
                        pos_moves.append([final_cord[0],int(final_cord[1])])

        pos_moves = RulesEnforcer.remove_outofbound_moves(pos_moves)
        return pos_moves
             
class Rook(object):
    """
    Movement behavior for this piece
    
    The rook can move horizontally and vertically
    """
    def moves(cords, color, chessboard = None):
        pos_moves = []
        
        x = cords[0] #g
        y = int(cords[1]) #5
        
        x_new = ord(cords[0]) #g
        y_new = int(cords[1]) #5
        #while x is between 'a' and 'h' and..
        while x_new >= 97 and x_new <= 104 and y_new >= 1 and y_new <= 8:
            x_new = x_new + 1
            

            move = [chr(x_new), y_new]

            if chessboard:
                if RulesEnforcer.collision_detection(move, color, chessboard) == "friend":
                    break
                elif RulesEnforcer.collision_detection(move, color, chessboard) == "enemy":
                    pos_moves.append(move)
                    break
                else:
                    pass

            pos_moves.append(move)
        
        x_new = ord(cords[0]) #g
        y_new = int(cords[1]) #5   
        while x_new >= 97 and x_new <= 104 and y_new >= 1 and y_new <= 8:
            x_new = x_new - 1
            

            move = [chr(x_new), y_new]

            if chessboard:
                if RulesEnforcer.collision_detection(move, color, chessboard) == "friend":
                    break
                elif RulesEnforcer.collision_detection(move, color, chessboard) == "enemy":
                    pos_moves.append(move)
                    break
                else:
                    pass

            pos_moves.append(move)
            
        x_new = ord(cords[0]) #g
        y_new = int(cords[1]) #5     
        while x_new >= 97 and x_new <= 104 and y_new >= 1 and y_new <= 8:
            y_new = y_new - 1

            move = [chr(x_new), y_new]

            if chessboard:
                if RulesEnforcer.collision_detection(move, color, chessboard) == "friend":
                    break
                elif RulesEnforcer.collision_detection(move, color, chessboard) == "enemy":
                    pos_moves.append(move)
                    break
                else:
                    pass

            pos_moves.append(move)
            
        x_new = ord(cords[0]) #g
        y_new = int(cords[1]) #5       
        while x_new >= 97 and x_new <= 104 and y_new >= 1 and y_new <= 8:
            y_new = y_new + 1

            move = [chr(x_new), y_new]

            if chessboard:
                if RulesEnforcer.collision_detection(move, color, chessboard) == "friend":
                    break
                elif RulesEnforcer.collision_detection(move, color, chessboard) == "enemy":
                    pos_moves.append(move)
                    break
                else:
                    pass

            pos_moves.append(move)
        
        pos_moves = RulesEnforcer.remove_outofbound_moves(pos_moves)
        return pos_moves

    
class Knight(object):
    """
    Movement behavior for this piece
    
    The knight moves in an L shape
    """
    def moves(cords, color, chessboard = None):
        """
        takes as input the coordinate and color of the piece, outputs the possible moves
        
        """
        pos_hor1 = [chr(ord(cords[0]) + 1), chr(ord(cords[0]) - 1)]
        pos_hor2 = [chr(ord(cords[0]) + 2), chr(ord(cords[0]) - 2)]

        pos_ver1 = [int(cords[1]) + 1, int(cords[1]) - 1]
        pos_ver2 = [int(cords[1]) + 2, int(cords[1]) - 2]

        pos_moves1 = list(itertools.product(pos_hor1, pos_ver2))  
        pos_moves2 = list(itertools.product(pos_hor2, pos_ver1))  

        pos_moves = pos_moves1 + pos_moves2        
        pos_moves = [list(i) for i in pos_moves]
        
        pos_moves = RulesEnforcer.remove_outofbound_moves(pos_moves)
        
        #If chessboard exists, then run collision detection
        if chessboard:
            for i in range(len(pos_moves)):
                move = pos_moves[i]
                
                if RulesEnforcer.collision_detection(move, color, chessboard) == "friend":
                    pos_moves[i] = 'remove'

            while 'remove' in pos_moves:
                pos_moves.remove('remove')

        return pos_moves
    
class Bishop(object):
    """Movement behavior for this piece"""

    def moves(cords, color, chessboard = None):
        pos_moves = []
        
        x = cords[0] #g
        y = int(cords[1]) #5
        
        x_new = ord(cords[0]) #g
        y_new = int(cords[1]) #5
        #while x is between 'a' and 'h' and..
        while x_new >= 97 and x_new <= 104 and y_new >= 1 and y_new <= 8:
            x_new = x_new + 1
            y_new = y_new + 1

            move = [chr(x_new), y_new]

            if chessboard:
                if RulesEnforcer.collision_detection(move, color, chessboard) == "friend":
                    break
                elif RulesEnforcer.collision_detection(move, color, chessboard) == "enemy":
                    pos_moves.append(move)
                    break
                else:
                    pass

            pos_moves.append(move)
        
        x_new = ord(cords[0]) #g
        y_new = int(cords[1]) #5   
        while x_new >= 97 and x_new <= 104 and y_new >= 1 and y_new <= 8:
            x_new = x_new - 1
            y_new = y_new - 1

            move = [chr(x_new), y_new]

            if chessboard:
                if RulesEnforcer.collision_detection(move, color, chessboard) == "friend":
                    break
                elif RulesEnforcer.collision_detection(move, color, chessboard) == "enemy":
                    pos_moves.append(move)
                    break
                else:
                    pass

            pos_moves.append(move)
            
        x_new = ord(cords[0]) #g
        y_new = int(cords[1]) #5     
        while x_new >= 97 and x_new <= 104 and y_new >= 1 and y_new <= 8:
            x_new = x_new + 1
            y_new = y_new - 1

            move = [chr(x_new), y_new]

            if chessboard:
                if RulesEnforcer.collision_detection(move, color, chessboard) == "friend":
                    break
                elif RulesEnforcer.collision_detection(move, color, chessboard) == "enemy":
                    pos_moves.append(move)
                    break
                else:
                    pass

            pos_moves.append(move)
            
        x_new = ord(cords[0]) #g
        y_new = int(cords[1]) #5       
        while x_new >= 97 and x_new <= 104 and y_new >= 1 and y_new <= 8:
            x_new = x_new - 1
            y_new = y_new + 1

            move = [chr(x_new), y_new]

            if chessboard:
                if RulesEnforcer.collision_detection(move, color, chessboard) == "friend":
                    break
                elif RulesEnforcer.collision_detection(move, color, chessboard) == "enemy":
                    pos_moves.append(move)
                    break
                else:
                    pass

            pos_moves.append(move)
        
        pos_moves = RulesEnforcer.remove_outofbound_moves(pos_moves)
        return pos_moves
    
class Queen(object):
    """Movement behavior for this piece
    Should be a combination of the bishop and the rook, so we should be able to combine those two movements
    """

    def moves(cords, color, chessboard = None):
        pos_moves_verthor = Rook.moves(cords,color,chessboard)
        pos_moves_diag    = Bishop.moves(cords,color,chessboard)
        
        pos_moves = pos_moves_verthor + pos_moves_diag
        pos_moves = RulesEnforcer.remove_outofbound_moves(pos_moves)
        return pos_moves
        
class King(object):
    """Movement behavior for the King.
    The king is a special piece because there is the concept of "check."  
    If the king is under check, then something has to be done to move the king out of check. 
    The state of "check" will be stored here so that we can incorporate this logic.
    """

    def _init_(self, color):
        self.color = color
        self.check = False
        
    def moves(cords, color, chessboard = None):
        x = cords[0] #g
        y = int(cords[1]) #5
        
        pos_moves = []
        
        x_new = ord(cords[0]) #g
        y_new = int(cords[1]) #5
        
        #8 possible locations where the king can move
        pos_moves.append([chr(x_new + 1), y_new])
        pos_moves.append([chr(x_new - 1), y_new])
        pos_moves.append([chr(x_new), y_new + 1])
        pos_moves.append([chr(x_new), y_new - 1])
        pos_moves.append([chr(x_new + 1), y_new + 1])
        pos_moves.append([chr(x_new + 1), y_new - 1])
        pos_moves.append([chr(x_new - 1), y_new + 1])
        pos_moves.append([chr(x_new - 1), y_new - 1])
                                    
        pos_moves = RulesEnforcer.remove_outofbound_moves(pos_moves)

        #If chessboard exists, then run collision detection
        if chessboard:
            for i in range(len(pos_moves)):
                move = pos_moves[i]
                
                if RulesEnforcer.collision_detection(move, color, chessboard) == "friend":
                    pos_moves[i] = 'remove'

            while 'remove' in pos_moves:
                pos_moves.remove('remove')

        return pos_moves

class RulesEnforcer(object):

    """
    Enforces the rules of the game
    Examines the move, and determines whether its a valid move or not.
    """

    letter_dict = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7}
    pos_letters = letter_dict.keys()
    pos_nums = [1,2,3,4,5,6,7,8]
    letter_dict_rev = dict((v,k) for k,v in letter_dict.items())
    possible_pieces = ['p','r','n','b','q','k']
    
    def _init_(self):
        pass

    def check_square(chessboard, coordinate):
        """
        Takes as input a chess board and coordinate and outputs
        what is inside that space
        This is useful for a variable of purposes
        """
        mycord = RulesEnforcer.coordinate_map(coordinate)

        first = mycord[0]
        second = mycord[1]

        return chessboard[first][second]


  
  
    def possible_moves(chessboard, color, piece, coordinate):
        """return possible moves of a piece
        
        a number of things need to be taken into a count
        1. whether we are allowed to move the piece
        
        input:  piece, color, and coordinate of piece
        output: all possible moves of the piece (lists of lists)
        
        Example of a cooridinate: a2
        """

        #if the coordinate is an array 
        if type(coordinate) == list:
            coordinate = RulesEnforcer.coordinate_map_reverser(coordinate)

        #break out coordinate into a list of len(2)
        cords = list(coordinate)
        cords[1] = int(cords[1])

        #pawns
        if piece == 'p':
            pos_moves = Pawn.moves(cords, color, chessboard)  

        #rook
        elif piece == 'r':
            pos_moves = Rook.moves(cords, color, chessboard)
        
        #knight
        elif piece == 'n':
            pos_moves = Knight.moves(cords, color, chessboard)
        
        #bishop
        elif piece == 'b':
            pos_moves = Bishop.moves(cords, color, chessboard)
        
        #queen
        elif piece == "q":
            pos_moves = Queen.moves(cords, color, chessboard)
        
        #king
        elif piece == "k":
            pos_moves = King.moves(cords, color, chessboard)
            
        else:                 
            return "invalid inputs!"
            

        return pos_moves

  
    def all_possible_moves(chessboard, color):
        """takes as input a chessboard and generates all possible moves
        input: 
            color: color that you want to generate moves for, 'w' or 'b'
            chessboard: 8x8 chessboard
        output: dict of all possible moves 
            key: piece and position
            value: list of list of possible moves
        """

        #dict for storing all the moves
        all_moves = defaultdict()

        for cor1, row in enumerate(chessboard):
            for cor2, square in enumerate(row):
                if square.split('-')[0] == color:
                    piece = square.split('-')[1]
                    coordinate = [cor1, cor2]
                    
                    moves = RulesEnforcer.possible_moves(chessboard, color, piece, coordinate)

                    if moves:
                        all_moves[RulesEnforcer.coordinate_map_reverser(coordinate)] = moves

        return all_moves

   
    def remove_outofbound_moves(pos_moves):
        """remove moves that are out of range of the board
        input: list of list of moves
        output: list of list of moves, with out of bound moves removed
        """

        to_remove = []
        for i in range(len(pos_moves)):
            if pos_moves[i][0] not in RulesEnforcer.pos_letters or pos_moves[i][1] not in RulesEnforcer.pos_nums:
                to_remove.append(pos_moves[i])                                          
        for i in to_remove:
            pos_moves.remove(i)
            
        return pos_moves
        

    def collision_detection(move, color, chessboard):
        """
        Collision detection for the chess game.  
        
        input:
            move: the move i.e ['a',7]
            color: white ('w') or black ('b')
            chessboard: chessboard object
        output: "friend" or "enemy" depending on what color you are and what the enemy color is
        
        """ 
        try:
            move = RulesEnforcer.coordinate_map(move)
        except:
            return False

        x = move[0]
        y = move[1]

        try:
            piece = chessboard[x][y]
        except:
            return False

        if color == 'w' and piece.split('-')[0] == 'w':
            return "friend"
        elif color == 'b' and piece.split('-')[0] == 'b':
            return "friend"
        if color == 'w' and piece.split('-')[0] == 'b':
            return "enemy"
        elif color == 'b' and piece.split('-')[0] == 'w':
            return "enemy"
        else:
            return "empty"
         

    def coordinate_map(mycoordinate):
        """takes as input a chess coordinate and maps it to the coordinate in the array
        
        input: chess coordinate (ie a5)
        output: coordinate of the array to be used in the chessboard 
                for example: [0,2]
        
        """
        mycoordinate  = list(mycoordinate)

        starthor  = RulesEnforcer.letter_dict[mycoordinate[0]]
        startver  = 7 - (int(mycoordinate[1]) - 1)
        
        return [startver, starthor]

    def coordinate_map_reverser(myarray):
        """
        Does the opposite of coordinate_map().  Takes as input array coordinates (ie. [0,5])
        This method is useful if you 
        
        input: a length 2 list of array coordinates
        output: chess coordinate (str)
        example:
        [7,0] -> a1
        """

        #letter of cor
        first_cor  = RulesEnforcer.letter_dict_rev[myarray[1]]
        #number of cor
        second_cor = 8 - myarray[0] 
        
        return str(first_cor) + str(second_cor)
        

class ChessGame(RulesEnforcer,ChessAi):
    def _init_(self, ai_depth):
       
        ChessAi._init_(self, ai_depth)
        RulesEnforcer._init_(self)
        self.ai_depth = ai_depth
        self.chessboard = [["0-0"]*9 for i in range(9)]
        
        """Track aspects of the game"""
        #track which pieces have been taken
        self.white_taken = []
        self.black_taken = []
        
        #track which moves have been made in the game, key: move number, value: len 2 list of white and black move
        self.moves_made = {}
        
        #track the number of moves made
        self.move_count = 0
        
        #track whose turn it is (white always starts)
        self.current_turn = "w"
        
        #create pawns
        for i in range(8):
            self.chessboard[1][i] = 'b-p'
            self.chessboard[6][i] = 'w-p'
        
        #create rooks
        self.chessboard[0][0] = 'b-r'
        self.chessboard[0][7] = 'b-r'
        self.chessboard[7][0] = 'w-r'
        self.chessboard[7][7] = 'w-r'
        
        #create knights
        self.chessboard[0][1] = 'b-n'
        self.chessboard[0][6] = 'b-n'
        self.chessboard[7][1] = 'w-n'
        self.chessboard[7][6] = 'w-n'
        
        #create bishops
        self.chessboard[0][2] = 'b-b'
        self.chessboard[0][5] = 'b-b'
        self.chessboard[7][2] = 'w-b'
        self.chessboard[7][5] = 'w-b'
        
        #create queen and king
        self.chessboard[0][3] = 'b-q'
        self.chessboard[0][4] = 'b-k'
        self.chessboard[7][3] = 'w-q'
        self.chessboard[7][4] = 'w-k'

        self.game_over = False
            
    def see_board(self):
        """see the current state of the chessboard"""
        for i in self.chessboard:
            print(i)


    def recommend_move(self, depth_override = None):
        """
        Use the AI to recommend a move
        """
        if not depth_override:
            depth_override = self.ai_depth

        self.tree_generator(depth_override)
        return self.minimax(self.current_game_state, 0)

    def make_move_ai(self, depth_override = None):
        """
        Let the AI make the move
        """
        if not depth_override:
            depth_override = self.ai_depth

        myoutput = self.recommend_move(depth_override)
        start  = myoutput[2]
        finish = myoutput[3]

        self.make_move(start, finish)
        print(start)
        print(finish)

        return self.chessboard


    def make_move(self, start, finish):
        """
        Make a move
        
        input:
        starting coordinate: example "e4"
        ending coordinate: example "e5"
        
        """
        
        #map start and finish to gameboard coordinates
        start  = RulesEnforcer.coordinate_map(start)
        finish = RulesEnforcer.coordinate_map(finish)
        
        #need to move alot of this logic to the rules enforcer
        start_cor0  = start[0]
        start_cor1  = start[1]
        
        finish_cor0 = finish[0]
        finish_cor1 = finish[1]

        start_color = self.chessboard[start_cor0][start_cor1].split('-')[0]
        start_piece = self.chessboard[start_cor0][start_cor1].split('-')[1]
 
        destination_color = self.chessboard[finish_cor0][finish_cor1].split('-')[0]
        destination_piece = self.chessboard[finish_cor0][finish_cor1].split('-')[1]
        
        #cannot move if starting square is empty
        if start_color == '0':
            return "Starting square is empty!"
        
        #cannot move the other person's piece
        if self.current_turn != start_color:
            return "Cannot move the other person's piece!"
        
        #cannot take your own piece 
        if self.current_turn == destination_color:
            return "invalid move, cannot take your own piece!"
        elif self.current_turn != destination_color and destination_color != '0':
            if destination_piece == 'k':
                self.game_over = True
                return "game over, " + self.current_turn + " has won"
            elif self.current_turn == 'w':
                self.black_taken.append(destination_piece)
            elif self.current_turn == 'b':
                self.white_taken.append(destination_piece)     
        else:
            pass
        
        mypiece = self.chessboard[start_cor0][start_cor1]
        self.chessboard[start_cor0][start_cor1] = '0-0'
        self.chessboard[finish_cor0][finish_cor1] = mypiece
        
        #if the move is a success, change the turn state
        if self.current_turn == "w":
            self.current_turn = "b"
        elif self.current_turn == "b":
            self.current_turn = "w"
        
        return self.chessboard


    
    # def current_position_score(self):
    #     """
    #     Get the position score of the current game being played
    #     """
    #     return self.position_evaluator(self.chessboard)





if __name__ == '_main_':

    current_game = ChessGame(3)
    print("Lets play chess!!! Here is the board:\n")
    current_game.see_board()
    print('\n')

    while not current_game.game_over:
        print("Your turn: ")
        start_point = input("Enter starting point coordinate: ")
        end_point = input("Enter ending point coordinate: ")
        current_game.make_move(start_point,end_point)
        print("You have made a move!\n")
        current_game.see_board()
        print('\n')
        
        if current_game.current_turn == 'b':
            print("AI is thinking...")
            current_game.make_move_ai(3)
            print('AI has made a move!\n\n')
            current_game.see_board()
            print('\n')