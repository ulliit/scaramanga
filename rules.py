import copy
from log import Log #_260412

class Rules:

# covers the rules for Chess, Tactalaris and ChessXp
# python translation of the original js file which was
# 100% hand-coded
# Copyright (c) 2025 Ulrich Schwekendiek
# license: MIT
# https://opensource.org/license/mit
# tactalaris.com

    def __init__(self):

        self.logger = Log(); #_260412

        # misc
        self.invalidFen = ""

        # mode params
        self.abilities = {}

        self.boardSize = 0  # number of columns resp. rows

        self.cSquares = {}  # squares that are affected by castling

        self.files = []  # ["a","b","c", ... "j"]

        self.squareNumbers = {}  # a10: 0, ..., j1: 99

        self.squares = []  # ["a10","b10", ... "j1"]

        self.squaresCnt = 0  # 64 or 100

        # game params
        self.active = ""

        self.capturedCnt = 0 # number of captured pieces. used when checking for insufficient material.

        self.castlings = ""

        self.epField1 = ""
        self.epField2 = ""

        self.fullmoveNr = 0
        self.halfmoveClock = 0

        self.lastRemainingPiece = "" # last remaining piece on the board that is not a king. used when checking for insufficient material.

        self.kingSquares = {"k": "", "K": ""}

        self.position = {}

        # -------------------- initialize abilities ---------------------

        self.abilities64 = {}
        self.abilities100 = {}

        self.cSquares64 = {
            "w00": ["e1", "f1", "g1", "h1"],
            "w000": ["e1", "d1", "c1", "b1", "a1"],
            "b00": ["e8", "f8", "g8", "h8"],
            "b000": ["e8", "d8", "c8", "b8", "a8"]
        }

        # rooks on b & i is the default pattern for 10x10

        self.cSquares100 = {
            "w00": ["f1", "g1", "h1", "i1"],
            "w000": ["f1", "e1", "d1", "c1", "b1"],
            "b00": ["f10", "g10", "h10", "i10"],
            "b000": ["f10", "e10", "d10", "c10", "b10"]
        }

        # _240809
        # rooks on a & h is an accepted pattern as well

        self.cSquares100Long = {
            "w00": ["f1", "g1", "h1", "i1", "j1"],
            "w000": ["f1", "e1", "d1", "c1", "b1", "a1"],
            "b00": ["f10", "g10", "h10", "i10", "j10"],
            "b000": ["f10", "e10", "d10", "c10", "b10", "a10"]
        }

        # _240809
        self.files64 = ["a", "b", "c", "d", "e", "f", "g", "h"]
        self.files100 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

        # offsets are the numbers that must be added to the index of the square the piece is located on,
        # to get the indices of the fields the piece could reach from this square on an empty board.

        # ranks are used indicate an offboard status. if the rank the detected field is on is not the same
        # as the rank specified for the the offset, the offset led to an offboard field but was incorrectly
        # assigned to a field on the next rank. example: for a knight on e5 an offset -19 matches the field
        # f7 (correct), but for a knight on j5 it matches the field a6 (wrong).

        # separate offsets for black and white pawns, since they can only move in one direction

        self.offsets64 = {
            "P": [{"off": -9, "ranks": 1}, {"off": -8, "ranks": 1}, {"off": -16, "ranks": 2}, {"off": -7, "ranks": 1}],
            "p": [{"off": 7, "ranks": 1}, {"off": 8, "ranks": 1}, {"off": 16, "ranks": 2}, {"off": 9, "ranks": 1}],
            "n": [{"off": -15, "ranks": 2}, {"off": -6, "ranks": 1}, {"off": 10, "ranks": 1}, {"off": 17, "ranks": 2},
                  {"off": 15, "ranks": 2}, {"off": 6, "ranks": 1}, {"off": -10, "ranks": 1}, {"off": -17, "ranks": 2}],
            "b": [{"off": -7}, {"off": 9}, {"off": 7}, {"off": -9}],
            "r": [{"off": 1, "ranks": 0}, {"off": 8}, {"off": -1, "ranks": 0}, {"off": -8}],
            "q": [{"off": -7}, {"off": 1, "ranks": 0}, {"off": 9}, {"off": 8},
                  {"off": 7}, {"off": -1, "ranks": 0}, {"off": -9}, {"off": -8}],
            "k": [{"off": -7, "ranks": 1}, {"off": 1, "ranks": 0}, {"off": 9, "ranks": 1}, {"off": 8, "ranks": 1},
                  {"off": 7, "ranks": 1}, {"off": -1, "ranks": 0}, {"off": -9, "ranks": 1}, {"off": -8, "ranks": 1}]
        }

        self.offsets100 = {
            "P": [{"off": -11, "ranks": 1}, {"off": -10, "ranks": 1}, {"off": -20, "ranks": 2},
                  {"off": -30, "ranks": 3}, {"off": -9, "ranks": 1}],
            "p": [{"off": 9, "ranks": 1}, {"off": 10, "ranks": 1}, {"off": 20, "ranks": 2},
                  {"off": 30, "ranks": 3}, {"off": 11, "ranks": 1}],
            "c": [{"off": -29, "ranks": 3}, {"off": -7, "ranks": 1}, {"off": 13, "ranks": 1}, {"off": 31, "ranks": 3},
                  {"off": 29, "ranks": 3}, {"off": 7, "ranks": 1}, {"off": -13, "ranks": 1}, {"off": -31, "ranks": 3}],  # _240422
            "z": [{"off": -28, "ranks": 3}, {"off": -17, "ranks": 2}, {"off": 23, "ranks": 2}, {"off": 32, "ranks": 3},
                  {"off": 28, "ranks": 3}, {"off": 17, "ranks": 2}, {"off": -23, "ranks": 2}, {"off": -32, "ranks": 3}],  # _240422
            "n": [{"off": -19, "ranks": 2}, {"off": -8, "ranks": 1}, {"off": 12, "ranks": 1}, {"off": 21, "ranks": 2},
                  {"off": 19, "ranks": 2}, {"off": 8, "ranks": 1}, {"off": -12, "ranks": 1}, {"off": -21, "ranks": 2}],
            "b": [{"off": -9}, {"off": 11}, {"off": 9}, {"off": -11}],
            "f": [{"off": -29, "ranks": 3}, {"off": -28, "ranks": 3}, {"off": -17, "ranks": 2}, {"off": -7, "ranks": 1},
                  {"off": 13, "ranks": 1}, {"off": 23, "ranks": 2}, {"off": 32, "ranks": 3}, {"off": 31, "ranks": 3},
                  {"off": 29, "ranks": 3}, {"off": 28, "ranks": 3}, {"off": 17, "ranks": 2}, {"off": 7, "ranks": 1},
                  {"off": -13, "ranks": 1}, {"off": -23, "ranks": 2}, {"off": -32, "ranks": 3}, {"off": -31, "ranks": 3}],
            "r": [{"off": 1, "ranks": 0}, {"off": 10}, {"off": -1, "ranks": 0}, {"off": -10}],
            "q": [{"off": -9}, {"off": 1, "ranks": 0}, {"off": 11}, {"off": 10},
                  {"off": 9}, {"off": -1, "ranks": 0}, {"off": -11}, {"off": -10}],
            "k": [{"off": -9, "ranks": 1}, {"off": 1, "ranks": 0}, {"off": 11, "ranks": 1}, {"off": 10, "ranks": 1},
                  {"off": 9, "ranks": 1}, {"off": -1, "ranks": 0}, {"off": -11, "ranks": 1}, {"off": -10, "ranks": 1}]
        }

        self.pieces64 = ["P", "p", "n", "b", "r", "q", "k"]  # _240422
        self.pieces100 = ["P", "p", "c", "z", "n", "b", "f", "r", "q", "k"]  # _240422

        self.squareNumbers64 = {}
        self.squareNumbers100 = {}

        self.squares64 = []
        self.squares100 = []

        self.generateSquares({
            "boardSize": 8,
            "squareNumbers": self.squareNumbers64,
            "squares": self.squares64
        })

        self.generateSquares({
            "boardSize": 10,
            "squareNumbers": self.squareNumbers100,
            "squares": self.squares100
        })

        for piece in self.pieces64:
            self.generateAbilities({"boardSize": 8, "piece": piece})

        for piece in self.pieces100:
            # logger.log({function: "initializing", description: "generating abilities for 100", data: {piece: piece}})
            self.generateAbilities({"boardSize": 10, "piece": piece})
    
# -------------------- initialize abilities ---------------------
            
    def get_abilities(self, o):  # boardSize, piece, square

        boardSize = o.get("boardSize", 8)

        # logger.log({function: "chess100 abilities ", description: "before returning abilities", data: {boardSize: boardSize}})

        if boardSize == 8:

            return self.abilities64[o["piece"]][o["square"]]

        else:

            return self.abilities100[o["piece"]][o["square"]]
            
    def arrayToFen(self, o):  # array

        # turns a position array (see this.fenToArray) into a fen.

        boardSize = 8 if len(o["array"]) <= 72 else 10
        squaresCnt = boardSize * boardSize
        emptyFieldCounter = 0
        fen = ""

        for i in range(squaresCnt):

            if o["array"][i] == "":

                emptyFieldCounter = emptyFieldCounter + 1

            else:

                if emptyFieldCounter > 0:

                    fen = fen + str(emptyFieldCounter)
                    emptyFieldCounter = 0

                fen = fen + o["array"][i]

            if (i + 1) % boardSize == 0:

                if emptyFieldCounter > 0:

                    fen = fen + str(emptyFieldCounter)

                if i != squaresCnt - 1:

                    emptyFieldCounter = 0
                    fen = fen + "/"

        # handling the meta part if it exists:

        if len(o["array"]) > squaresCnt:

            meta = ""
            # _190623
            for i in range(squaresCnt, len(o["array"])):

                if i < squaresCnt + 1 or i > squaresCnt + 4:
                    # console.log("Adding:" + i + " " + o.array[i])
                    meta = meta + " " + o["array"][i]

                if i == squaresCnt + 1:

                    meta = meta + " "

                if (i > squaresCnt and i < squaresCnt + 5) and o["array"][i] != "-":  # condition for handling the castling data

                    meta = meta + o["array"][i]

                if i == squaresCnt + 4 and len(meta) == 3:

                    # no castling permissions left
                    meta = meta + "-"

            # _190623
            fen = fen + meta

        return fen
        
    def fen(self):

        emptyFieldCounter = 0
        fen = ""

        for i in range(self.squaresCnt):

            if self.position[self.squares[i]]["piece"] == "":

                emptyFieldCounter = emptyFieldCounter + 1

            else:

                if emptyFieldCounter > 0:

                    fen = fen + str(emptyFieldCounter)
                    emptyFieldCounter = 0

                fen = fen + self.position[self.squares[i]]["piece"]

            if (i + 1) % self.boardSize == 0:

                if emptyFieldCounter > 0:

                    fen = fen + str(emptyFieldCounter)

                if i != self.squaresCnt - 1:

                    emptyFieldCounter = 0
                    fen = fen + "/"

        # meta part

        fen = fen + " " + self.active

        fen = fen + " "

        if self.castlings == "":

            fen = fen + "-"

        else:

            if self.castlings.find(self.cSquares["w00"][2] + " ") != -1:

                fen = fen + "K"

            if self.castlings.find(self.cSquares["w000"][2] + " ") != -1:

                fen = fen + "Q"

            if self.castlings.find(self.cSquares["b00"][2] + " ") != -1:

                fen = fen + "k"

            if self.castlings.find(self.cSquares["b000"][2] + " ") != -1:

                fen = fen + "q"

        if self.epField1 == "":

            fen = fen + " -"

        else:

            fen = fen + " " + self.epField1 + self.epField2

        fen = fen + " " + str(self.halfmoveClock)

        fen = fen + " " + str(self.fullmoveNr)

        return fen

    def fenToArray(self, o):  # addMeta, fen
        # _190630 logit.log({function: "Chess.fenToArray", description: "start", data: {o: o}});

        # turns a fen into an array with the pattern [...]
        # if addMeta is true, meta info is appended

        squareNumbers = {
            "a8": 0, "b8": 1, "c8": 2, "d8": 3, "e8": 4, "f8": 5, "g8": 6, "h8": 7,
            "a7": 8, "b7": 9, "c7": 10, "d7": 11, "e7": 12, "f7": 13, "g7": 14, "h7": 15,
            "a6": 16, "b6": 17, "c6": 18, "d6": 19, "e6": 20, "f6": 21, "g6": 22, "h6": 23,
            "a5": 24, "b5": 25, "c5": 26, "d5": 27, "e5": 28, "f5": 29, "g5": 30, "h5": 31,
            "a4": 32, "b4": 33, "c4": 34, "d4": 35, "e4": 36, "f4": 37, "g4": 38, "h4": 39,
            "a3": 40, "b3": 41, "c3": 42, "d3": 43, "e3": 44, "f3": 45, "g3": 46, "h3": 47,
            "a2": 48, "b2": 49, "c2": 50, "d2": 51, "e2": 52, "f2": 53, "g2": 54, "h2": 55,
            "a1": 56, "b1": 57, "c1": 58, "d1": 59, "e1": 60, "f1": 61, "g1": 62, "h1": 63
        }

        emptyFieldsCount = 0
        fenArray = o["fen"].split(" ")
        position = []

        # _190630 logit.log({function: "Chess.fenToArray", description: "with moves", data: {o: o}});
                
        i = 0
        while i < len(fenArray[0]): # while instead of for - see below

            if fenArray[0][i] == " ":
                break

            elif fenArray[0][i].isdigit():

                emptyFieldsCount = int(fenArray[0][i])

                # exception handling for "10" on 10x10 board
                if i < len(fenArray[0]) - 1 and fenArray[0][i + 1] == "0":
                    emptyFieldsCount = 10
                    i += 1  # note: incrementing i inside a for loop wouldn't work in a for loop in python, that's why a while loop is used here

                # add empty fields
                for _ in range(emptyFieldsCount):
                    position.append("")

            elif fenArray[0][i] != "/":

                position.append(fenArray[0][i])

            i += 1

        # _191001 logit.log({function: "Chess.fenToArray", description: "before adding meta", data: {...}});

        # adding meta
        if o.get("addMeta") == True:

            # index where castling section starts
            firstIndexCastlingSection = len(position) + 1

            for i in range(1, len(fenArray)):

                if i == 2:
                    # _190630 logit.log({function: "Chess.fenToArray", description: "before specifying castling rights", data: {position: position}});

                    position.extend(["-", "-", "-", "-"])

                    if fenArray[i] != "-":

                        # check castling rights
                        castlings = ["K", "Q", "k", "q"]
                        classicCastling = False

                        for j in range(4):

                            if castlings[j] in fenArray[i]:

                                classicCastling = True
                                position[firstIndexCastlingSection + j] = castlings[j]

                        if classicCastling == False:

                            castlings960 = list(fenArray[i])
                            # _190630 logit.log({function: ..., data: {castlings960: castlings960}});

                            for j in range(len(castlings960)):

                                if castlings960[j].isupper():

                                    if position.index("K") > squareNumbers[castlings960[j].lower() + "1"]:

                                        # O-O-O
                                        position[firstIndexCastlingSection] = castlings960[j]

                                    else:

                                        # O-O
                                        position[firstIndexCastlingSection + 1] = castlings960[j]

                                else:

                                    if position.index("k") > squareNumbers[castlings960[j].lower() + "8"]:

                                        position[firstIndexCastlingSection + 2] = castlings960[j]

                                    else:

                                        position[firstIndexCastlingSection + 3] = castlings960[j]

                else:
                    # _190630 logit.log({function: "Chess.fenToArray", description: "pushing non-castling section"});
                    position.append(fenArray[i])

        # _190630 logit.log({function: "Chess.fenToArray", description: "before returning position", data: {position: position}});

        return position

    def generateAbilities(self, o):  # boardSize, piece
        # logger.log({function: "chess100 generate abilities", description: "start", data: {options: o, piece: o.piece}});
        currentAbilities = []

        abilities = self.abilities64
        offsets = self.offsets64
        squares = self.squares64

        if o["boardSize"] == 10:

            abilities = self.abilities100
            offsets = self.offsets100
            squares = self.squares100

        abilities[o["piece"]] = {}
        # logger.log({function: "chess100 generate abilities", description: "after init", data: {abilities: abilities[o.piece]}});

        for i in range(o["boardSize"] * o["boardSize"]):

            # loop through all squares to determine the move abilities of the piece for each square

            abilities[o["piece"]][squares[i]] = []

            currentAbilities = abilities[o["piece"]][squares[i]]  # abilities for the piece on the current square to inspect

            for j in range(len(offsets[o["piece"]])):
                # logger.log({function: "chess100 knight moves", description: "", data: {ranks: Math.abs(Math.floor(i/10)-Math.floor((i+offsets.n[j].off)/10))}});

                if o["piece"].lower() == "p":
                    # logger.log({function: "chess100 generate abilities", description: "for pawn", data: {ranks: Math.abs(Math.floor(i/10)-Math.floor((i+offsets.n[j].off)/10))}});

                    target_index = i + offsets[o["piece"]][j]["off"]

                    if (0 <= target_index < len(squares) and
                        offsets[o["piece"]][j]["ranks"] == abs((i // o["boardSize"]) - (target_index // o["boardSize"]))):

                        if o["boardSize"] == 8:

                            if not (((i < 8 or (i > 15 and j == 2)) and o["piece"] == "p") or
                                    ((i > 55 or (i < 48 and j == 2)) and o["piece"] == "P")):

                                # if the offsets are not invalid because the pawn is located on the initial square or rank 3-5
                                currentAbilities.append(squares[target_index])

                        else:

                            if not (((i < 10 or (i > 19 and j == 3) or (i > 49 and j == 2)) and o["piece"] == "p") or
                                    ((i > 89 or (i < 80 and j == 3) or (i < 50 and j == 2)) and o["piece"] == "P")):

                                # if the offsets are not invalid because the pawn is located on the initial square or rank 3-5
                                currentAbilities.append(squares[target_index])

                elif o["piece"] in ["c", "z", "n", "f", "k"]:  # _240422

                    # generate abilities for aviators and the king

                    target_index = i + offsets[o["piece"]][j]["off"]

                    if (0 <= target_index < len(squares) and
                        offsets[o["piece"]][j]["ranks"] == abs((i // o["boardSize"]) - (target_index // o["boardSize"]))):

                        currentAbilities.append(squares[target_index])

                else:

                    # generate abilities for pedestrians

                    currentOffset = offsets[o["piece"]][j]["off"]
                    # logger.log({function: "generateAbilities", description: "before detecting offsets for pedestrians", data: {currentOffset: currentOffset, square: squares[i]}});

                    for k in range(o["boardSize"]):

                        target_index = i + currentOffset

                        if (0 <= target_index < len(squares) and
                            ((offsets[o["piece"]][j].get("ranks", k + 1)) ==
                             abs((i // o["boardSize"]) - (target_index // o["boardSize"])) )):

                            currentAbilities.append(squares[target_index])
                            currentOffset += offsets[o["piece"]][j]["off"]

                            # logger.log({function: "generateAbilities", description: "after adjusting currentOffset", data: {currentOffset: currentOffset, square: squares[i]}});

                        else:

                            if k > 0:
                                currentAbilities.append("")  # add an empty space as marker for a direction change

                            break

            if len(currentAbilities) > 0 and currentAbilities[-1] == "":

                # remove the last empty space element
                currentAbilities.pop()
        # logger.log({function: "chess100 generate abilities", description: "after generating abilities", data: {abilities: abilities[o.piece], piece: o.piece}});        
    def generateSquares(self, o):  # boardSize, columns, squareNumbers, squares //_191108

        columns = list("abcdefgh")

        if o["boardSize"] == 10:
            
            columns = list("abcdefghij")

        squareCnt = 0

        for i in range(o["boardSize"]):

            for j in range(o["boardSize"]):

                key = columns[j] + str(o["boardSize"] - i)
                o["squareNumbers"][key] = squareCnt
                o["squares"].append(key)

                squareCnt += 1
        
    def inCheck(self, o=None):  # king, square
        # logger.log({"function": "chess100 inCheck", "description": "start", "data": {"options": o}})
        
        if o is None:
            o = {
                "king": "K" if self.active == "w" else "k",
                "square": self.kingSquares["K"] if self.active == "w" else self.kingSquares["k"]
            }

        controlledBy_ = self.position[o["square"]]["controlledBy"].split(" ")
        controlledBy_.pop()

        for i in range(len(controlledBy_)):

            # the following term uses a pattern with a leading " " and a "> 0" comparison
            # because "pnbfrqk".indexOf("") = 0 and "pnbfrqk".indexOf("p") = 0.

            piece = self.position[controlledBy_[i]]["piece"]
            if ((" pcznbfrqk".find(piece) > 0 and o["king"] == "K") or
                (" PCZNBFRQK".find(piece) > 0 and o["king"] == "k")  # _240422
               ):
                # logger.log({"function": "chess100 inCheck", "description": "before returning true", "data": {"controlledBy_": self.position[controlledBy_[i]], "options": o}})
                return True

        return False
    
    def _isAbsolutelyPinned(self, o):  # kSquare, square
        # logger.log({"function": "chess100 _isAbsolutelyPinned", "description": "start", "data": {"options": o}})
        # checks if a certain piece is pinned and the shielded piece is its king

        controlledBy_ = self.position[o["square"]]["controlledBy"].split(" ")
        controlledBy_.pop()
        # logger.log({"function": "chess100 _isAbsolutelyPinned", "description": "after detecting controlledBy", "data": {"controlledBy": controlledBy_}})

        pAbilities = []  # abilities of the potentially pinning piece
        isPDirection = False  # check the up to four directions, in which bishop, rook and queen can fire

        for k in range(len(controlledBy_)):

            piece = self.position[controlledBy_[k]]["piece"]
            if " BRQbrq".find(piece) > 0:
                # only bishop, rook and queen can pin

                if ((" BRQ".find(piece) > 0 and self.position[o["kSquare"]]["piece"] == "k") or
                    (" brq".find(piece) > 0 and self.position[o["kSquare"]]["piece"] == "K")):
                    # enemy bishop, rook or queen found that controls the square the examined piece is located on.
                    # the piece is pinned if its king is located on a square behind, in the same direction the enemy piece fires.

                    pAbilities = self.abilities[piece.lower()][controlledBy_[k]]
                    # logger.log({"function": "chess100 _isAbsolutelyPinned", "description": "potential pinner",
                    #             "data": {"fen": self.fen(), "pAbilities": pAbilities,
                    #                      "squareParams": self.position[controlledBy_[k]], "square": controlledBy_[k]}})

                    for l in range(len(pAbilities)):
                        # looping forward through the abilities of the enemy piece and not the squares it controls,
                        # since the enemy piece only controls the fields between itself and the examined piece.
                        # the friendly king would be located on a square behind the square of the examined piece, if applicable.

                        if pAbilities[l] == "":
                            isPDirection = False
                        elif pAbilities[l] == o["square"]:
                            isPDirection = True
                        elif (self.position[pAbilities[l]]["piece"].lower() != "k" and isPDirection):
                            # a piece is located between the piece to be evaluated and its king - no pin
                            break
                        elif pAbilities[l] == o["kSquare"] and isPDirection:
                            # the friendly piece and its king are located on the same direction - friendly piece is pinned
                            # logger.log({"function": "chess100 _isAbsolutelyPinned", "description": "before returning true", "data": {"fen": self.fen()}})
                            return True

        # logger.log({"function": "chess100 _isAbsolutelyPinned", "description": "before returning false", "data": {"fen": self.fen(), "options": o}})
        return False
        
    def _updateSquares(self, o):  # causedByBlockedUpdate, previousPiece, sqFrom, sqTo //_201013
        # _201012 logger.log({"function": "chess100 update squares ", "description": "start",
        #                    "data": {"callstack": print(Exception().stack), "piece": self.position[o["sqTo"]]["piece"], "o": o}})
        # updates the controlledBy and controls property of the squares (the elements of position[])
        # this is the only place where these properties are manipulated.
        # requires that the piece properties for the from and to square were updated before
        # a piece controls all squares where it could potentially beat another piece

        piece = self.position[o["sqTo"]]["piece"].lower()

        if o.get("causedByBlockedUpdate") and " BRQbrq".find(self.position[o["sqTo"]]["piece"]) < 1:
            return

        if o.get("sqFrom"):  # can be empty if the piece didn't move but was put on the board during a position setup
            # controls from previous square must be removed
            controls_ = self.position[o["sqFrom"]]["controls"].split(" ")
            controls_.pop()  # remove last empty element from split
            # logger.log({"function": "chess100 update squares ", "description": "after controls_ init", "data": {"controls": self.position[o["sqFrom"]]["controls"], "controls_": controls_}})

            for i in range(len(controls_)):
                self.position[controls_[i]]["controlledBy"] = self.position[controls_[i]]["controlledBy"].replace(o["sqFrom"] + " ", "")

            self.position[o["sqFrom"]]["controls"] = ""
            # self.position[o["sqFrom"]]["piece"] = ""

        self.position[o["sqTo"]]["controls"] = ""

        # _201012
        if o.get("previousPiece"):
            # if a piece was beaten, update the fields it controlled
            previousPiece = "P" if o["previousPiece"] == "P" else o["previousPiece"].lower()
            # logger.log({"function": "chess100 update squares ", "description": "before checking abilities for beaten piece",
            #             "data": {"abilities": self.abilities[previousPiece][o["sqTo"]], "previousPiece": previousPiece}})
            for i in range(len(self.abilities[previousPiece][o["sqTo"]])):
                if self.abilities[previousPiece][o["sqTo"]][i] != "":
                    self.position[self.abilities[previousPiece][o["sqTo"]][i]]["controlledBy"] = \
                        self.position[self.abilities[previousPiece][o["sqTo"]][i]]["controlledBy"].replace(o["sqTo"] + " ", "")

        # _201012                                            
        if piece == "p":
            # check list special abilities:
            # moves only forward
            # beats differently
            # move abilities depend on the rank
            # no first and eighth rank
            # promotion
            # en passant

            pAbilities = self.abilities[self.position[o["sqTo"]]["piece"]][o["sqTo"]]
            # logger.log({"function": "chess100 update squares ", "description": "set controls for pawn field",
            #             "data": {"abilities": self.abilities["P"], "piece": piece, "sqTo": o["sqTo"]}})
            
            if o["sqTo"][:1] != self.files[0]:  # left border pawns cannot beat to both sides
                self.position[pAbilities[0]]["controlledBy"] += o["sqTo"] + " "
                self.position[o["sqTo"]]["controls"] += pAbilities[0] + " "

            if o["sqTo"][:1] != self.files[-1]:  # right border pawns
                self.position[pAbilities[-1]]["controlledBy"] += o["sqTo"] + " "
                self.position[o["sqTo"]]["controls"] += pAbilities[-1] + " "

        else:
            piece = piece.lower()
            # _240422 if(piece == "n" || piece == "f" || piece == "k"){
            if piece in ["c", "z", "n", "f", "k"]:  # _240422
                for i in range(len(self.abilities[piece][o["sqTo"]])):
                    # add flags
                    self.position[self.abilities[piece][o["sqTo"]][i]]["controlledBy"] += o["sqTo"] + " "
                    # the controls property doesn't need to be set for these piece types, but for simplification:
                    self.position[o["sqTo"]]["controls"] += self.abilities[piece][o["sqTo"]][i] + " "
            else:
                # logger.log({"function": "chess100.load ", "description": "before checking pedestrian", "data": {"piece": piece}})
                # for the pedestrians directions can be blocked by other pieces, so this has to be checked
                blocked = False

                for i in range(len(self.abilities[piece][o["sqTo"]])):
                    if self.abilities[piece][o["sqTo"]][i] != "":
                        if o.get("causedByBlockedUpdate"):
                            # logger.log({"function": "chess100.load ", "description": "checking pedestrian - delete old informations",
                            #             "data": {"controls": self.position[o["sqTo"]]["controls"], "abilities": self.abilities[piece][o["sqTo"]][i]}})
                            self.position[self.abilities[piece][o["sqTo"]][i]]["controlledBy"] = \
                                self.position[self.abilities[piece][o["sqTo"]][i]]["controlledBy"].replace(o["sqTo"] + " ", "")
                            self.position[o["sqTo"]]["controls"] = self.position[o["sqTo"]]["controls"].replace(
                                self.abilities[piece][o["sqTo"]][i] + " ", "")

                        if not blocked:
                            self.position[self.abilities[piece][o["sqTo"]][i]]["controlledBy"] += o["sqTo"] + " "
                            self.position[o["sqTo"]]["controls"] += self.abilities[piece][o["sqTo"]][i] + " "

                            if self.position[self.abilities[piece][o["sqTo"]][i]]["piece"] != "":
                                blocked = True
                    else:
                        blocked = False

        if o.get("sqFrom"):
            # logger.log({"function": "chess100 _updateSquares ", "description": "updating blocked", "data": {}})
            # update blocked flags for pedestrians that control the from or the to square
            sqToCB_ = self.position[o["sqTo"]]["controlledBy"].split(" ")
            sqToCB_.pop()
            if len(sqToCB_) > 0:
                # logger.log({"function": "chess100 _updateSquares ", "description": "updating blocked", "data": {"sqToCB_": sqToCB_}})
                for i in range(len(sqToCB_)):
                    if sqToCB_[i] != o["sqFrom"]:
                        self._updateSquares({"sqTo": sqToCB_[i], "causedByBlockedUpdate": True})

            sqFromCB_ = self.position[o["sqFrom"]]["controlledBy"].split(" ")
            sqFromCB_.pop()
            # logger.log({"function": "chess100 _updateSquares ", "description": "updating blocked from sqFrom", "data": {"sqFromCB_": sqFromCB_}})
            if len(sqFromCB_) > 0:
                for i in range(len(sqFromCB_)):
                    self._updateSquares({"sqTo": sqFromCB_[i], "causedByBlockedUpdate": True})
                    
    def inCheckmate(self):
        # logger.log({"function": "chess100 inCheckmate", "description": "start", "data": {}})
        king = "K" if self.active == "w" else "k"
        square = self.kingSquares["K"] if self.active == "w" else self.kingSquares["k"]
        # _210428
        # detect how many pieces give check and note them

        controlledBy_ = self.position[square]["controlledBy"].split(" ")
        controlledBy_.pop()  # remove last empty element
        # logger.log({"function": "chess100 inCheckmate", "description": "detect check giving pieces", "data": {}})

        enemy = ""
        enemy2 = ""  # in case of double check

        for i in range(len(controlledBy_)):
            # _240422
            if ((" PCZNBFRQK".find(self.position[controlledBy_[i]]["piece"]) > 0 and self.active == "b") or
                (" pcznbfrqk".find(self.position[controlledBy_[i]]["piece"]) > 0 and self.active == "w")):
                # if the controller is an enemy piece
                if enemy != "":
                    enemy2 = controlledBy_[i]  # second check giving enemy piece
                else:
                    enemy = controlledBy_[i]

        if enemy == "":
            # no check, no mate
            return False

        # logger.log({"function": "chess100 inCheckmate", "description": "after detecting enemies", "data": {"enemy": enemy, "enemy2": enemy2}})
        # all neighbouring fields (nf) are in check or occupied by a friendly piece?

        for i in range(len(self.abilities[king.lower()][square])):
            ability = self.abilities[king.lower()][square][i]
            # logger.log({"function": "chess100 inCheckmate", "description": "looping through neighbouring squares",
            #            "data": {"indexOfB": " pnbfrqk".find(self.position[ability]["piece"]),
            #                     "indexOfW": " PNBFRQK".find(self.position[ability]["piece"]),
            #                     "square": ability, "squareParams": self.position[ability]}})

            if not self.inCheck({"king": king, "square": ability}):
                # if the nf is not in check
                if ((" PCZNBFRQK".find(self.position[ability]["piece"]) < 1 and self.active == "w") or
                    (" pcznbfrqk".find(self.position[ability]["piece"]) < 1 and self.active == "b")):
                    # if the nf is not occupied by a friendly piece
                    if " BRQbrq".find(self.position[enemy]["piece"]) > 0:
                        # the following special case applies only to the pedestrians
                        abilitiesEnemy = self.abilities[self.position[enemy]["piece"].lower()][enemy]
                        for j in range(len(abilitiesEnemy)):
                            if abilitiesEnemy[j] == ability:
                                # _220207
                                # if the nf is not in check, but part of the abilities of the check giving piece, there is a blocker
                                for k in range(j - 1, -1, -1):
                                    if self.position[abilitiesEnemy[k]]["piece"] == king:
                                        # if the king is the blocker it is a special case: the field behind the king
                                        break
                                    elif self.position[abilitiesEnemy[k]]["piece"] != king:
                                        # blocker is another piece, not mate
                                        return False
                                break

                        if j == len(abilitiesEnemy) - 1:
                            if enemy2 != "":
                                # double check case
                                abilitiesEnemy2 = self.abilities[self.position[enemy2]["piece"].lower()][enemy2]
                                for j in range(len(abilitiesEnemy2)):
                                    if abilitiesEnemy2[j] == ability:
                                        break
                                if j == len(abilitiesEnemy2):
                                    return False
                            else:
                                return False
                    else:
                        return False

        # check for double check
        if enemy2 != "":
            return True

        # evaluate if the piece that gives check can be beaten
        controlledByBy_ = self.position[enemy]["controlledBy"].split(" ")
        controlledByBy_.pop()
        for j in range(len(controlledByBy_)):
            if ((" PCZNBFRQ".find(self.position[controlledByBy_[j]]["piece"]) > 0 and self.active == "w") or
                (" pcznbfrq".find(self.position[controlledByBy_[j]]["piece"]) > 0 and self.active == "b")):
                if not self._isAbsolutelyPinned({"kSquare": square, "square": controlledByBy_[j]}):
                    return False
        # logger.log({"function": "chess100 inCheckmate", "description": "check for blockers", "data": {}})  # _210506

        # evaluate if the piece that gives check can be blocked
        if " BRQbrq".find(self.position[enemy]["piece"]) > 0:
            controls_ = self.position[enemy]["controls"].split(" ")
            controls_.pop()
            # logger.log({"function": "chess100 inCheckmate", "description": "check for BRQbrq blocks", "data": {"controls_": controls_}})
            aDirection = []
            abilitiesEnemy = self.abilities[self.position[enemy]["piece"].lower()][enemy]

            for i in range(len(abilitiesEnemy)):
                if abilitiesEnemy[i] == "":
                    aDirection = []
                elif abilitiesEnemy[i] == square:
                    break
                else:
                    aDirection.append(abilitiesEnemy[i])

            # evaluate the potentially blocking pieces
            for i in range(len(aDirection)):
                controlledBy_ = self.position[aDirection[i]]["controlledBy"].split(" ")
                controlledBy_.pop()
                for j in range(len(controlledBy_)):
                    if ((" CZNBFRQ".find(self.position[controlledBy_[j]]["piece"]) > 0 and self.active == "w") or
                        (" cznbfrq".find(self.position[controlledBy_[j]]["piece"]) > 0 and self.active == "b")):
                        if not self._isAbsolutelyPinned({"kSquare": square, "square": controlledBy_[j]}):
                            return False

                # check for pawns that could move forward to block the attacking piece
                direction = 1 if self.active == "w" else -1
                pawn = "P" if self.active == "w" else "p"

                for j in range(1, 4):
                    index = self.squareNumbers[aDirection[i]] + (self.boardSize * direction * j)
                    if 0 <= index < self.squaresCnt:
                        squareToTest = self.squares[index]
                        if self.position[squareToTest]["piece"] != pawn:
                            break
                        else:
                            for k in range(1, len(self.abilities[pawn][squareToTest]) - 1):
                                if self.abilities[pawn][squareToTest][k] == aDirection[i]:
                                    if not self._isAbsolutelyPinned({"kSquare": square, "square": squareToTest}):
                                        return False
                            break

        return True
        
    def insufficientMaterial(self):  # _210922
        # logger.log({"function": "chess100.insufficientMaterial", "description": "start", "data": {"capturedCnt": self.capturedCnt}})  # _210922
        # check for insufficient material

        if self.capturedCnt >= (self.boardSize * 4 - 3):

            if self.capturedCnt == (self.boardSize * 4 - 2):
                # bare kings left
                return True
            else:
                if self.lastRemainingPiece == "":
                    for i in range(self.squaresCnt):
                        if (self.position[self.squares[i]]["piece"] != "" and
                            self.position[self.squares[i]]["piece"].lower() != "k"):
                            self.lastRemainingPiece = self.position[self.squares[i]]["piece"]
                            break

                if self.lastRemainingPiece.lower() in ["b", "n"]:
                    return True

        return False
        
    def fiftyMoves(self):  # _210922
        # check for 50 move rule violation
        # logger.log({"function": "chess100.fiftyMoves", "description": "before checking halfmove clock", "data": {"halfmoveClock": self.halfmoveClock}})  # _210922
        # _250927 return (halfmoveClock >= 50)
        return self.halfmoveClock >= 100  # _250927
        
    def inStalemate(self):
        # logger.log({"function": "chess100 inStalemate", "description": "start", "data": {"fen": self.fen()}})
        controls = []
        kSquare = self.kingSquares["K"] if self.active == "w" else self.kingSquares["k"]
        pAbilities = []
        piece = ""

        for i in range(self.squaresCnt):

            piece = self.position[self.squares[i]]["piece"]

            if (("PCZNBFRQK".find(piece) >= 0 and self.active == "w") or
                ("pcznbfrqk".find(piece) >= 0 and self.active == "b")):  # _240422
                # logger.log({"function": "chess100 inStalemate", "description": "piece found", "data": {}})
                
                controls_ = self.position[self.squares[i]]["controls"].split(" ")
                controls_.pop()  # remove last empty element if exists

                for j in range(len(controls_)):
                    # logger.log({"function": "chess100 inStalemate", "description": "checking controlled squares", "data": {}})
                    target_piece = self.position[controls_[j]]["piece"]
                    if (("PCZNBFRQK".find(piece) != "PCZNBFRQK".find(target_piece)) or
                        ("pcznbfrqk".find(piece) != "pcznbfrqk".find(target_piece))):
                        # logger.log({"function": "chess100 inStalemate", "description": "no friendly piece on controlled square", "data": {}})
                        # if the controlled square isn't blocked by a friendly piece

                        if piece.lower() == "k":
                            if not self.inCheck({"king": piece, "square": controls_[j]}):
                                # king can move - no stalemate
                                return False
                        elif not self._isAbsolutelyPinned({"kSquare": kSquare, "square": self.squares[i]}):
                            return False

                # logger.log({"function": "chess100 inStalemate", "description": "vertical pawn movement?", "data": {"square": self.squares[i]}})
                if piece.lower() == "p":
                    # check possible vertical movements of a certain pawn
                    pAbilities = self.abilities[piece][self.squares[i]]

                    for k in range(1, len(pAbilities) - 1):
                        if self.position[pAbilities[k]]["piece"] == "":
                            # possible square is not blocked by a piece
                            if not self._isAbsolutelyPinned({"kSquare": kSquare, "square": self.squares[i]}):
                                # pawn can move
                                return False

        return True
        
    def get_legal_moves_for_square(self, o):  # o is a dict like {"square": "e4"}
    
        # returns all legal moves for a given fen and target square
        # the KISS solution that works without any preselection
        
        fen = self.fen()
        legals = []

        for i in range(len(self.squares)):
            self.load({"fen": fen})  # must be reloaded, since moving changes the state of the class instance
            retVal = self.move({"from": self.squares[i], "to": o["square"]})

            if retVal.get("from") is not None:
                legals.append(retVal)
                legals[-1]["fen"] = self.fen()

        return legals
    # _260412
    def get_legal_moves(self):
    
        # logger.log({function: "rules.get_legal_moves", description: "start", data: {fen: self.fen}})
        # returns all legal moves for the current fen

        fen = self.fen()
        legals = []

        def testMove(e):  # from, to, promotion
                # self.logger.log({"function": "rules.get_legal_moves", "description": "test move", "data": {"from": e.get("from"), "to": e.get("to"), "promotion": e.get("promotion")}}) #_260415
            retVal = self.move({
                "from": e.get("from"),
                "to": e.get("to"),
                "promotion": e.get("promotion")
            })

            if retVal.get("from") is not None:
                    # self.logger.log({"function": "rules.get_legal_moves", "description": "valid move found", "data": {"e": e}}) #_260415
                legals.append(retVal)
                legals[-1]["fen"] = self.fen()

            # must be reloaded, since moving changes the state of the class instance
            self.load({"fen": fen})

        for i in range(len(self.squares)):
                # self.logger.log({"function": "rules.get_legal_moves", "description": "looping squares", "data": {"active": self.active, "square": self.position[self.squares[i]]}}) #_260412
            sq = self.squares[i]
            piece = self.position[sq]["piece"]

            if ((" PCZNBFRQK".find(piece) > 0 and self.active == "w") or
                (" pcznbfrqk".find(piece) > 0 and self.active == "b")):
                    # self.logger.log({"function": "rules.get_legal_moves", "description": "testing move", "data": {"piece": piece, "square": self.position[self.squares[i]]}}) #_260415
                # if a piece of the color to move is placed on the current square
                # test all possible moves for this piece

                controls_ = self.position[sq]["controls"].split(" ")
                controls_.pop()  # the split generates [" "] as last array element, which must be removed.
                    # logger.log({function: "rules.get_legal_moves", description: "piece to test detected", data: {controls_: controls_, square: squares[i], squares: squares}})
                for j in range(len(controls_)):
                    testMove({"from": sq, "to": controls_[j]})

                if " pP".find(piece) > 0:

                    # vertical pawn moves must be checked separately, since they can't beat stuff and therefore aren't represented by the controls_ array.

                    file = sq[0]
                    rank = int(sq[1:2])

                    if len(sq) == 3:
                        rank = int(sq[1:3])  # 10th rank

                    for k in range(1, 4):

                        targetRank = rank + k if piece == "P" else rank - k

                        if targetRank == 1 or targetRank == 10:

                            promotions_ = (
                                ["C", "Z", "N", "B", "F", "R", "Q"]
                                if self.active == "w"
                                else ["c", "z", "n", "b", "f", "r", "q"]
                            )

                            for l in range(len(promotions_)):
                                testMove({
                                    "from": sq,
                                    "to": file + str(targetRank),
                                    "promotion": promotions_[l]
                                })

                        else:

                            testMove({
                                "from": sq,
                                "to": file + str(targetRank)
                            })

                if ((piece == "K" and ((sq == "e1" and self.boardSize == 8) or (sq == "f1" and self.boardSize == 10))) or
                    (piece == "k" and ((sq == "e8" and self.boardSize == 8) or (sq == "f10" and self.boardSize == 10)))):

                    # check castlings

                    file = sq[0]
                    rank = sq[1:2]

                    if len(sq) == 3:
                        rank = sq[1:2]  # 10th rank (kept identical to JS behavior)

                    for m in range(2, 4):

                        targetFile = self.files[(4 if self.boardSize == 8 else 5) + m]
                        testMove({"from": sq, "to": targetFile + rank})

                        targetFile = self.files[(4 if self.boardSize == 8 else 5) - m]
                        testMove({"from": sq, "to": targetFile + rank})

        return legals
    # _260412        
    def load(self, o):  # fen
        # _200920 logger.log({"function": "chess100.load", "description": "start", "data": {"o": o}})  # _200603
        # resets the instance to its initial state and loads a position

        # mode dependent initializations and validity check (to be extended and tested)
        fen_ = o["fen"].split("/")
        # logger.log({"function": "chess100.load", "description": "before initializing fixed vars", "data": {"fen_Length": len(fen_)}})  # _191106
        self.boardSize = len(fen_)
        self.squaresCnt = self.boardSize * self.boardSize

        self.fenInvalid = ""
        msgInvalid = "Invalid fen - "

        if self.boardSize not in [8, 10]:
            self.fenInvalid = f"board has {self.boardSize} ranks."
            return msgInvalid + self.fenInvalid

        if "k" not in o["fen"] or "K" not in o["fen"]:
            # console.log("Invalid fen - pawns on the first and/or last rank.")
            self.fenInvalid = "position needs two kings."
            return msgInvalid + self.fenInvalid

        if ("p" in fen_[0] or "P" in fen_[0] or
            "p" in fen_[self.boardSize - 1] or "P" in fen_[self.boardSize - 1]):
            # console.log("Invalid fen - pawns on the first and/or last rank.")
            self.fenInvalid = "pawns on the first and/or last rank."
            return msgInvalid + self.fenInvalid

        # board size specific arrays
        if self.boardSize == 8:
            self.abilities = self.abilities64
            self.files = self.files64
            self.squareNumbers = self.squareNumbers64
            self.squares = self.squares64
            self.cSquares = self.cSquares64
        else:
            self.abilities = self.abilities100
            self.files = self.files100
            self.squareNumbers = self.squareNumbers100
            self.squares = self.squares100
            self.cSquares = self.cSquares100  # can still change due to castling validation test  # _240809

        # game dependent initializations
        self.active = ""
        self.capturedCnt = self.boardSize * 4
        self.castlings = ""
        self.epField1 = ""
        self.epField2 = ""
        self.fullmoveNr = 0
        self.halfmoveClock = 0
        self.lastRemainingPiece = ""
        self.kingSquares = {"k": "", "K": ""}
        # logger.log({"function": "chess100.load", "description": "before clearing position", "data": {"position": self.position}})
        self.position = {}
        # logger.log({"function": "chess100.load", "description": "after initializing squares and squareNumbers", "data": {"squaresCnt": self.squaresCnt, "squares": self.squares, "squareNumbers": self.squareNumbers}})

        # load new position
        # _260329 var pos = chessToolsClient.fenToArray({addMeta: true, fen: o.fen})
        pos = self.fenToArray({"addMeta": True, "fen": o["fen"]})  # _260329
        # logger.log({"function": "chess100.load", "description": "after initializing pos", "data": {"pos": pos, "squaresCnt": self.squaresCnt}})
        
        for i in range(self.squaresCnt):
            self.position[self.squares[i]] = {"controlledBy": "", "controls": "", "piece": pos[i]}
            if pos[i].lower() == "k":
                self.kingSquares[pos[i]] = self.squares[i]
            if pos[i] != "":
                self.capturedCnt -= 1

        # logger.log({"function": "chess100.load", "description": "before updating squares", "data": {"position": self.position}})  # _191106
        for i in range(self.squaresCnt):
            if self.position[self.squares[i]]["piece"] != "":
                self._updateSquares({"sqTo": self.squares[i]})

        # handle meta part
        # init color to move
        self.active = pos[self.squaresCnt]

        def initCastlings():
            self.castlings = ""
            # init castling flags
            if pos[self.squaresCnt + 1] == "K":
                self.castlings += f"{self.cSquares['w00'][2]} "
            if pos[self.squaresCnt + 2] == "Q":
                self.castlings += f"{self.cSquares['w000'][2]} "
            if pos[self.squaresCnt + 3] == "k":
                self.castlings += f"{self.cSquares['b00'][2]} "
            if pos[self.squaresCnt + 4] == "q":
                self.castlings += f"{self.cSquares['b000'][2]} "

        initCastlings()  # _240809
        # logger.log({"function": "chess100.load", "description": "after setting castlings", "data": {"castlings": self.castlings}})
        
        self.epField1 = pos[self.squaresCnt + 5][:2]
        if len(pos[self.squaresCnt + 5]) == 4:
            self.epField2 = pos[self.squaresCnt + 5][2:4]

        # halfmoveClock
        self.halfmoveClock = int(pos[self.squaresCnt + 6])
        # fullmoveNr
        self.fullmoveNr = int(pos[self.squaresCnt + 7])
        # logger.log({"function": "chess100.load", "description": "after setting position", "data": {"epField1": self.epField1}})

        # make some validations after the position has been loaded
        def checkCastlings():  # _240809
            # check if the placement of kings & rooks corresponds with the castling flags set  # _240809
            if (self.kingSquares["K"] != self.cSquares["w00"][0] and 
                (self.castlings.find(f"{self.cSquares['w00'][2]} ") != -1 or self.castlings.find(f"{self.cSquares['w000'][2]} ") != -1)):
                self.fenInvalid = "white castling possibilities were enabled but the white king is not on its initial square."
                return msgInvalid + self.fenInvalid

            if (self.kingSquares["k"] != self.cSquares["b00"][0] and 
                (self.castlings.find(f"{self.cSquares['b00'][2]} ") != -1 or self.castlings.find(f"{self.cSquares['b000'][2]} ") != -1)):
                self.fenInvalid = "black castling possibilities were enabled but the black king is not on its initial square."
                return msgInvalid + self.fenInvalid

            if (self.castlings.find(f"{self.cSquares['w00'][2]} ") != -1 and
                self.position[self.cSquares['w00'][-1]]["piece"] != "R"):  # _240809
                self.fenInvalid = "short castling for white was enabled but the rook is not on its initial square."
                return msgInvalid + self.fenInvalid

            if (self.castlings.find(f"{self.cSquares['w000'][2]} ") != -1 and
                self.position[self.cSquares['w000'][-1]]["piece"] != "R"):  # _240809
                self.fenInvalid = "long castling for white was enabled but the rook is not on its initial square."
                return msgInvalid + self.fenInvalid

            if (self.castlings.find(f"{self.cSquares['b00'][2]} ") != -1 and
                self.position[self.cSquares['b00'][-1]]["piece"] != "r"):  # _240809
                self.fenInvalid = "short castling for black was enabled but the rook is not on its initial square."
                return msgInvalid + self.fenInvalid

            if (self.castlings.find(f"{self.cSquares['b000'][2]} ") != -1 and
                self.position[self.cSquares['b000'][-1]]["piece"] != "r"):  # _240809
                self.fenInvalid = "long castling for black was enabled but the rook is not on its initial square."
                return msgInvalid + self.fenInvalid

            return "ok"

        castlingsCheckMsg = checkCastlings()

        if castlingsCheckMsg != "ok" and self.boardSize == 10:
            # console.log("castlinglong")
            # if the "rooks on b & i" castling pattern didn't pass the test for 10x10, try "rooks on a & j"
            self.fenInvalid = ""
            self.cSquares = self.cSquares100Long
            # console.log("square: " + self.cSquares["b00"][-1])
            initCastlings()
            castlingsCheckMsg = checkCastlings()
            if castlingsCheckMsg != "ok":
                return castlingsCheckMsg
                
    def move(self, o):  # from, san, to, promotion
        # logger.log({"function": "chess100.move", "description": "start", "data": {"fen": self.fen(), "options": o, "position": self.position}})  # _200603 _201013

        if self.fenInvalid != "":
            # if the position is invalid, it's an invalid move anyway
            return {"err": f"Invalid position - {self.fenInvalid}"}

        if "san" in o:
            o = self.sanToFromTo({"san": o["san"]})

        if o["from"] not in self.position:
            # invalid from square
            return {"err": "Source square is invalid."}

        piece = self.position[o["from"]]["piece"]
        # _240925
        if piece == "":
            return {"err": "No piece on the source square."}
        # _240925

        retVal = {
            # optional:
            # ambiguous,
            # capture,
            # castling,
            # promotion #_260412
            "from": o["from"],
            "piece": piece,
            "to": o["to"]
        }

        if (piece in "PCZNBFRQK" and self.active == "b") or \
           (piece in "pcznbfrqk" and self.active == "w"):  # _240422
            # wrong color moved
            return {"err": "Wrong color moved."}

        if piece.lower() == "p" and \
           o["to"] + " " in self.position[o["from"]]["controls"] and \
           self.position[o["to"]]["piece"] == "" and \
           o["to"] not in [self.epField1, self.epField2]:
            return {"err": "The pawn controls the square but there is no enemy piece to beat on it."}

        previousValues = {
            "castlings": self.castlings,
            "epField1": self.epField1,
            "epField2": self.epField2,
            "kingSquares": {k: v for k, v in self.kingSquares.items()},
            "position": {k: v.copy() for k, v in self.position.items()}  # kiss solution; always saving the complete position
            # just to be ready for the rare case of restore after an illegal move seems to be
            # a bit of performance waste, but developing a restore solution using updateSquares
            # would be not trivial, especially when a beaten piece has to be placed again. //_201013
        }

        self.epField1 = ""
        self.epField2 = ""
        
        if o["to"] + " " not in self.position[o["from"]]["controls"]:
            # target square is not controlled by the piece to be moved

            if piece.lower() == "p":
                # check pawn squares where moving but not beating is allowed

                pAbilities = self.abilities[piece][o["from"]]
                checkCnt = 0

                firstValidIndex = 1 if o["from"][0] != self.files[0] else 0  # take account of border pawns. they can't beat to both sides
                firstInvalidIndex = len(pAbilities) - (1 if o["from"][0] != self.files[-1] else 0)  # see above
                # logger.log({"function": "chess100.move", "description": "check vertical pawn move", "data": {"firstValidIndex": firstValidIndex, "firstInvalidIndex": firstInvalidIndex, "pAbilities": pAbilities}})

                for i in range(firstValidIndex, firstInvalidIndex):

                    if self.position[pAbilities[i]]["piece"] != "":
                        return {"err": "Pawn move requested, but a needed square is blocked by a piece."}

                    checkCnt += 1

                    if pAbilities[i] == o["to"]:
                        # valid pawn square
                        if (self.squareNumbers[o["to"]] < self.boardSize or
                            self.squareNumbers[o["to"]] > (self.squaresCnt - self.boardSize - 1)) and o.get("promotion") is None:
                            return {"err": "A pawn moved to the last rank but wasn't promoted."}

                        direction = -1 if self.active == "w" else 1

                        if self.boardSize == 8:
                            if checkCnt == 2:
                                # pawn moved two squares forward - registering the ep field
                                self.epField1 = self.squares[self.squareNumbers[o["from"]] + 8 * direction]
                        else:
                            if checkCnt in (2, 3):
                                # pawn moved more than one square forward - registering ep field(s)
                                self.epField1 = self.squares[self.squareNumbers[o["from"]] + 10 * direction]
                                if checkCnt == 3:
                                    self.epField2 = self.squares[self.squareNumbers[o["from"]] + 20 * direction]

                        break
                else:
                    return {"err": "Pawn move requested, but the target square is invalid."}

            elif piece.lower() == "k":
                # logger.log({"function": "chess100.move", "description": "before checking castling move", "data": {"castlings": self.castlings, "to": o["to"]}})
                # check for a possible castling
                # the requirement "king and rook on their initial squares when castling is enabled" is checked by the .load function

                if o["to"] + " " not in self.castlings:
                    # logger.log({"function": "chess100.move", "description": "missing castling possibility", "data": {}})
                    return {"err": "Castling requested, but the target square is invalid."}
                elif ((o["to"] == self.cSquares["w00"][2] and (self.position[self.cSquares["w00"][1]]["piece"] != "" or
                                                             self.position[self.cSquares["w00"][2]]["piece"] != "" or
                                                             (self.position[self.cSquares["w00"][3]]["piece"] != "" and len(self.cSquares["w00"]) == 5))) or
                      (o["to"] == self.cSquares["b00"][2] and (self.position[self.cSquares["b00"][1]]["piece"] != "" or
                                                             self.position[self.cSquares["b00"][2]]["piece"] != "" or
                                                             (self.position[self.cSquares["b00"][3]]["piece"] != "" and len(self.cSquares["b00"]) == 5))) or
                      (o["to"] == self.cSquares["w000"][2] and (self.position[self.cSquares["w000"][1]]["piece"] != "" or
                                                              self.position[self.cSquares["w000"][2]]["piece"] != "" or
                                                              self.position[self.cSquares["w000"][3]]["piece"] != "" or
                                                              (self.position[self.cSquares["w000"][4]]["piece"] != "" and len(self.cSquares["w000"]) == 6))) or
                      (o["to"] == self.cSquares["b000"][2] and (self.position[self.cSquares["b000"][1]]["piece"] != "" or
                                                              self.position[self.cSquares["b000"][2]]["piece"] != "" or
                                                              self.position[self.cSquares["b000"][3]]["piece"] != "" or
                                                              (self.position[self.cSquares["b000"][4]]["piece"] != "" and len(self.cSquares["b000"]) == 6)))):
                    # logger.log({"function": "chess100.move", "description": "missing castling requirements", "data": {}})
                    return {"err": "Castling requested, but one of the needed squares is occupied by a piece."}
                elif ((o["to"] == self.cSquares["w00"][2] and (self.inCheck({"king": piece, "square": self.cSquares["w00"][2]}) or
                                                            self.inCheck({"king": piece, "square": self.cSquares["w00"][1]}) or
                                                            self.inCheck({"king": piece, "square": self.cSquares["w00"][0]}))) or
                      (o["to"] == self.cSquares["b00"][2] and (self.inCheck({"king": piece, "square": self.cSquares["b00"][2]}) or
                                                            self.inCheck({"king": piece, "square": self.cSquares["b00"][1]}) or
                                                            self.inCheck({"king": piece, "square": self.cSquares["b00"][0]}))) or
                      (o["to"] == self.cSquares["w000"][2] and (self.inCheck({"king": piece, "square": self.cSquares["w000"][2]}) or
                                                             self.inCheck({"king": piece, "square": self.cSquares["w000"][1]}) or
                                                             self.inCheck({"king": piece, "square": self.cSquares["w000"][0]}))) or
                      (o["to"] == self.cSquares["b000"][2] and (self.inCheck({"king": piece, "square": self.cSquares["b000"][2]}) or
                                                             self.inCheck({"king": piece, "square": self.cSquares["b000"][1]}) or
                                                             self.inCheck({"king": piece, "square": self.cSquares["b000"][0]})))):
                    # logger.log({"function": "chess100.move", "description": "missing castling requirements", "data": {}})
                    return {"err": "Castling requested, but the king is or would be in check."}
                else:
                    # if the castling is valid, place the rook
                    # the king move is considered as a normal piece move and the placement is done at the end of this function

                    if o["to"] == self.cSquares["w00"][2]:
                        self.position[self.cSquares["w00"][1]]["piece"] = "R"
                        self.position[self.cSquares["w00"][-1]]["piece"] = ""  # _240809
                        self._updateSquares({"sqFrom": self.cSquares["w00"][-1], "sqTo": self.cSquares["w00"][1]})
                        retVal["castling"] = "O-O"
                    elif o["to"] == self.cSquares["b00"][2]:
                        self.position[self.cSquares["b00"][1]]["piece"] = "r"
                        self.position[self.cSquares["b00"][-1]]["piece"] = ""
                        self._updateSquares({"sqFrom": self.cSquares["b00"][-1], "sqTo": self.cSquares["b00"][1]})
                        retVal["castling"] = "O-O"
                    elif o["to"] == self.cSquares["w000"][2]:
                        self.position[self.cSquares["w000"][1]]["piece"] = "R"
                        self.position[self.cSquares["w000"][-1]]["piece"] = ""  # _240809
                        self._updateSquares({"sqFrom": self.cSquares["w000"][-1], "sqTo": self.cSquares["w000"][1]})
                        retVal["castling"] = "O-O-O"
                    elif o["to"] == self.cSquares["b000"][2]:
                        self.position[self.cSquares["b000"][1]]["piece"] = "r"
                        self.position[self.cSquares["b000"][-1]]["piece"] = ""  # _240809
                        self._updateSquares({"sqFrom": self.cSquares["b000"][-1], "sqTo": self.cSquares["b000"][1]})
                        retVal["castling"] = "O-O-O"

            else:
                return {"err": "Target square is invalid."}
                
        if (self.position[o["to"]]["piece"] != "" and
            ((" PCZNBFRQK".find(piece) > 0 and " PCZNBFRQK".find(self.position[o["to"]]["piece"]) > 0) or
             (" pcznbfrqk".find(piece) > 0 and " pcznbfrqk".find(self.position[o["to"]]["piece"]) > 0))):
            return {"err": "Target square is blocked by a friendly piece."}

        # logger.log({"function": "chess100.move", "description": "before adjusting castling flags", "data": {"castlings": self.castlings, "piece": piece, "cSquares": self.cSquares}})
        # adjust castling flags, if applicable
        
        if self.castlings != "":

            if piece == "K":
                self.castlings = self.castlings.replace(self.cSquares["w000"][2] + " ", "")
                self.castlings = self.castlings.replace(self.cSquares["w00"][2] + " ", "")

            if piece == "k":
                self.castlings = self.castlings.replace(self.cSquares["b000"][2] + " ", "")
                self.castlings = self.castlings.replace(self.cSquares["b00"][2] + " ", "")

            # rook has moved
            if piece == "R":
                # _240809 if(o.from == cSquares["w000"][4]){
                if o["from"] == self.cSquares["w000"][-1]:  # _240809
                    self.castlings = self.castlings.replace(self.cSquares["w000"][2] + " ", "")
                # _240809 if(o.from == cSquares["w00"][3]){
                if o["from"] == self.cSquares["w00"][-1]:  # _240809
                    self.castlings = self.castlings.replace(self.cSquares["w00"][2] + " ", "")

            if piece == "r":
                # _240809 if(o.from == cSquares["b000"][4]){
                if o["from"] == self.cSquares["b000"][-1]:  # _240809
                    self.castlings = self.castlings.replace(self.cSquares["b000"][2] + " ", "")
                # _240809 if(o.from == cSquares["b00"][3]){
                if o["from"] == self.cSquares["b00"][-1]:  # _240809
                    self.castlings = self.castlings.replace(self.cSquares["b00"][2] + " ", "")

            # rook was beaten
            if self.position[o["to"]]["piece"] == "R":
                # _240809 if(o.to == cSquares["w000"][4]){
                if o["to"] == self.cSquares["w000"][-1]:  # _240809
                    self.castlings = self.castlings.replace(self.cSquares["w000"][2] + " ", "")
                # _240809 if(o.to == cSquares["w00"][3]){
                if o["to"] == self.cSquares["w00"][-1]:  # _240809
                    self.castlings = self.castlings.replace(self.cSquares["w00"][2] + " ", "")

            if self.position[o["to"]]["piece"] == "r":
                # _240809 if(o.to == cSquares["b000"][4]){
                if o["to"] == self.cSquares["b000"][-1]:  # _240809
                    self.castlings = self.castlings.replace(self.cSquares["b000"][2] + " ", "")
                # _240809 if(o.to == cSquares["b00"][3]){
                if o["to"] == self.cSquares["b00"][-1]:  # _240809
                    self.castlings = self.castlings.replace(self.cSquares["b00"][2] + " ", "")

        # logger.log({"function": "chess100.move", "description": "after adjusting castling flags", "data": {"castlings": self.castlings, "piece": piece, "cSquares": self.cSquares}})
        
        # handle promotions
        
        #_260415 if "promotion" in o:
        if o.get("promotion") is not None: # 260415

            if piece not in "pP":
                return {"err": "Promotion requested, but it's not a pawn that has been moved."}

            if not ((piece == "P" and self.squareNumbers[o["to"]] < self.boardSize) or
                    (piece == "p" and self.squareNumbers[o["to"]] > (self.squaresCnt - self.boardSize - 1))):
                # if the piece is not a pawn or the rank is invalid
                return {"err": "Promotion requested, but the target square is not on the first or last rank."}

            piece = o["promotion"].upper() if self.active == "w" else o["promotion"].lower()
            retVal["promotion"] = piece; #_260412
            # return success()

        if piece.lower() == "k":
            self.kingSquares[piece] = o["to"]
            
        if piece.lower() == "k":
            self.kingSquares[piece] = o["to"]

        # if applicable, detect the piece that was beaten
        
        previousPiece = self.position[o["to"]]["piece"]

        # update position
        
        # update position

        self.position[o["to"]]["piece"] = piece
        self.position[o["from"]]["piece"] = ""
        # logger.log({"function": "chess100.move ", "description": "before handling ep capture", "data": {"ep1": self.epField1, "ep2": self.epField2, "piece": piece}})
        if piece.lower() == "p" and (o["to"] == previousValues["epField1"] or o["to"] == previousValues["epField2"]):

            # handle en passant capture

            beatenByEpSquare = self.squares[
                self.squareNumbers[o["to"]] +
                self.boardSize *
                (2 if (previousValues["epField2"] != "" and o["to"] == previousValues["epField1"]) else 1) *
                (-1 if piece == "p" else 1)
            ]  # _201021
            previousPiece = self.position[beatenByEpSquare]["piece"]  # _200722
            previousValues["beatenByEp"] = {"piece": self.position[beatenByEpSquare]["piece"], "square": beatenByEpSquare}
            self.position[beatenByEpSquare]["piece"] = ""
            # logger.log({"function": "chess100.move ", "description": "after handling ep capture", "data": {"beatenByEp": previousValues["beatenByEp"], "beatenByEpSquare": beatenByEpSquare}})  # _201021

        # logger.log({"function": "chess100.move ", "description": "before updating squares"})
        self._updateSquares({"previousPiece": previousPiece, "sqFrom": o["from"], "sqTo": o["to"]})
        # logger.log({"function": "chess100.move", "description": "after updating squares", "data": {"active": self.active, "con1": (self.active == "w" and self.inCheck({"king": "K", "square": self.kingSquares["K"]})), "con2": (self.active == "b" and self.inCheck({"king": "k", "square": self.kingSquares["k"]})), "kingSquares": self.kingSquares}})

        if ((self.active == "w" and self.inCheck({"king": "K", "square": self.kingSquares["K"]})) or
            (self.active == "b" and self.inCheck({"king": "k", "square": self.kingSquares["k"]}))):
            
            # if in the resulting position the side that has moved would be in check, reset the position to the previous state

            if "promotion" in o:
                piece = "P" if self.active == "w" else "p"

            self.castlings = previousValues["castlings"]
            self.epField1 = previousValues["epField1"]
            self.epField2 = previousValues["epField2"]
            self.kingSquares = copy.deepcopy(previousValues["kingSquares"])
            self.position = copy.deepcopy(previousValues["position"])  # _201013

            return {"err": "Active color has moved, but is still in check."}
            
        # check if the move is ambiguous
        
        controlledBy_ = self.position[o["to"]]["controlledBy"].split(" ")
        controlledBy_.pop()  # remove the last empty element from split

        for sq in controlledBy_:
            if self.position[sq]["piece"] == retVal["piece"]:  # _200603
                # logger.log({"function": "chess100.move", "description": "is it ambiguous", "data": {"piece": piece, "previousPiece": previousPiece}})
                if not (retVal["piece"].lower() == "p" and previousPiece == ""):
                    # if the piece is not a pawn that moved forward on a field that is controlled by neighbouring friendly pawns.
                    if not self._isAbsolutelyPinned({"kSquare": self.kingSquares["K"] if self.active == "w" else self.kingSquares["k"], "square": sq}):
                        # if the twin piece isn't absolutely pinned.
                        # https://chess.stackexchange.com/questions/1864/is-this-case-considered-an-ambiguity-or-not
                        retVal["ambiguous"] = True

        # switch active color
        
        if self.active == "w":
            self.active = "b"
        else:
            self.active = "w"
            self.fullmoveNr += 1

        # update halfmove clock
        
        # logger.log({"function": "chess100.move", "description": "before checking halfmoveclock", "data": {"hmc": self.halfmoveClock, "piece": piece, "previousPiece": previousPiece}})  # _200603
        if retVal["piece"].lower() == "p" or previousPiece != "":
            self.halfmoveClock = 0
        else:
            self.halfmoveClock += 1

        # update capture count if a piece was taken
        
        if previousPiece != "":
            self.capturedCnt += 1
            retVal["capture"] = True

        # logger.log({"function": "chess100.move", "description": "before return", "data": {"castlings": self.castlings, "sqTo": o["to"], "sqToParams": self.position[o["to"]]}})

        # move was successful - return move data
        
        return retVal
        
    def get_position(self):
        # simply returns the current board position
        return self.position  # _211231

    def possibleEpSquares(self):
        # detects all possible en passant squares for a given position
        possibleEpSquares = []

        # console.log("active: " + active)
        if self.active == "b":
            # Black to move → check for white pawns that can be captured en passant
            for i in range(4, 8 if self.boardSize == 10 else 5):
                for j in range(self.boardSize):
                    if self.position[self.files[j] + str(i)]["piece"] == "P":
                        epSquares = ""

                        if j == 0:
                            if self.position[self.files[j+1] + str(i)]["piece"] == "p":
                                epSquares = self.files[j] + str(i - 1)
                            if i == 5 and self.position[self.files[j+1] + str(i-1)]["piece"] == "p":
                                epSquares = self.files[j] + str(i - 2) + epSquares

                        if j == (self.boardSize - 1):
                            if self.position[self.files[j-1] + str(i)]["piece"] == "p":
                                epSquares = self.files[j] + str(i - 1)
                            if i == 5 and self.position[self.files[j-1] + str(i-1)]["piece"] == "p":
                                epSquares = self.files[j] + str(i - 2) + epSquares

                        if 0 < j < (self.boardSize - 1):
                            if self.position[self.files[j+1] + str(i)]["piece"] == "p" or \
                               self.position[self.files[j-1] + str(i)]["piece"] == "p":
                                epSquares = self.files[j] + str(i - 1)
                            if i == 5 and (self.position[self.files[j+1] + str(i-1)]["piece"] == "p" or \
                                          self.position[self.files[j-1] + str(i-1)]["piece"] == "p"):
                                epSquares = self.files[j] + str(i - 2) + epSquares

                        if epSquares != "":
                            possibleEpSquares.append(epSquares)

        else:
            # White to move → check for black pawns that can be captured en passant
            for i in range(7, 3 if self.boardSize == 10 else 6, -1):
                for j in range(self.boardSize):
                    # console.log(JSON.stringify(files))
                    if self.position[self.files[j] + str(i)]["piece"] == "p":
                        # console.log("ep pawn to beat: " & position[files[j] + i])
                        epSquares = ""

                        if j == 0:
                            if self.position[self.files[j+1] + str(i)]["piece"] == "P":
                                epSquares = self.files[j] + str(i + 1)
                            if i == 6 and self.position[self.files[j+1] + str(i+1)]["piece"] == "P":
                                epSquares = self.files[j] + str(i + 2) + epSquares

                        if j == (self.boardSize - 1):
                            if self.position[self.files[j-1] + str(i)]["piece"] == "P":
                                epSquares = self.files[j] + str(i + 1)
                            if i == 6 and self.position[self.files[j-1] + str(i+1)]["piece"] == "P":
                                epSquares = self.files[j] + str(i + 2) + epSquares

                        if 0 < j < (self.boardSize - 1):
                            if self.position[self.files[j+1] + str(i)]["piece"] == "P" or \
                               self.position[self.files[j-1] + str(i)]["piece"] == "P":
                                epSquares = self.files[j] + str(i + 1)
                            if i == 6 and (self.position[self.files[j+1] + str(i+1)]["piece"] == "P" or \
                                          self.position[self.files[j-1] + str(i+1)]["piece"] == "P"):
                                epSquares = self.files[j] + str(i + 2) + epSquares

                        if epSquares != "":
                            possibleEpSquares.append(epSquares)

        return possibleEpSquares
    
    def sanToFromTo(self, o):  # san
        # console.log(new Error().stack);
        # converts a move from san notation to from: ... to: notation
        # san examples: e4, exd4, Nfd4, Nfxd4+, Nxd4, Nd4+, Nd4#, O-O, O-O-O, b8=Q, b8=Q+, axb8=Q, axb8=Q+

        # if the active color castled, the transformation is simple

        if o["san"].find("O-O-O") != -1:
            return {"from": self.cSquares[self.active + "000"][0], "to": self.cSquares[self.active + "000"][2]}

        if o["san"].find("O-O") != -1:
            return {"from": self.cSquares[self.active + "00"][0], "to": self.cSquares[self.active + "00"][2]}

        # otherwise the params must be evaluated by analyzing san and position

        # detecting piece

        ambiguous = ""
        piece = o["san"][0]

        if "CZNBFRQK".find(piece) == -1:  # _240422

            # piece is a pawn

            piece = "P"

        # detecting the target square

        san_ = list(o["san"])
        sqTo = ""
        # logger.log({function: "chess100.sanToFromTo  ", description: "before looping for target", data: {san: san_}});
        for i in range(len(san_) - 1, -1, -1):
            # logger.log({function: "chess100.sanToFromTo  ", description: "detecting target", data: {san: san_}});
            if "abcdefghij1234567890".find(san_[i]) != -1:  # _191209
                if (len(sqTo) == 2 and sqTo != "10") or len(sqTo) == 3:

                    if "abcdefghij1234567890".find(san_[i]) != -1:  # _191209
                        # ambiguous move, e.g. the f in Nfd4
                        # or pawn beat, e.g. the e in exd4

                        ambiguous = san_[i]
                        break

                else:

                    sqTo = san_[i] + sqTo

        # logger.log({function: "chess100.sanToFromTo  ", description: "after analyzing san", data: {ambiguous: ambiguous, controlledBy_: controlledBy_, fen: self.fen(), san: o.san, sqTo: sqTo}});

        # detecting the source square

        sqFrom = ""

        controlledBy_ = self.position[sqTo]["controlledBy"].split(" ")
        controlledBy_.pop()
        # logger.log({function: "chess100.sanToFromTo  ", description: "before detecting source", data: {ambiguous: ambiguous, controlledBy_: controlledBy_, fen: self.fen(), san: o.san, sqTo: sqTo}});
        for i in range(len(controlledBy_)):

            if self.position[controlledBy_[i]]["piece"].upper() == piece:

                if ((" PCZNBFRQK".find(self.position[controlledBy_[i]]["piece"]) > 0 and self.active == "w") or
                    (" pcznbfrqk".find(self.position[controlledBy_[i]]["piece"]) > 0 and self.active == "b")):

                    if ambiguous == "" or (ambiguous != "" and ambiguous in controlledBy_[i]):

                        if not (ambiguous == "" and piece == "P"):

                            # if ! a pawn moved and pawn(s) control the target square, but no piece was beaten (then a pawn moved forward, see below)

                            sqFrom = controlledBy_[i]
                            # _240131
                            if ambiguous == "":

                                pieceFound = False

                                for j in range(len(controlledBy_)):

                                    if self.position[controlledBy_[j]]["piece"] == self.position[controlledBy_[i]]["piece"]:

                                        if pieceFound is False:

                                            pieceFound = True

                                        else:

                                            # if the san notation wasn't ambiguous but the piece exists nevertheless twice in the controlledBy string. this can happen if one of both pieces is absolutely pinned.

                                            if self._isAbsolutelyPinned({
                                                "kSquare": (self.kingSquares["K"] if self.active == "w" else self.kingSquares["k"]),
                                                "square": controlledBy_[i]
                                            }) is True:

                                                sqFrom = ""

                                # _240131

                            if sqFrom != "":
                                # _240131
                                break

        if sqFrom == "":
            # logger.log({function: "chess100.sanToFromTo  ", description: "pawn moved forward", data: {san: o.san, sqTo: sqTo}});
            # pawn moved forward

            squareToTest = ""

            for i in range(1, 4):

                squareToTest = self.squares[
                    self.squareNumbers[sqTo] + self.boardSize * i * (1 if self.active == "w" else -1)
                ]

                if self.position[squareToTest]["piece"] != "":

                    sqFrom = squareToTest
                    break

        promotion = ""
        promotions_ = list("QFNZCBR")  # _240422
        # _240422 var promotions_ = "QNFBR".split("");

        for i in range(len(promotions_)):

            if ("=" + promotions_[i]) in o["san"]:

                promotion = promotions_[i]
                break

        retVal = {"from": sqFrom, "to": sqTo}

        if promotion != "":
            retVal["promotion"] = promotion

        # logger.log({function: "chess100.sanToFromTo  ", description: "before returning retval", data: {retVal: retVal}}); //_200608
        return retVal

    def turn(self):
        # returns the color of the active player
        return self.active
        
# renaming from legalMoves to get_legal_moves_for_square #_261102
# renaming from legalMovesAll to get_legal_moves #_261102
