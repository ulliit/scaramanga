import unittest
from rules import Rules

class TestTactalaris(unittest.TestCase):

    def test_part_1(self):
        rules = Rules()

        # abilities
        self.assertEqual(
            rules.get_abilities({"boardSize": 10, "piece": "n", "square": "b10"}),
            ['d9', 'c8', 'a8']
        )

        self.assertEqual(
            rules.get_abilities({"boardSize": 10, "piece": "b", "square": "i3"}),
            ['j4', '', 'j2', '', 'h2', 'g1', '', 'h4', 'g5', 'f6', 'e7', 'd8', 'c9', 'b10']
        )

        self.assertEqual(
            rules.get_abilities({"boardSize": 10, "piece": "f", "square": "c6"}),
            ['d9', 'e9', 'f8', 'f7', 'f5', 'f4', 'e3', 'd3', 'b3', 'a3', 'a9', 'b9']
        )

        self.assertEqual(
            rules.get_abilities({"boardSize": 10, "piece": "r", "square": "e10"}),
            ['f10', 'g10', 'h10', 'i10', 'j10', '', 'e9', 'e8', 'e7', 'e6', 'e5',
             'e4', 'e3', 'e2', 'e1', '', 'd10', 'c10', 'b10', 'a10']
        )

        self.assertEqual(
            rules.get_abilities({"boardSize": 10, "piece": "p", "square": "b8"}),
            ["a7", "b7", "b6", "c7"]
        )

        self.assertEqual(
            rules.get_abilities({"boardSize": 10, "piece": "P", "square": "b7"}),
            ["a8", "b8", "c8"]
        )

        # check if update squares works correctly for border pawns
        rules.load({"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"})
        self.assertEqual(
            rules.get_position()["h7"],
            {"controlledBy": "h8 ", "controls": "g6 ", "piece": "p"}
        )

        rules.load({"fen": "frnbqkbnrf/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/FRNBQKBNRF w KQkq - 0 1"})
        self.assertEqual(
            rules.get_position()["a9"],
            {"controlledBy": "c10 ", "controls": "b8 ", "piece": "p"}
        )

        rules.load({"fen": "6k3/1b8/10/10/10/10/10/10/8N1/7K2 w - - 0 1"})
        self.assertEqual(
            rules.get_position()["b9"],
            {"controlledBy": "", "controls": "c10 c8 d7 e6 f5 g4 h3 i2 a8 a10 ", "piece": "b"}
        )

        self.assertEqual(
            rules.get_position()["i2"],
            {"controlledBy": "b9 h1 ", "controls": "j4 g1 g3 h4 ", "piece": "N"}
        )

        # trying to move on a square that is not controlled
        self.assertEqual(
            rules.move({"from": "i2", "to": "d4"}),
            {"err": "Target square is invalid."}
        )

        # trying to move on a square that is occupied by a friendly piece
        self.assertEqual(
            rules.move({"from": "h1", "to": "i2"}),
            {"err": "Target square is blocked by a friendly piece."}
        )

        # position should be updated after legal knight move
        self.assertEqual(
            rules.move({"from": "i2", "to": "g3"}),
            {"from": "i2", "piece": "N", "to": "g3"}
        )

        self.assertEqual(
            rules.get_position()["b9"],
            {"controlledBy": "", "controls": "c10 c8 d7 e6 f5 g4 h3 i2 j1 a8 a10 ", "piece": "b"}
        )

        self.assertEqual(
            rules.move({"from": "b9", "to": "c8"}),
            {"from": "b9", "piece": "b", "to": "c8"}
        )

        self.assertEqual(
            rules.move({"from": "g3", "to": "i2"}),
            {"from": "g3", "piece": "N", "to": "i2"}
        )

        # black bishop on c8 should have lost control over j1 after white knight moved to i2
        self.assertEqual(
            rules.get_position()["c8"],
            {"controlledBy": "", "controls": "d9 e10 d7 e6 f5 g4 h3 i2 b7 a6 b9 a10 ", "piece": "b"}
        )

        # test some pawn movements from the start position
        rules.load({"fen": "frnbqkbnrf/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/FRNBQKBNRF w KQkq - 0 1"})
        self.assertEqual(
            rules.move({"from": "e2", "to": "e5"}),
            {"from": "e2", "piece": "P", "to": "e5"}
        )

        self.assertEqual(
            rules.move({"from": "e2", "to": "e6"}),
            {"err": "No piece on the source square."}
        )

        rules.load({"fen": "10/10/1pp3k3/1P8/10/10/10/10/6K3/10 w - - 0 1"})
        self.assertEqual(
            rules.move({"from": "b7", "to": "a8"}),
            {"err": "The pawn controls the square but there is no enemy piece to beat on it."}
        )

        self.assertEqual(
            rules.move({"from": "b7", "to": "c8"}),
            {"capture": True, "from": "b7", "piece": "P", "to": "c8"}
        )

        self.assertEqual(
            rules.get_position()["c8"],
            {"controlledBy": "", "controls": "b9 d9 ", "piece": "P"}
        )

        rules.load({"fen": "10/3k4P1/10/10/10/10/10/10/6K3/10 w - - 0 1"})
        self.assertEqual(
            rules.move({"from": "i9", "to": "i10", "promotion": "Q"}),
            {"from": "i9", "piece": "P", "promotion": "Q", "to": "i10"} #_260412
        )

        self.assertEqual(
            rules.get_position()["i10"],
            {
                "controlledBy": "",
                "controls": "j10 j9 i9 i8 i7 i6 i5 i4 i3 i2 i1 h9 g8 f7 e6 d5 c4 b3 a2 h10 g10 f10 e10 d10 c10 b10 a10 ",
                "piece": "Q"
            }
        )

        # series of moves including short castlings
        rules.load({"fen": "frnbqkbnrf/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/FRNBQKBNRF w KQkq - 0 1"})
        self.assertEqual(
            rules.move({"from": "h2", "to": "h3"}),
            {"from": "h2", "piece": "P", "to": "h3"}
        )
        self.assertEqual(
            rules.move({"from": "h9", "to": "h8"}),
            {"from": "h9", "piece": "p", "to": "h8"}
        )
        self.assertEqual(
            rules.move({"from": "g1", "to": "h2"}),
            {"from": "g1", "piece": "B", "to": "h2"}
        )
        self.assertEqual(
            rules.move({"from": "g10", "to": "h9"}),
            {"from": "g10", "piece": "b", "to": "h9"}
        )
        self.assertEqual(
            rules.move({"from": "h1", "to": "g3"}),
            {"from": "h1", "piece": "N", "to": "g3"}
        )
        self.assertEqual(
            rules.move({"from": "h10", "to": "g8"}),
            {"from": "h10", "piece": "n", "to": "g8"}
        )
        self.assertEqual(
            rules.move({"from": "f1", "to": "h1"}),
            {"castling": "O-O", "piece": "K", "from": "f1", "to": "h1"}
        )
        self.assertEqual(
            rules.move({"from": "f10", "to": "h10"}),
            {"castling": "O-O", "piece": "k", "from": "f10", "to": "h10"}
        )

        # black O-O-O
        rules.load({"fen": "r3k3/ppp1np2/1b1p4/4n2r/4P1bp/2NN4/PPPB2PP/2K1RB1R b q - 4 19"})
        self.assertEqual(
            rules.move({"from": "e8", "to": "c8"}),
            {"castling": "O-O-O", "from": "e8", "piece": "k", "to": "c8"}
        )

        # ep
        rules.load({"fen": "1r3r2/p2k4/7Q/P2p4/nPp5/8/6P1/6K1 b - b3 0 47"})
        self.assertEqual(
            rules.move({"from": "c4", "to": "b3"}),
            {"capture": True, "from": "c4", "piece": "p", "to": "b3"}
        )
        self.assertEqual(
            rules.fen(),
            "1r3r2/p2k4/7Q/P2p4/n7/1p6/6P1/6K1 w - - 0 48"
        )

        # capture of a white pawn on the fourth rank, after moving three steps from the initial square //_201021

        rules.load({"fen": "frnbqkbnrf/ppp1pppppp/10/10/10/2PpPP4/10/10/PP1P2PPPP/FRNBQKBNRF b - e3e4 0 3"})
        self.assertEqual(
            rules.move({"san": "dxe4"}),
            {"capture": True, "from": "d5", "piece": "p", "to": "e4"}
        )

        # capture of a black pawn on the 8th rank, after moving three steps from the initial square

        rules.load({"fen": "1r2q1r1k1/pppp3ppp/3n6/2f2P2f1/2b1p1p3/2B1PnpBb1/2F5F1/3P2N1P1/PPP1N2P1P/1R2QR2K1 w - g8g7 0 16"})
        self.assertEqual(
            rules.move({"san": "fxg8"}),
            {"capture": True, "from": "f7", "piece": "P", "to": "g8"}
        )

        # promotion

        rules.load({"fen": "3r4/1q2Pk2/8/8/8/8/5B2/5K2 w - - 0 64"})
        self.assertEqual(
            rules.move({"from": "e7", "to": "d8", "promotion": "N"}),
            {"capture": True, "from": "e7", "piece": "P", "promotion": "N", "to": "d8"} #_260412
        )
        self.assertEqual(
            rules.fen(),
            "3N4/1q3k2/8/8/8/8/5B2/5K2 b - - 0 64"
        )

        rules.load({"fen": "7Q/6K1/8/8/8/8/1pk5/8 b - - 0 71"})
        self.assertEqual(
            rules.move({"from": "b2", "to": "b1", "promotion": "Q"}),
            {"from": "b2", "piece": "p", "promotion": "q", "to": "b1"} #_260412
        )
        self.assertEqual(
            rules.fen(),
            "7Q/6K1/8/8/8/8/2k5/1q6 w - - 0 72"
        )

        rules.load({"fen": "10/k2P2q3/10/10/10/10/10/10/10/2K7 w - - 0 1"})
        self.assertEqual(
            rules.move({"from": "d9", "to": "d10", "promotion": "F"}),
            {"from": "d9", "piece": "P", "promotion": "F", "to": "d10"} #_260412
        )
        self.assertEqual(
            rules.fen(),
            "3F6/k5q3/10/10/10/10/10/10/10/2K7 b - - 0 1"
        )

        rules.load({"fen": "10/k2P2q3/10/10/10/10/10/10/10/2K7 w - - 0 1"})
        self.assertEqual(
            rules.move({"san": "d10=F+"}),
            {"from": "d9", "piece": "P", "promotion": "F", "to": "d10"} #_260412
        )
        self.assertEqual(
            rules.fen(),
            "3F6/k5q3/10/10/10/10/10/10/10/2K7 b - - 0 1"
        )
        
        # fen validation

        self.assertEqual(
            rules.load({"fen": "4k3r1/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/1R3K2R1 w KQkq - 0 1"}),
            "Invalid fen - black castling possibilities were enabled but the black king is not on its initial square."
        )

        # return turn

        self.assertEqual(rules.turn(), "w")

        # king can't move on an enemy controlled field
        # chess100 evaluates if the king of the side to move would be in check in the resulting position and should return to the previous position if not.

        rules.load({"fen": "10/10/5k1P2/10/8B1/10/10/10/10/7K2 b - - 0 1"})
        self.assertEqual(
            rules.move({"from": "f8", "to": "g8"}),
            {"err": "Active color has moved, but is still in check."}
        )
        self.assertEqual(
            rules.get_position()["e9"],
            {"controlledBy": "f8 ", "controls": "", "piece": ""}
        )
        self.assertEqual(
            rules.get_position()["g9"],
            {"controlledBy": "f8 h8 ", "controls": "", "piece": ""}
        )
        self.assertEqual(
            rules.get_position()["h9"],
            {"controlledBy": "", "controls": "", "piece": ""}
        )

        # special case: from the perspective of a check giving pedestrian, is the field behind the king marked as in check?

        rules.load({"fen": "10/10/6f3/10/5pkp2/5r1q2/10/1Q8/10/1K8 w - - 0 1"})
        rules.move({"from": "b3", "to": "g8"})
        self.assertEqual(
            rules.move({"from": "g6", "to": "g5"}),
            {"err": "Active color has moved, but is still in check."}
        )
        self.assertTrue(rules.inCheckmate())
        
        # should recognize that no friendly piece can block here
                                 
        rules.load({"fen": "8/8/8/7r/5Q1k/2K4p/6p1/8 b - - 0 1"})
        self.assertTrue(rules.inCheckmate())

        rules.load({"fen": "8/7k/2b2q2/2P5/2P5/K1q5/P5P1/8 w - - 1 50"})
        self.assertTrue(rules.inCheckmate())

        # mate by double check

        rules.load({"fen": "3k1K2/r7/3N4/3Q4/8/8/8/2R5 w - - 0 1"})
        rules.move({"from": "d6", "to": "f7"})
        self.assertTrue(rules.inCheckmate())

        rules.load({"fen": "3k1K2/r7/8/8/8/8/3B4/2RQ4 w - - 0 1"})
        rules.move({"from": "d2", "to": "g5"})
        self.assertTrue(rules.inCheckmate())

        rules.load({"fen": "1r6/p4P1p/4P2k/3p4/3q2p1/1N6/PK3R2/5R2 w - - 0 1"})
        rules.move({"from": "b2", "to": "a3"})
        rules.move({"from": "d4", "to": "b4"})
        self.assertFalse(rules.inCheckmate())
        
        # piece can't move because it's absolutely pinned

        rules.load({"fen": "10/6kppp/5r4/10/3B6/10/10/10/7ppp/8K1 b - - 0 1"})
        self.assertEqual(
            rules.move({"from": "f8", "to": "f1"}),
            {"err": "Active color has moved, but is still in check."}
        )

        # two knights can reach the same square but one is absolutely pinned

        rules.load({"fen": "10/10/4r1k3/10/10/10/4N1N3/4K5/10/10 w - - 0 1"})
        self.assertEqual(
            rules.move({"san": "Nf6+"}),
            {"from": "g4", "piece": "N", "to": "f6"}
        )

        # beating illegally a piece, because the king would be in check.
        # after the automatic position reset to the previous state, the beaten piece should be in place again
        # and control the fields it controlled before.

        rules.load({"fen": "3r6/1pk7/1pp3p3/10/5f4/10/3B1F4/1P1KPP3B/10/10 w - - 0 1"})
        self.assertEqual(
            rules.move({"from": "d4", "to": "f6"}),
            {"err": "Active color has moved, but is still in check."}
        )

        self.assertEqual(
            rules.get_position()["f6"],
            {"controlledBy": "d4 ", "controls": "g9 h9 i8 i7 i5 i4 h3 g3 e3 d3 c4 c5 c7 c8 d9 e9 ", "piece": "f"}
        )

        self.assertEqual(
            rules.fen(),
            "3r6/1pk7/1pp3p3/10/5f4/10/3B1F4/1P1KPP3B/10/10 w - - 0 1"
        )

        self.assertTrue(rules.inCheck({"king": "K", "square": "d3"}))
        
        # no pin here
        
        rules.load({"fen": "10/7kpp/5r4/10/3B6/10/10/10/7PPP/8K1 b - - 0 1"})
        self.assertEqual(
            rules.move({"from": "f8", "to": "f1"}),
            {"from": "f8", "piece": "r", "to": "f1"}
        )

        # wrong side moved
        
        rules.load({"fen": "frnbqkbnrf/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/FRNBQKBNRF w KQkq - 0 1"})
        self.assertEqual(
            rules.move({"from": "e9", "to": "e7"}),
            {"err": "Wrong color moved."}
        )

        # vintage start position shouldn't be checkmate
        
        rules.load({"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"})
        self.assertFalse(rules.inCheck({"king": "K", "square": "e1"}))
        self.assertFalse(rules.inCheckmate())

        # no checkmate, since one neighbouring field is free
        
        rules.load({"fen": "10/10/10/10/10/10/10/6K3/7Q2/6k3 b - - 0 1"})
        self.assertFalse(rules.inCheckmate())

        # after the next two moves it should be checkmate
        
        rules.move({"from": "g1", "to": "f1"})
        rules.move({"from": "h2", "to": "f2"})
        self.assertTrue(rules.inCheckmate())

        # no checkmate, pawn can block
        
        rules.load({"fen": "5N2/5pp1/6pk/8/7K/8/8/2B5 b - - 0 1"})
        self.assertFalse(rules.inCheckmate())

        # checkmate, pawn can't block, since it is pinned
        
        rules.load({"fen": "q4N2/5pp1/2R3pk/8/7K/8/8/2B5 b - - 0 1"})
        self.assertTrue(rules.inCheckmate())

        # no checkmate, rook can block
        
        rules.load({"fen": "6rk/2p1R3/2b5/8/2PR4/8/7P/7K w - - 0 1"})
        self.assertFalse(rules.inCheckmate())

        # piece takes a piece with the same move pattern, gives check and it shouldn't be checkmate by doublecheck
        # this was buggy until fix 201012 because the fields controlled by the beaten piece weren't updated
        
        rules.load({"fen": "4rrk3/pppp2pppp/2q7/3q4fF/5p1N2/2FQ2n3/3P2B3/3Nf5/PPP1K1P2P/7R2 w - - 0 1"})
        rules.move({"san": "Bxd7"})
        self.assertFalse(rules.inCheckmate())

        # simple stalemate
        
        rules.load({"fen": "8k1/7P2/8K1/10/10/10/10/10/10/10 b - - 0 1"})
        self.assertFalse(rules.inStalemate())
        rules.move({"from": "i10", "to": "h10"})
        rules.move({"from": "i8", "to": "h8"})
        self.assertTrue(rules.inStalemate())

        # fen output
        
        self.assertEqual(
            rules.fen(),
            "7k2/7P2/7K2/10/10/10/10/10/10/10 b - - 2 2"
        )

        # 50 moves
        
        rules.load({"fen": "5r4/10/10/4k5/10/5K4/10/10/3R6/10 w - - 98 120"})
        self.assertFalse(rules.fiftyMoves())
        rules.move({"from": "f5", "to": "g5"})
        rules.move({"from": "f10", "to": "g10"})
        self.assertTrue(rules.fiftyMoves())  
        
        # insufficient material
        
        rules.load({"fen": "10/10/6B3/4k5/10/5K4/10/10/p9/10 w - - 0 1"})
        self.assertFalse(rules.insufficientMaterial())
        rules.move({"from": "g8", "to": "a2"})
        self.assertTrue(rules.insufficientMaterial())

        # san conversion / move tests
        # by executing the first moves of the dragon variation
        
        rules.load({"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"})
        self.assertEqual(rules.sanToFromTo({"san": "e4"}), {"from": "e2", "to": "e4"})
        rules.move({"san": "e4"})
        self.assertEqual(rules.fen(), "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        rules.move({"san": "c5"})
        self.assertEqual(rules.fen(), "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2")
        self.assertEqual(rules.sanToFromTo({"san": "Nf3"}), {"from": "g1", "to": "f3"})
        rules.move({"san": "Nf3"})
        self.assertEqual(rules.fen(), "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
        self.assertEqual(rules.sanToFromTo({"san": "d6"}), {"from": "d7", "to": "d6"})
        rules.move({"san": "d6"})
        rules.move({"san": "d4"})
        self.assertEqual(rules.fen(), "rnbqkbnr/pp2pppp/3p4/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq d3 0 3")
        self.assertEqual(rules.sanToFromTo({"san": "cxd4"}), {"from": "c5", "to": "d4"})
        rules.move({"san": "cxd4"})
        self.assertEqual(rules.sanToFromTo({"san": "Nxd4"}), {"from": "f3", "to": "d4"})
        rules.move({"san": "Nxd4"})
        rules.move({"san": "Nf6"})
        self.assertEqual(rules.fen(), "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5")
        self.assertEqual(rules.sanToFromTo({"san": "Nc3"}), {"from": "b1", "to": "c3"})
        rules.move({"san": "Nc3"})
        self.assertEqual(rules.fen(), "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 2 5")
        rules.move({"san": "g6"})
        rules.move({"san": "Be3"})
        rules.move({"san": "Bg7"})
        rules.move({"san": "f3"})
        self.assertEqual(rules.fen(), "rnbqk2r/pp2ppbp/3p1np1/8/3NP3/2N1BP2/PPP3PP/R2QKB1R b KQkq - 0 7")
        rules.move({"san": "O-O"})
        self.assertEqual(rules.fen(), "rnbq1rk1/pp2ppbp/3p1np1/8/3NP3/2N1BP2/PPP3PP/R2QKB1R w KQ - 1 8")
        rules.move({"san": "Dd2"})
        rules.move({"san": "Nc6"})
        rules.move({"san": "Bc4"})
        rules.move({"san": "Bd7"})
        self.assertEqual(rules.fen(), "r2q1rk1/pp1bppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 5 10")
        self.assertEqual(rules.sanToFromTo({"san": "O-O-O"}), {"from": "e1", "to": "c1"})
        rules.move({"san": "O-O-O"})
        self.assertEqual(rules.fen(), "r2q1rk1/pp1bppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/2KR3R b - - 6 10")
        
        # san promotion
        
        rules.load({"fen": "3r4/1q2Pk2/8/8/8/8/5B2/5K2 w - - 0 1"})
        self.assertEqual(rules.sanToFromTo({"san": "exd8=N"}), {"from": "e7", "to": "d8", "promotion": "N"})

        # san: pawn beats with check
        
        rules.load({"fen": "r2qn3/pp1kbpB1/2p1b3/1P2Q2p/3P2p1/2NP1pP1/P6P/4RRK1 w - - 1 24"})
        self.assertEqual(rules.move({"san": "bxc6+"}), {"capture": True, "from": "b5", "piece": "P", "to": "c6"})

        # tbd:
        #    test for
        #        writing more tests for castlings, ep, checkmate, stalemate, ambiguousity
        #        correct position reconstruction after "if in the resulting position the side that has moved would be in check, reset the position to the previous state"

        # testing some border pawn moves
        
        rules.load({"fen": "r2qk3/pp2bpBp/2p1bn2/4Q3/3P2p1/2NP1pP1/PP5P/4RRK1 b q - 1 21"})
        self.assertEqual(rules.move({"from": "h7", "to": "h5"}), {"from": "h7", "piece": "p", "to": "h5"})

        rules.load({"fen": "kbnnbrqr/pppppppp/8/8/8/8/PPPPPPPP/RQRBNNBK w - - 0 1"})
        self.assertEqual(rules.move({"from": "a2", "to": "a4"}), {"from": "a2", "piece": "P", "to": "a4"})
        self.assertEqual(rules.move({"from": "h7", "to": "h5"}), {"from": "h7", "piece": "p", "to": "h5"})

        rules.load({"fen": "rqrbnnbk/pppppppp/8/8/8/8/PPPPPPPP/KBNNBRQR w - - 0 1"})
        self.assertEqual(rules.move({"from": "h2", "to": "h4"}), {"from": "h2", "piece": "P", "to": "h4"})
        self.assertEqual(rules.move({"from": "a7", "to": "a5"}), {"from": "a7", "piece": "p", "to": "a5"})

        # short castling for black should be eliminated after beating the rook on h8
        
        rules.load({"fen": "r1bqk1nr/ppp2pb1/2np4/6p1/3PPp2/2NQ1N2/PPP3P1/R1B1KB1R w KQkq - 0 9"})
        rules.move({"from": "h1", "to": "h8"})
        self.assertEqual(rules.fen(), "r1bqk1nR/ppp2pb1/2np4/6p1/3PPp2/2NQ1N2/PPP3P1/R1B1KB2 b Qq - 0 9")

        rules.load({"fen": "5k3r/8R1/10/10/10/10/10/10/P1N7/6K3 w k - 0 1"})
        self.assertEqual(rules.move({"from": "c2", "to": "e3"}), {"from": "c2", "piece": "N", "to": "e3"})
        rules.load({"fen": "5k3r/8R1/10/10/10/10/10/10/P1N7/6K3 w k - 0 1"})
        self.assertEqual(rules.move({"from": "a2", "to": "a5"}), {"from": "a2", "piece": "P", "to": "a5"})
        self.assertEqual(rules.fen(), "5k3r/8R1/10/10/10/P9/10/10/2N7/6K3 b k a3a4 0 1")
        self.assertEqual(rules.move({"from": "f10", "to": "h10"}), {"castling": "O-O", "from": "f10", "piece": "k", "to": "h10"})
        self.assertEqual(rules.fen(), "6rk2/8R1/10/10/10/P9/10/10/2N7/6K3 w - - 1 2")

        # rook checkmate on the back rank
        
        rules.load({"fen": "7k/5R1P/4pP2/8/pp6/1n6/1P2K2R/3n4 w - - 3 40"})
        rules.move({"san": "Rf8#"})
        self.assertTrue(rules.inCheckmate())
        
        rules.load({"fen": "frnbqkbnr1/pppp1ppppp/10/4p5/10/5P4/4P4f/4B5/PPPP2PPPP/FRNBQK1NRF w KQkq - 3 4"})
        self.assertEqual(rules.move({"from": "j1", "to": "i4"}), {"from": "j1", "piece": "F", "to": "i4"})

        # remove related castling option after rook movement
        
        rules.load({"fen": "frnbqkbnr1/pppp2pppp/10/5p4/4p5/5P4/4P3Ff/4B5/PPPP2PPPP/FRNBQK1NR1 w KQkq f8 0 5"})
        rules.move({"from": "i1", "to": "j1"})
        self.assertEqual(rules.fen(), "frnbqkbnr1/pppp2pppp/10/5p4/4p5/5P4/4P3Ff/4B5/PPPP2PPPP/FRNBQK1N1R b Qkq - 1 5")

        # detect ambiguousity
        
        rules.load({"fen": "k6r2/p6q2/4N5/3N6/10/10/10/10/8PP/1R7K w - - 0 1"})
        self.assertEqual(rules.move({"from": "e8", "to": "c9"}), {"ambiguous": True, "from": "e8", "piece": "N", "to": "c9"})

        rules.load({"fen": "4r5/1kpP1P1q2/1p8/10/10/10/10/3N4P1/2P7/2K7 w - - 0 1"})
        self.assertEqual(rules.move({"from": "f9", "promotion": "F", "to": "e10"}), {"ambiguous": True, "capture": True, "from": "f9", "piece": "P", "promotion": "F", "to": "e10"}) #_260412

        rules.load({"fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p1N1/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 6 5"})
        self.assertEqual(
            rules.get_legal_moves_for_square({"square": "f7"}),
            [
                {"from": "g5", "piece": "N", "to": "f7", "capture": True, "fen": "r1bqk2r/pppp1Npp/2n2n2/2b1p3/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 5"},
                {"from": "c4", "piece": "B", "to": "f7", "capture": True, "fen": "r1bqk2r/pppp1Bpp/2n2n2/2b1p1N1/4P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 5"}
            ]
        )

        rules.load({"fen": "rnbqkbnr/pp1ppppp/8/8/1p2P3/P7/2PP1PPP/RNBQKBNR b KQkq - 0 1"})
        self.assertEqual(
            rules.get_legal_moves_for_square({"square": "b3"}),
            [{"from": "b4", "piece": "p", "to": "b3", "fen": "rnbqkbnr/pp1ppppp/8/8/4P3/Pp6/2PP1PPP/RNBQKBNR w KQkq - 0 2"}]
        )
        
        # get all legal moves for a certain position

        rules.load({"fen": "qcnbrrbzck/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/KCZBRRBNCQ w - - 0 1"})
            # logger.log({"function": "rules-spec", "description": "test get_legal_moves", "data": {"legalMoves": rules.get_legal_moves()}})
        self.assertEqual(rules.get_legal_moves(), [
            {"from": "a2", "piece": "P", "to": "a3", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/10/P9/1PPPPPPPPP/KCZBRRBNCQ b - - 0 1"},
            {"from": "a2", "piece": "P", "to": "a4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/P9/10/1PPPPPPPPP/KCZBRRBNCQ b - a3 0 1"},
            {"from": "a2", "piece": "P", "to": "a5", "fen": "qcnbrrbzck/pppppppppp/10/10/10/P9/10/10/1PPPPPPPPP/KCZBRRBNCQ b - a3a4 0 1"},

            {"from": "b2", "piece": "P", "to": "b3", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/10/1P8/P1PPPPPPPP/KCZBRRBNCQ b - - 0 1"},
            {"from": "b2", "piece": "P", "to": "b4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/1P8/10/P1PPPPPPPP/KCZBRRBNCQ b - b3 0 1"},
            {"from": "b2", "piece": "P", "to": "b5", "fen": "qcnbrrbzck/pppppppppp/10/10/10/1P8/10/10/P1PPPPPPPP/KCZBRRBNCQ b - b3b4 0 1"},

            {"from": "c2", "piece": "P", "to": "c3", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/10/2P7/PP1PPPPPPP/KCZBRRBNCQ b - - 0 1"},
            {"from": "c2", "piece": "P", "to": "c4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/2P7/10/PP1PPPPPPP/KCZBRRBNCQ b - c3 0 1"},
            {"from": "c2", "piece": "P", "to": "c5", "fen": "qcnbrrbzck/pppppppppp/10/10/10/2P7/10/10/PP1PPPPPPP/KCZBRRBNCQ b - c3c4 0 1"},

            {"from": "d2", "piece": "P", "to": "d3", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/10/3P6/PPP1PPPPPP/KCZBRRBNCQ b - - 0 1"},
            {"from": "d2", "piece": "P", "to": "d4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/3P6/10/PPP1PPPPPP/KCZBRRBNCQ b - d3 0 1"},
            {"from": "d2", "piece": "P", "to": "d5", "fen": "qcnbrrbzck/pppppppppp/10/10/10/3P6/10/10/PPP1PPPPPP/KCZBRRBNCQ b - d3d4 0 1"},

            {"from": "e2", "piece": "P", "to": "e3", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/10/4P5/PPPP1PPPPP/KCZBRRBNCQ b - - 0 1"},
            {"from": "e2", "piece": "P", "to": "e4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/4P5/10/PPPP1PPPPP/KCZBRRBNCQ b - e3 0 1"},
            {"from": "e2", "piece": "P", "to": "e5", "fen": "qcnbrrbzck/pppppppppp/10/10/10/4P5/10/10/PPPP1PPPPP/KCZBRRBNCQ b - e3e4 0 1"},

            {"from": "f2", "piece": "P", "to": "f3", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/10/5P4/PPPPP1PPPP/KCZBRRBNCQ b - - 0 1"},
            {"from": "f2", "piece": "P", "to": "f4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/5P4/10/PPPPP1PPPP/KCZBRRBNCQ b - f3 0 1"},
            {"from": "f2", "piece": "P", "to": "f5", "fen": "qcnbrrbzck/pppppppppp/10/10/10/5P4/10/10/PPPPP1PPPP/KCZBRRBNCQ b - f3f4 0 1"},

            {"from": "g2", "piece": "P", "to": "g3", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/10/6P3/PPPPPP1PPP/KCZBRRBNCQ b - - 0 1"},
            {"from": "g2", "piece": "P", "to": "g4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/6P3/10/PPPPPP1PPP/KCZBRRBNCQ b - g3 0 1"},
            {"from": "g2", "piece": "P", "to": "g5", "fen": "qcnbrrbzck/pppppppppp/10/10/10/6P3/10/10/PPPPPP1PPP/KCZBRRBNCQ b - g3g4 0 1"},

            {"from": "h2", "piece": "P", "to": "h3", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/10/7P2/PPPPPPP1PP/KCZBRRBNCQ b - - 0 1"},
            {"from": "h2", "piece": "P", "to": "h4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/7P2/10/PPPPPPP1PP/KCZBRRBNCQ b - h3 0 1"},
            {"from": "h2", "piece": "P", "to": "h5", "fen": "qcnbrrbzck/pppppppppp/10/10/10/7P2/10/10/PPPPPPP1PP/KCZBRRBNCQ b - h3h4 0 1"},

            {"from": "i2", "piece": "P", "to": "i3", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/10/8P1/PPPPPPPP1P/KCZBRRBNCQ b - - 0 1"},
            {"from": "i2", "piece": "P", "to": "i4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/8P1/10/PPPPPPPP1P/KCZBRRBNCQ b - i3 0 1"},
            {"from": "i2", "piece": "P", "to": "i5", "fen": "qcnbrrbzck/pppppppppp/10/10/10/8P1/10/10/PPPPPPPP1P/KCZBRRBNCQ b - i3i4 0 1"},

            {"from": "j2", "piece": "P", "to": "j3", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/10/9P/PPPPPPPPP1/KCZBRRBNCQ b - - 0 1"},
            {"from": "j2", "piece": "P", "to": "j4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/9P/10/PPPPPPPPP1/KCZBRRBNCQ b - j3 0 1"},
            {"from": "j2", "piece": "P", "to": "j5", "fen": "qcnbrrbzck/pppppppppp/10/10/10/9P/10/10/PPPPPPPPP1/KCZBRRBNCQ b - j3j4 0 1"},

            {"from": "b1", "piece": "C", "to": "c4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/2C7/10/PPPPPPPPPP/K1ZBRRBNCQ b - - 1 1"},
            {"from": "b1", "piece": "C", "to": "a4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/C9/10/PPPPPPPPPP/K1ZBRRBNCQ b - - 1 1"},

            {"from": "c1", "piece": "Z", "to": "e4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/4Z5/10/PPPPPPPPPP/KC1BRRBNCQ b - - 1 1"},
            {"from": "c1", "piece": "Z", "to": "f3", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/10/5Z4/PPPPPPPPPP/KC1BRRBNCQ b - - 1 1"},
            {"from": "c1", "piece": "Z", "to": "a4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/Z9/10/PPPPPPPPPP/KC1BRRBNCQ b - - 1 1"},

            {"from": "h1", "piece": "N", "to": "i3", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/10/8N1/PPPPPPPPPP/KCZBRRB1CQ b - - 1 1"},
            {"from": "h1", "piece": "N", "to": "g3", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/10/6N3/PPPPPPPPPP/KCZBRRB1CQ b - - 1 1"},

            {"from": "i1", "piece": "C", "to": "j4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/9C/10/PPPPPPPPPP/KCZBRRBN1Q b - - 1 1"},
            {"from": "i1", "piece": "C", "to": "h4", "fen": "qcnbrrbzck/pppppppppp/10/10/10/10/7C2/10/PPPPPPPPPP/KCZBRRBN1Q b - - 1 1"}
        ])

        rules.load({"fen": "8k1/2ppq1r1z1/6PQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 b - - 3 47"})
            # logger.log({function: "rules-spec",  description: "test get_legal_moves",  data: {legalMoves: rules.get_legal_moves()}})
        self.assertEqual(rules.get_legal_moves(),  [
            {"from":"i10", "piece":"k", "to":"j10", "fen":"9k/2ppq1r1z1/6PQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"i10", "piece":"k", "to":"j9", "fen":"10/2ppq1r1zk/6PQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            
            {"from":"c9", "piece":"p", "to":"c8", "fen":"8k1/3pq1r1z1/2p3PQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 0 48"},
            {"from":"c9", "piece":"p", "to":"c7", "fen":"8k1/3pq1r1z1/6PQ2/2p4N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - c8 0 48"},
            {"from":"c9", "piece":"p", "to":"c6", "fen":"8k1/3pq1r1z1/6PQ2/7N1p/r1p7/5p4/ppP7/1P2P5/PB1P6/2KB6 w - c8c7 0 48"},
            
            {"from":"d9", "piece":"p", "to":"d8", "fen":"8k1/2p1q1r1z1/3p2PQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 0 48"},
            {"from":"d9", "piece":"p", "to":"d7", "fen":"8k1/2p1q1r1z1/6PQ2/3p3N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - d8 0 48"},
            {"from":"d9", "piece":"p", "to":"d6", "fen":"8k1/2p1q1r1z1/6PQ2/7N1p/r2p6/5p4/ppP7/1P2P5/PB1P6/2KB6 w - d8d7 0 48"},
            
            {"from":"e9", "piece":"q", "to":"f10", "fen":"5q2k1/2pp2r1z1/6PQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"f9", "fen":"8k1/2pp1qr1z1/6PQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"f8", "fen":"8k1/2pp2r1z1/5qPQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"g7", "fen":"8k1/2pp2r1z1/6PQ2/6qN1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"h6", "fen":"8k1/2pp2r1z1/6PQ2/7N1p/r6q2/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"i5", "fen":"8k1/2pp2r1z1/6PQ2/7N1p/r9/5p2q1/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"j4", "fen":"8k1/2pp2r1z1/6PQ2/7N1p/r9/5p4/ppP6q/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"e8", "fen":"8k1/2pp2r1z1/4q1PQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"e7", "fen":"8k1/2pp2r1z1/6PQ2/4q2N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"e6", "fen":"8k1/2pp2r1z1/6PQ2/7N1p/r3q5/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"e5", "fen":"8k1/2pp2r1z1/6PQ2/7N1p/r9/4qp4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"e4", "fen":"8k1/2pp2r1z1/6PQ2/7N1p/r9/5p4/ppP1q5/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"e3", "capture":True, "fen":"8k1/2pp2r1z1/6PQ2/7N1p/r9/5p4/ppP7/1P2q5/PB1P6/2KB6 w - - 0 48"},
            {"from":"e9", "piece":"q", "to":"d8", "fen":"8k1/2pp2r1z1/3q2PQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"c7", "fen":"8k1/2pp2r1z1/6PQ2/2q4N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"b6", "fen":"8k1/2pp2r1z1/6PQ2/7N1p/rq8/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"a5", "fen":"8k1/2pp2r1z1/6PQ2/7N1p/r9/q4p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"d10", "fen":"3q4k1/2pp2r1z1/6PQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"e9", "piece":"q", "to":"e10", "fen":"4q3k1/2pp2r1z1/6PQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            
            {"from":"g9", "piece":"r", "to":"h9", "fen":"8k1/2ppq2rz1/6PQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"g9", "piece":"r", "to":"g8", "capture":True, "fen":"8k1/2ppq3z1/6rQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 0 48"},
            {"from":"g9", "piece":"r", "to":"f9", "fen":"8k1/2ppqr2z1/6PQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"g9", "piece":"r", "to":"g10", "fen":"6r1k1/2ppq3z1/6PQ2/7N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            
            {"from":"i9", "piece":"z", "to":"g6", "fen":"8k1/2ppq1r3/6PQ2/7N1p/r5z3/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"i9", "piece":"z", "to":"f7", "fen":"8k1/2ppq1r3/6PQ2/5z1N1p/r9/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            
            {"from":"j7", "piece":"p", "to":"j6", "fen":"8k1/2ppq1r1z1/6PQ2/7N2/r8p/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 0 48"},
            {"from":"j7", "piece":"p", "to":"j5", "fen":"8k1/2ppq1r1z1/6PQ2/7N2/r9/5p3p/ppP7/1P2P5/PB1P6/2KB6 w - j6 0 48"},
            
            {"from":"a6", "piece":"r", "to":"b6", "fen":"8k1/2ppq1r1z1/6PQ2/7N1p/1r8/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"a6", "piece":"r", "to":"c6", "fen":"8k1/2ppq1r1z1/6PQ2/7N1p/2r7/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"a6", "piece":"r", "to":"d6", "fen":"8k1/2ppq1r1z1/6PQ2/7N1p/3r6/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"a6", "piece":"r", "to":"e6", "fen":"8k1/2ppq1r1z1/6PQ2/7N1p/4r5/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"a6", "piece":"r", "to":"f6", "fen":"8k1/2ppq1r1z1/6PQ2/7N1p/5r4/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"a6", "piece":"r", "to":"g6", "fen":"8k1/2ppq1r1z1/6PQ2/7N1p/6r3/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"a6", "piece":"r", "to":"h6", "fen":"8k1/2ppq1r1z1/6PQ2/7N1p/7r2/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"a6", "piece":"r", "to":"i6", "fen":"8k1/2ppq1r1z1/6PQ2/7N1p/8r1/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"a6", "piece":"r", "to":"j6", "fen":"8k1/2ppq1r1z1/6PQ2/7N1p/9r/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"a6", "piece":"r", "to":"a5", "fen":"8k1/2ppq1r1z1/6PQ2/7N1p/10/r4p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"a6", "piece":"r", "to":"a7", "fen":"8k1/2ppq1r1z1/6PQ2/r6N1p/10/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"a6", "piece":"r", "to":"a8", "fen":"8k1/2ppq1r1z1/r5PQ2/7N1p/10/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"a6", "piece":"r", "to":"a9", "fen":"8k1/r1ppq1r1z1/6PQ2/7N1p/10/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            {"from":"a6", "piece":"r", "to":"a10", "fen":"r7k1/2ppq1r1z1/6PQ2/7N1p/10/5p4/ppP7/1P2P5/PB1P6/2KB6 w - - 4 48"},
            
            {"from":"f5", "piece":"p", "to":"f4", "fen":"8k1/2ppq1r1z1/6PQ2/7N1p/r9/10/ppP2p4/1P2P5/PB1P6/2KB6 w - - 0 48"},
            
            {"from":"a4", "piece":"p", "to":"b3", "capture":True, "fen":"8k1/2ppq1r1z1/6PQ2/7N1p/r9/5p4/1pP7/1p2P5/PB1P6/2KB6 w - - 0 48"},
            {"from":"a4", "piece":"p", "to":"a3", "fen":"8k1/2ppq1r1z1/6PQ2/7N1p/r9/5p4/1pP7/pP2P5/PB1P6/2KB6 w - - 0 48"}
        ])

        # get all legal moves for a certain position (castling)

        rules.load({"fen": "r4k3r/9p/10/10/10/10/10/10/P9/R4K3R w KQkq - 0 1"})

        self.assertEqual(rules.get_legal_moves(), [
            {"from": "a2", "piece": "P", "to": "a3", "fen": "r4k3r/9p/10/10/10/10/10/P9/10/R4K3R b KQkq - 0 1"},
            {"from": "a2", "piece": "P", "to": "a4", "fen": "r4k3r/9p/10/10/10/10/P9/10/10/R4K3R b KQkq a3 0 1"},
            {"from": "a2", "piece": "P", "to": "a5", "fen": "r4k3r/9p/10/10/10/P9/10/10/10/R4K3R b KQkq a3a4 0 1"},
            
            {"from": "a1", "piece": "R", "to": "b1", "fen": "r4k3r/9p/10/10/10/10/10/10/P9/1R3K3R b Kkq - 1 1"},
            {"from": "a1", "piece": "R", "to": "c1", "fen": "r4k3r/9p/10/10/10/10/10/10/P9/2R2K3R b Kkq - 1 1"},
            {"from": "a1", "piece": "R", "to": "d1", "fen": "r4k3r/9p/10/10/10/10/10/10/P9/3R1K3R b Kkq - 1 1"},
            {"from": "a1", "piece": "R", "to": "e1", "fen": "r4k3r/9p/10/10/10/10/10/10/P9/4RK3R b Kkq - 1 1"},
            
            {"from": "f1", "piece": "K", "to": "g2", "fen": "r4k3r/9p/10/10/10/10/10/10/P5K3/R8R b kq - 1 1"},
            {"from": "f1", "piece": "K", "to": "g1", "fen": "r4k3r/9p/10/10/10/10/10/10/P9/R5K2R b kq - 1 1"},
            {"from": "f1", "piece": "K", "to": "e1", "fen": "r4k3r/9p/10/10/10/10/10/10/P9/R3K4R b kq - 1 1"},
            {"from": "f1", "piece": "K", "to": "e2", "fen": "r4k3r/9p/10/10/10/10/10/10/P3K5/R8R b kq - 1 1"},
            {"from": "f1", "piece": "K", "to": "f2", "fen": "r4k3r/9p/10/10/10/10/10/10/P4K4/R8R b kq - 1 1"},
            
            {"from": "f1", "piece": "K", "to": "h1", "castling": "O-O", "fen": "r4k3r/9p/10/10/10/10/10/10/P9/R5RK2 b kq - 1 1"},
            {"from": "f1", "piece": "K", "to": "d1", "castling": "O-O-O", "fen": "r4k3r/9p/10/10/10/10/10/10/P9/3KR4R b kq - 1 1"},
            
            {"from": "j1", "piece": "R", "to": "i1", "fen": "r4k3r/9p/10/10/10/10/10/10/P9/R4K2R1 b Qkq - 1 1"},
            {"from": "j1", "piece": "R", "to": "h1", "fen": "r4k3r/9p/10/10/10/10/10/10/P9/R4K1R2 b Qkq - 1 1"},
            {"from": "j1", "piece": "R", "to": "g1", "fen": "r4k3r/9p/10/10/10/10/10/10/P9/R4KR3 b Qkq - 1 1"},
            
            {"from": "j1", "piece": "R", "to": "j2", "fen": "r4k3r/9p/10/10/10/10/10/10/P8R/R4K4 b Qkq - 1 1"},
            {"from": "j1", "piece": "R", "to": "j3", "fen": "r4k3r/9p/10/10/10/10/10/9R/P9/R4K4 b Qkq - 1 1"},
            {"from": "j1", "piece": "R", "to": "j4", "fen": "r4k3r/9p/10/10/10/10/9R/10/P9/R4K4 b Qkq - 1 1"},
            {"from": "j1", "piece": "R", "to": "j5", "fen": "r4k3r/9p/10/10/10/9R/10/10/P9/R4K4 b Qkq - 1 1"},
            {"from": "j1", "piece": "R", "to": "j6", "fen": "r4k3r/9p/10/10/9R/10/10/10/P9/R4K4 b Qkq - 1 1"},
            {"from": "j1", "piece": "R", "to": "j7", "fen": "r4k3r/9p/10/9R/10/10/10/10/P9/R4K4 b Qkq - 1 1"},
            {"from": "j1", "piece": "R", "to": "j8", "fen": "r4k3r/9p/9R/10/10/10/10/10/P9/R4K4 b Qkq - 1 1"},
            
            {"from": "j1", "piece": "R", "to": "j9", "capture": True, "fen": "r4k3r/9R/10/10/10/10/10/10/P9/R4K4 b Qkq - 0 1"}
        ])
        
        # get all legal moves for a certain position (promotions)

        rules.load({"fen": "7K2/6P2P/10/10/3b6/10/10/10/10/4k5 w - - 0 1"})

        self.assertEqual(rules.get_legal_moves(), [
            {"from": "h10", "piece": "K", "to": "i10", "fen": "8K1/6P2P/10/10/3b6/10/10/10/10/4k5 b - - 1 1"},
            {"from": "h10", "piece": "K", "to": "i9", "fen": "10/6P1KP/10/10/3b6/10/10/10/10/4k5 b - - 1 1"},
            {"from": "h10", "piece": "K", "to": "h9", "fen": "10/6PK1P/10/10/3b6/10/10/10/10/4k5 b - - 1 1"},
            {"from": "h10", "piece": "K", "to": "g10", "fen": "6K3/6P2P/10/10/3b6/10/10/10/10/4k5 b - - 1 1"},
            
            {"from": "j9", "piece": "P", "to": "j10", "promotion": "C", "fen": "7K1C/6P3/10/10/3b6/10/10/10/10/4k5 b - - 0 1"},
            {"from": "j9", "piece": "P", "to": "j10", "promotion": "Z", "fen": "7K1Z/6P3/10/10/3b6/10/10/10/10/4k5 b - - 0 1"},
            {"from": "j9", "piece": "P", "to": "j10", "promotion": "N", "fen": "7K1N/6P3/10/10/3b6/10/10/10/10/4k5 b - - 0 1"},
            {"from": "j9", "piece": "P", "to": "j10", "promotion": "B", "fen": "7K1B/6P3/10/10/3b6/10/10/10/10/4k5 b - - 0 1"},
            {"from": "j9", "piece": "P", "to": "j10", "promotion": "F", "fen": "7K1F/6P3/10/10/3b6/10/10/10/10/4k5 b - - 0 1"},
            {"from": "j9", "piece": "P", "to": "j10", "promotion": "R", "fen": "7K1R/6P3/10/10/3b6/10/10/10/10/4k5 b - - 0 1"},
            {"from": "j9", "piece": "P", "to": "j10", "promotion": "Q", "fen": "7K1Q/6P3/10/10/3b6/10/10/10/10/4k5 b - - 0 1"}
        ])
                             
if __name__ == "__main__":
    unittest.main()
    
# renaming from legalMoves to get_legal_moves_for_square #_261102
# renaming from legalMovesAll to get_legal_moves #_261102
