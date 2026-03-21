function Settings(){


    
    this.drawValue = -0.5; // is 0 for an equal start position. if white has an advantage, it should be a value < 0 and >= -0.5 black vice versa.

    // engine opponent for the nn. learn from hard opponents ... because a dumb opponent (a untrained nn in self-play)) might incentivize dumb moves if they lead to success due to its dumb answer. like in real life ...
    // sample: for 2k7/10/2K7/10/10/10/10/R9/10/10 w - - 0 1 the dumb 1. Kd8 would lead to success if black answers with 1. ... Kd10 and white now finds 2. Ra10#
    //arguments against an engine opponent: learning is assymetrical - the nn only learns the patterns for one side and if the outcome is the only reward, learning will take forever since stockfish will allow a dumb nn very few wins even for a completely won start position like k+r vs k.
    // on the other hand for nn vs nn the signaling can be weak as well: if one side blunders way the rook in a k+r vs k+r and the game ends as a draw anyway because the nn is too dumb to mate this, the blunder won't be penalized.
    // so a better approach is probably to use stockfish not as an opponent but as a supervisor: each net move is valued by stockfish - but not too strict to let the net develop its own style.
        
    this.engine = {hash: 512, evaluation: {bNote: true}, installationId: "68a93411f061142a2c95528b", multiPV: {1: [6, 25], 5: [4, 40]}, name: "Fairy-Stockfish faf18f3", path: "Documents/cafe64/engines/sf_fairy_945daf1_asahi/src/stockfish", randomizedTimeFractionFirstMove: 0.05, threads: 2, time: 50, uciOptions: [{name: "VariantPath", value: "variants.ini"}, {name: "UCI_Variant", value: "10x10chess"}], username: "ScaramangaBot"}; 

//    this.fens = ["10/10/10/5k4/10/5R4/10/4K5/10/10 b - - 1 1","10/10/10/4k5/10/5R4/10/4K5/10/10 w - - 2 2","10/10/10/4k5/10/5R4/3K6/10/10/10 b - - 3 2","10/10/10/10/4k5/5R4/3K6/10/10/10 w - - 4 3","10/10/10/10/4k5/10/3K6/10/10/5R4 b - - 5 3","10/10/10/3k6/10/10/3K6/10/10/5R4 w - - 6 4","10/10/10/3k6/10/3K6/10/10/10/5R4 b - - 7 4","10/10/10/4k5/10/3K6/10/10/10/5R4 w - - 8 5","10/10/10/4k5/10/3K1R4/10/10/10/10 b - - 9 5","10/10/10/3k6/10/3K1R4/10/10/10/10 w - - 10 6","10/10/10/3k6/10/3KR5/10/10/10/10 b - - 11 6","10/10/10/2k7/10/3KR5/10/10/10/10 w - - 12 7","10/10/10/2k1R5/10/3K6/10/10/10/10 b - - 13 7","10/10/3k6/4R5/10/3K6/10/10/10/10 w - - 14 8","10/10/3k6/4R5/3K6/10/10/10/10/10 b - - 15 8","10/10/2k7/4R5/3K6/10/10/10/10/10 w - - 16 9","10/10/2k7/3R6/3K6/10/10/10/10/10 b - - 17 9","10/10/1k8/3R6/3K6/10/10/10/10/10 w - - 18 10","10/10/1k8/2R7/3K6/10/10/10/10/10 b - - 19 10","10/1k8/10/2R7/3K6/10/10/10/10/10 w - - 20 11","10/1k8/10/2RK6/10/10/10/10/10/10 b - - 21 11","10/10/1k8/2RK6/10/10/10/10/10/10 w - - 22 12","10/10/1k1K6/2R7/10/10/10/10/10/10 b - - 23 12","10/1k8/3K6/2R7/10/10/10/10/10/10 w - - 24 13","10/1k8/3K6/1R8/10/10/10/10/10/10 b - - 25 13","10/10/k2K6/1R8/10/10/10/10/10/10 w - - 26 14","10/10/k1K7/1R8/10/10/10/10/10/10 b - - 27 14","10/k9/2K7/1R8/10/10/10/10/10/10 w - - 28 15","10/k9/2K7/R9/10/10/10/10/10/10 b - - 29 15","1k8/10/2K7/R9/10/10/10/10/10/10 w - - 30 16","1k8/10/2K7/10/10/10/10/R9/10/10 b - - 31 16","2k7/10/2K7/10/10/10/10/R9/10/10 w - - 32 17"] // fens to train from
    
    this.halfMoveMax = 2;
    
    this.maxGames = 1000; // number of games fairy stockfish has to play in a trainsv (supervised) session
    
    this.netPlays = "w";
    
    this.startFen = function(){
    
        return randomKRvKR_FEN();

        // return "qcnbrrbzck/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/KCZBRRBNCQ w - - 0 1";
        
   }

    this.rounds = 5; // number of rounds to circle trough the fens array provided
    
    this.successRate = 0.8; // percentage of training games where the position was solved successfully. note: this works currently only for a position that is won for white










    
// Returns a random legal FEN for K+R vs K+R on 10×10 (White to move)
// Both sides have exactly king + rook, empty board otherwise
function randomKRvKR_FEN() {
  const files = 'abcdefghij';
  const ranks = '1234567890';

  const rand = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
  const square = (f, r) => files[f] + ranks[r];

  let wk, wr, bk, br;
  let attempts = 0;
  const MAX_ATTEMPTS = 1000;

  while (attempts++ < MAX_ATTEMPTS) {
    // White pieces
    wk = { file: rand(0,9), rank: rand(0,9) };
    do {
      wr = { file: rand(0,9), rank: rand(0,9) };
    } while (wr.file === wk.file && wr.rank === wk.rank);

    // Black pieces
    bk = { file: rand(0,9), rank: rand(0,9) };
    do {
      br = { file: rand(0,9), rank: rand(0,9) };
    } while (br.file === bk.file && br.rank === bk.rank);

    // Safety checks
    const kingsAdjacent = Math.max(Math.abs(wk.file - bk.file), Math.abs(wk.rank - bk.rank)) <= 1;
    const whiteKingAttackedByBlackRook = (bk.file === br.file || bk.rank === br.rank) && 
                                         (wk.file === br.file || wk.rank === br.rank);
    const blackKingAttackedByWhiteRook = (wk.file === wr.file || wk.rank === wr.rank) && 
                                         (bk.file === wr.file || bk.rank === wr.rank);
    const wrAttackedByBK = Math.max(Math.abs(wr.file - bk.file), Math.abs(wr.rank - bk.rank)) <= 1;
    const brAttackedByWK = Math.max(Math.abs(br.file - wk.file), Math.abs(br.rank - wk.rank)) <= 1;

    if (!kingsAdjacent &&
        !whiteKingAttackedByBlackRook &&
        !blackKingAttackedByWhiteRook &&
        !wrAttackedByBK &&
        !brAttackedByWK) {
      break;
    }
  }

  if (attempts >= MAX_ATTEMPTS) {
    // Safe fallback (you can improve this later)
    return "4k3R3/8/8/8/8/8/8/4K2r w - - 0 1".replace(/8/g, "..........");
  }

  // Build board (rank 10 = index 0)
  const board = Array(10).fill().map(() => Array(10).fill('.'));

  board[9 - wk.rank][wk.file] = 'K';
  board[9 - wr.rank][wr.file] = 'R';
  board[9 - bk.rank][bk.file] = 'k';
  board[9 - br.rank][br.file] = 'r';

  // Convert to FEN
  const fenRows = [];
  for (let r = 0; r < 10; r++) {
    let row = '';
    let empty = 0;
    for (let f = 0; f < 10; f++) {
      const sq = board[r][f];
      if (sq === '.') {
        empty++;
      } else {
        if (empty > 0) { row += empty; empty = 0; }
        row += sq;
      }
    }
    if (empty > 0) row += empty;
    fenRows.push(row);
  }

  return `${fenRows.join('/')} w - - 0 1`;
}

}

exports.Settings = Settings;
