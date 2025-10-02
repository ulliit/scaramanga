function Settings(){

    // engine opponent for the nn. learn from hard opponents ... because a dumb opponent (a untrained nn in self-play)) might incentivize dumb moves if they lead to success due to its dumb answer. like in real life ...
    // sample: for 2k7/10/2K7/10/10/10/10/R9/10/10 w - - 0 1 the dumb 1. Kd8 would lead to success if black answers with 1. ... Kd10 and white now finds 2. Ra10#
    
    var drawValue = -0.5; // is 0 for an equal start position. if white has an advantage, it should be a value < 0 and >= -0.5 black vice versa.
    
    this.engine = {hash: 512, evaluation: {bNote: true}, installationId: "68a93411f061142a2c95528b", multiPV: {1: [6, 25], 5: [4, 40]}, name: "Fairy-Stockfish faf18f3", path: "Documents/cafe64/engines/sf_fairy_945daf1_asahi/src/stockfish", randomizedTimeFractionFirstMove: 0.05, threads: 2, time: 50, uciOptions: [{name: "VariantPath", value: "variants.ini"}, {name: "UCI_Variant", value: "10x10chess"}], username: "ScaramangaBot"}; 

    this.fens = ["10/10/10/5k4/10/5R4/10/4K5/10/10 b - - 1 1","10/10/10/4k5/10/5R4/10/4K5/10/10 w - - 2 2","10/10/10/4k5/10/5R4/3K6/10/10/10 b - - 3 2","10/10/10/10/4k5/5R4/3K6/10/10/10 w - - 4 3","10/10/10/10/4k5/10/3K6/10/10/5R4 b - - 5 3","10/10/10/3k6/10/10/3K6/10/10/5R4 w - - 6 4","10/10/10/3k6/10/3K6/10/10/10/5R4 b - - 7 4","10/10/10/4k5/10/3K6/10/10/10/5R4 w - - 8 5","10/10/10/4k5/10/3K1R4/10/10/10/10 b - - 9 5","10/10/10/3k6/10/3K1R4/10/10/10/10 w - - 10 6","10/10/10/3k6/10/3KR5/10/10/10/10 b - - 11 6","10/10/10/2k7/10/3KR5/10/10/10/10 w - - 12 7","10/10/10/2k1R5/10/3K6/10/10/10/10 b - - 13 7","10/10/3k6/4R5/10/3K6/10/10/10/10 w - - 14 8","10/10/3k6/4R5/3K6/10/10/10/10/10 b - - 15 8","10/10/2k7/4R5/3K6/10/10/10/10/10 w - - 16 9","10/10/2k7/3R6/3K6/10/10/10/10/10 b - - 17 9","10/10/1k8/3R6/3K6/10/10/10/10/10 w - - 18 10","10/10/1k8/2R7/3K6/10/10/10/10/10 b - - 19 10","10/1k8/10/2R7/3K6/10/10/10/10/10 w - - 20 11","10/1k8/10/2RK6/10/10/10/10/10/10 b - - 21 11","10/10/1k8/2RK6/10/10/10/10/10/10 w - - 22 12","10/10/1k1K6/2R7/10/10/10/10/10/10 b - - 23 12","10/1k8/3K6/2R7/10/10/10/10/10/10 w - - 24 13","10/1k8/3K6/1R8/10/10/10/10/10/10 b - - 25 13","10/10/k2K6/1R8/10/10/10/10/10/10 w - - 26 14","10/10/k1K7/1R8/10/10/10/10/10/10 b - - 27 14","10/k9/2K7/1R8/10/10/10/10/10/10 w - - 28 15","10/k9/2K7/R9/10/10/10/10/10/10 b - - 29 15","1k8/10/2K7/R9/10/10/10/10/10/10 w - - 30 16","1k8/10/2K7/10/10/10/10/R9/10/10 b - - 31 16","2k7/10/2K7/10/10/10/10/R9/10/10 w - - 32 17"] // fens to train from
    
    this.halfMoveMax = 10;
    
    this.maxGames = 1000; // number of games fairy stockfish has to play in a trainsv (supervised) session
    
    this.netPlays = "w";
    
    this.startFen = function(){
    
        return randomKRKFen();

        // return "qcnbrrbzck/pppppppppp/10/10/10/10/10/10/PPPPPPPPPP/KCZBRRBNCQ w - - 0 1";
        
   }

    this.rounds = 5; // number of rounds to circle trough the fens array provided
    
    this.successRate = 0.8; // percentage of training games where the position was solved succesfully. note: this works currently only for a position that is won for white










    
    // helper functions
    
    var rules = new (require('../client/js/chess100.js').Chess)(); //_t

	var randomKRKFen = function() {

		const files = "abcdefghij"; // 10x10 board
		const ranks = [...Array(10).keys()].map(i => (10 - i).toString());

		function squareToCoord(file, rank) {
		    return file + rank;
		}

		function randomSquare() {
		    return files[Math.floor(Math.random() * 10)] + ranks[Math.floor(Math.random() * 10)];
		}

		// helper to check if two kings are adjacent
		function kingsAdjacent(k1, k2) {
		    const f1 = files.indexOf(k1[0]);
		    const r1 = parseInt(k1[1]);
		    const f2 = files.indexOf(k2[0]);
		    const r2 = parseInt(k2[1]);
		    return Math.abs(f1 - f2) <= 1 && Math.abs(r1 - r2) <= 1;
		}

		let wK, bK, wR;
		do {
		    wK = randomSquare();
		    bK = randomSquare();
		} while (wK === bK || kingsAdjacent(wK, bK));

		do {
		    wR = randomSquare();
		} while (wR === wK || wR === bK);

		// Construct FEN-like string for 10x10
		// Start with empty 10x10
		let board = Array.from({ length: 10 }, () => Array(10).fill("1"));

		function place(square, piece) {
		    const f = files.indexOf(square[0]);
		    const r = 10 - parseInt(square[1]);
		    board[r][f] = piece;
		}

		place(wK, "K");
		place(bK, "k");
		place(wR, "R");

		// Compress each rank (like FEN)
		const fenRanks = board.map(row => {
		    let out = "";
		    let count = 0;
		    for (const cell of row) {
		        if (cell === "1") {
		            count++;
		        } else {
		            if (count > 0) {
		                out += count;
		                count = 0;
		            }
		            out += cell;
		        }
		    }
		    if (count > 0) out += count;
		    return out;
		});

		var fen = fenRanks.join("/") + " w - - 0 1";
		
	    var check = rules.load({fen: fen});
		
		if(check != undefined){
		
		    // if the fen is invalid due to a bug, repeat to get a valid one
		
		    return randomKRKFen();

		   // throw "fen invalid: " + check
		
		}
		
		return fen
		
	}

}

exports.Settings = Settings;
