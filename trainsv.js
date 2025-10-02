//	ok another plan to speed things up

//	stockfish plays a game against itself

//	for every move it randomly chooses between all candidate moves not, let's say, 0.5 pawn units worse than the best move.

//	the nn gets the position and all candidate moves get the same amount on reinforcement, no matter if stockfish detected a quality difference.

//	net is trainde after each move.

//	a training after the game has finished might be added.

var logger = new (require('../client/js/log.js').logger);
var settings = new (require('./settings.js').Settings)({});
var boardEncoder = new (require('./boardencoder.js').BoardEncoder)({});
var chessToolsClient = Object.create(require('../client/js/chesstoolsclient.js').chessToolsClient);
var game = new(require("../client/js/moves.js").asMoves)(logger);
var fairyStockfish = new (require('../server/engine.js').Engine)({engine: settings.engine, multiPV: settings.engine.multiPV});
var moveEncoder = new (require('./moveencoder.js').MoveEncoder)({});
var network = new (require('./network.js').Network)({});
var validator = new (require('./validator.js').Validator)();

network.loadFromDisk("scaramanga_sl.json");

// Replay buffer for one game

var gameHistory = [];
var numberOfGames = 0;

function requestMove() {

    // ask Stockfish for candidate moves

    fairyStockfish.move({ fen: game.currentPosition().fen, time: `movetime ${settings.engine.time}` });
    
}

function startGame(){
    
    game.reset({initialFen: settings.startFen(), positions: []});

    gameHistory = [];
    
    requestMove();

}

fairyStockfish.on("engine ready", function() {

    console.log("[TrainerSV] Engine is ready. Starting first game.");
    
    startGame();
    
});

fairyStockfish.on("moved", function(o) {

    // o contains a best moves object. sample:

	//	{"bestMoves":[
	//	{"cp":91,"variant":["f2f5","e9e6","g1c5","f9f8","f5e6","i10h7","h1g3","e10e6","d2d5","b9b6","c5a3","e6h6","e2e5","h6h2","d1f3","a9a7","i1j4"],"move":"f2f5","nr":1},
	//	{"cp":82,"variant":["b1c4","c9c8","f2f5","f9f6","e2e4","h10f7","j2j5","e9e6","f5e6","e10e6","i1j4","j9j8","d2d5","e6e5","j4g3","b10c7"],"move":"b1c4","nr":2},
	//	{"cp":67,"variant":["e2e5","f9f6","f2f4","e9e7","h1g3","f6e5","f4e5","d10h6","g1e3","h6e3","f1f10","e10f10","e1e3","b10c7","c2c3","c10d8"],"move":"e2e5","nr":3}]}}

    var bestMoves = o.bestMoves;
    
    // if the position is a clear win, select only the best move. this will prevent the nn from hovering around in trivial win situations (like k+r vs k).
    
    if(bestMoves[0].cp > 300){bestMoves = [bestMoves[0]]};

    // Convert Stockfish moves to your move format
    
    const candidates = bestMoves.map(m => chessToolsClient.convertMove({ move: m.move }));

    // pick one move randomly (to simulate varied Stockfish play)
    
    const chosen = candidates[Math.floor(Math.random() * candidates.length)];

    // Record this training step
    
    gameHistory.push({
    
        fen: game.currentPosition().fen, // <-- FEN of position BEFORE chosen move
        candidates: candidates,
        chosen: chosen
        
    });

    // apply the chosen move to the board
    
    game.moveAdd(chosen);

    // detect game over here
    
    if (game.gameStatus() != "running") {
    
        numberOfGames ++;
        finalizeGame();
        
    } else {
    
        requestMove();
        
    }
    
});

// --- Finish game and train network

function finalizeGame() {
                                                                                     console.log("[TrainerSV] Finalizing game...");
    for (const step of gameHistory) {
    
        // Encode candidate moves
        
        const moveIndices = step.candidates.map(c => moveEncoder.encode(c.from, c.to));
        const probabilitiesForMoves = Array(network.outputSize).fill(0);

        // Spread probability mass evenly across candidate moves
        
        for (const idx of moveIndices) {
        
            probabilitiesForMoves[idx] = 1 / moveIndices.length;
            
        }

        // Forward pass

	    const vector = boardEncoder.encode(step.fen); // fen encoded in a format suitable for network and training
	    const predictions = network.forward({vector});

        // No value training here (policy imitation only)
        
        const result = 0;

        // Backprop
        
        network.backprop(
            vector,
            probabilitiesForMoves,
            result,
            predictions,
            0.01 // small supervised LR
        );
    }

    console.log("[TrainerSV] Game training complete.");
    
    // Save network after training
    
    network.saveToDisk("scaramanga_sl.json");
    
    if(numberOfGames < settings.maxGames){
    
        // start next game
        
        startGame();
        
    }else{
    
        // run validation after training
    
        for(var i=0;i<5;i++){
        
            validator.validateNetwork({nn: network, numGames: 1, startFen: settings.startFen()});
           
        }
    
    }
    
}

