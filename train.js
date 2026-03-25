var boardEncoder = new (require('./boardencoder.js').BoardEncoder)({});
var fairyStockfish = new (require('../server/engine.js').Engine)({engine: settings.engine, multiPV: settings.engine.multiPV}); // using fairy stockfish evals as a guidance for accelatered learning. 1) if these evals are not wanted, a dummy class with an appropriate interface can be used here.
var logger = new (require('../client/js/log.js').logger);
var settings = new (require('./settings.js').Settings)({});
// var moveEncoder = new (require('./moveencoder.js').MoveEncoder)({});
var network = new (require('./network.js').Network)({});
var trainer = new (require('./trainer.js').Trainer)({});
var validator = new (require('./validator.js').Validator)();

var game = new(require("../client/js/moves.js").asMoves)(logger);

// choose a start position
// transform the position into an input vector that can be understood by the nn with boardEncoder.encode
// then create a nn using the input vector
// create a probabilities vector and a value for this position with nn.forward. each element of probabilities is the probability for a square pair to be the best move. value is the estimated score for the position BEFORE a move is made.
// detect which elements of the probabilities vector represent legal moves using the Chess instance
// pick one of the legal moves with a preference for the ones with better policy values
// add selected move to the history
// get a new position by executing the selected move
// loop through the steps above until we have a game result
// Assign result (+1 win / 0 draw / −1 loss).
// train the net: compute losses, update the weights with backpropagation
// start a new game and repeat all the steps above
// save net on hard disk after session

var history = []; // contains a history of all selected moves where vector, predictions, move index and the position score are saved
var drawValue = settings.drawValue;
var rounds = 0;
var gamesPlayed = 0;
var results = [];
var startFen = "";
var successRate = 0;
var numberOfGames = 300; // currently replaced by settings.successRate
var halfMoveClock = 0;
var halfMoveMax = settings.halfMoveMax;
const MOVE_PENALTY = 0.1;

network.loadFromDisk("scaramanga.json");

var currentFenIndex = settings.fens.length;

function newGame(){

	startFen = settings.startFen();
	
	startFen_ = startFen.split(" ");
	startFen_[4] = 0; // reset the half move clock
	startFen_[5] = 1; // reset the full move nr
	
	startFen = startFen_.join(" ");	 
	
	game.reset({initialFen: startFen, positions: []});
	halfMoveClock = 0;
	history = [];

}

function makeMove(){

    const fen = game.currentPosition().fen;
    const vector = boardEncoder.encode(fen); // fen encoded in a format suitable for network and training
    const predictions = network.forward({vector});
                                                            // logger.log({function: "nextMove", description: "after board vector generation", data: {boardVector: vector, predictions: predictions}});
    const validMoves = validator.getValidMoves({fen, probabilities: predictions.probabilities});
                                                            // logger.log({function: "nextMove", description: "after valid moves detection", data: {validMoves: validMoves}});
    const selectedMove = validator.pickByPolicy({validMoves: validMoves, successRate: successRate});
    
    halfMoveClock = parseInt(game.currentPosition().fen.split(" ")[4], 10);

    history.push({
            color: game.turn(),
         selectedMove: selectedMove, // for debugging
        moveIndex: selectedMove.index, // e.g. 324
        predictions: predictions,
        vector: vector // e.g. [0,0,1,0,...,1]
    });
                                                                 // console.log("turn", game.turn())
    selectedMove.move.internal = false;
    selectedMove.move.returnvalue = "";
    game.moveAdd(selectedMove.move);
    
    // ask Stockfish for an eval

    fairyStockfish.move({fen: fen, time: "movetime " + settings.engine.time});
        
}

fairyStockfish.on("engine ready", function() {

    console.log("Engine is ready. Starting self-play.");
    
    newGame();
    makeMove();

});

fairyStockfish.on("moved", function(o) {

    // add stockfish eval to the last move
    
    // o contains a best moves object. sample:

	//	{"bestMoves":[
	//	{"cp":91,"variant":["f2f5","e9e6","g1c5","f9f8","f5e6","i10h7","h1g3","e10e6","d2d5","b9b6","c5a3","e6h6","e2e5","h6h2","d1f3","a9a7","i1j4"],"move":"f2f5","nr":1},
	//	{"cp":82,"variant":["b1c4","c9c8","f2f5","f9f6","e2e4","h10f7","j2j5","e9e6","f5e6","e10e6","i1j4","j9j8","d2d5","e6e5","j4g3","b10c7"],"move":"b1c4","nr":2},
	//	{"cp":67,"variant":["e2e5","f9f6","f2f4","e9e7","h1g3","f6e5","f4e5","d10h6","g1e3","h6e3","f1f10","e10f10","e1e3","b10c7","c2c3","c10d8"],"move":"e2e5","nr":3}]}}
    
    history[history.length-1].sfPolicyTarget = validator.moveIndex({from: o.bestMoves[0].move.substring(0,2), to: o.bestMoves[0].move.substring(2,4));
   
    // Why cp/400? It's an empirically tuned scale factor that maps cp scores roughly to win probability (similar to logistic/Pawn=100 scaling in traditional engines).+400 cp ≈ +0.76 value (strong advantage)
    // +800 cp ≈ +0.96 value (very winning)
    // 0 cp ≈ 0.0 value (roughly equal)
    // -∞ cp (mate in few) → approaches -1
    // tanh bounds it nicely to [-1, +1], which matches your value head output (tanh activation, win/loss framing).

    history[history.length-1].sfValueTarget = Math.tanh(o.bestMoves[0].cp / 400) 

	if(game.gameStatus() === "running" && halfMoveClock < halfMoveMax) {
	
        makeMove();
	    
	}else{
	
		// Game finished, compute result
		
		let result = game.gameStatus();
		if (result === "1-0") result = 1;
		else if (result === "0-1") result = -1;
		else result = drawValue; // draw
		
		let numberOfResultsForWinRate = 200;
		
		results.push(result);
		if(results.length > numberOfResultsForWinRate) results.shift();
		
		let wins = 0;
		
		for(var i=0;i<results.length;i++){
		
		    if (results[i] === 1) wins++;
		
		}
		
//		var resultWithoutPenalization = result;

//		if(MOVE_PENALTY != -1 && result == 1){
//		
//			let penalty = 1 - MOVE_PENALTY * game.currentPosition().moveNr; // penalize long mates (encourages shorter mating sequences).
//			penalty = Math.max(0, penalty);  // no negative -> the penalized result won't flip sign in extreme cases.
//			result = result * penalty;
//			
//		}

		trainer.trainFromGame({nn: network, history, result});
		gamesPlayed++;
		
		if((gamesPlayed / 200) == Math.trunc(gamesPlayed / 200)){
		
		    // save nn to disk after a certain number of games
		
		    network.saveToDisk("scaramanga.json");
		    
		}
		
        successRate = (wins / results.length);
													logger.log({
														function: "trainingLoop",
														description: "training game finished",
														data: {
															result: result,
															resultWithoutPenalization: resultWithoutPenalization,
															fenIndex: currentFenIndex,
															fensStudied: fensStudied,
															finalFen: game.currentPosition().fen,
															// startFen: startFen,
															successRate: successRate,
															pgn: game.positionsToPgn({positions: game.getPositions()}).replace("&#189", "") + (game.currentPosition().kingState || "max halfmove nr of " + halfMoveMax + " exceeded.")
														}
													});
	    
        if(successRate > settings.successRate || results.length >= 50) {
        
            rounds = rounds + 1;
        
            if(rounds < settings.rounds){
            
                results = [];
                successRate = 0;
                
                newGame();
                makeMove();
            
            }else{
 
				// Optional: run validation after training

				validator.validateNetwork({nn: network, numGames: 5, startFen});           
            
            }
            
        }else{
        
            newGame();
            makeMove();
                        
        }
        			
	}
    
});

// 1) "This is often called supervised + reinforcement hybrid, imitation learning bootstrap, teacher-student distillation, or value/policy labeling ..."
// https://x.com/i/grok?conversation=2034098359535898775




