var boardEncoder = new (require('./boardencoder.js').BoardEncoder)({});
var logger = new (require('../client/js/log.js').logger);
var settings = new (require('./settings.js').Settings)({});
var fairyStockfish = new (require('../server/engine.js').Engine)({engine: settings.engine, multiPV: settings.engine.multiPV});
var moveEncoder = new (require('./moveencoder.js').MoveEncoder)({});
var network = new (require('./network.js').Network)({});
var trainer = new (require('./trainer.js').Trainer)({});
var validator = new (require('./validator.js').Validator)();

var game = new(require("../client/js/moves.js").asMoves)(logger);

// choose a start position
// transform the position into an input vector that can be understood by the nn with boardEncoder.encode
// create a nn using the input vector
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
var drawValue = -0.5; // is 0 for an equal start position. if white has an advantage, it should be a value < 0 and >= -1 black vice versa.
var fensStudied = 0;
var gamesPlayed = 0;
var startFen = "10/8k1/10/6R1K1/10/10/10/10/10/10 w - - 0 1";
var numberOfGames = 300; // currently replaced by settings.successRate
var halfMoveClock = 0;
var halfMoveMax = 2;
const MOVE_PENALTY = -1; // 0.002;

network.loadFromDisk("scaramanga.json");

var botMove = {empty: true};

var currentFenIndex = settings.fens.length;

function trainLoop(){

	while (fensStudied < (settings.rounds * settings.fens.length)){

		currentFenIndex = currentFenIndex -1;
		
		halfMoveMax = halfMoveMax + 2;
		
		if(currentFenIndex == -1){
		
		    currentFenIndex = settings.fens.length - 1;
		    halfMoveMax = 10;
		    
		}
		
		startFen = settings.fens[currentFenIndex];
		
		startFen_ = startFen.split(" ");
		startFen_[4] = 0; // reset the half move clock
		startFen_[5] = 1; // reset the full move nr
		
		startFen = startFen_.join(" ");
		
		console.log(startFen)
		
		let successRate = 0;
		
		let results = [];

		while (successRate < settings.successRate || results.length < 50) {

			game.reset({initialFen: startFen, positions: []});
			halfMoveClock = 0;
			history = [];

			while (game.gameStatus() === "running" && halfMoveClock < halfMoveMax) {
			
				const fen = game.currentPosition().fen;
				
				if(game.turn() == settings.netPlays){
				
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
					
				}else{
				
				    botIsThinking = true;
				
				    fairyStockfish.move({fen: fen, time: "movetime 500"});
				    
				    while(botMove.empty == true){
				    
				    
				    }
				    
				    game.moveAdd(botMove.move);
				    
				    botMove = {empty: true};
				
				}

			}

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
			
			var resultWithoutPenalization = result;

			if(MOVE_PENALTY != -1){
			
				let penalty = 1 - MOVE_PENALTY * game.currentPosition().moveNr; // penalize long mates (encourages shorter mating sequences).
				penalty = Math.max(0, penalty);  // no negative -> the penalized result won't flip sign in extreme cases.
				result = result * penalty;
				
			}

			trainer.trainFromGame({nn: network, history, result, resultWithoutPenalization});
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
				    result,
				    resultWithoutPenalization,
				    fenIndex: currentFenIndex,
				    fensStudied: fensStudied,
				    finalFen: game.currentPosition().fen,
				    startFen: startFen,
				    successRate: successRate,
				    pgn: game.positionsToPgn({positions: game.getPositions()}).replace("&#189", "") + (game.currentPosition().kingState || "max halfmove nr of " + halfMoveMax + " exceeded.")
				}
			});
			
		}

		fensStudied ++
		
	}

	// Save network after training
	network.saveToDisk("scaramanga.json");

	// Optional: run validation after training
	validateNetwork({nn: network, numGames: 5, startFen});
	
}

fairyStockfish.on("engine ready", function(o){
                                                        logger.log({function: "stockfish ready", description: "start", data: {o: o}})
   trainLoop();

});

fairyStockfish.on("moved", function(o){
                                                                   logger.log({function: "gBotMatch on moved", description: "start", data: {o: o}})
	var nextMove = {}

	if(o.bestMoves.length > 1){

		// multiPV > 1
		
		////// nextMove = moveSelector.chooseRandomMove({bestMoves: o.bestMoves, fen: gBot.sMoves.currentPosition().fen, multiPV: botSettings.multiPV, evaluation: botSettings.evaluation});
		o.bestMoves[0];

	}
	else{

		nextMove = o.bestMoves[0];

	}
	
	botMove = chessToolsClient.convertMove({move: nextMove.move});
	                                                                             logger.log({function: "fairy stockfish el moved", description: "after generating bot movee", data: {move: move, nextMove: nextMove, o: o}})
});

function validateNetwork({nn, numGames, startFen}) {

    // play a few gamess to see if things are improving

    let results = [];
    let validationGame = new (require("../client/js/moves.js").asMoves)(logger);

    for (let g = 0; g < numGames; g++) {
        validationGame.reset({initialFen: startFen, positions: []});
        let history = [];

        while (validationGame.gameStatus() === "running" &&
               validationGame.currentPosition().moveNr <= 100) {
            
            let fen = validationGame.currentPosition().fen;
            let vector = boardEncoder.encode(fen);
            let predictions = nn.forward({vector});
            let validMoves = validator.getValidMoves({fen, probabilities: predictions.probabilities});
            let selectedMove = validator.pickByPolicy({temperature: 0, validMoves: validMoves});

            validationGame.moveAdd(selectedMove.move);
        }

        let result = validationGame.gameStatus();
        results.push({
			result: result,
			finalFen: validationGame.currentPosition().fen,
			pgn: validationGame.positionsToPgn({positions: validationGame.getPositions()}).replace("&#189", "") 
				 + validationGame.currentPosition().kingState
		});
        if (result === "1-0") results.push(1);
        else if (result === "0-1") results.push(-1);
        else results.push(0);
    }

//    // summarize
//    const wins = results.filter(r => r === 1).length;
//    const losses = results.filter(r => r === -1).length;
//    const draws = results.filter(r => r === 0).length;
//    
    logger.log({function: "validateNetwork", description: "before next move for running", data: {results: results}});
//     console.log(`Validation: ${wins}/${numGames} wins, ${draws} draws, ${losses} losses`);
}


