function Validator(){

    var self = this;

    var logger = new (require('../client/js/log.js').logger);
    var rules = new (require('../client/js/chess100.js').Chess)();
    
    var squareIndices = {}; // {"a1":0,"b1":1,"c1":2, ...
    var indexSquares = {}; // {"0":"a1","1":"b1","2":"c1", ...

    function generateSquares(){
        
        var boardSize = 10;    
        var columns = "abcdefghij".split("");
        var squareCnt = 0

		for(var i=0;i<boardSize;i++){

		    for(var j=0;j<boardSize;j++){

		        squareIndices[columns[j] + (i+1).toString()] = squareCnt;
		        indexSquares[squareCnt] = columns[j] + (i + 1).toString();

		        squareCnt++;

		    }

		}

    }
    
    generateSquares();
                                             // logger.log({function: "Validator", description: "after generating squares", data: {squareNumbers: squareIndices, squares: indexSquares}});
	this.getValidMoves = function(o){ // fen, probabilities
                                      // logger.log({function: "Validator.getValidMoves", description: "start", data: {o: o}});
		let moves = [];

		// 10×10 board → 100 squares
		
		for (let from = 0; from < 100; from++) {
		
		    for (let to = 0; to < 100; to++) {
		    
		        let index = from * 100 + to;
		        let score = o.probabilities[index];

		        // build move object
		        
		        let move = {from: indexSquares[from.toString()], to: indexSquares[to.toString()], promotion: undefined };

		        // ask validator
		        
		        rules.load({fen: o.fen});
		        
		        let result = rules.move(move);

		        if (!result.err) {
		        
		            moves.push({ move: move, score: score, index: index });
		            
		        }
		        
		    }
		    
		}

		return moves;
		
	}
	
//	this.pickByPolicy = function(validMoves) {
//	
//		if (validMoves.length === 0) return null;

//		// normalize scores
//		const total = validMoves.reduce((sum, m) => sum + m.score, 0);
//		let r = Math.random() * total;

//		for (const m of validMoves) {
//		r -= m.score;
//		if (r <= 0) return m; // return only the move object
//		}

//		// fallback in case of floating point issues
//		return validMoves[validMoves.length - 1];
//	  
//	}
	
	this.pickByPolicy = function(o = {}) {
	
		// With probability epsilon, pick a random legal move.

		// Otherwise, sample using temperature scaling (softmax-like).

		// If temperature = 0, it always picks the best-scoring move.
		
		const {validMoves = [], successRate = 0.9, temperature = 1} = o;
	
		if (o.validMoves.length === 0) return null;
		
		// --- greedy choice if temperature == 0
		
		if (o.temperature === 0) {
		    return o.validMoves.reduce((a, b) => (a.score > b.score ? a : b));
		}
		
		// --- epsilon schedule adapts to the development of the success rate: more random move selection when success rate is low, less when its high  ---
		
		const epsilonMax = 0.3;   // exploration when failing
		const epsilonMin = 0.05;  // minimal exploration
		
		// Linear decay (simple and robust):
		let epsilon = epsilonMax - (epsilonMax - epsilonMin) * o.successRate;

		// Optional: Exponential decay instead of linear
		// const k = 3; // steepness
		// let epsilon = epsilonMin + (epsilonMax - epsilonMin) * Math.exp(-k * successRate);

		// --- epsilon-greedy: random move with probability epsilon
		
		if (Math.random() < epsilon) {
		    return o.validMoves[Math.floor(Math.random() * o.validMoves.length)];
		}

		// --- temperature-scaled softmax sampling
		
		const scaledScores = o.validMoves.map(m => Math.pow(Math.max(m.score, 1e-12), 1 / o.temperature));
		const total = scaledScores.reduce((sum, s) => sum + s, 0);

		let r = Math.random() * total;
		for (let i = 0; i < o.validMoves.length; i++) {
		    r -= scaledScores[i];
		    if (r <= 0) return o.validMoves[i];
		}

		// --- fallback (floating point safety)
		
		return o.validMoves[o.validMoves.length - 1];
		
	};
	
	this.validateNetwork = function({nn, numGames, startFen}) {

		// play a few games to see if things are improving

		let results = [];
		let boardEncoder = new (require('./boardencoder.js').BoardEncoder)({});
		let validationGame = new (require("../client/js/moves.js").asMoves)(logger);

		for (let g = 0; g < numGames; g++) {
		
		    validationGame.reset({initialFen: startFen, positions: []});
		    let history = [];

		    while (validationGame.gameStatus() === "running" &&
		           validationGame.currentPosition().moveNr <= 100) {
		        
		        let fen = validationGame.currentPosition().fen;
		        let vector = boardEncoder.encode(fen);
		        let predictions = nn.forward({vector});
		        let validMoves = self.getValidMoves({fen, probabilities: predictions.probabilities});
		        let selectedMove = self.pickByPolicy({temperature: 0, validMoves: validMoves});

		        validationGame.moveAdd(selectedMove.move);
		        
		    }

		    let result = validationGame.gameStatus();
		    
		    results.push({
		    
		        startFen: startFen,
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

}

exports.Validator = Validator
