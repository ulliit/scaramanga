var logger = new (require('../client/js/log.js').logger);

function Trainer(){

	// --- Replay buffer setup ---
	
	const MAX_BUFFER_SIZE = 1; // keep last 1000 finished games
	const buffer = [];

	// --- Training after each game ---

	this.trainFromGame = function(o) { // batch, nn, history, result, resultWithoutPenalization
	
		// trainFromGame(history, result) is called once the game ends.
		// Its job is to:

		// Take all the states and NN outputs you stored during self-play (history).

		// Compare them to the actual game result.

		// Compute losses
		// Value loss = how far off the NN’s guess was from the true result.
		// Policy loss = how poorly the NN matched the chosen move.

		// Call the NN’s backprop to update weights.

		// --- Store current game into replay buffer ---
		// Keep a buffer of recent games and train on random samples from it. Whereas a training only from the latest game would make a net highly unstable (catastrophic forgetting).
		
		buffer.push({history: JSON.parse(JSON.stringify(o.history)), result: o.result, resultWithoutPenalization: o.resultWithoutPenalization});
		if (buffer.length > MAX_BUFFER_SIZE) buffer.shift();

		const batch = [];

        if(o.batch){
        
            // --- Sample a minibatch of games from buffer ---

			for (let i = 0; i < o.batch; i++) {
				batch.push(buffer[Math.floor(Math.random() * buffer.length)]);
			}
			
		}else{
		
		    batch.push(buffer[buffer.length-1]);
		
		}

		// --- Train from sampled minibatch ---
		for (const game of batch) {

			// --- Determine if this game should be replayed (automatic winning sequence replay) ---
			var isWin = (game.resultWithoutPenalization == 1 ? true : false); // consider only wins for replay
			const maxRepeats = 1;          // maximum times to replay a winning step
			const decay = 0.9;             // decay factor for repetitions
			
			let reps = (isWin == true ? maxRepeats : 1); // replay multiple times if win 1)
				    logger.log({function: "trainFromGame", description: "after setting number of repetitions", data: {isWin: isWin, reps: reps, resultWithoutPenalization: game.resultWithoutPenalization}});
			for (const step of game.history) {
			                                         if(step.selectedMove.move.from == "c8" && step.selectedMove.move.to == "d8"){
			                                              logger.log({function: "trainFromGame", description: "history step", data: {move: step.selectedMove , moveProbabilty: step.predictions.probabilities[step.moveIndex]}});
			                                         }
			    if (step.color === "b") {continue} //_t train only using whites moves
			
			    let repetitions = reps;

				while(repetitions > 0){
				
				    // one-hot target for played move

					var probabilitiesForPlayedMove = Array(step.predictions.probabilities.length).fill(0); // probabilities for the move that was played
					probabilitiesForPlayedMove[step.moveIndex] = 1; 

					// Forward pass (get current outputs) 1)
//					                                                                       console.log("step.predictions.probabilities.slice(0,5)", step.predictions.probabilities.slice(0,5));
//					                                                                      console.log("step.predictions.value", step.predictions.value);
//                                                                                           console.log("selectedMove: " + JSON.stringify(step.selectedMove.move) + ", step.color: " + step.color)
					const predictions = o.nn.forward({vector: step.vector});
					
					let result = game.result;
					
					if (step.color === "b") {

						result = -game.result; // flip perspective for Black
						
					}
					
					// advantage = (target - baseline)
					
                    let advantage = result - predictions.value;

//					// Loss diagnostics optional for debugging, te real loss computing happens in the nn.backprop.			
////					                                                                       console.log("probabilitiesForPlayedMove", JSON.stringify(probabilitiesForPlayedMove));
////					                                                                       console.log("predictions.value", predictions.value);
////                                                                                           console.log("predictions.probabilities.slice(0,5)", predictions.probabilities.slice(0,5));
////                                                                                           console.log("result", game.result)
//			
//					const valueLoss = (predictions.value - result) ** 2;
//					                                           
//					var probabilitiesLoss = 0;
//					
//					for (let i = 0; i < predictions.probabilities.length; i++) {
//					
//						if (probabilitiesForPlayedMove[i] > 0) {
//						
//							probabilitiesLoss -= Math.log(predictions.probabilities[i] + 1e-12); // epsilon to avoid log(0)
//						
//						}
//					
//					}
//			  
//					const totalLoss = valueLoss + probabilitiesLoss;
//					                             logger.log({function: "trainFromGame", description: "after computing losses", data: {result: result, totalLoss: totalLoss, value: predictions.value, valueLoss: valueLoss}});
					var baseLearningRate = 0.05
					
					// --- Adaptive learning rate ---
				    // Win/loss -> full LR
				    // Draw -> smaller LR
				    
					let lrScale = (Math.abs(game.resultWithoutPenalization) === 1) ? 1.0 : 0.5;
					let effectiveLearningRate = baseLearningRate * lrScale;
				
					// Update weights via backprop
					
					o.nn.backprop(step.vector, probabilitiesForPlayedMove, result, predictions, effectiveLearningRate, advantage);

					// reduce repetitions for winning sequences
					repetitions *= decay;
					repetitions = Math.floor(repetitions);

				} // end while repetition

			} // end for each step
		} // end for each game in batch
	  
	} // end trainFromGame
	
}

exports.Trainer = Trainer;

//	1)

//	If the nn found a successful path for a win, would in make sense to let it repeat the move sequence several times?

//	Yes — that’s a classic curriculum / reinforcement trick. The idea is: once the network discovers a good trajectory, you let it “rehearse” it multiple times so the gradient updates reinforce that successful path.

//	Here’s why it helps:

//	Stronger signal: Winning sequences are rare at first, so seeing them repeatedly boosts the learning signal.

//	Faster convergence: The network gets more confident about moves that lead to victory.

//	Stability: Multiple passes over the same successful sequence reduce variance in training.

//	smart version: only repeat sequences that actually improved the network’s predicted value, and decay repetitions over time to avoid overfitting. This is closer to how AlphaZero handles “self-play experience replay.”
