var logger = new (require('../client/js/log.js').logger);

function Trainer(){

	// --- Replay buffer setup ---
	
	const MAX_BUFFER_SIZE = 16; // how many of the most recent games are used by trainFromGame
	const buffer = [];

	// --- Training after each game ---

	this.trainFromGame = function(o) { // batch, nn, history, result
	
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
		
	   buffer.push({history: JSON.parse(JSON.stringify(o.history)), result: o.result});
		if (buffer.length > MAX_BUFFER_SIZE) buffer.shift();

		const batch = o.batch 
		    ? Array.from({length: o.batch}, () => buffer[Math.floor(Math.random() * buffer.length)])
		    : [buffer[buffer.length - 1]];

		for (const game of batch) {

		    const isWin = (o.result === 1);
		    let reps = isWin ? 1 : 1;           // you can increase replay for wins later

		    for (const step of game.history) {

		        let repetitions = reps;
		        while (repetitions > 0) {

		            // Forward pass
		            const predictions = o.nn.forward({vector: step.vector});

		            // Target value (flip for black)
		            let targetValue = (step.color === "b") ? -game.result : game.result;

		            // ====================== VALUE LOSS ======================
		            let valueLoss = (predictions.value - targetValue) ** 2;

		            // NEW: Stockfish auxiliary value loss (dense supervision)
		            let auxValueLoss = 0;
		            if (step.sfValueTarget !== undefined && step.sfValueTarget !== null) {
		                auxValueLoss = (predictions.value - step.sfValueTarget) ** 2;
		            }

		            // Combine (start with strong SF weight)
		            const totalValueLoss = valueLoss + 0.8 * auxValueLoss;

		            // ====================== POLICY LOSS ======================
		            // One-hot for the move actually played (your original REINFORCE)
		            let playedTarget = Array(step.predictions.probabilities.length).fill(0);
		            playedTarget[step.moveIndex] = 1;

		            let policyLoss = 0;
		            for (let i = 0; i < predictions.probabilities.length; i++) {
		                if (playedTarget[i] > 0) {
		                    policyLoss -= Math.log(predictions.probabilities[i] + 1e-12);
		                }
		            }

		            // NEW: Stockfish policy distillation (imitate best move)
		            let distillationLoss = 0;
		            if (step.sfPolicyTarget) {
		            
		                if (step.sfPolicyTarget >= 0 && step.sfPolicyTarget < predictions.probabilities.length) {
		                    const prob = predictions.probabilities[step.sfPolicyTarget];
		                    distillationLoss = -Math.log(Math.max(prob, 1e-12));
		                }
		                
		            }

		            const totalPolicyLoss = policyLoss + 0.7 * distillationLoss;

		            // Total loss
		            const totalLoss = totalValueLoss + totalPolicyLoss;

		            // Optional: log to see the new terms
		            // logger.log({function: "trainFromGame", sfValueLoss: auxValueLoss, sfPolicyLoss: distillationLoss});

		            // Backpropagation
		            const advantage = targetValue - predictions.value;
		            o.nn.backprop(
		                step.vector, 
		                playedTarget,           // still use played move for main policy
		                targetValue, 
		                predictions, 
		                0.05,                   // your base LR
		                advantage
		            );

		            repetitions *= 0.9;
		            repetitions = Math.floor(repetitions);
		        }
		    }
		}
	  
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
