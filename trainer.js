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

        buffer.push({
            history: JSON.parse(JSON.stringify(o.history)),
            result: o.result
        });
        if (buffer.length > MAX_BUFFER_SIZE) buffer.shift();

        // Mini-batch
        const batch = o.batch 
            ? Array.from({length: o.batch}, () => buffer[Math.floor(Math.random() * buffer.length)])
            : [buffer[buffer.length - 1]];

        for (const game of batch) {

            const targetResult = game.result;   // +1 / 0 / -1

            for (const step of game.history) {

                let repetitions = (targetResult === 1) ? 1 : 1;

                while (repetitions > 0) {

                    // Forward pass
                    const predictions = o.nn.forward({vector: step.vector});

                    // Target value (flip for black)
                    let targetValue = (step.color === "b") ? -targetResult : targetResult;

                    // ====================== VALUE LOSS ======================
                    let valueLoss = (predictions.value - targetValue) ** 2;

                    // Stockfish auxiliary value loss
                    let auxValueLoss = 0;
                    if (typeof step.sfValueTarget === "number") {
                        auxValueLoss = (predictions.value - step.sfValueTarget) ** 2;
                    }

                    const totalValueLoss = valueLoss + 0.8 * auxValueLoss;

                    // ====================== POLICY LOSS with LEGAL MASKING ======================
                    // One-hot for the move actually played
                    let playedTarget = new Array(predictions.probabilities.length).fill(0);
                    playedTarget[step.moveIndex] = 1;

                    let policyLoss = 0;

                    // Create legal move mask (1 = legal, 0 = illegal)
                    let legalMask = new Array(predictions.probabilities.length).fill(0);

                    if (step.legalMoves && Array.isArray(step.legalMoves)) {
                        for (let m of step.legalMoves) {
                            if (m.index !== undefined) {
                                legalMask[m.index] = 1;
                            }
                        }
                    }

                    // Compute policy loss only on legal moves + played move
                    for (let i = 0; i < predictions.probabilities.length; i++) {
                        if (playedTarget[i] > 0 || legalMask[i] === 1) {
                            policyLoss -= Math.log(Math.max(predictions.probabilities[i], 1e-12));
                        }
                    }

                    // Stockfish policy distillation (only if SF move is legal)
                    let distillationLoss = 0;
                    if (typeof step.sfPolicyTarget === "number") {
                        const idx = step.sfPolicyTarget;
                        if (idx >= 0 && idx < predictions.probabilities.length && legalMask[idx] === 1) {
                            distillationLoss = -Math.log(Math.max(predictions.probabilities[idx], 1e-12));
                        }
                    }

                    const totalPolicyLoss = policyLoss + 0.7 * distillationLoss;

                    // Total loss
                    const totalLoss = totalValueLoss + totalPolicyLoss;

                    // Optional logging
                    // logger.log({ auxValue: auxValueLoss.toFixed(4), distLoss: distillationLoss.toFixed(4), legalMoves: step.legalMoves?.length || 0 });

                    // Backpropagation
                    const advantage = targetValue - predictions.value;
                    o.nn.backprop(
                        step.vector, 
                        playedTarget,
                        targetValue, 
                        predictions, 
                        0.05,
                        advantage
                    );

                    repetitions *= 0.9;
                    repetitions = Math.floor(repetitions);
                }
            }
        }
    };
	
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
