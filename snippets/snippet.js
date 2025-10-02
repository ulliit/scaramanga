function CurriculumTrainer(nn, validator, trainer, engine) {
    this.fens = [];          // List of FENs along a Stockfish-generated mate line
    this.currentDepth = 1;   // Start with mate in 1
    this.successThreshold = 0.8; // e.g. 80% success before moving deeper
    this.evalGames = 50;     // Number of evaluation games per stage

    // Load a Stockfish-generated game sequence (from mate in N down to mate)
    this.loadGameLine = function(fenSequence) {
        this.fens = fenSequence;
    };

    this.trainStage = async function() {
        const startIndex = this.fens.length - this.currentDepth; 
        const startFen = this.fens[startIndex]; 

        console.log(`\n=== Training stage: mate in ${this.currentDepth} (FEN: ${startFen}) ===`);

        let wins = 0;
        for (let i = 0; i < this.evalGames; i++) {
            let game = new (require("../client/js/moves.js").asMoves)({});
            game.reset({ initialFen: startFen, positions: [] });

            // Play a self-play game from this FEN
            const result = playSelfPlayGame(game, nn, validator, trainer);
            
            if (result === 1) wins++;
        }

        const winRate = wins / this.evalGames;
        console.log(`Stage ${this.currentDepth} finished with win rate: ${winRate}`);

        if (winRate >= this.successThreshold) {
            console.log(`✔ Success! Advancing to mate in ${this.currentDepth + 1}`);
            this.currentDepth++;
        } else {
            console.log(`✘ Not yet stable. Repeating mate in ${this.currentDepth}`);
        }
    };

    this.runCurriculum = async function() {
        while (this.currentDepth <= this.fens.length) {
            await this.trainStage();
        }
        console.log("🎉 Curriculum complete: network mastered the full endgame!");
    };
}

// helper: self-play one game from a starting FEN
function playSelfPlayGame(game, nn, validator, trainer) {
    let history = [];
    let moveCount = 0;
    const MAX_MOVES = 100;

    while (game.gameStatus() === "running" && moveCount < MAX_MOVES) {
        const fen = game.currentPosition().fen;
        const vector = boardEncoder.encode(fen);
        const predictions = nn.forward({ vector });
        const validMoves = validator.getValidMoves({ fen, probabilities: predictions.probabilities });
        const selectedMove = validator.pickByPolicy(validMoves, 0.1); // epsilon-greedy

        history.push({
            moveIndex: selectedMove.index,
            predictions,
            vector,
            color: game.currentPosition().turn
        });

        game.moveAdd(selectedMove.move);
        moveCount++;
    }

    let result = game.gameStatus();
    if (result === "1-0") result = 1;
    if (result === "0-1") result = -1;
    if (result === "draw") result = 0;

    trainer.trainFromGame({ nn, history, result });
    return result;
}

