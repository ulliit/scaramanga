function MoveEncoder(o) {

    // options: { boardSize: 10 }
    var boardSize  = o && o.boardSize || 10;
    var numSquares = boardSize * boardSize;      // 100
    var policySize = numSquares * numSquares;    // 10,000

    // encode a move (fromSquare, toSquare) -> policy index
    this.encode = function(from, to) {
        return from * numSquares + to;
    };

    // decode a policy index -> { from, to }
    this.decode = function(index) {
        return {
            from: Math.floor(index / numSquares),
            to:   index % numSquares
        };
    };

    // convert (row, col) -> square index
    this.squareIndex = function(row, col) {
        return row * boardSize + col;
    };

    // convert square index -> { row, col }
    this.squareCoords = function(index) {
        return {
            row: Math.floor(index / boardSize),
            col: index % boardSize
        };
    };

    // mask a full policy vector down to legal moves and renormalize
    this.maskPolicy = function(policyVector, legalMoves) {
        var mask = Array(policySize).fill(0);
        for (var i = 0; i < legalMoves.length; i++) {
            var m = legalMoves[i];
            mask[this.encode(m.from, m.to)] = 1;
        }
        var masked = policyVector.map(function(p, i){ return p * mask[i]; });
        var sum = masked.reduce(function(a, b){ return a + b; }, 0);
        return sum > 0 ? masked.map(function(x){ return x / sum; }) : mask;
    };

    // expose sizes if needed
    this.boardSize  = boardSize;
    this.numSquares = numSquares;
    this.policySize = policySize;
    
}

exports.MoveEncoder = MoveEncoder
