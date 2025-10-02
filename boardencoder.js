function BoardEncoder() {

	// --- Input boardEncoder ---
	// Encodes a 10x10 board into 
	// why 1601: one vector slot for each piece/square combination and one for the side to move -> 16 (pieces) * 100 (squares) + 1 = 1601 //_us
	// note: nns need a fixed vector length and can't handle a dynamic one //_us

    // --- Piece mapping (extend if needed) ---
    const pieceMap = {
        'P': 0,  'p': 1,   // pawns
        'N': 2,  'n': 3,   // knights
        'B': 4,  'b': 5,   // bishops
        'R': 6,  'r': 7,   // rooks
        'Q': 8,  'q': 9,   // queens
        'K': 10, 'k': 11,  // kings
        'C': 12, 'c': 13,  // camels
        'Z': 14, 'z': 15   // zebras
    };

    // reverse mapping for decode
    const reverseMap = Object.fromEntries(
        Object.entries(pieceMap).map(([k,v]) => [v,k])
    );

    const squareChannels = 16;   // one-hot channels per square
    const boardSize = 100;       // 10 × 10
    const inputSize = boardSize * squareChannels + 1; // +1 for side-to-move

    // --- Encode: FEN → input vector ---
    this.encode = function(fen) {
        const input = Array(inputSize).fill(0);
        const parts = fen.trim().split(" ");
        const boardPart = parts[0];
        const stmPart = parts[1] || "w";

        const rows = boardPart.split("/");
        for (let rank = 0; rank < 10; rank++) {
            let file = 0;
            let row = rows[rank];
            let i = 0;
            while (i < row.length) {
                let char = row[i];
                if (/[0-9]/.test(char)) {
                    // read full number (can be "10")
                    let numStr = "";
                    while (i < row.length && /[0-9]/.test(row[i])) {
                        numStr += row[i];
                        i++;
                    }
                    file += parseInt(numStr);
                    continue;
                } else {
                    const idx = pieceMap[char];
                    if (idx !== undefined) {
                        let square = rank * 10 + file;
                        let offset = square * squareChannels + idx;
                        input[offset] = 1;
                    }
                    file++;
                    i++;
                }
            }
        }

        // Side to move
        input[inputSize - 1] = (stmPart === "w" ? 1 : 0);

        return input;
    };

    // --- Decode: input vector → FEN ---
    this.decode = function(input) {
        let rows = [];
        for (let rank = 0; rank < 10; rank++) {
            let rowStr = "";
            let emptyCount = 0;
            for (let file = 0; file < 10; file++) {
                let square = rank * 10 + file;
                let base = square * squareChannels;
                let pieceIdx = input.slice(base, base + squareChannels).findIndex(x => x === 1);
                if (pieceIdx === -1) {
                    emptyCount++;
                } else {
                    if (emptyCount > 0) {
                        rowStr += emptyCount.toString();
                        emptyCount = 0;
                    }
                    rowStr += reverseMap[pieceIdx] || "?";
                }
            }
            if (emptyCount > 0) rowStr += emptyCount.toString();
            rows.push(rowStr);
        }
        const stm = (input[inputSize - 1] === 1 ? "w" : "b");
        return rows.join("/") + " " + stm + " - - 0 1";
    };

    this.inputSize = inputSize;
}

exports.BoardEncoder = BoardEncoder;
