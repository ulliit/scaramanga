function Network(o){ // inputSize, hiddenSize, probabilitiesSize

    // the neural net

	// --- Helper math functions ---
	function zeros(n) { return Array(n).fill(0); }
	function randomMatrix(rows, cols) {
	  return Array.from({ length: rows }, () =>
		Array.from({ length: cols }, () => (Math.random() - 0.5) * 0.01)
	  );
	}
	function matmul(A, x) { return A.map(row => row.reduce((sum, val, j) => sum + val * x[j], 0)); }
	function add(a, b) { return a.map((val, i) => val + b[i]); }
	function relu(v) { return v.map(x => Math.max(0, x)); }
	function tanh(v) { return v.map(x => Math.tanh(x)); }
	function softmax(v) {
	  const max = Math.max(...v);
	  const exps = v.map(x => Math.exp(x - max));
	  const sum = exps.reduce((a, b) => a + b, 0);
	  return exps.map(x => x / sum);
	}

    var inputSize = o.inputSize || 1601; // all possible combinations of a square and a piece type, for a 10x10 board: 100 * 16  = 1600; + 1 for side to move → 1601 features
    var hiddenSize = o.hiddenSize || 128; // Hidden Layer: 128 neurons, ReLU
    var probabilitiesSize = o.probabilitiesSize || 10000; // all possible pairs of two squares, for a 10x10 board: 100 * 100  = 10000

    // Hidden layer
    
    var w1 = randomMatrix(hiddenSize, inputSize);
    var b1 = zeros(hiddenSize);

    // probabilities head
    
    var wp = randomMatrix(probabilitiesSize, hiddenSize);
    var bp = zeros(probabilitiesSize);

    // Value Head: 1 neuron, tanh output (-1 = loss, +1 = win)
    
    var wv = randomMatrix(1, hiddenSize);
    var bv = zeros(1);
    
    // Backward pass
    
	this.backprop = function(vector, probabilitiesForPlayedMove, result, predictions, learningRate, advantage) {
		
		learningRate = learningRate || 0.3;

		const { probabilities, value, h, z1 } = predictions;

		// --- Value loss gradient (MSE) ---
		var dvalue = 2 * (value - result);
		var dzv = dvalue * (1 - value * value);

		// --- Advantage scaling ---
		// fallback: if not passed, use (target - prediction) as advantage
		if (typeof advantage !== "number") {
		    advantage = result - value;
		}
		// clip for stability
		const ADV_CLIP = 2.0;
		if (advantage > ADV_CLIP) advantage = ADV_CLIP;
		if (advantage < -ADV_CLIP) advantage = -ADV_CLIP;

		// --- Policy gradient ---
		// REINFORCE: grad = A * (p - y)
		var dprobabilities = probabilities.map((p, i) =>
		    advantage * (p - probabilitiesForPlayedMove[i])
		);

		// --- Gradients for wp, bp ---
		var dwp = wp.map((row, i) => row.map((_, j) => dprobabilities[i] * h[j]));
		var dbp = dprobabilities;

		// --- Gradients for wv, bv ---
		var dwv = wv.map((row, i) => row.map((_, j) => dzv * h[j]));
		var dbv = [dzv];

		// --- Backprop hidden layer ---
		var dh = zeros(h.length);
		for (var i = 0; i < probabilitiesSize; i++) {
		    for (var j = 0; j < hiddenSize; j++) {
		        dh[j] += dprobabilities[i] * wp[i][j];
		    }
		}
		for (var j = 0; j < hiddenSize; j++) {
		    dh[j] += dzv * wv[0][j];
		}
		var dz1 = z1.map((x, j) => (x > 0 ? 1 : 0) * dh[j]);

		// --- Gradients for w1, b1 ---
		var dw1 = w1.map((row, i) => row.map((_, j) => dz1[i] * vector[j]));
		var db1 = dz1;

		// --- Update weights ---
		for (var i = 0; i < hiddenSize; i++) {
		    for (var j = 0; j < inputSize; j++) {
		        w1[i][j] -= learningRate * dw1[i][j];
		    }
		    b1[i] -= learningRate * db1[i];
		}
		for (var i = 0; i < probabilitiesSize; i++) {
		    for (var j = 0; j < hiddenSize; j++) {
		        wp[i][j] -= learningRate * dwp[i][j];
		    }
		    bp[i] -= learningRate * dbp[i];
		}
		for (var j = 0; j < hiddenSize; j++) {
		    wv[0][j] -= learningRate * dwv[0][j];
		}
		bv[0] -= learningRate * dbv[0];
	};

	this.forward = function(e) {
	
	     // takes the given position and makes the following predictions about it:
	     
         // probabilities: a vector with 10000 elements, with each element representing the "best move" probabilty for a certain combination of two squares. means, combinations that could never represent a legal move are included. 
         
         // sample: probabilities = [0.0, 0.0003, 0.15, 0.0, 0.2, ...]
         
         // this function is called BEFORE checking legality.

         // so things like “a1 → a1” or “king jumps 5 squares” all get numbers too.
         
         // why? The NN’s job is pattern recognition, not rules enforcement.

         // Value = a floating number representing the estimated value of the position BEFORE a move was made.

         // Hidden = intermediate features the net has built (not directly meaningful to you, but useful for learning).

        const z1 = add(matmul(w1, e.vector), b1);
        const h  = relu(z1);
        const zp = add(matmul(wp, h), bp);
        const probabilities = softmax(zp);
        const zv = add(matmul(wv, h), bv);
        const value = tanh(zv)[0];
        return { probabilities, value, h, z1, zv, zp }; // keep activations for backprop
        
    };
    
	this.loadFromDisk = function(path){

		const fs = require("fs");
		const pathFinder = require("path");
		
		path = pathFinder.join(__dirname, (path || "network.json")); // in Node.js, process.cwd() is the current working directory from where you run the script.
		
		if (!fs.existsSync(path)) {
		
		    console.error("Network file not found: Starting a new net. Path", path);
		    return
   
        }
		
		const state = JSON.parse(fs.readFileSync(path));
		w1 = state.w1;
		b1 = state.b1;
		wp = state.wp;
		bp = state.bp;
		wv = state.wv;
		bv = state.bv;
		console.log("Network loaded from", path);
		
	};
		
    this.saveToDisk = function(path){

		const fs = require("fs");
		const pathFinder = require("path");
		
		path = pathFinder.join(__dirname, (path || "network.json"));
		
		const state = {
		    w1, b1,
		    wp, bp,
		    wv, bv
		};
		fs.writeFileSync(path, JSON.stringify(state));
		console.log("Network saved to", path);
		
	};

}

exports.Network = Network;
