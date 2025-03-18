/**
 * MNIST digits classification.
 *
 * A simple JS neural network implementation, written from scratch
 * as part of my personal "programming kata" routine.
 * It was originally written without using the internet,
 * books and even my own code snippets.
 * To be honest, initially there were a lot of bugs.
 *
 * Since I haven't implemented convolutional layers,
 * the accuracy of the network is not as high as I wanted.
 * Some models are good at classifying a certain set of numbers, 
 * others recognize another set. So, to improve predictive power, 
 * I decided to use four models and average their results.
 *
 * The models were trained on the MNIST dataset.
 *
 * p.s.: Guys don't use JS for such tasks,
 * because python provides much better tools!
 *
 * @version 0.1.7
 * @author Denis Khakimov <denisdude@gmail.com>
 */

// array of NNs
const nns = [];
const init = event => {
  const INVERT_IMAGE = true;
  let DIAMETER = 50;
  let AUTOUPDATE_PAUSE = 250;
  let MULTIPLY_EACH_PIXEL_BY = 1.0;
  let AUTO_UPDATE = false;

  // @TODO download and create NNs
  if (window.DNN_1) nns.push(createNN(window.DNN_1, "DNN_1"));
  if (window.DNN_2) nns.push(createNN(window.DNN_2, "DNN_2"));
  if (window.DNN_3) nns.push(createNN(window.DNN_3, "DNN_3"));
  if (window.DNN_4) nns.push(createNN(window.DNN_4, "DNN_4"));
  //console.log(nns)
  // create UI elements for NNs
  createNetworkElements(predict);

  const root = document.getElementById("board");
  const trainRoot = document.getElementById("s-train-image");
  const canvas = createCanvas(root);
  const canvasTrain = createCanvas(trainRoot);
  canvasTrain.width = 28;
  canvasTrain.height = 28;
  const ctx = canvas.getContext("2d");
  const ctxTrain = canvasTrain.getContext("2d");

  DIAMETER = 50 * canvas.width / 500;

  if (true) {
    ctx.filter = "blur(2px)";
    ctx.imageSmoothingEnabled = true;
    ctxTrain.imageSmoothingEnabled = true;
    ctxTrain.imageSmoothingQuality = "high";
  }

  let PAINT = false;
  ctx.fillStyle = "rgb(0,0,0)";
  clear(ctx);
  let coord = { x: 0, y: 0 };
  const setCoords = e => {
    coord.x = e.clientX - canvas.offsetLeft;
    coord.y = e.clientY - canvas.offsetTop;
  };
  canvas.addEventListener("pointermove", e => {
    if (PAINT) {
      ctx.beginPath();
      ctx.lineWidth = DIAMETER;
      ctx.lineCap = "round";
      ctx.strokeStyle = "rgb(0,0,0)";
      ctx.moveTo(coord.x, coord.y);
      setCoords(e);
      ctx.lineTo(coord.x, coord.y);
      ctx.stroke();
    }
  });
  canvas.addEventListener("pointerdown", e => {
    PAINT = true;
    setCoords(e);
  });
  canvas.addEventListener("pointerup", e => {
    PAINT = false;
  });

  /**
   * Collects all the necessary data,
   * gets a prediction and shows the result.
   */
  function predict() {
    ctxTrain.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, canvasTrain.width, canvasTrain.height);
    const imageData = ctxTrain.getImageData(0, 0, canvasTrain.width, canvasTrain.height);
    // gathering image data
    const input = [];
    for (let i = 0; i < imageData.data.length; i += 4) {
      let pixel =
      (imageData.data[i + 0] +
      imageData.data[i + 1] +
      imageData.data[i + 2]) /
      765;
      //pixel = 0.01 + pixel * 0.99
      pixel = INVERT_IMAGE ? 1 - pixel : pixel;
      pixel = pixel * MULTIPLY_EACH_PIXEL_BY;
      input.push(pixel);
    }
    // get a prediction
    let prediction = networkPrediction(input, nns);
    if (!prediction) {
      showPrediction("");
      return;
    }
    //console.log(prediction)
    const pred = argmax(prediction, 4);
    showPrediction(`
      The number is <b class="A">${pred[0].oldIndex}</b> (${(
    pred[0].value * 100).
    toFixed(2)}%)
      <br />
      But also can be: <br />
      <b class="B">${pred[1].oldIndex}</b> (${(pred[1].value * 100).toFixed(
    2)
    }%);<br />
      <b class="C">${pred[2].oldIndex}</b> (${(pred[2].value * 100).toFixed(
    2)
    }%);<br />
      <b class="D">${pred[3].oldIndex}</b> (${(pred[3].value * 100).toFixed(
    2)
    }%);<br />
    `);
  }

  let interval = setInterval(() => {
    if (AUTO_UPDATE) {
      // predict
      btnrecognize.setAttribute("disabled", true);
      predict();
    } else {
      // nothing
      btnrecognize.removeAttribute("disabled");
    }
  }, AUTOUPDATE_PAUSE);

  // UI -- begin
  const btnrecognize = document.querySelector("button[name=btnrecognize]");
  btnrecognize.addEventListener("click", e => {
    predict();
  });
  const btnclear = document.querySelector("button[name=btnclear]");
  btnclear.addEventListener("click", e => {
    clear(ctx);
  });
  const autoupdate = document.querySelector("input[name=autoupdate]");
  autoupdate.addEventListener("change", e => {
    AUTO_UPDATE = e.target.checked;
  });
  // UI -- end
};
/*
window.addEventListener('load', (event) => {
  document.body.classList.add("loaded");
  // run main logic
  init(event);
});
//*/
//*
document.addEventListener("DOMContentLoaded", event => {
  document.body.classList.add("loaded");
  // run main logic
  init(event);
});
//*/

const createCanvas = container => {
  const canvas = document.createElement("canvas");
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;
  canvas.style.left = 0;
  canvas.style.top = 0;
  canvas.style.position = "absolute";
  container.appendChild(canvas);
  return canvas;
};
const clear = ctx => {
  const oldStyle = ctx.fillStyle;
  ctx.fillStyle = "rgb(255,255,255)";
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.fillStyle = oldStyle;
};
const showPrediction = prediction => {
  const predDiv = document.getElementById("s-prediction");
  predDiv.innerHTML = prediction;
};
/**
 *
 */
const networkPrediction = (input, nns) => {
  const predictions = [];
  //
  for (let nn of nns) {
    if (!nn.active) continue;
    let prediction = nn.predict(input);
    predictions.push(softmax(prediction.matrix.elements));
    //predictions.push(prediction.matrix.elements)
  }
  if (predictions.length === 0) return null;
  //
  const prediction = new Array(predictions[0].length).fill(0);
  for (let i = 0; i < predictions.length; i++)
  for (let j = 0; j < predictions[i].length; j++)
  prediction[j] += predictions[i][j];

  return prediction.map(a => a / predictions.length);
};
/**
 * Returns an NN object created from the passed JSON file.
 */
const createNN = (jsonObject, nnID) => {
  const nn = new NN(1e-3);
  nn.loadNetwork(jsonObject);
  nn.ID = nnID;
  nn.active = true;
  return nn;
};
/**
 * Creates simple UI elements for every NN
 */
const createNetworkElements = predictFn => {
  const parent = document.getElementById("s-predictions");
  const onCheckboxChange = event => {
    const checkbox = event.target;
    const nnID = checkbox.value;
    const nn = nns.filter(v => v.ID == nnID);
    nn[0].active = checkbox.checked;
    predictFn();
    //console.log({ cc: checkbox.checked, nn: nn })
  };
  for (let nn of nns) {
    const element = document.createElement("div");
    element.className = "s-item";
    element.setAttribute("data-id", nn.ID);
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = nn.ID;
    checkbox.checked = nn.active;
    checkbox.addEventListener("change", onCheckboxChange);
    const label = document.createElement("label");
    const span = document.createElement("span");
    span.textContent = nn.ID;
    label.appendChild(checkbox);
    label.appendChild(span);
    element.appendChild(label);
    parent.appendChild(element);
  }
};

// ----------

// Useful functions -- begin
const OHE = (num, outOf, min = 0.0, max = 1.0) => {
  const A = new Array(outOf).fill(min);
  A[num] = max;
  return A;
};
const argmax = (A, num) => {
  const R = A.map((v, i, a) => ({ oldIndex: i, value: v }));
  R.sort((a, b) => b.value - a.value);
  return R.slice(0, num);
};
const softmax = A => {
  const expSum = A.reduce((p, c) => p + Math.exp(c), 0);
  const R = A.map(a => Math.exp(a) / expSum);
  return R;
};
const randomMatrix = (rows, cols, min, max) => {
  const random = (min, max) => min + Math.random() * (max - min);
  const M = [];
  for (let i = 0; i < rows; i++) {
    M[i] = [];
    for (let j = 0; j < cols; j++) M[i][j] = random(min, max);
  }
  return M;
};
// Useful functions -- end

// Matrix -- begin
class Matrix {
  constructor(rows, cols, ...args) {
    this.elements = new Array(rows * cols).fill(0);
    this.shape = [rows, cols];
    for (let i = 0; i < args.length; i++) this.elements[i] = args[i];
  }

  set shape(v) {
    this._shape = v;
  }

  get shape() {
    return this._shape;
  }

  get(r, c) {
    return this.elements[r * this.shape[1] + c];
  }

  set(r, c, v) {
    this.elements[r * this.shape[1] + c] = v;
  }
  apply(fn) {
    for (let i = 0; i < this.elements.length; i++)
    this.elements[i] = fn(this.elements[i]);
    return this;
  }

  plus(V) {
    if (V.shape[0] != this.shape[0] || V.shape[1] != this.shape[1])
    throw new Error("Shape of the V doesn't fit");
    const R = new Matrix(this.shape[0], this.shape[1]);
    for (let i = 0; i < this.elements.length; i++)
    R.elements[i] = this.elements[i] + V.elements[i];
    return R;
  }

  minus(V) {
    if (V.shape[0] != this.shape[0] || V.shape[1] != this.shape[1])
    throw new Error("Shape of the V doesn't fit");
    const R = new Matrix(this.shape[0], this.shape[1]);
    for (let i = 0; i < this.elements.length; i++)
    R.elements[i] = this.elements[i] - V.elements[i];
    return R;
  }

  prod(V) {
    if (V.shape[0] != this.shape[0] || V.shape[1] != this.shape[1])
    throw new Error("Shape of the V doesn't fit");
    const R = new Matrix(this.shape[0], this.shape[1]);
    for (let i = 0; i < this.elements.length; i++)
    R.elements[i] = this.elements[i] * V.elements[i];
    return R;
  }

  outerProd(V) {
    let rows = this.shape[0];
    let cols = V.T.shape[0];
    const R = new Matrix(rows, cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        R.set(i, j, this.get(i, 0) * V.T.get(j, 0));
      }
    }
    return R;
  }

  xScalar(S) {
    const R = new Matrix(this.shape[0], this.shape[1]);
    for (let i = 0; i < this.elements.length; i++)
    R.elements[i] = this.elements[i] * S;
    return R;
  }

  xVector(V) {
    if (V.shape[0] != this.shape[1])
    throw new Error("Shape of the V doesn't fit");
    const R = new Matrix(this.shape[0], 1);
    for (let i = 0; i < this.shape[0]; i++)
    for (let j = 0; j < this.shape[1]; j++)
    R.set(i, 0, R.get(i, 0) + this.get(i, j) * V.get(j, 0));
    return R;
  }

  xMatrix(V) {
    if (this.shape[1] != V.shape[0])
    throw new Error("Shape of the V doesn't fit");
    const R = new Matrix(this.shape[0], V.shape[1]);
    for (let i = 0; i < this.shape[0]; i++)
    for (let j = 0; j < V.shape[1]; j++)
    for (let k = 0; k < this.shape[1]; k++)
    R.set(i, j, R.get(i, j) + this.get(i, k) * V.get(k, j));
    return R;
  }

  get T() {
    let data = new Array(this.shape[0] * this.shape[1]).fill(0);
    for (let i = 0; i < this.shape[0]; i++)
    for (let j = 0; j < this.shape[1]; j++)
    data[j * this.shape[0] + i] = this.elements[i * this.shape[1] + j];
    return new Matrix(this.shape[1], this.shape[0], ...data);
  }

  toString() {
    let str = "";
    str += " [\n";
    for (let i = 0; i < this.shape[0]; i++) {
      str += "    [ ";
      for (let j = 0; j < this.shape[1]; j++) {
        if (j > 0) str += ", ";
        str += this.elements[i * this.shape[1] + j];
      }
      str += " ],\n";
    }
    str += " ]\n";
    return str;
  }

  toArray() {
    let A = [];
    for (let i = 0; i < this.shape[0]; i++) {
      A[i] = [];
      for (let j = 0; j < this.shape[1]; j++)
      A[i][j] = this.elements[i * this.shape[1] + j];
    }
    return A;
  }

  static maybeArray(ELEMENT) {
    return Array.isArray(ELEMENT) ? ELEMENT[0] : ELEMENT;
  }

  static fromArray(A) {
    let rows = A.length;
    let cols = 1;
    if (Array.isArray(A[0])) cols = A[0].length;
    const elements = new Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      if (cols === 1) {
        elements[i] = this.maybeArray(A[i]);
      } else {
        for (let j = 0; j < cols; j++)
        elements[j + i * cols] = this.maybeArray(A[i][j]);
      }
    }
    const M = new Matrix(rows, cols);
    M.elements = elements;
    return M;
  }}

// Matrix -- end

// Activators -- begin
class ActivatorEmpty {
  constructor() {}
  forward(A) {
    if (Array.isArray(A)) return [...A];else
    return A;
  }
  back(A) {
    if (Array.isArray(A)) return A.map(a => 1);else
    return A.apply(a => 1);
  }}

class ActivatorLinear {
  constructor() {}
  forward(A) {
    if (Array.isArray(A)) return [...A];else
    return A;
  }
  back(A) {
    if (Array.isArray(A)) return A.map(a => 1);else
    return A.apply(a => 1);
  }}

class ActivatorTanh {
  constructor() {}
  tanh(a) {
    return 2 / (1 + Math.exp(-2 * a)) - 1;
  }
  forward(A) {
    if (Array.isArray(A)) return A.map(a => this.tanh(a));else
    return A.apply(a => this.tanh(a));
  }
  back(A) {
    if (Array.isArray(A)) return A.map(a => 1 - Math.pow(this.tanh(a), 2));else
    return A.apply(a => 1 - Math.pow(this.tanh(a), 2));
  }}

class ActivatorSigmoid {
  constructor() {}
  forward(A) {
    if (Array.isArray(A))
    return A.map(a => {
      if (Array.isArray(a)) return a.map(b => 1 / (1 + Math.exp(-b)));else
      return 1 / (1 + Math.exp(-a));
    });else
    return A.apply(a => 1 / (1 + Math.exp(-a)));
  }
  back(A) {
    if (Array.isArray(A))
    return A.map(a => {
      if (Array.isArray(a)) {
        return a.map(b => {
          const ex = Math.exp(-b);
          return ex / Math.pow(1 + ex, 2);
        });
      } else {
        const ex = Math.exp(-a);
        return ex / Math.pow(1 + ex, 2);
      }
    });else

    return A.apply(a => {
      const ex = Math.exp(-a);
      return ex / Math.pow(1 + ex, 2);
    });
  }}

class ActivatorReLU {
  constructor() {}
  forward(A) {
    if (Array.isArray(A))
    return A.map(a => {
      if (Array.isArray(a)) return a.map(b => Math.max(b, 0));else
      return Math.max(a, 0);
    });else
    return A.apply(a => Math.max(a, 0));
  }
  back(A) {
    if (Array.isArray(A))
    return A.map(a => {
      if (Array.isArray(a)) return a.map(b => b > 0 ? 1 : 0);else
      return a > 0 ? 1 : 0;
    });else
    return A.apply(a => a > 0 ? 1 : 0);
  }}

// Activators -- end

// Layer -- begin
class Layer {
  constructor(size, activator = null, dropout = 0) {
    this.size = size;
    this.activator = activator;
    this.dropoutRate = dropout;
    this.dropoutMask = null;
    if (!this.activator) {
      this.activator = new ActivatorEmpty();
    }
    this.matrix = new Matrix(1, this.size);
    this.delta = new Matrix(1, this.size);
  }

  createDropoutMask() {
    if (this.dropoutRate == 0) return;

    this.dropoutMask = new Array(this.size);
    for (let i = 0; i < this.size; i++)
    this.dropoutMask[i] = Math.random() > this.dropoutRate ? 1 : 0;
    return this.dropoutMask;
  }

  dropout(A) {
    if (this.dropoutRate == 0) return;

    if (Array.isArray(A)) {
      for (let i = 0; i < A.length; i++)
      for (let j = 0; j < A[i].length; j++)
      A[i][j] = this.dropoutMask[j + i * A[i].length];
    } else {
      for (let i = 0; i < A.elements.length; i++)
      A.elements[i] *= this.dropoutMask[i];
    }
    return A;
  }

  setData(A) {
    if (Array.isArray(A)) this.matrix = new Matrix(1, this.size, ...A);else
    this.matrix.elements = A.elements;
  }

  get(x, y) {
    return this.matrix.get(x, y);
  }

  set(x, y, v) {
    this.matrix.set(x, y, v);
  }

  forward(L, W) {
    return this.activator.forward(L.matrix.xMatrix(W));
  }

  back(A) {
    return this.activator.back(A);
  }}

// Layer -- end

// NN -- begin
class NN {
  constructor(alpha) {
    this.layers = [];
    this.weights = [];
    this.weights_deltas = [];

    this.alpha = alpha;
    this.cum_error = 0;
    this.success_counter = 0;
    this.ratio = 0;

    this.testInputs = [];
    this.testTargets = [];
  }
  get last() {
    return this.layers.length - 1;
  }
  get size() {
    return this.layers.length;
  }
  get prediction() {
    return this.layers[this.last];
  }
  /**
   *
   * @param {Layer} layer
   */
  addLayer(layer) {
    this.layers.push(layer);
    if (this.layers.length > 1) {
      const data = randomMatrix(
      this.layers[this.last - 1].size,
      this.layers[this.last].size,
      -0.25,
      0.25);

      const M = Matrix.fromArray(data);
      this.weights.push(M);
    }
  }
  setTestData(inputs, targets) {
    this.testInputs = inputs;
    this.testTargets = targets;
  }
  predict(input) {
    this.forward(input);
    return this.prediction;
  }
  forward(input) {
    this.layers[0].setData(input);
    for (let li = 1; li < this.size; li++) {
      this.layers[li].matrix = this.layers[li].forward(
      this.layers[li - 1],
      this.weights[li - 1]);

    }
  }
  log(...msg) {
    console.log(...msg);
  }
  describeNetwork() {
    const object = {};
    object.weights = this.weights;
    object.layers = [];
    object.alpha = this.alpha;
    for (let i = 0; i < this.size; i++) {
      object.layers.push({
        size: this.layers[i].size,
        activator: this.layers[i].activator.constructor.name,
        dropout: this.layers[i].dropout });

    }
    return object;
  }
  loadNetwork(jsonObject) {
    for (let i = 0; i < jsonObject.layers.length; i++) {
      let activator = null;
      switch (jsonObject.layers[i].activator) {
        case "ActivatorEmpty":
          activator = new ActivatorEmpty();
          break;
        case "ActivatorReLU":
          activator = new ActivatorReLU();
          break;
        case "ActivatorTanh":
          activator = new ActivatorTanh();
          break;
        case "ActivatorSigmoid":
          activator = new ActivatorSigmoid();
          break;
        case "ActivatorLinear":
          activator = new ActivatorLinear();
          break;
        default:
          activator = new ActivatorEmpty();
          break;}

      this.addLayer(new Layer(jsonObject.layers[i].size, activator));
    }
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i].elements = jsonObject.weights[i].elements;
      this.weights[i].shape = jsonObject.weights[i]._shape;
    }
  }}

// NN -- end