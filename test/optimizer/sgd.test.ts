import Learn from "../../src/index";

test("Linear Regression", function () {
  let train_X = Learn.create([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]);
  let train_Y = Learn.create([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]);
  let W_init = Learn.create(0.5);
  let b_init = Learn.create(0.5);

  let X = Learn.variable([train_X.length], "X");
  let Y = Learn.variable([train_Y.length], "Y");

  let W = Learn.parameter(W_init, "weight");
  let b = Learn.parameter(b_init, "bias");

  let y_hat = W.multiply(X).add(b);
  let loss = y_hat.subtract(Y).square().reduceSum();

  let optimizer = Learn.gradientDescent(0.001);
  let trainStep = optimizer.minimize(loss, [W, b]);

  for (let i = 0; i < 10000; i++) {
    X.value = train_X;
    Y.value = train_Y;
    let lossVal = loss.eval();
    if (i % 100 === 0) {
      console.log(lossVal);
    }

    trainStep.eval();
  }

  console.log("W = ", W.value);
  console.log("b = ", b.value);
});