import Learn from "../../src/index";
import {Tensor} from "tensor4js";

test("Linear Regression", function () {
  let train_X = Tensor.create([[3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]]);
  let train_Y = Tensor.create([[1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]]);
  let W_init = Tensor.create([[0.5]]);
  let b_init = Tensor.create([[0.5]]);

  let X = Learn.variable([1, train_X.length], "X");
  let Y = Learn.variable([1, train_Y.length], "Y");
  X.value = train_X;
  Y.value = train_Y;

  let W = Learn.parameter(W_init, "weight");
  let b = Learn.parameter(b_init, "bias");

  let y_hat = W.multiply(X).add(b);
  let loss = y_hat.subtract(Y).square().reduceSum();

  let optimizer = Learn.gradientDescent(0.001);
  let trainStep = optimizer.minimize(loss, [W, b]);

  // let lr = Learn.constant(Tensor.create(0.001), "lr");
  // let gradients = Learn.gradients(loss, [W, b]);
  // let grad_W = gradients[0];
  // let grad_b = gradients[1];
  //
  // let newW = W.subtract(grad_W.multiply(lr));
  // let newb = b.subtract(grad_b.multiply(lr));

  for (let i = 0; i < 10000; i++) {
    // console.log("--------- LOOP " + (i+1) + " ----------------")
      console.log("loss: ", loss.eval().toString());

    // console.log("grad_W: ",grad_W.eval().toString());
    // console.log("grad_b: ",grad_b.eval().toString());

    // let newWval = newW.eval();
    // let newbVal = newb.eval();
    // W.value = newWval;
    // b.value = newbVal;

    trainStep.eval();

  }

  console.log("W", W.value.toString());
  console.log("b", b.value.toString());
});