import Learn4js, {Tensor} from "../../src/index";

test("expm1", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.constant(tensorA);
  let result = a.expm1();
  // console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  // console.log(gradA.value.toString());

  let expected = Tensor.create([[Math.exp(1), Math.exp(2), Math.exp(3)], [Math.exp(4), Math.exp(5), Math.exp(6)]]);

  expect(gradA.value).toEqual(expected);
});

test("log1p", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.constant(tensorA);
  let result = a.log1p();
  // console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  // console.log(gradA.value.toString());

  let expected = Tensor.create([[1 / 2, 1 / 3, 1 / 4], [1 / 5, 1 / 6, 1 / 7]]);

  expect(gradA.value).toEqual(expected);
});

test("reciprocal", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.constant(tensorA);
  let result = a.reciprocal();
  // console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  // console.log(gradA.value.toString());

  let expected = Tensor.create([[-1, -1 / 4, -1 / 9], [-1 / 16, -1 / 25, -1 / 36]]);

  expect(gradA.value).toEqual(expected);
});

test("abs", function () {

  let tensorA = Learn4js.create([[-1, -2, 1], [1, -3, 0]]);
  let a = Learn4js.constant(tensorA);
  let result = a.abs();

  let expectedVal = Learn4js.create([[1, 2, 1], [1, 3, 0]]);
  expect(result.value).toEqual(expectedVal);

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  let expectedGrad = Tensor.create([[-1, -1, 1], [1, -1, 0]]);
  expect(gradA.value).toEqual(expectedGrad);
});

test("exp", function () {

  let tensorA = Learn4js.create([[-1, -2, 1], [1, -3, 0]]);
  let a = Learn4js.constant(tensorA);
  let result = a.exp();

  let expectedVal = Learn4js.create([[Math.exp(-1), Math.exp(-2), Math.exp(1)],
    [Math.exp(1), Math.exp(-3), Math.exp(0)]]);
  expect(result.value).toEqual(expectedVal);

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  let expectedGrad = Tensor.create([[Math.exp(-1), Math.exp(-2), Math.exp(1)],
    [Math.exp(1), Math.exp(-3), Math.exp(0)]]);
  expect(gradA.value).toEqual(expectedGrad);
});

test("log", function () {

  let tensorA = Learn4js.create([[-1, -2, 1], [1, -3, 0]]);
  let a = Learn4js.constant(tensorA);
  let result = a.log();

  let expectedVal = Learn4js.create([[Math.log(-1), Math.log(-2), Math.log(1)],
    [Math.log(1), Math.log(-3), Math.log(0)]]);
  expect(result.value).toEqual(expectedVal);

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  let expectedGrad = Tensor.create([[1 / -1, 1 / -2, 1 / 1],
    [1 / 1, 1 / -3, Number.POSITIVE_INFINITY]]);
  expect(gradA.value).toEqual(expectedGrad);
});

test("negate", function () {

  let tensorA = Learn4js.create([[-1, -2, 1], [1, -3, 0]]);
  let a = Learn4js.constant(tensorA);
  let result = a.negate();

  let expectedVal = Learn4js.create([[1, 2, -1], [-1, 3, -0]]);
  expect(result.value).toEqual(expectedVal);

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  let expectedGrad = Tensor.create([[-1, -1, -1],
    [-1, -1, -1]]);
  expect(gradA.value).toEqual(expectedGrad);
});

test("relu", function () {

  let tensorA = Learn4js.create([[-1, -2, 2], [4, -3, 0]]);
  let a = Learn4js.constant(tensorA);
  let result = a.relu();

  let expectedVal = Learn4js.create([[0, 0, 2], [4, 0, 0]]);
  expect(result.value).toEqual(expectedVal);

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  let expectedGrad = Tensor.create([
    [0, 0, 1],
    [1, 0, 0]
  ]);
  expect(gradA.value).toEqual(expectedGrad);
});

test("elu", function () {

  let tensorA = Learn4js.create([[1, -2, 3], [-5, 4, 0]]);
  let a = Learn4js.constant(tensorA);
  let result = a.elu();

  let expectedVal = Learn4js.create([[1, Math.expm1(-2), 3], [Math.expm1(-5), 4, 0]]);
  expect(result.value).toEqual(expectedVal);

  console.log(result.value.toString());
  //
  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  //
  let expectedGrad = Tensor.create([
    [1, Math.exp(-2), 1],
    [Math.exp(-5), 1, 1]
  ]);
  expect(gradA.value).toEqual(expectedGrad);
  console.log(gradA.value.toString());
});

test("round", function () {

  let tensorA = Learn4js.create([[-1.1, -2.1, 2.2], [4.1, -3.3, 0.8]]);
  let a = Learn4js.constant(tensorA);
  let result = a.round();

  let expectedVal = Learn4js.create([
    [-1, -2, 2], [4, -3, 1]
  ]);
  expect(result.value).toEqual(expectedVal);

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  let expectedGrad = Tensor.sparseZeros([2, 3]);
  expect(gradA.value).toEqual(expectedGrad);
});

test("floor", function () {

  let tensorA = Learn4js.create([[-1.1, -2.1, 2.2], [4.1, -3.3, 0.8]]);
  let a = Learn4js.constant(tensorA);
  let result = a.floor();

  let expectedVal = Learn4js.create([
    [-2, -3, 2], [4, -4, 0]
  ]);
  expect(result.value).toEqual(expectedVal);

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  let expectedGrad = Tensor.sparseZeros([2, 3]);
  expect(gradA.value).toEqual(expectedGrad);
});

test("step", function () {

  let tensorA = Learn4js.create([[-1.1, -2.1, 2.2], [0, -3.3, 0.8]]);
  let a = Learn4js.constant(tensorA);
  let result = a.step();

  let expectedVal = Learn4js.create([
    [0, 0, 1], [0, 0, 1]
  ]);
  expect(result.value).toEqual(expectedVal);

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  let expectedGrad = Tensor.sparseZeros([2, 3]);
  expect(gradA.value).toEqual(expectedGrad);
});

test("sign", function () {

  let tensorA = Learn4js.create([[-1.1, -2.1, 2.2], [0, -3.3, 0.8]]);
  let a = Learn4js.constant(tensorA);
  let result = a.sign();

  let expectedVal = Learn4js.create([
    [-1, -1, 1], [0, -1, 1]
  ]);
  expect(result.value).toEqual(expectedVal);

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  let expectedGrad = Tensor.sparseZeros([2, 3]);
  expect(gradA.value).toEqual(expectedGrad);
});

test("ceil", function () {

  let tensorA = Learn4js.create([[-1.1, -2.1, 2.2], [4.1, -3.3, 0.8]]);
  let a = Learn4js.constant(tensorA);
  let result = a.ceil();

  let expectedVal = Learn4js.create([
    [-1, -2, 3], [5, -3, 1]
  ]);
  expect(result.value).toEqual(expectedVal);

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  let expectedGrad = Tensor.sparseZeros([2, 3]);
  expect(gradA.value).toEqual(expectedGrad);
});

test("sigmoid", function () {

  let tensorA = Learn4js.create([[1, 2, 3], [0, -1, -2]]);
  let a = Learn4js.constant(tensorA);
  let result = a.sigmoid();
  // console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  // console.log(gradA.value.toString());

  let expected = Tensor.create([
    [Math.exp(-1) / (Math.exp(-1) + 1) ** 2, Math.exp(-2) / (Math.exp(-2) + 1) ** 2, Math.exp(-3) / (Math.exp(-3) + 1) ** 2],
    [Math.exp(0) / (Math.exp(0) + 1) ** 2, Math.exp(1) / (Math.exp(1) + 1) ** 2, Math.exp(2) / (Math.exp(2) + 1) ** 2]
  ]);

  expect(gradA.value).toEqual(expected);
});

test("square", function () {

  let tensorA = Learn4js.create([[1, 2, 3], [0, -1, -2]]);
  let a = Learn4js.constant(tensorA);
  let result = a.square();
  // console.log(result.value.toString());

  let expectedVal = Learn4js.create([
    [1, 4, 9], [0, 1, 4]
  ]);
  expect(result.value).toEqual(expectedVal);

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  // console.log(gradA.value.toString());

  let expected = Tensor.create([
    [2, 4, 6],
    [0, -2, -4]
  ]);

  expect(gradA.value).toEqual(expected);
});

test("sqrt", function () {

  let tensorA = Learn4js.create([[1, 2, 3], [0, 2, 4]]);
  let a = Learn4js.constant(tensorA);
  let result = a.sqrt();
  // console.log(result.value.toString());

  let expectedVal = Learn4js.create([
    [Math.sqrt(1), Math.sqrt(2), Math.sqrt(3)], [0, Math.sqrt(2), 2]
  ]);
  expect(result.value).toEqual(expectedVal);

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  // console.log(gradA.value.toString());

  let expected = Tensor.create([
    [0.5 / Math.sqrt(1), 0.5 / Math.sqrt(2), 0.5 / Math.sqrt(3)],
    [0.5 / Math.sqrt(0), 0.5 / Math.sqrt(2), 0.5 / Math.sqrt(4)]
  ]);

  expect(gradA.value).toEqual(expected);
});

test("softplus", function () {

  let tensorA = Learn4js.create([[1, 2, 3], [4, 5, 6]]);
  let a = Learn4js.constant(tensorA);
  let result = a.softplus();
  console.log(result.value.toString());

  let expectedVal = Learn4js.create([
    [Math.log(Math.exp(1) + 1), Math.log(Math.exp(2) + 1), Math.log(Math.exp(3) + 1)],
    [Math.log(Math.exp(4) + 1), Math.log(Math.exp(5) + 1), Math.log(Math.exp(6) + 1)]
  ]);
  expect(result.value).toEqual(expectedVal);
  //
  // let k = a.sigmoid();
  // console.log(k.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  // console.log(gradA.value.toString());
  //
  let expected = Tensor.create([
    [sigmoid(1), sigmoid(2), sigmoid(3)],
    [sigmoid(4), sigmoid(5), sigmoid(6)]
  ]);

  expect(gradA.value).toEqual(expected);
  console.log(gradA.value.toString());
});

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}