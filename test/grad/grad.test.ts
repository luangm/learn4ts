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

test("reshape", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  // console.log(tensorA.toString());

  let a = Learn4js.constant(tensorA);
  let result = a.reshape([1, 6]);
  // console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  // console.log(gradA.value.toString());

  let expected = Tensor.create([[1, 1, 1], [1, 1, 1]]);

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

test("sinh", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.constant(tensorA);
  let result = a.sinh();
  // console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  // console.log(gradA.value.toString());

  let expected = Tensor.create([[Math.cosh(1), Math.cosh(2), Math.cosh(3)], [Math.cosh(4), Math.cosh(5), Math.cosh(6)]]);

  expect(gradA.value).toEqual(expected);
});

test("cosh", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.constant(tensorA);
  let result = a.cosh();
  // console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  // console.log(gradA.value.toString());

  let expected = Tensor.create([[Math.sinh(1), Math.sinh(2), Math.sinh(3)], [Math.sinh(4), Math.sinh(5), Math.sinh(6)]]);

  expect(gradA.value).toEqual(expected);
});

test("tanh", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.constant(tensorA);
  let result = a.tanh();
  // console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  // console.log(gradA.value.toString());

  let expected = Tensor.create([
    [1 / Math.cosh(1) / Math.cosh(1), 1 / Math.cosh(2) / Math.cosh(2), 1 / Math.cosh(3) / Math.cosh(3)],
    [1 / Math.cosh(4) / Math.cosh(4), 1 / Math.cosh(5) / Math.cosh(5), 1 / Math.cosh(6) / Math.cosh(6)]
  ]);

  expect(gradA.value).toEqual(expected);
});