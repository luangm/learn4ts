import Learn4js, {Tensor} from "../../src/index";

test("sin", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.constant(tensorA);
  let result = a.sin();
  // console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  // console.log(gradA.value.toString());

  let expected = Tensor.create([
    [Math.cos(1), Math.cos(2), Math.cos(3)],
    [Math.cos(4), Math.cos(5), Math.cos(6)]
  ]);

  expect(gradA.value).toEqual(expected);
});

test("cos", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.constant(tensorA);
  let result = a.cos();
  // console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  // console.log(gradA.value.toString());

  let expected = Tensor.create([
    [-Math.sin(1), -Math.sin(2), -Math.sin(3)],
    [-Math.sin(4), -Math.sin(5), -Math.sin(6)]
  ]);

  expect(gradA.value).toEqual(expected);
});

test("tan", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.constant(tensorA);
  let result = a.tan();
  // console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  // console.log(gradA.value.toString());

  let expected = Tensor.create([
    [1 / Math.cos(1) ** 2, 1 / Math.cos(2) ** 2, 1 / Math.cos(3) ** 2],
    [1 / Math.cos(4) ** 2, 1 / Math.cos(5) ** 2, 1 / Math.cos(6) ** 2]
  ]);

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

test("asin", function () {

  let tensorA = Learn4js.create([[-1, -0.5, 0], [0, 0.5, 1]]);
  let a = Learn4js.constant(tensorA);
  let result = a.asin();

  let expVal = Tensor.create([[Math.asin(-1), Math.asin(-0.5), Math.asin(0)],
    [Math.asin(0), Math.asin(0.5), Math.asin(1)]]);

  expect(result.value).toEqual(expVal);
  console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];
  //
  console.log(gradA.value.toString());
  //
  let expected = Tensor.create([
    [1 / Math.sqrt(1 - (-1) ** 2), 1 / Math.sqrt(1 - 0.5 ** 2), 1 / Math.sqrt(1 - 0 ** 2)],
    [1 / Math.sqrt(1 - 0 ** 2), 1 / Math.sqrt(1 - 0.5 ** 2), 1 / Math.sqrt(1 - 1 ** 2)]
  ]);
  //
  expect(gradA.value).toEqual(expected);
});

test("asinh", function () {

  let tensorA = Learn4js.create([[-1, -0.5, 0], [0, 0.5, 1]]);
  let a = Learn4js.constant(tensorA);
  let result = a.asinh();

  let expVal = Tensor.create([[Math.asinh(-1), Math.asinh(-0.5), Math.asinh(0)],
    [Math.asinh(0), Math.asinh(0.5), Math.asinh(1)]]);

  expect(result.value).toEqual(expVal);
  console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];
  //
  console.log(gradA.value.toString());
  //
  let expected = Tensor.create([
    [1 / Math.sqrt(1 + (-1) ** 2), 1 / Math.sqrt(1 + 0.5 ** 2), 1 / Math.sqrt(1 + 0 ** 2)],
    [1 / Math.sqrt(1 + 0 ** 2), 1 / Math.sqrt(1 + 0.5 ** 2), 1 / Math.sqrt(1 + 1 ** 2)]
  ]);
  //
  expect(gradA.value).toEqual(expected);
});

test("acos", function () {

  let tensorA = Learn4js.create([[-1, -0.5, 0], [0, 0.5, 1]]);
  let a = Learn4js.constant(tensorA);
  let result = a.acos();

  let expVal = Tensor.create([[Math.acos(-1), Math.acos(-0.5), Math.acos(0)],
    [Math.acos(0), Math.acos(0.5), Math.acos(1)]]);

  expect(result.value).toEqual(expVal);
  console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  console.log(gradA.value.toString());

  let expected = Tensor.create([
    [-1 / Math.sqrt(1 - (-1) ** 2), -1 / Math.sqrt(1 - 0.5 ** 2), -1 / Math.sqrt(1 - 0 ** 2)],
    [-1 / Math.sqrt(1 - 0 ** 2), -1 / Math.sqrt(1 - 0.5 ** 2), -1 / Math.sqrt(1 - 1 ** 2)]
  ]);

  expect(gradA.value).toEqual(expected);
});

test("acosh", function () {

  let tensorA = Learn4js.create([[1, 2, 3], [4, 5, 6]]);
  let a = Learn4js.constant(tensorA);
  let result = a.acosh();

  let expVal = Tensor.create([[Math.acosh(1), Math.acosh(2), Math.acosh(3)],
    [Math.acosh(4), Math.acosh(5), Math.acosh(6)]]);

  expect(result.value).toEqual(expVal);
  console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  console.log(gradA.value.toString());

  let k = 0.20412413775920868;
  let expected = Tensor.create([
    [1 / Math.sqrt((1) ** 2 - 1), 1 / Math.sqrt(2 ** 2 - 1), 1 / Math.sqrt(3 ** 2 - 1)],
    [1 / Math.sqrt(4 ** 2 - 1), k, 1 / Math.sqrt(6 ** 2 - 1)]
  ]);

  expect(gradA.value).toEqual(expected);
});

test("atan", function () {

  let tensorA = Learn4js.create([[1, 2, 3], [4, 5, 6]]);
  let a = Learn4js.constant(tensorA);
  let result = a.atan();

  let expVal = Tensor.create([[Math.atan(1), Math.atan(2), Math.atan(3)],
    [Math.atan(4), Math.atan(5), Math.atan(6)]]);

  expect(result.value).toEqual(expVal);
  console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  console.log(gradA.value.toString());

  let expected = Tensor.create([
    [1 / (1 ** 2 + 1), 1 / (2 ** 2 + 1), 1 / (3 ** 2 + 1)],
    [1 / (4 ** 2 + 1), 1 / (5 ** 2 + 1), 1 / (6 ** 2 + 1)]
  ]);

  expect(gradA.value).toEqual(expected);
});


test("atanh", function () {

  let tensorA = Learn4js.create([[-1, -0.5, 0], [0, 0.5, 1]]);
  let a = Learn4js.constant(tensorA);
  let result = a.atanh();

  let expVal = Tensor.create([[Math.atanh(-1), Math.atanh(-0.5), Math.atanh(0)],
    [Math.atanh(0), Math.atanh(0.5), Math.atanh(1)]]);

  expect(result.value).toEqual(expVal);
  console.log(result.value.toString());

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  console.log(gradA.value.toString());

  let k = 2.777778148651123;
  let expected = Tensor.create([
    [ 1 / (1-(1) ** 2 ), 1 / (1-(-0.5) ** 2 ), 1 / (1-0 ** 2 )],
    [1 / (1-0 ** 2 ), 1 / (1-0.5 ** 2 ), 1 / (1-(1) ** 2 )]
  ]);

  expect(gradA.value).toEqual(expected);
});