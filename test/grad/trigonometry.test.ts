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
    [1/Math.cos(1)**2, 1/Math.cos(2)**2, 1/Math.cos(3)**2],
    [1/Math.cos(4)**2, 1/Math.cos(5)**2, 1/Math.cos(6)**2]
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