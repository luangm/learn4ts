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