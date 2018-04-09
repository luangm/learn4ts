import Learn4js, {Tensor} from "../../src/index";

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

test("repeat", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  // console.log(tensorA.toString());

  let a = Learn4js.constant(tensorA);
  let result = a.repeat(2, 0);
  // console.log(result.value.toString());

  let expectedVal = Tensor.create([[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]]);
  expect(result.value).toEqual(expectedVal);

  // let grads = Learn4js.gradients(result, [a]);
  // let gradA = grads[0];
  //
  // // console.log(gradA.value.toString());
  //
  // let expected = Tensor.create([[1, 1, 1], [1, 1, 1]]);
  //
  // expect(gradA.value).toEqual(expected);
});

test("tile", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  // console.log(tensorA.toString());

  let a = Learn4js.constant(tensorA);
  let result = a.tile([2, 2]);
  // console.log(result.value.toString());

  let expectedVal = Tensor.create([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6], [1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]]);
  expect(result.value).toEqual(expectedVal);

  // let grads = Learn4js.gradients(result, [a]);
  // let gradA = grads[0];
  //
  // // console.log(gradA.value.toString());
  //
  // let expected = Tensor.create([[1, 1, 1], [1, 1, 1]]);
  //
  // expect(gradA.value).toEqual(expected);
});

test("transpose", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.constant(tensorA);
  let result = a.transpose();
  // console.log(result.value.toString());

  let expectedVal = Tensor.create([[1, 4], [2, 5], [3, 6]]);
  expect(result.value.dup()).toEqual(expectedVal);

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];
  //
  // console.log(gradA.value.toString());
  //
  let expected = Tensor.create([[1, 1, 1], [1, 1, 1]]);
  //
  expect(gradA.value.dup()).toEqual(expected);
});

test("transpose with mul", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([3, 2]);
  let tensorB = Learn4js.linspace(2, 7, 6).reshape([2, 3]);
  let a = Learn4js.constant(tensorA);
  let b = Learn4js.constant(tensorB);
  let mul = a.transpose().multiply(b);
  let result = mul;
  // console.log(result.value.toString());

  let expectedVal = Tensor.create([[1 * 2, 3 * 3, 4 * 5], [2 * 5, 4 * 6, 6 * 7]]);
  expect(result.value.dup()).toEqual(expectedVal);

  let grads = Learn4js.gradients(result, [a, b]);
  let gradA = grads[0];
  let gradB = grads[1];

  let expectedA = Tensor.create([[2, 5], [3, 6], [4, 7]]);
  expect(gradA.value.dup()).toEqual(expectedA);

  let expectedB = Tensor.create([[1, 3, 5], [2, 4, 6]]);
  expect(gradB.value.dup()).toEqual(expectedB);

  console.log(gradA.value);
  console.log(gradB.value);
});