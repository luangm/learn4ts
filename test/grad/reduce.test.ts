import Learn4js, {Tensor} from "../../src/index";

test("reduce all", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.parameter(tensorA);
  let add = a.reduceSum();

  let expectedValue = Tensor.create(21);
  expect(add.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(add, [a]);
  let gradA = grads[0];

  let expectedGradA = Tensor.create([[1, 1, 1], [1, 1, 1]]);

  expect(gradA.value).toEqual(expectedGradA);

});

test("reduce 0", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.parameter(tensorA);
  let add = a.reduceSum(0);

  let expectedValue = Tensor.create([5, 7, 9]);
  expect(add.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(add, [a]);
  let gradA = grads[0];

  // let expectedGradA = Tensor.create([[]]);

  // expect(gradA.value).toEqual(expectedGradA);

  console.log(gradA.value.toString());
});

test("reduce 1", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.parameter(tensorA);
  let add = a.reduceSum(1);

  let expectedValue = Tensor.create([6, 15]);
  expect(add.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(add, [a]);
  let gradA = grads[0];

  // let expectedGradA = Tensor.create([[]]);

  // expect(gradA.value).toEqual(expectedGradA);

  console.log(gradA.value.toString());
});

test("reduce 0 1", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.parameter(tensorA);
  let add = a.reduceSum([0, 1]);

  let expectedValue = Tensor.create(21);
  expect(add.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(add, [a]);
  let gradA = grads[0];

  // let expectedGradA = Tensor.create([[]]);

  // expect(gradA.value).toEqual(expectedGradA);

  console.log(gradA.value.toString());
});

test("reduce mean all", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.parameter(tensorA);
  let add = a.reduceMean();

  let expectedValue = Tensor.create(3.5);
  expect(add.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(add, [a]);
  let gradA = grads[0];

  let expectedGradA = Tensor.create([[1 / 6, 1 / 6, 1 / 6], [1 / 6, 1 / 6, 1 / 6]]);

  expect(gradA.value).toEqual(expectedGradA);

});

test("reduce mean 0", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.parameter(tensorA);
  let add = a.reduceMean(0);

  let expectedValue = Tensor.create([2.5, 3.5, 4.5]);
  expect(add.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(add, [a]);
  let gradA = grads[0];

  let expectedGradA = Tensor.create([[1 / 2, 1 / 2, 1 / 2], [1 / 2, 1 / 2, 1 / 2]]);

  expect(gradA.value).toEqual(expectedGradA);

});

test("reduce mean 1", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.parameter(tensorA);
  let add = a.reduceMean(1);

  let expectedValue = Tensor.create([2, 5]);
  expect(add.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(add, [a]);
  let gradA = grads[0];

  let expectedGradA = Tensor.create([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]]);

  expect(gradA.value).toEqual(expectedGradA);

});

test("reduce max all", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.parameter(tensorA);
  let add = a.reduceMax();

  let expectedValue = Tensor.create(6);
  expect(add.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(add, [a]);
  let gradA = grads[0];

  let expectedGradA = Tensor.create([[0, 0, 0], [0, 0, 1]]);

  expect(gradA.value).toEqual(expectedGradA);

});

test("reduce max 0", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.parameter(tensorA);
  let add = a.reduceMax(0);

  let expectedValue = Tensor.create([4, 5, 6]);
  expect(add.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(add, [a]);
  let gradA = grads[0];

  let expectedGradA = Tensor.create([[0, 0, 0], [1, 1, 1]]);

  expect(gradA.value).toEqual(expectedGradA);

});

test("reduce max 1", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.parameter(tensorA);
  let add = a.reduceMax(1);

  let expectedValue = Tensor.create([3, 6]);
  expect(add.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(add, [a]);
  let gradA = grads[0];

  let expectedGradA = Tensor.create([[0, 0, 1], [0, 0, 1]]);

  expect(gradA.value).toEqual(expectedGradA);

});

test("reduce min 1", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.parameter(tensorA);
  let add = a.reduceMin(1);

  let expectedValue = Tensor.create([1, 4]);
  expect(add.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(add, [a]);
  let gradA = grads[0];

  let expectedGradA = Tensor.create([[1, 0, 0], [1, 0, 0]]);

  expect(gradA.value).toEqual(expectedGradA);

});

test("reduce prod", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.parameter(tensorA);
  let add = a.reduceProd();

  let expectedValue = Tensor.create(720);
  expect(add.value).toEqual(expectedValue);
  //
  // let grads = Learn4js.gradients(add, [a]);
  // let gradA = grads[0];
  //
  // let expectedGradA = Tensor.create([[1, 0, 0], [1, 0, 0]]);
  //
  // expect(gradA.value).toEqual(expectedGradA);

});

test("reduce logsumexp", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.parameter(tensorA);
  let result = a.reduceLogSumExp();

  // console.log(result.value.toString());
  let expectedValue = Tensor.create(6.45619345);
  expect(result.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(result, [a]);
  let gradA = grads[0];

  let expectedGradA = Tensor.create([
    [Math.exp(1 - 6.45619345), Math.exp(2 - 6.45619345), Math.exp(3 - 6.45619345)],
    [Math.exp(4 - 6.45619345), Math.exp(5 - 6.45619345), Math.exp(6 - 6.45619345)]
  ]);

  expect(gradA.value).toEqual(expectedGradA);

  console.log(gradA.value.toString());

});

test("reduce logsumexp expanded ", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.parameter(tensorA);
  let b = Learn4js.parameter(tensorA);
  let result = a.reduceLogSumExp().multiply(b);

  // console.log(result.value.toString());
  // let expectedValue = Tensor.create(6.45619345);
  // expect(result.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(result, [a, b]);
  let gradA = grads[0];
  let gradB = grads[1];
  //
  // let expectedGradA = Tensor.create([
  //   [Math.exp(1 - 6.45619345), Math.exp(2 - 6.45619345), Math.exp(3 - 6.45619345)],
  //   [Math.exp(4 - 6.45619345), Math.exp(5 - 6.45619345), Math.exp(6 - 6.45619345)]
  // ]);
  //
  // expect(gradA.value).toEqual(expectedGradA);

  console.log(gradA.value.toString());
  console.log(gradB.value.toString());

});

test("l1norm", function () {

  let tensorA = Learn4js.create([[[-1, -2, 3], [4, 5, 6]], [[1, 2, 3], [2, 3, 4]]]);
  let a = Learn4js.parameter(tensorA);
  let b = Learn4js.constant(Tensor.create([[1, 2, 3], [4, 5, 6]]));
  let result = a.l1Norm().multiply(b);

  // console.log(result.value.toString());
  let expectedValue = Tensor.create([[36, 72, 108], [144, 180, 216]]);
  expect(result.value).toEqual(expectedValue);
  //
  let grads = Learn4js.gradients(result, [a, b]);
  let gradA = grads[0];
  let gradB = grads[1];

  // console.log(gradA.value.toString());
  // console.log(grads[1].value.toString());

  let expectedGradA = Tensor.create([[[-21, -21, 21], [21, 21, 21]], [[21, 21, 21], [21, 21, 21]]]);
  expect(gradA.value).toEqual(expectedGradA);

  let expectedGradB = Tensor.create([[36, 36, 36], [36, 36, 36]]);
  expect(gradB.value).toEqual(expectedGradB);
});

test("l1norm axis 0", function () {

  let tensorA = Learn4js.create([[[-1, -2, 3], [4, 5, 6]], [[1, 2, 3], [2, 3, 4]]]);
  let a = Learn4js.parameter(tensorA);
  let b = Learn4js.constant(Tensor.create([[1, 2, 3], [4, 5, 6]]));
  let result = a.l1Norm(0).multiply(b);

  console.log(result.value.toString());
  let expectedValue = Tensor.create([[2, 8, 18], [24, 40, 60]]);
  expect(result.value).toEqual(expectedValue);
  //
  let grads = Learn4js.gradients(result, [a, b]);
  let gradA = grads[0];
  let gradB = grads[1];

  // console.log(gradA.value.toString());
  // console.log(grads[1].value.toString());

  let expectedGradA = Tensor.create([[[-1, -2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]);
  expect(gradA.value).toEqual(expectedGradA);

  let expectedGradB = Tensor.create([[2, 4, 6], [6, 8, 10]]);
  expect(gradB.value).toEqual(expectedGradB);
});

test("l1norm axis 1", function () {

  let tensorA = Learn4js.create([[[-1, -2, 3], [4, 5, 6]], [[1, 2, 3], [2, 3, 4]]]);
  let a = Learn4js.parameter(tensorA);
  let b = Learn4js.constant(Tensor.create([[1, 2, 3], [4, 5, 6]]));
  let result = a.l1Norm(1).multiply(b);

  console.log(result.value.toString());
  // let expectedValue = Tensor.create([[2, 8, 18], [24, 40, 60]]);
  // expect(result.value).toEqual(expectedValue);
  //
  let grads = Learn4js.gradients(result, [a, b]);
  let gradA = grads[0];
  let gradB = grads[1];

  console.log(gradA.value.toString());
  console.log(grads[1].value.toString());

  // let expectedGradA = Tensor.create([[[-1, -2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]);
  // expect(gradA.value).toEqual(expectedGradA);

  // let expectedGradB = Tensor.create([[2, 4, 6], [6, 8, 10]]);
  // expect(gradB.value).toEqual(expectedGradB);
});

test("l2norm", function () {

  let tensorA = Learn4js.create([[[-1, -2, 3], [4, 5, 6]], [[1, 2, 3], [2, 3, 4]]]);
  let a = Learn4js.parameter(tensorA);
  let b = Learn4js.constant(Tensor.create([[1, 2, 3], [4, 5, 6]]));
  let result = a.l2Norm().multiply(b);

  // console.log(result.value.toString());
  let expectedValue = Tensor.create([[11.575837, 23.151674, 34.727512],
    [46.30335 , 57.879185, 69.455025]]);
  expect(result.value).toEqual(expectedValue);
  //
  let grads = Learn4js.gradients(result, [a, b]);
  let gradA = grads[0];
  let gradB = grads[1];
  //
  console.log(gradA.value.toString());
  console.log(gradB.value.toString());
  //
  // let expectedGradA = Tensor.create([[[-1.8141237, -3.6282475,  5.442371 ],
  //   [ 7.2564945,  9.070618 , 10.884742 ]],
  //
  //   [[ 1.8141236,  3.6282473,  5.442371 ],
  //     [ 3.6282473,  5.442371 ,  7.2564945]]]);
  // expect(gradA.value).toEqual(expectedGradA);
  //
  // let expectedGradB = Tensor.create([[11.575837, 11.575837, 11.575837],
  //   [11.575837, 11.575837, 11.575837]]);
  // expect(gradB.value).toEqual(expectedGradB);
});

test("infNorm", function () {

  let tensorA = Learn4js.create([[[-1, -2, 3], [4, 5, 6]], [[1, 2, 3], [2, 3, 4]]]);
  let a = Learn4js.parameter(tensorA);
  let b = Learn4js.constant(Tensor.create([[1, 2, 3], [4, 5, 6]]));
  let result = a.infNorm().multiply(b);

  console.log(result.value.toString());
  // let expectedValue = Tensor.create([[11.575837, 23.151674, 34.727512],
  //   [46.30335 , 57.879185, 69.455025]]);
  // expect(result.value).toEqual(expectedValue);
  //
  let grads = Learn4js.gradients(result, [a, b]);
  let gradA = grads[0];
  let gradB = grads[1];
  //
  console.log(gradA.value.toString());
  console.log(gradB.value.toString());
  //
  // let expectedGradA = Tensor.create([[[-1.8141237, -3.6282475,  5.442371 ],
  //   [ 7.2564945,  9.070618 , 10.884742 ]],
  //
  //   [[ 1.8141236,  3.6282473,  5.442371 ],
  //     [ 3.6282473,  5.442371 ,  7.2564945]]]);
  // expect(gradA.value).toEqual(expectedGradA);
  //
  // let expectedGradB = Tensor.create([[11.575837, 11.575837, 11.575837],
  //   [11.575837, 11.575837, 11.575837]]);
  // expect(gradB.value).toEqual(expectedGradB);
});

test("pNorm", function () {

  let tensorA = Learn4js.create([[[-1, -2, 3], [4, 5, 6]], [[1, 2, 3], [2, 3, 4]]]);
  let a = Learn4js.parameter(tensorA);
  let b = Learn4js.constant(Tensor.create([[1, 2, 3], [4, 5, 6]]));
  let result = a.pNorm(5).multiply(b);

  console.log(result.value.toString());
  // let expectedValue = Tensor.create([[11.575837, 23.151674, 34.727512],
  //   [46.30335 , 57.879185, 69.455025]]);
  // expect(result.value).toEqual(expectedValue);
  //
  let grads = Learn4js.gradients(result, [a, b]);
  let gradA = grads[0];
  let gradB = grads[1];
  //
  console.log(gradA.value.toString());
  console.log(gradB.value.toString());
  //
  // let expectedGradA = Tensor.create([[[-1.8141237, -3.6282475,  5.442371 ],
  //   [ 7.2564945,  9.070618 , 10.884742 ]],
  //
  //   [[ 1.8141236,  3.6282473,  5.442371 ],
  //     [ 3.6282473,  5.442371 ,  7.2564945]]]);
  // expect(gradA.value).toEqual(expectedGradA);
  //
  // let expectedGradB = Tensor.create([[11.575837, 11.575837, 11.575837],
  //   [11.575837, 11.575837, 11.575837]]);
  // expect(gradB.value).toEqual(expectedGradB);
});