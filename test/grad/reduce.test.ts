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