import Learn4js, {Tensor} from '../../src/index';

test('reduce all', function () {

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

test('reduce 0', function () {

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

test('reduce 1', function () {

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

test('reduce 0 1', function () {

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