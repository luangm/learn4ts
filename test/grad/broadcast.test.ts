import Learn4js, {Tensor} from '../../src/index';

test('broad add', function () {

  let tensorA = Learn4js.linspace(1, 3, 3).reshape([1, 3]); // [[1,2,3]]
  let tensorB = Learn4js.linspace(1, 2, 2).reshape([2, 1]); // [[1], [2]]
  let a = Learn4js.parameter(tensorA);
  let b = Learn4js.parameter(tensorB);
  let add = a.add(b);

  let expectedValue = Tensor.create([[2, 3, 4], [3, 4, 5]]);
  expect(add.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(add, [a, b]);
  let gradA = grads[0];
  let gradB = grads[1];

  let expectedGradA = Tensor.create([[2, 2, 2]]);
  let expectedGradB = Tensor.create([[3], [3]]);

  expect(gradA.value).toEqual(expectedGradA);
  expect(gradB.value).toEqual(expectedGradB);

});

test('broad sub', function () {

  let tensorA = Learn4js.linspace(1, 3, 3).reshape([1, 3]); // [[1,2,3]]
  let tensorB = Learn4js.linspace(1, 2, 2).reshape([2, 1]); // [[1], [2]]
  let a = Learn4js.parameter(tensorA);
  let b = Learn4js.parameter(tensorB);
  let add = a.subtract(b);

  let expectedValue = Tensor.create([[0, 1, 2], [-1, 0, 1]]);
  expect(add.value).toEqual(expectedValue);

  let grads = Learn4js.gradients(add, [a, b]);
  let gradA = grads[0];
  let gradB = grads[1];

  let expectedGradA = Tensor.create([[2, 2, 2]]);
  let expectedGradB = Tensor.create([[-3], [-3]]);

  expect(gradA.value).toEqual(expectedGradA);
  expect(gradB.value).toEqual(expectedGradB);

});