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