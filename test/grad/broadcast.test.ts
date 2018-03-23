import Learn4js from '../../src/index';

test('broad add', function () {

  let tensorA = Learn4js.linspace(1, 3, 3).reshape([1, 3]);
  let tensorB = Learn4js.linspace(1, 2, 2).reshape([2, 1]);
  let a = Learn4js.parameter(tensorA);
  let b = Learn4js.parameter(tensorB);

  let add = a.add(b);
  console.log(add.value);

  let grads = Learn4js.gradients(add, [a, b]);
  let gradA = grads[0];
  let gradB = grads[1];
  //
  console.log(gradA.value);
  console.log(gradB.value);
});