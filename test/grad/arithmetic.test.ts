import Learn4js, {Tensor} from '../../src/index';

test('multiply', function () {

    let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
    let tensorB = Learn4js.linspace(2, 7, 6).reshape([2, 3]);
    let a = Learn4js.parameter(tensorA);
    let b = Learn4js.parameter(tensorB);
    let add = a.multiply(b);

    let expectedValue = Tensor.create([[1 * 2, 2 * 3, 3 * 4], [4 * 5, 5 * 6, 6 * 7]]);
    expect(add.value).toEqual(expectedValue);

    let grads = Learn4js.gradients(add, [a, b]);
    let gradA = grads[0];
    let gradB = grads[1];

    let expectedGradA = Tensor.create([[2, 3, 4], [5, 6, 7]]);
    let expectedGradB = Tensor.create([[1, 2, 3], [4, 5, 6]]);

    expect(gradA.value).toEqual(expectedGradA);
    expect(gradB.value).toEqual(expectedGradB);

});


test('divide', function () {

    let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
    let tensorB = Learn4js.linspace(2, 7, 6).reshape([2, 3]);
    let a = Learn4js.parameter(tensorA);
    let b = Learn4js.parameter(tensorB);
    let add = a.divide(b);

    let expectedValue = Tensor.create([[1 / 2, 2 / 3, 3 / 4], [4 / 5, 5 / 6, 6 / 7]]);
    expect(add.value).toEqual(expectedValue);

    let grads = Learn4js.gradients(add, [a, b]);
    let gradA = grads[0];
    let gradB = grads[1];

    let expectedGradA = Tensor.create([[1 / 2, 1 / 3, 1 / 4], [1 / 5, 1 / 6, 1 / 7]]);
    let expectedGradB = Tensor.create([[-1 / 2 / 2, -2 / 3 / 3, -3 / 4 / 4], [-4 / 5 / 5, -0.138888880610466, -6 / 7 / 7]]);

    // console.log(-5/6/6);
    expect(gradA.value).toEqual(expectedGradA);
    expect(gradB.value).toEqual(expectedGradB);

});

test('matmul', function () {

    let tensorA = Learn4js.linspace(1, 4, 4).reshape([2, 2]); // [[1,2], [3,4]]
    let tensorB = Learn4js.linspace(2, 5, 4).reshape([2, 2]); // [[2, 3], [4,5]]
    let a = Learn4js.parameter(tensorA);
    let b = Learn4js.parameter(tensorB);
    let add = a.matmul(b);

    let expectedValue = Tensor.create([[2 + 8, 3 + 10], [6 + 16, 9 + 20]]);
    expect(add.value).toEqual(expectedValue);

    let grads = Learn4js.gradients(add, [a, b]);
    let gradA = grads[0];
    let gradB = grads[1];

    let expectedGradA = Tensor.create([[2 + 3, 4 + 5], [2 + 3, 4 + 5]]); // [[1,1],[1,1]] @ [[2,4],[3,5]]
    let expectedGradB = Tensor.create([[1 + 3, 1 + 3], [2 + 4, 2 + 4]]); // [[1,3],[2,4]] @ [[1,1],[1,1]

    expect(gradA.value).toEqual(expectedGradA);
    expect(gradB.value).toEqual(expectedGradB);

});