import Learn4js, {Tensor} from "../../src/index";

test("reciprocal", function () {

  let tensorA = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let a = Learn4js.constant(tensorA);
  let y = Learn4js.constant(Tensor.scalar(-1));
  let result = a.pow(y);
  // console.log(result.value.toString());
  let expectedVal = Tensor.create([[1 / 1, 1 / 2, 1 / 3], [1 / 4, 1 / 5, 1 / 6]]);
  expect(result.value).toEqual(expectedVal);
  //
  // let grads = Learn4js.gradients(result, [a]);
  // let gradA = grads[0];
  //
  // // console.log(gradA.value.toString());
  //
  // let expected = Tensor.create([[-1, -1 / 4, -1 / 9], [-1 / 16, -1 / 25, -1 / 36]]);
  //
  // expect(gradA.value).toEqual(expected);
});

test("square", function () {

  let tensorA = Learn4js.create([[1, 2, 3], [0, -1, -2]]);
  let a = Learn4js.constant(tensorA);
  let y = Learn4js.constant(Tensor.scalar(2));
  let result = a.pow(y);
  // console.log(result.value.toString());

  let expectedVal = Learn4js.create([
    [1, 4, 9], [0, 1, 4]
  ]);
  expect(result.value).toEqual(expectedVal);
  //
  // let grads = Learn4js.gradients(result, [a]);
  // let gradA = grads[0];
  //
  // // console.log(gradA.value.toString());
  //
  // let expected = Tensor.create([
  //   [2, 4, 6],
  //   [0, -2, -4]
  // ]);
  //
  // expect(gradA.value).toEqual(expected);
});

test("sqrt", function () {

  let tensorA = Learn4js.create([[1, 2, 3], [0, 2, 4]]);
  let a = Learn4js.constant(tensorA);
  let y = Learn4js.constant(Tensor.scalar(0.5));
  let result = a.pow(y);

  // console.log(result.value.toString());

  let expectedVal = Learn4js.create([
    [Math.sqrt(1), Math.sqrt(2), Math.sqrt(3)], [0, Math.sqrt(2), 2]
  ]);
  expect(result.value).toEqual(expectedVal);
  //
  // let grads = Learn4js.gradients(result, [a]);
  // let gradA = grads[0];
  //
  // // console.log(gradA.value.toString());
  //
  // let expected = Tensor.create([
  //   [0.5 / Math.sqrt(1), 0.5 / Math.sqrt(2), 0.5 / Math.sqrt(3)],
  //   [0.5 / Math.sqrt(0), 0.5 / Math.sqrt(2), 0.5 / Math.sqrt(4)]
  // ]);
  //
  // expect(gradA.value).toEqual(expected);
});