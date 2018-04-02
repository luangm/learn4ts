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

  let expectedVal = Tensor.create([[1, 2, 3], [1,2, 3], [4,5,6],[4,5,6]]);
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

  let expectedVal = Tensor.create([[1, 2, 3, 1,2,3], [4,5,6,4,5,6], [1, 2, 3, 1,2,3], [4,5,6,4,5,6]]);
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