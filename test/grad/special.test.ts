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