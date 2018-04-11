import Learn4js from "../../src/index";

test("addN", function () {
  let aVal = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let bVal = Learn4js.linspace(2, 7, 6).reshape([2, 3]);
  let cVal = Learn4js.linspace(3, 8, 6).reshape([2, 3]);
  let x = Learn4js.constant(aVal, "x");
  let y = Learn4js.constant(bVal, "y");
  let z = Learn4js.constant(cVal, "z");

  let addN = Learn4js.addN([x, y, z]);
  console.log(addN.value);

  let customGrad = Learn4js.constant([[1, 2, 1], [2, 3, 2]]);
  let grad = Learn4js.gradients(addN, [x, y, z], customGrad);
  console.log(grad[0].value);
  console.log(grad[1].value);
  console.log(grad[2].value);
});

test("addN use in grad", function () {
  let aVal = Learn4js.linspace(1, 6, 6).reshape([2, 3]);
  let bVal = Learn4js.linspace(2, 7, 6).reshape([2, 3]);

  let x = Learn4js.constant(aVal, "x");
  let y = Learn4js.constant(bVal, "y");

  let sum = x.add(y);
  let sub = x.subtract(y);
  let mul = sum.multiply(sub);

  console.log(mul.value);

  let grad = Learn4js.gradients(mul, [x, y]);
  console.log(grad[0].value);
  console.log(grad[1].value);
});