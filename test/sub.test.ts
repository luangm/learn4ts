import Learn4js from "../src/index";

test("AA", function () {
  let xVal = Learn4js.create([-1, 2]);
  let yVal = Learn4js.create([-2, 3]);

  let x = Learn4js.constant(xVal, "x");
  let y = Learn4js.constant(yVal, "y");

  let z = Learn4js.factory.test(x, y); // (left + right) * (left - right)

  console.log(z.value);
});

test("l2norm", function () {
  let tensorA = Learn4js.create([1, 2, 3]);
  let a = Learn4js.constant(tensorA);
  let result = a.l2Norm();

  console.log(result);

  let grad = Learn4js.gradients(result,[a]);

  console.log(grad[0].value);
  // for (let [key, node] of Learn4js.activeGraph.nodes) {
  //   console.log(key, node.type);
  // }

});