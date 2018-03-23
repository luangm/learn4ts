import Learn4js from "../src/index";
import EvaluationVisitor from "../src/visitor/EvaluationVisitor";
import ReverseGradientVisitor from "../src/visitor/ReverseGradientVisitor";

test("AA", function () {
  let xVal = Learn4js.create([-1, 2]);
  let yVal = Learn4js.create([-2, 3]);

  let x = Learn4js.constant(xVal, "x");
  let y = Learn4js.constant(yVal, "y");
  let sum = Learn4js.add(x, y, "add");
  let abs = Learn4js.abs(sum);
  // console.log(sum);

  let sess = abs.graph.session;
  let evalVisitor = new EvaluationVisitor(sess);
  evalVisitor.visit(abs);

  console.log(sess.getValue(abs));

  let gradVisitor = new ReverseGradientVisitor(abs.graph);
  gradVisitor.visit(abs);

  console.log(abs.graph);
});