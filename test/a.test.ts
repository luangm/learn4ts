import Learn4js from "../src/index";
import EvaluationVisitor from "../src/visitor/EvaluationVisitor";
import Session from "../src/Session";

test("AA", function () {
  let xVal = Learn4js.create([1, 2]);
  let yVal = Learn4js.create([2, 3]);

  let x = Learn4js.constant(xVal, "x");
  let y = Learn4js.constant(yVal, "y");
  let sum = Learn4js.add(x, y, "add");

  // console.log(sum);

  let evalVisitor = new EvaluationVisitor(new Session(null));
  evalVisitor.visit(sum);
});