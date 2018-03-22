import BinaryExpression from "./BinaryExpression";
import Expression from "../Expression";
import Graph from "../../Graph";
import {ShapeUtils, TensorMath} from "tensor4js";
import EvaluationVisitor from "../../visitor/EvaluationVisitor";

export default class Add extends BinaryExpression {

  static TYPE = "Add";

  private _shape: number[];

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._shape = ShapeUtils.broadcastShapes(left.shape, right.shape);
  }

  get shape() {
    return this._shape;
  }

  get type() {
    return Add.TYPE;
  }

  static evaluate(node: Add, visitor: EvaluationVisitor, params?: any): void {
    console.log("Add.Eval");
    if (visitor.session.isValid(node)) {
      return;
    }
    node.left.accept(visitor, params);
    node.right.accept(visitor, params);
    let left = visitor.getValue(node.left);
    let right = visitor.getValue(node.right);
    let result = TensorMath.add(left, right);
    visitor.setValue(node, result);
    console.log(result);
  }
}