import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Reciprocal extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.Reciprocal;
  }

  static evaluate(node: Reciprocal): Tensor {
    let base = node.base.value;
    return TensorMath.reciprocal(base);
  }

}