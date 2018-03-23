import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Tanh extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.Tanh;
  }

  static evaluate(node: Tanh): Tensor {
    let base = node.graph.session.getValue(node.base);
    return TensorMath.tanh(base);
  }

}