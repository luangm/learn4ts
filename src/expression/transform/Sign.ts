import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Sign extends TransformExpression {

  get type() {
    return ExpressionTypes.Sign;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(node: Sign): Tensor {
    let base = node.base.value;
    return TensorMath.sign(base);
  }

}