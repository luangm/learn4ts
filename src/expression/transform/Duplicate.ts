import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Duplicate extends TransformExpression {

  get type() {
    return ExpressionTypes.Duplicate;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Duplicate;
    let base = node.base.value;
    return TensorMath.dup(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    return [grad];
  }
}