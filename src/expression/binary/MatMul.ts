import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import BinaryExpression from "./BinaryExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class MatMul extends BinaryExpression {

  private readonly _shape: number[];
  private readonly _transposeLeft: boolean;
  private readonly _transposeRight: boolean;

  get shape() {
    return this._shape;
  }

  get transposeLeft() {
    return this._transposeLeft;
  }

  get transposeRight() {
    return this._transposeRight;
  }

  get type() {
    return ExpressionTypes.MatMul;
  }


  get params() {
    return {
      type: this.type,
      name: this.name,
      left: this.left.id,
      right: this.right.id,
      transposeLeft: this.transposeLeft,
      transposeRight: this.transposeRight
    }
  }

  constructor(left: Expression, right: Expression, transposeLeft: boolean, transposeRight: boolean, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._transposeLeft = transposeLeft || false;
    this._transposeRight = transposeRight || false;

    this._shape = [0, 0];
    this._shape[0] = transposeLeft ? left.shape[1] : left.shape[0];
    this._shape[1] = transposeRight ? right.shape[0] : right.shape[1];
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as MatMul;
    let left = node.left.value;
    let right = node.right.value;
    return TensorMath.matmul(left, right, node.transposeLeft, node.transposeRight);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as MatMul;
    let leftGrad = grad.matmul(node.right, false, true);
    let rightGrad = node.left.matmul(grad, true, false);
    return [leftGrad, rightGrad];
  }

}