import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import BinaryExpression from "./BinaryExpression";

export default class MatMul extends BinaryExpression {

  private _shape: number[];
  get shape() {
    return this._shape;
  }

  private _transposeLeft: boolean;
  get transposeLeft() {
    return this._transposeLeft;
  }

  private _transposeRight: boolean;
  get transposeRight() {
    return this._transposeRight;
  }

  get type() {
    return ExpressionTypes.MatMul;
  }

  constructor(left: Expression, right: Expression, transposeLeft: boolean, transposeRight: boolean, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._transposeLeft = transposeLeft || false;
    this._transposeRight = transposeRight || false;

    this._shape = [0, 0];
    this._shape[0] = transposeLeft ? left.shape[1] : left.shape[0];
    this._shape[1] = transposeRight ? right.shape[0] : right.shape[1];
  }

  static evaluate(node: MatMul): Tensor {
    let left = node.left.value;
    let right = node.right.value;
    return TensorMath.matmul(left, right, node.transposeLeft, node.transposeRight);
  }

  static gradients(node: MatMul, grad: Expression): Expression[] {
    let leftGrad = grad.matmul(node.right, false, true);
    let rightGrad = node.left.matmul(grad, true, false);
    return [leftGrad, rightGrad];
  }

}