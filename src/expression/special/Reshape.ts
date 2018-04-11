import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Reshape extends Expression {

  private readonly _base: Expression;
  private readonly _shape: number[];

  get base() {
    return this._base;
  }

  get dependencies(): Expression[] {
    return [this.base];
  }

  get params() {
    return {
      type: this.type,
      name: this.name,
      base: this.base.id,
      shape: this.shape
    };
  }

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Reshape;
  }

  constructor(base: Expression, newShape: number[], graph: Graph, name?: string) {
    super(graph, name);
    this._base = base;

    let length = 1;
    for (let i = 0; i < base.shape.length; i++) {
      length *= base.shape[i];
    }

    for (let i = 0; i < newShape.length; i++) {
      if (newShape[i] == -1) {
        let prod = 1;
        for (let j = 0; j < newShape.length; j++) {
          if (j !== i) {
            prod *= newShape[j];
          }
        }
        newShape[i] = length / prod;
      }
    }
    this._shape = newShape;
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Reshape;
    let base = node.base.value;
    return base.reshape(node.shape);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Reshape;
    let baseGrad = grad.reshape(node.base.shape);
    return [baseGrad];
  }
}