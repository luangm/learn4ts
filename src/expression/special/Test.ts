import {ShapeUtils} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import BinaryExpression from "../binary/BinaryExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Test extends BinaryExpression {

  private readonly _shape: number[];

  get params() {
    return {
      type: this.type,
      name: this.name,
      left: this.left.id,
      right: this.right.id
    };
  }

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Test;
  }

  constructor(left: Expression, right: Expression, graph: Graph, name?: string) {
    super(left, right, graph, name);
    this._shape = ShapeUtils.broadcastShapes(left.shape, right.shape);
  }

  protected buildInternal(): Expression | undefined {
    let add = this.left.add(this.right);
    let subtract = this.left.subtract(this.right);
    return add.multiply(subtract);
  }
}