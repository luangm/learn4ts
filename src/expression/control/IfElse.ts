import Graph from "../../Graph";
import Expression from "../Expression";
import {Tensor} from "tensor4js";
import {ExpressionTypes} from "../ExpressionTypes";

export default class IfElse extends Expression {

  private readonly _condition: Expression;
  private readonly _falsy: Expression;
  private readonly _truthy: Expression;

  get condition() {
    return this._condition;
  }

  /**
   * Note: the IfElse depends on condition node.
   */
  get dependencies() {
    return [this._condition];
  }

  get falsy() {
    return this._falsy;
  }

  get params() {
    return {
      type: this.type,
      name: this.name
    };
  }

  get shape(): number[] {
    return this.truthy.shape;
  }

  get truthy() {
    return this._truthy;
  }

  get type(): string {
    return ExpressionTypes.IfElse;
  }

  constructor(condition: Expression, truthy: Expression, falsy: Expression, graph: Graph, name?: string) {
    super(graph, name);
    this._condition = condition;
    this._truthy = truthy;
    this._falsy = falsy;
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as IfElse;
    let condVal = node.condition.value;
    if (condVal.data[0]) {
      return node.truthy.value;
    } else {
      return node.falsy.value;
    }
  }

}